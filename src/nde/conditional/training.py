"""Local full-layout denoising, separate from Phase 6 reconstruction contracts."""

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field

from nde.conditional.data import CONDITION_ID, Conditions, load_dataset
from nde.conditional.layout import LAYOUT_ID, Capacity, decode_layout, encode_target
from nde.corpus.records_io import (
    check_snapshot,
    create_record,
    identity,
    read_json,
    read_sealed,
    report_files,
    save_json,
    snapshot,
)
from nde.llm.config import Strict
from nde.reconstruction import g3
from nde.reconstruction.codec import EOS, MASK, PAD, seal, tokenize
from nde.reconstruction.model import build, pack, runtime, unpack
from nde.reconstruction.resources import TimeAllowance
from nde.reconstruction.training import inspect_model

REPRESENTATION = "nde-conditional-denoising-v1"
GUIDANCE_POLICY = "classifier-free-logit-interpolation-v1"
MAX_GUIDANCE = 8


class TrainingConfig(Strict):
    version: Literal["1"] = "1"
    dataset: Path
    base_model: Path
    capacity: Capacity = Field(default_factory=Capacity)
    adaptation: Literal["native-full-finetune"] = "native-full-finetune"
    condition_drop_probability: float = Field(default=0.5, gt=0, lt=1, allow_inf_nan=False)
    max_condition_tokens: int = Field(default=1024, ge=64, le=8192)
    sampling_temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    seed: int = 7
    learning_rate: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    max_steps: int = Field(default=10, ge=1, le=100000)
    max_tokens: int = Field(default=1000000, ge=1)
    max_attention_cells: int = Field(default=100000000, ge=1)
    max_seconds: float = Field(default=3600, gt=0, allow_inf_nan=False)
    inference_max_tokens: int = Field(default=1000000, ge=1)
    inference_max_attention_cells: int = Field(default=100000000, ge=1)
    inference_max_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    test_only: bool = False
    g3_protocol: Path | None = None


def _dataset(path: Path) -> dict[str, Any]:
    return load_dataset(path, optimizer=True)


def _context(config: TrainingConfig, conditions: Conditions | None) -> list[int]:
    tokens = tokenize(
        {
            "representation": REPRESENTATION,
            "capacity": config.capacity.model_dump(),
            "conditions": conditions.model_dump(mode="json") if conditions is not None else None,
        }
    )
    if len(tokens) > config.max_condition_tokens:
        raise ValueError("conditions exceed registered context capacity; truncation forbidden")
    return tokens + [PAD] * (config.max_condition_tokens - len(tokens))


def _scientific(config: TrainingConfig) -> dict[str, Any]:
    if config.test_only:
        return {"status": "test-only mechanics", "inputs": {}}
    if config.g3_protocol is None:
        raise ValueError("real conditional training requires genuine passed G3 evidence")
    decision = g3.assess(config.g3_protocol)
    if decision["test_only"] or decision["outcome"] != "passed":
        raise ValueError("real conditional training requires genuine passed G3 evidence")
    return {"status": "passed", "decision_id": decision["id"], "inputs": decision["inputs"]}


def preflight(config_path: Path) -> dict[str, Any]:
    config = TrainingConfig.model_validate(read_json(config_path))
    dataset = _dataset(config.dataset)
    base = inspect_model(config.base_model)
    if bool(dataset["test_only"] or base["test_only"]) != config.test_only:
        raise ValueError("fixture data/base requires explicit test_only training")
    layouts = [encode_target(row["target"], config.capacity) for row in dataset["examples"]]
    if not layouts:
        raise ValueError("conditional dataset has no eligible training examples")
    for example in dataset["examples"]:
        _context(config, Conditions.model_validate(example["conditions"]))
    if config.max_condition_tokens + config.capacity.max_tokens > base["max_positions"]:
        raise ValueError("registered layout capacity exceeds base positions; truncation forbidden")
    path = config.dataset / "dataset.json" if config.dataset.is_dir() else config.dataset
    return {
        "representation": REPRESENTATION,
        "condition_id": CONDITION_ID,
        "guidance_policy": GUIDANCE_POLICY,
        "training_condition_policy": (
            "Bernoulli dropout of entire condition block; all output masked"
        ),
        "layout_id": LAYOUT_ID,
        "dataset_id": dataset["id"],
        "target_set_id": identity({"layouts": [row["id"] for row in layouts]}),
        "base_id": base["id"],
        "initial_parameter_sha256": base["parameter_sha256"],
        "parameter_count": base["parameter_count"],
        "width": base["width"],
        "max_positions": base["max_positions"],
        "architecture": base["architecture"],
        "adaptation": config.adaptation,
        "config": config.model_dump(mode="json"),
        "config_path": str(config_path.resolve()),
        "scientific_evidence": _scientific(config),
        "inputs": snapshot(
            [
                config_path,
                path,
                config.base_model / "model.json",
                config.base_model / "weights.json",
            ]
        ),
    }


def inspect_checkpoint(path: Path, *, validate_inputs: bool = False) -> dict[str, Any]:
    checkpoint = read_sealed(path)
    if checkpoint.get("kind") != "conditional-checkpoint":
        raise ValueError(
            "not a conditional checkpoint; reconstruction checkpoints require adaptation"
        )
    binding = checkpoint["binding"]
    if (
        binding.get("architecture") != "nde-single-attention-v1"
        or binding.get("adaptation") != "native-full-finetune"
    ):
        raise ValueError("unsupported conditional checkpoint architecture/adaptation")
    if (
        binding.get("representation") != REPRESENTATION
        or binding.get("layout_id") != LAYOUT_ID
        or binding.get("condition_id") != CONDITION_ID
        or binding.get("guidance_policy") != GUIDANCE_POLICY
    ):
        raise ValueError("unsupported conditional representation")
    TrainingConfig.model_validate(binding["config"])
    if validate_inputs:
        check_snapshot(binding["inputs"])
        check_snapshot(binding["scientific_evidence"]["inputs"])
        if preflight(Path(binding["config_path"])) != binding:
            raise ValueError("checkpoint transitive inputs changed")
    if identity(checkpoint["model_state"]) != checkpoint["parameter_sha256"]:
        raise ValueError("checkpoint model parameters changed")
    return checkpoint


def _save_checkpoint(out: Path, state: dict[str, Any]) -> dict[str, Any]:
    path = out / "checkpoints" / f"step-{state['step']:08d}.json"
    pending = path.with_suffix(".pending")
    if path.exists() or pending.exists():
        raise ValueError("immutable checkpoint already exists or publication interrupted")
    checkpoint = seal(state)
    pending.parent.mkdir(parents=True, exist_ok=True)
    save_json(pending, checkpoint)
    path.hardlink_to(pending)
    pending.unlink()
    save_json(
        out / "latest.json", {"checkpoint": str(path.relative_to(out)), "id": checkpoint["id"]}
    )
    return checkpoint


def _restore(
    binding: dict[str, Any], config: TrainingConfig, out: Path, *, resume: bool
) -> tuple[Any, Any, dict[str, Any]]:
    torch = runtime()
    base = inspect_model(config.base_model)
    state: dict[str, Any]
    if resume:
        candidates = sorted((out / "checkpoints").glob("step-*.json"))
        if not candidates:
            raise ValueError("no checkpoint to resume")
        previous = inspect_checkpoint(candidates[-1], validate_inputs=True)
        if previous["binding"] != binding:
            raise ValueError("resume configuration/condition/guidance inputs changed")
        state = {key: value for key, value in previous.items() if key not in {"id", "version"}}
        model_state = state["model_state"]
    else:
        if (out / "latest.json").exists() or (out / "checkpoints").exists():
            raise ValueError("training run exists; use resume")
        model_state = read_sealed(config.base_model / "weights.json")["weights"]
        state = {
            "kind": "conditional-checkpoint",
            "binding": binding,
            "step": 0,
            "sampler_position": 0,
            "tokens": 0,
            "attention_cells": 0,
            "elapsed_seconds": 0.0,
            "losses": [],
            "conditioning_modes": [],
            "optimizer_state": None,
            "rng_state": None,
            "reserved_failed_tokens": 0,
            "reserved_failed_attention_cells": 0,
        }
    model = build(base["width"], base["max_positions"], config.seed, model_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if state["optimizer_state"] is not None:
        optimizer_state = unpack(state["optimizer_state"])
        optimizer_state["state"] = {
            int(key): value for key, value in optimizer_state["state"].items()
        }
        optimizer.load_state_dict(optimizer_state)
        torch.set_rng_state(torch.tensor(state["rng_state"], dtype=torch.uint8))
    else:
        state["model_state"] = model.state()
        state["parameter_sha256"] = identity(state["model_state"])
        state["optimizer_state"] = pack(optimizer.state_dict())
        state["rng_state"] = torch.get_rng_state().tolist()
        state["status"] = "partial"
        _save_checkpoint(out, state)
    return model, optimizer, state


def _recover_work(out: Path, binding: dict[str, Any], state: dict[str, Any]) -> None:
    work_path = out / "work.json"
    if work_path.exists():
        pending = read_sealed(work_path)
        if pending["binding_id"] != identity(binding):
            raise ValueError("training work reservation changed")
        if pending["step"] > state["step"]:
            for field in ("tokens", "attention_cells"):
                key = "reserved_failed_" + field
                state[key] = max(state[key], pending[field] - state[field])


def _update(
    model: Any, optimizer: Any, state: dict[str, Any], batch: dict[str, Any], timer: TimeAllowance
) -> None:
    torch = runtime()
    context, width, target = batch["context"], batch["width"], batch["target"]
    optimizer.zero_grad()
    logits = model.forward(context + [MASK] * width, causal=False, context_length=len(context))
    labels = torch.tensor(target + [PAD] * (width - len(target)), dtype=torch.long)
    weights = torch.tensor(batch["weights"] + [1.0] * (width - len(target)), dtype=torch.float32)
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    loss = (losses * weights).sum() / weights.sum()
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite conditional objective; interrupted work reservation retained")
    loss.backward()
    optimizer.step()
    state["step"] += 1
    state["sampler_position"] += 1
    state["tokens"] += batch["length"]
    state["attention_cells"] += batch["cells"]
    state["losses"].append(float(loss.item()))
    state["conditioning_modes"].append(batch["conditioning_mode"])
    state["elapsed_seconds"] = timer.elapsed()
    state["model_state"] = model.state()
    state["parameter_sha256"] = identity(state["model_state"])
    state["optimizer_state"] = pack(optimizer.state_dict())
    state["rng_state"] = torch.get_rng_state().tolist()


def train(
    config_path: Path, out: Path, *, resume: bool = False, max_steps_this_call: int | None = None
) -> dict[str, Any]:
    started = time.monotonic()
    if max_steps_this_call is not None and max_steps_this_call < 1:
        raise ValueError("steps-this-call must be positive")
    binding = preflight(config_path)
    config = TrainingConfig.model_validate(binding["config"])
    dataset = _dataset(config.dataset)
    torch = runtime()
    torch.set_num_threads(1)
    model, optimizer, state = _restore(binding, config, out, resume=resume)
    timer = TimeAllowance(
        out / "training-time.json",
        identity(binding),
        config.max_seconds,
        started,
        elapsed_floor=state["elapsed_seconds"],
    )
    _recover_work(out, binding, state)
    stop = min(config.max_steps, state["step"] + (max_steps_this_call or config.max_steps))
    state["status"] = "complete" if state["step"] == config.max_steps else "partial"
    while state["step"] < stop:
        example = dataset["examples"][state["sampler_position"] % len(dataset["examples"])]
        layout = encode_target(example["target"], config.capacity)
        target = layout["tokens"]
        drop = float(torch.rand(1).item()) < config.condition_drop_probability
        conditions = None if drop else Conditions.model_validate(example["conditions"])
        context = _context(config, conditions)
        width = config.capacity.max_tokens
        length = len(context) + width
        cells = width * length
        if (
            timer.expired()
            or state["tokens"] + state["reserved_failed_tokens"] + length > config.max_tokens
            or state["attention_cells"] + state["reserved_failed_attention_cells"] + cells
            > config.max_attention_cells
        ):
            state["status"] = "resource-stop"
            break
        save_json(
            out / "work.json",
            seal(
                {
                    "binding_id": identity(binding),
                    "step": state["step"] + 1,
                    "tokens": state["tokens"] + state["reserved_failed_tokens"] + length,
                    "attention_cells": state["attention_cells"]
                    + state["reserved_failed_attention_cells"]
                    + cells,
                }
            ),
        )
        timer.save(tokens=state["tokens"], cells=state["attention_cells"])
        _update(
            model,
            optimizer,
            state,
            {
                "context": context,
                "target": target,
                "weights": layout["loss_weights"],
                "width": width,
                "length": length,
                "cells": cells,
                "conditioning_mode": "unconditional" if drop else "conditional",
            },
            timer,
        )
        state["status"] = "complete" if state["step"] == config.max_steps else "partial"
        if timer.expired():
            state["status"] = "resource-stop"
        _save_checkpoint(out, state)
        timer.save(tokens=state["tokens"], cells=state["attention_cells"], status="complete")
        if state["status"] == "resource-stop":
            break
    timer.save(tokens=state["tokens"], cells=state["attention_cells"], status="complete")
    report = {
        key: value
        for key, value in state.items()
        if key not in {"model_state", "optimizer_state", "rng_state"}
    }
    report_files(out, "Conditional layout training", report)
    return report


def candidate_cost(
    checkpoint: dict[str, Any], conditions: Conditions, guidance_scale: float = 1.0
) -> tuple[int, int]:
    if not math.isfinite(guidance_scale) or not 0 <= guidance_scale <= MAX_GUIDANCE:
        raise ValueError("guidance scale must be finite and between 0 and 8")
    if checkpoint["step"] < 1 or (
        checkpoint["parameter_sha256"] == checkpoint["binding"]["initial_parameter_sha256"]
    ):
        raise ValueError("conditional sampling requires actual trained parameter updates")
    modes = set(checkpoint["conditioning_modes"])
    if "conditional" not in modes:
        raise ValueError("conditional sampling requires recorded conditional training exposure")
    if guidance_scale != 1 and "unconditional" not in modes:
        raise ValueError("dropped/guided sampling requires unconditional training exposure")
    config = TrainingConfig.model_validate(checkpoint["binding"]["config"])
    _context(config, Conditions.model_validate(conditions))
    passes = 1 if guidance_scale in {0.0, 1.0} else 2
    length = config.max_condition_tokens + config.capacity.max_tokens
    return length * passes, config.capacity.max_tokens * length * passes


def _guided_logits(
    model: Any,
    context_for: Callable[[bool], list[int]],
    guidance_scale: float,
    deadline: float,
    output: list[int],
) -> tuple[Any, int, dict[str, str]]:
    conditional: Any = None
    unconditional: Any = None
    passes = 0
    fingerprints: dict[str, str] = {}
    modes = [True] if guidance_scale == 1 else [False] if guidance_scale == 0 else [False, True]
    for conditioned in modes:
        if time.monotonic() >= deadline:
            break
        context = context_for(conditioned)
        logits = model.forward(context + output, causal=False, context_length=len(context))
        passes += 1
        fingerprints["conditional" if conditioned else "unconditional"] = hashlib.sha256(
            logits.detach().cpu().numpy().tobytes()
        ).hexdigest()
        if conditioned:
            conditional = logits
        else:
            unconditional = logits
    if passes != len(modes):
        return None, passes, fingerprints
    if guidance_scale == 0:
        return unconditional, passes, fingerprints
    if guidance_scale == 1:
        return conditional, passes, fingerprints
    return unconditional + guidance_scale * (conditional - unconditional), passes, fingerprints


# PLR0913: keyword-only sampling controls shared by full generation and inpainting.
def sample_candidate(  # noqa: PLR0913
    checkpoint_path: Path,
    conditions: Conditions | None = None,
    *,
    seed: int = 7,
    guidance_scale: float = 1.0,
    deadline: float | None = None,
    observed: list[int] | None = None,
) -> dict[str, Any]:
    """Denoise MASK positions; `observed` output tokens are clamped for inpainting."""
    started = time.monotonic()
    conditions = Conditions.model_validate(conditions or {})
    checkpoint = inspect_checkpoint(checkpoint_path)
    config = TrainingConfig.model_validate(checkpoint["binding"]["config"])
    binding = checkpoint["binding"]
    tokens_required, cells_required = candidate_cost(checkpoint, conditions, guidance_scale)
    deadline = min(
        deadline if deadline is not None else float("inf"), started + config.inference_max_seconds
    )
    width = config.capacity.max_tokens
    length = config.max_condition_tokens + width
    output = [MASK] * width if observed is None else observed
    if len(output) != width:
        raise ValueError("observed output must fill the registered capacity exactly")
    result: dict[str, Any] = {
        "status": "invalid",
        "view": None,
        "generated_tokens": [],
        "diagnostics": [],
        "checkpoint_id": checkpoint["id"],
        "layout_id": LAYOUT_ID,
        "condition_id": CONDITION_ID,
        "conditions": conditions.model_dump(mode="json"),
        "guidance_scale": guidance_scale,
        "guidance_policy": GUIDANCE_POLICY,
        "seed": seed,
        "work": {"tokens": 0, "attention_cells": 0, "passes": 0, "elapsed_seconds": 0.0},
    }
    if (
        time.monotonic() >= deadline
        or tokens_required > config.inference_max_tokens
        or cells_required > config.inference_max_attention_cells
    ):
        result.update(
            status="resource-stop", diagnostics=["registered inference work cap exceeded"]
        )
        return result
    torch = runtime()
    torch.set_num_threads(1)
    model = build(binding["width"], binding["max_positions"], seed, checkpoint["model_state"])
    with torch.no_grad():
        logits, passes, fingerprints = _guided_logits(
            model,
            lambda conditioned: _context(config, conditions if conditioned else None),
            guidance_scale,
            deadline,
            output,
        )
        tokens: list[int]
        if logits is None:
            tokens = []
        elif config.sampling_temperature:
            probabilities = torch.softmax(logits / config.sampling_temperature, dim=-1)
            tokens = cast(list[int], torch.multinomial(probabilities, 1).squeeze(-1).tolist())
        else:
            tokens = cast(list[int], logits.argmax(dim=-1).tolist())
    if tokens:
        tokens = [
            token if known == MASK else known for token, known in zip(tokens, output, strict=True)
        ]
    elapsed = time.monotonic() - started
    result["generated_tokens"] = tokens
    result["pass_logit_sha256"] = fingerprints
    result["work"] = {
        "tokens": length * passes,
        "attention_cells": width * length * passes,
        "passes": passes,
        "elapsed_seconds": elapsed,
    }
    try:
        if time.monotonic() >= deadline or logits is None:
            result["status"] = "resource-stop"
            raise ValueError("registered inference time cap exceeded")
        if EOS not in tokens:
            raise ValueError("layout generation did not terminate with EOS")
        view = decode_layout(tokens[: tokens.index(EOS) + 1], config.capacity)
        if conditions.scene_range is not None and not (
            conditions.scene_range[0] <= len(view["scenes"]) <= conditions.scene_range[1]
        ):
            raise ValueError("sampled scene count falls outside caller-permitted range")
        result.update(status="complete", view=view)
    except (ValueError, UnicodeDecodeError, TypeError) as error:
        result["diagnostics"] = [str(error)]
    return result


def sample(
    checkpoint_path: Path,
    out: Path,
    *,
    conditions: Conditions | None = None,
    seed: int = 7,
    guidance_scale: float = 1.0,
) -> dict[str, Any]:
    started = time.monotonic()
    if (out / "sample.json").exists():
        raise ValueError("immutable sample already exists; choose a new output directory")
    conditions = Conditions.model_validate(conditions or {})
    checkpoint = inspect_checkpoint(checkpoint_path)
    config = TrainingConfig.model_validate(checkpoint["binding"]["config"])
    tokens, cells = candidate_cost(checkpoint, conditions, guidance_scale)
    timer = TimeAllowance(
        out / "inference-time.json",
        identity(
            {
                "checkpoint_id": checkpoint["id"],
                "conditions": conditions.model_dump(mode="json"),
                "seed": seed,
                "guidance_scale": guidance_scale,
            }
        ),
        config.inference_max_seconds,
        started,
    )
    before_tokens, before_cells = timer.tokens_before, timer.cells_before
    if (
        before_tokens + tokens > config.inference_max_tokens
        or before_cells + cells > config.inference_max_attention_cells
    ):
        result = {
            "status": "resource-stop",
            "view": None,
            "generated_tokens": [],
            "diagnostics": ["durable inference work allowance exhausted"],
            "work": {
                "tokens": before_tokens,
                "attention_cells": before_cells,
                "elapsed_seconds": timer.elapsed(),
                "passes": 0,
            },
        }
    else:
        # Reservation survives an interrupted forward pass; recovery retains the allowance.
        timer.save(tokens=before_tokens + tokens, cells=before_cells + cells)
        result = sample_candidate(
            checkpoint_path,
            conditions,
            seed=seed,
            guidance_scale=guidance_scale,
            deadline=timer.deadline,
        )
        result["work"]["tokens"] += before_tokens
        result["work"]["attention_cells"] += before_cells
    result = create_record(out / "sample.json", result)
    timer.save(
        tokens=result["work"]["tokens"], cells=result["work"]["attention_cells"], status="complete"
    )
    report_files(out, "Conditional learned layout proposal", result)
    return result


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="conditional_training_command", required=True)
    for name in ("preflight", "train", "resume"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        if name != "preflight":
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--steps-this-call", type=int)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("checkpoint", type=Path)
    sampling = commands.add_parser("sample")
    sampling.add_argument("checkpoint", type=Path)
    sampling.add_argument("--out", type=Path, required=True)
    sampling.add_argument("--seed", type=int, default=7)
    sampling.add_argument("--conditions", type=Path)
    sampling.add_argument("--guidance-scale", type=float, default=1.0)


def run(args: argparse.Namespace) -> int:
    try:
        if args.conditional_training_command == "preflight":
            result = preflight(args.config)
        elif args.conditional_training_command in {"train", "resume"}:
            result = train(
                args.config,
                args.out,
                resume=args.conditional_training_command == "resume",
                max_steps_this_call=args.steps_this_call,
            )
        elif args.conditional_training_command == "sample":
            conditions = (
                Conditions.model_validate(read_json(args.conditions)) if args.conditions else None
            )
            result = sample(
                args.checkpoint,
                args.out,
                conditions=conditions,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
            )
        else:
            result = inspect_checkpoint(args.checkpoint)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"nde conditional-train: {error}", file=sys.stderr)
        return 2
