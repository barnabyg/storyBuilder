"""Bounded local CPU training and atomic, inspectable reconstruction checkpoints."""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field

from nde import g1_report
from nde.corpus import g2
from nde.corpus.export import validate_export
from nde.corpus.records_io import (
    check_snapshot,
    create_record,
    evidence_files,
    file_sha,
    identity,
    read_json,
    read_sealed,
    report_files,
    save_json,
    snapshot,
)
from nde.llm.config import Strict
from nde.reconstruction.codec import (
    BOS,
    BYTE_OFFSET,
    EOS,
    MASK,
    PAD,
    VOCABULARY_SIZE,
    seal,
    tokenize,
)
from nde.reconstruction.dataset import load_dataset
from nde.reconstruction.model import build, pack, runtime, unpack
from nde.reconstruction.tasks import model_input
from nde.reconstruction.training_data import dataset_file, load_training_dataset

MIN_WIDTH, MIN_POSITIONS = 4, 8


class TrainingConfig(Strict):
    version: Literal["1"] = "1"
    dataset: Path
    base_model: Path
    objective: Literal["denoising", "ar"] = "denoising"
    seed: int = 7
    learning_rate: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    max_steps: int = Field(default=10, ge=1, le=100000)
    max_tokens: int = Field(default=1000000, ge=1)
    max_attention_cells: int = Field(default=100000000, ge=1)
    max_seconds: float = Field(default=3600, gt=0, allow_inf_nan=False)
    max_output_tokens: int = Field(default=256, ge=4)
    inference_max_tokens: int = Field(default=1000000, ge=1)
    budget_policy: Literal["padded-token-attention-caps-v1"] = "padded-token-attention-caps-v1"
    test_only: bool = False
    g1_batch: Path | None = None
    g1_realisation: list[Path] = Field(default_factory=list[Path])
    g2_gate: Path | None = None


def initialize_model(
    out: Path, *, width: int = 16, max_positions: int = 4096, seed: int = 7
) -> dict[str, Any]:
    """Create local test weights; these are not evidence of pretrained literary competence."""
    if width < MIN_WIDTH or max_positions < MIN_POSITIONS:
        raise ValueError("backbone dimensions are too small")
    model = build(width, max_positions, seed)
    out.mkdir(parents=True, exist_ok=True)
    weights = create_record(out / "weights.json", {"weights": model.state(), "test_only": True})
    return create_record(
        out / "model.json",
        {
            "architecture": "nde-single-attention-v1",
            "width": width,
            "max_positions": max_positions,
            "vocabulary_size": VOCABULARY_SIZE,
            "seed": seed,
            "test_only": True,
            "initialization": "local-random-fixture",
            "parameter_sha256": identity(weights["weights"]),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "weights_sha256": file_sha(out / "weights.json"),
        },
    )


class LocalPretraining(Strict):
    width: int = Field(ge=4)
    max_positions: int = Field(ge=8)
    model_id: str = Field(min_length=1)
    pretraining_description: str = Field(min_length=1)
    artifacts: list[Path] = Field(min_length=1)
    test_only: bool = False


def register_local_model(weights_path: Path, config_path: Path, out: Path) -> dict[str, Any]:
    """Register pretrained weights in the native architecture with bound source artifacts."""
    config = LocalPretraining.model_validate(read_json(config_path))
    weights = read_sealed(weights_path)
    if not isinstance(weights.get("test_only"), bool) or (
        weights["test_only"] and not config.test_only
    ):
        raise ValueError("fixture or unclassified base weights cannot authorize real training")
    model = build(config.width, config.max_positions, 0, weights["weights"])
    torch = runtime()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise ValueError("pretrained local weights must be finite")
    out.mkdir(parents=True, exist_ok=True)
    create_record(out / "weights.json", {"weights": model.state(), "test_only": config.test_only})
    return create_record(
        out / "model.json",
        {
            "architecture": "nde-single-attention-v1",
            "width": config.width,
            "max_positions": config.max_positions,
            "vocabulary_size": VOCABULARY_SIZE,
            "seed": 0,
            "test_only": config.test_only,
            "initialization": "registered-local-pretrained",
            "pretraining": config.model_dump(mode="json"),
            "inputs": snapshot([weights_path, config_path, *config.artifacts]),
            "parameter_sha256": identity(model.state()),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "weights_sha256": file_sha(out / "weights.json"),
        },
    )


def inspect_model(path: Path) -> dict[str, Any]:
    model = read_sealed(path / "model.json")
    check_snapshot(model.get("inputs", {}))
    if file_sha(path / "weights.json") != model["weights_sha256"]:
        raise ValueError("local base weights changed")
    weights = read_sealed(path / "weights.json")
    if identity(weights["weights"]) != model["parameter_sha256"]:
        raise ValueError("local base parameter identity changed")
    if (
        model["architecture"] != "nde-single-attention-v1"
        or model["vocabulary_size"] != VOCABULARY_SIZE
    ):
        raise ValueError("unsupported backbone or vocabulary")
    return model


def _dataset(path: Path, *, optimizer: bool = False) -> dict[str, Any]:
    return load_training_dataset(path) if optimizer else load_dataset(path)


def task_set_id(tasks: list[dict[str, Any]]) -> str:
    return identity({"task_ids": sorted(task["id"] for task in tasks)})


def dataset_tasks(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    return [task for example in dataset["examples"] for task in example["tasks"]]


def _scientific_evidence(config: TrainingConfig, dataset: dict[str, Any]) -> dict[str, Any]:
    if config.test_only:
        return {"status": "test-only mechanics", "inputs": {}}
    if config.g1_batch is None or config.g2_gate is None:
        raise ValueError("real Phase 6 training requires genuine passed G1/G2 evidence")
    first = g1_report.decision(config.g1_batch, config.g1_realisation)
    second = g2.validate_gate(config.g2_gate)
    if any(
        decision["outcome"] != "passed" or decision["test_only"] for decision in (first, second)
    ):
        raise ValueError("real Phase 6 training requires genuine passed G1/G2 evidence")
    original = dataset
    while "parent_path" in original:
        original = load_training_dataset(Path(original["parent_path"]))
    export = validate_export(Path(original["export_path"]))
    if export["gate_id"] != second["id"]:
        raise ValueError("training export differs from registered G2 gate")
    files = evidence_files(config.g1_batch) + evidence_files(config.g2_gate)
    for packet in config.g1_realisation:
        files += evidence_files(packet)
    return {
        "status": "passed",
        "g1_id": first["id"],
        "g2_id": second["id"],
        "inputs": snapshot(files),
    }


def preflight(config_path: Path) -> dict[str, Any]:
    config = TrainingConfig.model_validate(read_json(config_path))
    dataset = _dataset(config.dataset, optimizer=True)
    base = inspect_model(config.base_model)
    if bool(dataset["test_only"] or base["test_only"]) != config.test_only:
        raise ValueError("synthetic/local-random material requires explicit test_only training")
    scientific = _scientific_evidence(config, dataset)
    tasks = dataset_tasks(dataset)
    if not tasks or not any(task["targets"] for task in tasks):
        raise ValueError("training dataset has no eligible reconstruction targets")
    for task in tasks:
        if len(model_input(task)) + config.max_output_tokens > base["max_positions"]:
            raise ValueError("task exceeds local model capacity; no silent truncation")
        if len(_target_tokens(task)[0]) > config.max_output_tokens:
            raise ValueError("target exceeds registered output capacity")
    return {
        "scientific_evidence": scientific,
        "config": config.model_dump(mode="json"),
        "dataset_id": dataset["id"],
        "task_set_id": task_set_id(tasks),
        "codec_id": dataset["codec_id"],
        "vocabulary_id": dataset["vocabulary_id"],
        "base_id": base["id"],
        "initial_parameter_sha256": base["parameter_sha256"],
        "parameter_count": base["parameter_count"],
        "budget_policy": config.budget_policy,
        "inputs": snapshot(
            [
                config_path,
                dataset_file(config.dataset),
                config.base_model / "model.json",
                config.base_model / "weights.json",
            ]
        ),
    }


def _target_tokens(task: dict[str, Any]) -> tuple[list[int], list[float]]:
    """Only training calls this: a fixed output block prevents gold-length disclosure."""
    # Syntax is reliable; only value bytes inherit the channel target weight.
    tokens: list[int] = [BOS, ord("[") + BYTE_OFFSET]
    weights: list[float] = [1.0, 1.0]
    for index, target in enumerate(task["targets"]):
        prefix = ("," if index else "") + '{"position":' + str(target["position"]) + ',"value":'
        prefix_tokens = list(prefix.encode("utf-8"))
        tokens.extend(byte + BYTE_OFFSET for byte in prefix_tokens)
        weights.extend([1.0] * len(prefix_tokens))
        value = tokenize(target["value"])[1:-1]
        tokens.extend(value)
        weights.extend([float(target["weight"])] * len(value))
        tokens.append(ord("}") + BYTE_OFFSET)
        weights.append(1.0)
    tokens.extend([ord("]") + BYTE_OFFSET, EOS])
    weights.extend([1.0, 1.0])
    return tokens, weights


def _save_checkpoint(out: Path, state: dict[str, Any]) -> dict[str, Any]:
    checkpoint = {**state, "version": "1"}
    checkpoint["id"] = identity(checkpoint)
    directory = out / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"step-{state['step']:08d}.json"
    if target.exists():
        raise ValueError("immutable checkpoint already exists")
    pending = directory / f".pending-{uuid.uuid4().hex}.json"
    save_json(pending, checkpoint)
    target.hardlink_to(pending)
    pending.unlink()
    save_json(
        out / "latest.json",
        {"checkpoint": str(target.relative_to(out)), "checkpoint_id": checkpoint["id"]},
    )
    return checkpoint


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = read_sealed(path)
    check_snapshot(checkpoint["binding"]["inputs"])
    check_snapshot(checkpoint["binding"]["scientific_evidence"]["inputs"])
    binding = checkpoint["binding"]
    config = TrainingConfig.model_validate(binding["config"])
    dataset = _dataset(config.dataset, optimizer=True)
    if (
        dataset["id"] != binding["dataset_id"]
        or task_set_id(dataset_tasks(dataset)) != binding["task_set_id"]
        or inspect_model(config.base_model)["id"] != binding["base_id"]
        or _scientific_evidence(config, dataset) != binding["scientific_evidence"]
    ):
        raise ValueError("checkpoint transitive training inputs changed")
    if identity(checkpoint["model_state"]) != checkpoint["parameter_sha256"]:
        raise ValueError("checkpoint parameters changed")
    return checkpoint


def _latest(out: Path) -> tuple[Path, dict[str, Any]]:
    latest = read_json(out / "latest.json")
    path = out / latest["checkpoint"]
    checkpoint = inspect_checkpoint(path)
    if checkpoint["id"] != latest["checkpoint_id"]:
        raise ValueError("latest checkpoint identity changed")
    # A crash after publication but before pointer update must not replay optimizer work.
    for candidate in sorted((out / "checkpoints").glob("step-*.json")):
        recovered = inspect_checkpoint(candidate)
        if recovered["binding"] != checkpoint["binding"]:
            raise ValueError("checkpoint series binding changed")
        if recovered["step"] > checkpoint["step"]:
            path, checkpoint = candidate, recovered
    return path, checkpoint


def _training_state(
    binding: dict[str, Any], config: TrainingConfig, out: Path, *, resume: bool
) -> tuple[Any, dict[str, Any], Any]:
    base = inspect_model(config.base_model)
    torch = runtime()
    state: dict[str, Any]
    if resume:
        _, previous = _latest(out)
        if previous["binding"] != binding:
            raise ValueError("resume inputs/configuration changed")
        state = {key: value for key, value in previous.items() if key not in {"id", "version"}}
        model = build(base["width"], base["max_positions"], config.seed, state["model_state"])
    else:
        if (out / "latest.json").exists():
            raise ValueError("run exists; use resume")
        weights = read_sealed(config.base_model / "weights.json")["weights"]
        model = build(base["width"], base["max_positions"], config.seed, weights)
        state = {
            "binding": binding,
            "step": 0,
            "sampler_position": 0,
            "losses": [],
            "tokens": 0,
            "attention_cells": 0,
            "elapsed_seconds": 0.0,
            "model_state": model.state(),
            "optimizer_state": None,
            "rng_state": torch.get_rng_state().tolist(),
            "status": "partial",
            "parameter_sha256": identity(model.state()),
        }
        _save_checkpoint(out, state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if state["optimizer_state"] is not None:
        restored = unpack(state["optimizer_state"])
        restored["state"] = {int(key): value for key, value in restored["state"].items()}
        optimizer.load_state_dict(restored)
    torch.set_rng_state(torch.tensor(state["rng_state"], dtype=torch.uint8))
    journal = out / "work.json"
    if journal.exists():
        reserved = read_sealed(journal)
        if reserved["binding_id"] != identity(binding):
            raise ValueError("training work reservation binding changed")
        if reserved["step_intent"] > state["step"]:
            lost = max(0, reserved["tokens"] - state["tokens"])
            state["reserved_failed_tokens"] = state.get("reserved_failed_tokens", 0) + lost
            state["tokens"] = max(state["tokens"], reserved["tokens"])
            state["attention_cells"] = max(state["attention_cells"], reserved["attention_cells"])
            state["elapsed_seconds"] = max(
                state["elapsed_seconds"],
                reserved["elapsed_seconds"] + max(0.0, time.time() - reserved["started_wall_time"]),
            )
    return model, state, optimizer


def _optimizer_step(
    model: Any,
    optimizer: Any,
    context: list[int],
    target_data: tuple[list[int], list[float]],
    config: TrainingConfig,
) -> float:
    torch = runtime()
    width = config.max_output_tokens
    target, weights = target_data
    output = [MASK] * width
    if config.objective == "ar":
        output = [BOS, *target[:-1]] + [PAD] * (width - len(target))
    logits = model.forward(
        context + output, causal=config.objective == "ar", context_length=len(context)
    )
    labels = torch.tensor(target + [PAD] * (width - len(target)), dtype=torch.long)
    loss_weights = torch.tensor(weights + [0.0] * (width - len(weights)))
    if float(loss_weights.sum()) == 0:
        raise ValueError("task has no measurable weighted targets")
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    loss = (losses * loss_weights).sum() / loss_weights.sum()
    if not bool(torch.isfinite(loss)):
        raise ValueError("non-finite training loss; last checkpoint preserved")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def train(
    config_path: Path, out: Path, *, resume: bool = False, max_steps_this_call: int | None = None
) -> dict[str, Any]:
    """Train one optimizer on approved tasks, preserving budgets across resumptions."""
    if max_steps_this_call is not None and max_steps_this_call < 1:
        raise ValueError("steps-this-call must be positive")
    binding = preflight(config_path)
    config = TrainingConfig.model_validate(binding["config"])
    tasks = dataset_tasks(_dataset(config.dataset, optimizer=True))
    torch = runtime()
    torch.set_num_threads(1)
    model, state, optimizer = _training_state(binding, config, out, resume=resume)
    started = time.monotonic()
    stop_step = min(config.max_steps, state["step"] + (max_steps_this_call or config.max_steps))
    while state["step"] < stop_step:
        task = tasks[(int(state["sampler_position"]) + config.seed) % len(tasks)]
        context = model_input(task)
        target, weights = _target_tokens(task)
        width = config.max_output_tokens
        length = len(context) + width
        if (
            state["elapsed_seconds"] >= config.max_seconds
            or state["tokens"] + length > config.max_tokens
            or (state["attention_cells"] + width * length > config.max_attention_cells)
        ):
            state["status"] = "resource-stop"
            break
        save_json(
            out / "work.json",
            seal(
                {
                    "binding_id": identity(binding),
                    "step_intent": state["step"] + 1,
                    "tokens": state["tokens"] + length,
                    "attention_cells": state["attention_cells"] + width * length,
                    "elapsed_seconds": state["elapsed_seconds"],
                    "started_wall_time": time.time(),
                }
            ),
        )
        loss = _optimizer_step(model, optimizer, context, (target, weights), config)
        state["step"] += 1
        state["sampler_position"] += 1
        state["tokens"] += length
        state["attention_cells"] += width * length
        state["losses"].append(loss)
        state["model_state"] = model.state()
        state["parameter_sha256"] = identity(state["model_state"])
        state["optimizer_state"] = pack(optimizer.state_dict())
        state["rng_state"] = torch.get_rng_state().tolist()
        state["elapsed_seconds"] += time.monotonic() - started
        started = time.monotonic()
        state["status"] = "complete" if state["step"] == config.max_steps else "partial"
        _save_checkpoint(out, state)
    if state["status"] == "resource-stop":
        # Progress checkpoints remain immutable; a separate status retains the stop reason.
        save_json(out / "status.json", {"status": "resource-stop", "step": state["step"]})
    report = {
        "status": state["status"],
        "step": state["step"],
        "tokens": state["tokens"],
        "attention_cells": state["attention_cells"],
        "losses": state["losses"],
        "reserved_failed_tokens": state.get("reserved_failed_tokens", 0),
        "work_accounting": "completed work plus durable interrupted-step reservations",
        "parameter_sha256": state["parameter_sha256"],
        "binding": binding,
    }
    report_files(out, "Reconstruction training", report)
    return report


def _decode_prediction(tokens: list[int], task: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[int] = []
    for token in tokens:
        if token == EOS:
            break
        if token >= BYTE_OFFSET:
            content.append(token - BYTE_OFFSET)
        elif token not in {PAD, BOS}:
            raise ValueError("generated special token inside JSON")
    decoded = json.loads(bytes(content).decode("utf-8"))
    if not isinstance(decoded, list):
        raise ValueError("generated infill must be a replacement list")
    rows = cast(list[Any], decoded)
    if any(
        not isinstance(row, dict) or set(cast(dict[str, Any], row)) != {"position", "value"}
        for row in rows
    ):
        raise ValueError("malformed replacement entry")
    replacements = cast(list[dict[str, Any]], rows)
    positions = [row["position"] for row in replacements]
    if sorted(positions) != sorted(task["allowed_positions"]):
        raise ValueError("generated positions differ from registered prediction mask")
    return replacements


def _generate(model: Any, context: list[int], config: TrainingConfig) -> tuple[list[int], int]:
    torch = runtime()
    width = config.max_output_tokens
    generated: list[int] = []
    passes = 0
    with torch.no_grad():
        if config.objective == "ar":
            for _ in range(width):
                output = [BOS, *generated] + [PAD] * (width - len(generated) - 1)
                logits = model.forward(context + output, causal=True, context_length=len(context))
                generated.append(int(logits[len(generated)].argmax().item()))
                passes += 1
                if generated[-1] == EOS:
                    break
        else:
            logits = model.forward(
                context + [MASK] * width, causal=False, context_length=len(context)
            )
            generated = logits.argmax(dim=-1).tolist()
            passes = 1
    return generated, passes


def reconstruct(checkpoint_path: Path, dataset_path: Path, out: Path) -> dict[str, Any]:
    checkpoint = inspect_checkpoint(checkpoint_path)
    config = TrainingConfig.model_validate(checkpoint["binding"]["config"])
    dataset = _dataset(dataset_path)
    if dataset["codec_id"] != checkpoint["binding"]["codec_id"] or (
        dataset["vocabulary_id"] != checkpoint["binding"]["vocabulary_id"]
    ):
        raise ValueError("evaluation codec/vocabulary differs from checkpoint")
    tasks = dataset_tasks(dataset)
    base = inspect_model(config.base_model)
    model = build(base["width"], base["max_positions"], config.seed, checkpoint["model_state"])
    torch = runtime()
    torch.set_num_threads(1)
    tokens_used = cells_used = 0
    predictions: list[dict[str, Any]] = []
    for task in tasks:
        # The only task function called inside inference is model_input; it excludes targets.
        context = model_input(task)
        width = config.max_output_tokens
        length = len(context) + width
        passes = width if config.objective == "ar" else 1
        row: dict[str, Any] = {
            "task_id": task["id"],
            "status": "invalid",
            "replacements": [],
            "diagnostics": [],
        }
        if tokens_used + length * passes > config.inference_max_tokens:
            row["status"] = "resource-stop"
            row["diagnostics"] = ["registered inference token cap exhausted"]
            predictions.append(row)
            continue
        if length > model.max_positions:
            row["diagnostics"] = ["evaluation task exceeds position capacity"]
            predictions.append(row)
            continue
        generated, actual_passes = _generate(model, context, config)
        tokens_used += length * actual_passes
        cells_used += width * length * actual_passes
        try:
            row["replacements"] = _decode_prediction(generated, task)
            row["status"] = "complete"
        except (ValueError, UnicodeDecodeError, TypeError) as error:
            row["diagnostics"] = [str(error)]
        row["generated_tokens"] = generated
        predictions.append(row)
    matching = {
        key: checkpoint["binding"][key]
        for key in (
            "base_id",
            "initial_parameter_sha256",
            "parameter_count",
            "dataset_id",
            "task_set_id",
            "vocabulary_id",
            "budget_policy",
        )
    }
    matching["seed"] = config.seed
    matching["training_budget"] = {
        key: config.model_dump()[key]
        for key in (
            "max_steps",
            "max_tokens",
            "max_attention_cells",
            "max_seconds",
            "max_output_tokens",
            "learning_rate",
        )
    }
    matching["inference_budget"] = {"max_tokens": config.inference_max_tokens}
    result = create_record(
        out / "predictions.json",
        {
            "kind": "reconstruction-predictions",
            "dataset_id": dataset["id"],
            "task_set_id": task_set_id(tasks),
            "codec_id": dataset["codec_id"],
            "vocabulary_id": dataset["vocabulary_id"],
            "checkpoint_id": checkpoint["id"],
            "checkpoint_path": str(checkpoint_path.resolve()),
            "objective": config.objective,
            "seed": config.seed,
            "training_config_id": identity(config.model_dump(mode="json")),
            "matching": matching,
            "test_only": config.test_only or dataset["test_only"],
            "work": {
                "training_steps": checkpoint["step"],
                "training_tokens": checkpoint["tokens"],
                "reserved_failed_training_tokens": checkpoint.get("reserved_failed_tokens", 0),
                "training_attention_cells": checkpoint["attention_cells"],
                "training_elapsed_seconds": checkpoint["elapsed_seconds"],
                "inference_tokens": tokens_used,
                "inference_attention_cells": cells_used,
            },
            "predictions": predictions,
            "inputs": snapshot([checkpoint_path, dataset_file(dataset_path)]),
        },
    )
    report_files(out, "Reconstruction predictions", result)
    return result


def inspect_attention(
    base_path: Path, visible: list[int], infill: list[int], *, objective: Literal["denoising", "ar"]
) -> list[list[float]]:
    """Inspect native backbone logits to verify objective attention without training."""
    base = inspect_model(base_path)
    if not infill or any(token < 0 or token >= VOCABULARY_SIZE for token in [*visible, *infill]):
        raise ValueError("attention probe requires valid visible/infill token IDs")
    state = read_sealed(base_path / "weights.json")["weights"]
    model = build(base["width"], base["max_positions"], 0, state)
    with runtime().no_grad():
        return model.forward(
            visible + infill, causal=objective == "ar", context_length=len(visible)
        ).tolist()


def matched_configs(candidate: Path, comparator: Path) -> dict[str, Any]:
    left, right = preflight(candidate), preflight(comparator)
    if {left["config"]["objective"], right["config"]["objective"]} != {"denoising", "ar"}:
        raise ValueError("matched comparison requires denoising and AR")
    for key in (
        "dataset_id",
        "task_set_id",
        "codec_id",
        "vocabulary_id",
        "base_id",
        "initial_parameter_sha256",
        "parameter_count",
        "budget_policy",
    ):
        if left[key] != right[key]:
            raise ValueError(f"unmatched {key}")
    left_config = {k: v for k, v in left["config"].items() if k != "objective"}
    right_config = {k: v for k, v in right["config"].items() if k != "objective"}
    if left_config != right_config:
        raise ValueError("unmatched training/inference resource configuration")
    return {"matched": True, "policy": left["budget_policy"], "actual_work_required": True}


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="training_command", required=True)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--out", type=Path, required=True)
    initialize.add_argument("--width", type=int, default=16)
    initialize.add_argument("--max-positions", type=int, default=4096)
    initialize.add_argument("--seed", type=int, default=7)
    local = commands.add_parser("register-local")
    local.add_argument("weights", type=Path)
    local.add_argument("config", type=Path)
    local.add_argument("--out", type=Path, required=True)
    for name in ("preflight", "train", "resume"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        if name != "preflight":
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--steps-this-call", type=int)
    reconstruction = commands.add_parser("reconstruct")
    reconstruction.add_argument("checkpoint", type=Path)
    reconstruction.add_argument("dataset", type=Path)
    reconstruction.add_argument("--out", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("checkpoint", type=Path)
    match = commands.add_parser("match")
    match.add_argument("candidate", type=Path)
    match.add_argument("comparator", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        if args.training_command == "initialize":
            result = initialize_model(
                args.out, width=args.width, max_positions=args.max_positions, seed=args.seed
            )
        elif args.training_command == "register-local":
            result = register_local_model(args.weights, args.config, args.out)
        elif args.training_command == "preflight":
            result = preflight(args.config)
        elif args.training_command in {"train", "resume"}:
            result = train(
                args.config,
                args.out,
                resume=args.training_command == "resume",
                max_steps_this_call=args.steps_this_call,
            )
        elif args.training_command == "reconstruct":
            result = reconstruct(args.checkpoint, args.dataset, args.out)
        elif args.training_command == "match":
            result = matched_configs(args.candidate, args.comparator)
        else:
            result = inspect_checkpoint(args.checkpoint)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"nde reconstruction-train: {error}", file=sys.stderr)
        return 2
