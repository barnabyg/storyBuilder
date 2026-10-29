"""Fill an explicit text manifest without granting structural edit authority."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from pydantic import Field, field_validator

from nde.conditional.generation import read_proposal
from nde.corpus.records_io import (
    check_snapshot,
    create_record,
    identity,
    read_sealed,
    report_files,
    save_json,
    snapshot,
)
from nde.files import write_text_lf
from nde.llm.cache import ResponseCache
from nde.llm.client import Message
from nde.llm.config import ModelBinding, ProviderConfig, Strict
from nde.llm.errors import ProviderError
from nde.llm.gateway import Completion, Gateway, Prompt
from nde.llm.ledger import BudgetExceededError, CostLedger, Price, Pricing
from nde.llm.structured import parse, repair_messages, response_format, with_schema_instruction
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import rendered_files
from nde.plan.revision import changed_units, selector_value

PROMPT = (
    "Complete only the requested text fields of this generated fictional plan. "
    "Return an object with one key, fields, mapping exactly the requested selectors "
    "to nonempty text. Do not add evidence, entities, references, approval, provenance, "
    "or structural changes. Preserve the represented meaning and uncertainty. "
    "This is generated fiction, not a source annotation or an owner-approved reading."
)
PLACEHOLDER = "Unfilled generated text"
MAX_TEXT_CHARACTERS = 16000
MAX_PROVIDER_RETRIES = 3


class CompletionConfig(Strict):
    proposal: Path
    fields: list[str] = Field(min_length=1)
    provider: ProviderConfig
    model: ModelBinding
    price: Price
    max_repairs: int = Field(default=2, ge=0, le=8)
    seed: int = 7
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    max_cost_usd: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    cache: Path | None = None
    ledger: Path | None = None


class TextCompletion(Strict):
    fields: dict[str, str]

    @field_validator("fields")
    @classmethod
    def _nonempty(cls, fields: dict[str, str]) -> dict[str, str]:
        if any(not value.strip() or len(value) > MAX_TEXT_CHARACTERS for value in fields.values()):
            raise ValueError("completion text must be nonempty and at most 16000 characters")
        if any(value.strip() == PLACEHOLDER for value in fields.values()):
            raise ValueError("completion must replace placeholder text")
        return fields


def _has_evidence(value: Any) -> bool:
    if isinstance(value, dict):
        content = cast(dict[str, Any], value)
        return bool(content.get("evidence")) or any(_has_evidence(v) for v in content.values())
    if isinstance(value, list):
        return any(_has_evidence(v) for v in cast(list[Any], value))
    return False


def replace_text(content: dict[str, Any], selector: str, value: str) -> None:
    container: Any = content
    parts = selector.split("/")
    for part in parts[:-1]:
        if isinstance(container, dict):
            container = cast(dict[str, Any], container)[part]
        else:
            container = next(
                row for row in cast(list[dict[str, Any]], container) if row["id"] == part
            )
    if isinstance(container, list):
        cast(list[str], container)[int(parts[-1])] = value
    else:
        container[parts[-1]] = value


def _unauthorized_changes(
    before: NarrativePlan, after: NarrativePlan, fields: list[str]
) -> set[str]:
    unauthorized = set(changed_units(before, after)) - set(fields)
    alternative_lists = {
        field.rsplit("/", 1)[0] for field in fields if "/reading/alternatives/" in field
    }
    for parent in alternative_lists:
        previous = cast(list[str], selector_value(before, parent))
        current = cast(list[str], selector_value(after, parent))
        if len(previous) == len(current) and all(
            original == changed or f"{parent}/{index}" in fields
            for index, (original, changed) in enumerate(zip(previous, current, strict=True))
        ):
            unauthorized.discard(parent)
    return unauthorized


def _candidate(
    text: str, before: NarrativePlan, config: CompletionConfig
) -> tuple[NarrativePlan, TextCompletion]:
    completed = parse(text, TextCompletion)
    if set(completed.fields) != set(config.fields):
        raise ValueError("response fields must exactly match authorized selectors")
    content = copy.deepcopy(before.model_dump(mode="json"))
    for selector, value in completed.fields.items():
        replace_text(content, selector, value)
    candidate = NarrativePlan.model_validate(content)
    unauthorized = _unauthorized_changes(before, candidate, config.fields)
    if unauthorized or candidate.sources or _has_evidence(content):
        raise ValueError(
            f"completion changed unauthorized structure/evidence: {sorted(unauthorized)}"
        )
    validation = validate_plan(candidate)
    if not validation.valid:
        raise ValueError(validation.model_dump_json())
    return candidate, completed


def _call_record(call: Completion) -> dict[str, Any]:
    return {
        "model": call.binding.model_dump(mode="json"),
        "reported_model": call.result.model,
        "prompt_id": identity({"messages": call.prompt.messages}),
        "seed": call.seed,
        "cached": call.cached,
        "input_tokens": call.result.input_tokens,
        "output_tokens": call.result.output_tokens,
        "cost_usd": call.cost_usd,
        "retries": call.retries,
        "latency_seconds": call.result.latency_s,
        "finish_reason": call.result.finish_reason,
    }


def _gateway(config: CompletionConfig, out: Path) -> Gateway:
    params = dict(config.model.params)
    if params.get("stream") or params.get("n", 1) != 1:
        raise ValueError("completion requires a single non-streaming gateway response")
    if config.provider.retry.max_retries > MAX_PROVIDER_RETRIES:
        raise ValueError("completion provider retries must be bounded at three")
    return Gateway(
        providers={config.model.provider: config.provider},
        environ=os.environ,
        cache=ResponseCache(config.cache or out / "cache.sqlite"),
        pricing=Pricing({config.model.label: config.price}),
        ledger=CostLedger(
            config.ledger or out / "ledger.sqlite",
            per_run_usd=config.max_cost_usd,
            run_id=identity({"output": str(out.resolve())}),
        ),
    )


def _save_artifacts(plan: NarrativePlan, directory: Path) -> None:
    _, files = rendered_files(plan)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        write_text_lf(directory / name, content)


def _messages(
    proposal: dict[str, Any], plan: NarrativePlan, config: CompletionConfig
) -> tuple[Message, ...]:
    return (
        {"role": "system", "content": PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "conditions": proposal.get("conditions", {}),
                    "plan": plan.model_dump(mode="json"),
                    "fields": config.fields,
                    "locks": proposal.get("locks", {}),
                },
                sort_keys=True,
            ),
        },
    )


def _attempts(
    gateway: Gateway,
    config: CompletionConfig,
    source: tuple[dict[str, Any], NarrativePlan],
    out: Path,
    progress: dict[str, Any],
) -> NarrativePlan | None:
    proposal, before = source
    messages = _messages(proposal, before, config)
    response = (
        response_format(TextCompletion) if gateway.supports_json_schema(config.model) else None
    )
    if response is None:
        messages = with_schema_instruction(messages, TextCompletion)
    params = {k: v for k, v in config.model.params.items() if k != "max_completion_tokens"}
    binding = config.model.model_copy(
        update={"params": {**params, "max_tokens": config.max_tokens}}
    )
    for index in range(config.max_repairs + 1):
        attempt: dict[str, Any] = {"number": index + 1, "status": "pending", "diagnostics": []}
        progress["attempts"].append(attempt)
        prompt = Prompt("conditional_text_completion", 1, messages)
        attempt["prompt_id"] = identity({"messages": messages})
        save_json(out / f"attempt-{index + 1:02d}.json", {**attempt, "messages": messages})
        save_json(out / "progress.json", progress)
        try:
            call = gateway.complete(binding, prompt, seed=config.seed, response_format=response)
        except (ProviderError, BudgetExceededError) as error:
            attempt.update(status="failed", diagnostics=[str(error)], usage="unavailable")
            progress["status"] = (
                "budget-stop" if isinstance(error, BudgetExceededError) else "provider-failed"
            )
            save_json(out / f"attempt-{index + 1:02d}.json", attempt)
            return None
        write_text_lf(out / f"response-{index + 1:02d}.txt", call.text)
        attempt.update(call=_call_record(call), response=call.text)
        if gateway.ledger is not None and gateway.ledger.run_spent > config.max_cost_usd:
            attempt.update(
                status="rejected", diagnostics=["actual cost exceeded completion budget"]
            )
            progress["status"] = "budget-stop"
            save_json(out / f"attempt-{index + 1:02d}.json", attempt)
            return None
        try:
            if call.result.finish_reason != "stop":
                raise ValueError(
                    f"incomplete completion: finish_reason={call.result.finish_reason}"
                )
            candidate, completed = _candidate(call.text, before, config)
        except ValueError as error:
            attempt.update(status="rejected", diagnostics=[str(error)])
            messages = (*messages, *repair_messages(call.text, [str(error)]))
        else:
            attempt.update(status="accepted", fields=completed.fields)
            progress["selected_fields"] = completed.fields
            save_json(out / f"attempt-{index + 1:02d}.json", attempt)
            return candidate
        save_json(out / f"attempt-{index + 1:02d}.json", attempt)
        save_json(out / "progress.json", progress)
    progress["status"] = "repair-exhausted"
    return None


def complete(config_path: Path, out: Path) -> dict[str, Any]:
    config = CompletionConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    proposal_path = (
        config.proposal / "proposal.json" if config.proposal.is_dir() else config.proposal
    )
    proposal = read_proposal(proposal_path)
    before = NarrativePlan.model_validate(proposal["plan"])
    if len(set(config.fields)) != len(config.fields) or not set(config.fields) <= set(
        proposal["completion_fields"]
    ):
        raise ValueError("selected fields must be distinct members of the completion manifest")
    out.mkdir(parents=True, exist_ok=False)
    gateway = _gateway(config, out)
    progress: dict[str, Any] = {
        "kind": "conditional-text-completion",
        "status": "interrupted",
        "proposal_id": proposal["id"],
        "checkpoint_id": proposal.get("checkpoint_id"),
        "test_only": proposal.get("test_only", False),
        "config": config.model_dump(mode="json"),
        "config_id": identity(config.model_dump(mode="json")),
        "prompt_id": identity({"template": PROMPT, "version": 1}),
        "response_schema_id": identity(TextCompletion.model_json_schema()),
        "plan_schema_id": identity(NarrativePlan.model_json_schema()),
        "resolved_provider": config.provider.resolved_base_url(os.environ),
        "origin": (
            "generated-fiction; learned structure with LLM-filled text; "
            "no source evidence or owner approval"
        ),
        "attempts": [],
        "selected_fields": {},
        "inputs": snapshot([config_path, proposal_path]),
    }
    try:
        (out / "source-proposal.json").write_bytes(proposal_path.read_bytes())
        _save_artifacts(before, out / "last-valid")
        candidate = _attempts(gateway, config, (proposal, before), out, progress)
        if candidate is not None:
            check_snapshot(progress["inputs"])
            check_snapshot(proposal["inputs"])
            _save_artifacts(candidate, out / "last-valid")
            progress["status"] = (
                "complete"
                if set(config.fields) == set(proposal["completion_fields"])
                else "partial"
            )
            if progress["status"] == "complete":
                _save_artifacts(candidate, out)
    except Exception as error:
        progress.update(status="failed", diagnostics=[str(error)])
        raise
    finally:
        progress["calls"] = [_call_record(call) for call in gateway.history]
        progress["tokens"] = sum(
            call.result.input_tokens + call.result.output_tokens for call in gateway.history
        )
        progress["cost_usd"] = sum(call.cost_usd for call in gateway.history)
        progress["pending_call_usage"] = (
            "unknown; durable intent/cache retained"
            if any(a["status"] == "pending" for a in progress["attempts"])
            else None
        )
        save_json(out / "progress.json", progress)
        progress["artifacts"] = snapshot(
            [
                path
                for path in out.rglob("*")
                if path.is_file() and path.suffix in {".json", ".txt", ".md", ".html"}
            ]
        )
        create_record(out / "completion.json", progress)
        report_files(out, "Conditional text completion", progress)
        gateway.close()
    return progress


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="completion_command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("config", type=Path)
    execute.add_argument("--out", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("path", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        if args.completion_command == "run":
            result = complete(args.config, args.out)
        else:
            path = args.path / "completion.json" if args.path.is_dir() else args.path
            result = read_sealed(path)
            check_snapshot(result["inputs"])
            check_snapshot(result["artifacts"])
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] in {"complete", "partial"} else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"nde conditional-complete: {error}", file=sys.stderr)
        return 2
