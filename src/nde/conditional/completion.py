"""Fill an explicit text manifest without granting structural edit authority."""

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
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
PROMPT_ID = identity({"template": PROMPT, "version": 1})
# The template name and version that call records give completion requests.
PROMPT_NAME = "conditional_text_completion"
PROMPT_VERSION = 1
MAX_TEXT_CHARACTERS = 16000
MAX_PROVIDER_RETRIES = 3


class CompletionRequest(Strict):
    """What to complete and on which model; the caller supplies the gateway and its caps."""

    proposal: Path
    fields: list[str] = Field(min_length=1)
    model: ModelBinding
    max_repairs: int = Field(default=2, ge=0, le=8)
    seed: int = 7
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class CompletionConfig(CompletionRequest):
    """A standalone completion with its own provider, price, spending cap, cache and ledger."""

    provider: ProviderConfig
    price: Price
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


def apply_fields(before: NarrativePlan, fields: dict[str, str]) -> NarrativePlan:
    """The validated plan with exactly these authorised text fields filled; raises ValueError."""
    content = copy.deepcopy(before.model_dump(mode="json"))
    for selector, value in fields.items():
        replace_text(content, selector, value)
    candidate = NarrativePlan.model_validate(content)
    unauthorized = _unauthorized_changes(before, candidate, list(fields))
    if unauthorized or candidate.sources or _has_evidence(content):
        raise ValueError(
            f"completion changed unauthorized structure/evidence: {sorted(unauthorized)}"
        )
    validation = validate_plan(candidate)
    if not validation.valid:
        raise ValueError(validation.model_dump_json())
    return candidate


def _candidate(
    text: str, before: NarrativePlan, config: CompletionRequest
) -> tuple[NarrativePlan, TextCompletion]:
    completed = parse(text, TextCompletion)
    if set(completed.fields) != set(config.fields):
        raise ValueError("response fields must exactly match authorized selectors")
    return apply_fields(before, completed.fields), completed


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
    proposal: dict[str, Any], plan: NarrativePlan, config: CompletionRequest
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


@dataclass(frozen=True)
class _Limits:
    """Where records go, the standalone spending cap, and whether interruptions propagate."""

    out: Path
    max_cost_usd: float | None
    propagate: bool


def _attempts(
    gateway: Gateway,
    config: CompletionRequest,
    source: tuple[dict[str, Any], NarrativePlan],
    progress: dict[str, Any],
    limits: _Limits,
) -> NarrativePlan | None:
    proposal, before = source
    out = limits.out
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
        prompt = Prompt(PROMPT_NAME, PROMPT_VERSION, messages)
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
            if limits.propagate:
                # A run's own budget and provider failures stop the run, as for any step.
                raise
            return None
        write_text_lf(out / f"response-{index + 1:02d}.txt", call.text)
        attempt.update(call=_call_record(call), response=call.text)
        if (
            limits.max_cost_usd is not None
            and gateway.ledger is not None
            and gateway.ledger.run_spent > limits.max_cost_usd
        ):
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


def _source(request: CompletionRequest) -> tuple[Path, dict[str, Any], NarrativePlan]:
    proposal_path = (
        request.proposal / "proposal.json" if request.proposal.is_dir() else request.proposal
    )
    proposal = read_proposal(proposal_path)
    before = NarrativePlan.model_validate(proposal["plan"])
    if len(set(request.fields)) != len(request.fields) or not set(request.fields) <= set(
        proposal["completion_fields"]
    ):
        raise ValueError("selected fields must be distinct members of the completion manifest")
    return proposal_path, proposal, before


def _complete(
    gateway: Gateway,
    config: CompletionRequest,
    inputs: tuple[Path, str | None],
    limits: _Limits,
) -> dict[str, Any]:
    config_path, resolved_provider = inputs
    out = limits.out
    proposal_path, proposal, before = _source(config)
    first = len(gateway.history)
    progress: dict[str, Any] = {
        "kind": "conditional-text-completion",
        "status": "interrupted",
        "proposal_id": proposal["id"],
        "checkpoint_id": proposal.get("checkpoint_id"),
        "test_only": proposal.get("test_only", False),
        "config": config.model_dump(mode="json"),
        "config_id": identity(config.model_dump(mode="json")),
        "prompt_id": PROMPT_ID,
        "response_schema_id": identity(TextCompletion.model_json_schema()),
        "plan_schema_id": identity(NarrativePlan.model_json_schema()),
        "resolved_provider": resolved_provider,
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
        candidate = _attempts(gateway, config, (proposal, before), progress, limits)
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
        # A run's budget/provider stop keeps the precise status its attempt recorded.
        if not (limits.propagate and isinstance(error, (ProviderError, BudgetExceededError))):
            progress.update(status="failed", diagnostics=[str(error)])
        raise
    finally:
        history = gateway.history[first:]
        progress["calls"] = [_call_record(call) for call in history]
        progress["tokens"] = sum(
            call.result.input_tokens + call.result.output_tokens for call in history
        )
        progress["cost_usd"] = sum(call.cost_usd for call in history)
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
    return progress


def complete(config_path: Path, out: Path) -> dict[str, Any]:
    config = CompletionConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    _source(config)
    out.mkdir(parents=True, exist_ok=False)
    gateway = _gateway(config, out)
    try:
        return _complete(
            gateway,
            config,
            (config_path, config.provider.resolved_base_url(os.environ)),
            _Limits(out=out, max_cost_usd=config.max_cost_usd, propagate=False),
        )
    finally:
        gateway.close()


def complete_with_gateway(
    gateway: Gateway, request_path: Path, out: Path, *, resolved_provider: str | None
) -> dict[str, Any]:
    """Run the same completion on a caller's gateway, such as an experiment run's.

    The caller's cache, pricing, cost ledger and caps apply. Its provider and
    budget failures propagate once the attempt is recorded, so a run stops as it
    does for any other step; the completion record is still written.
    """
    request = CompletionRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    _source(request)
    out.mkdir(parents=True, exist_ok=False)
    return _complete(
        gateway,
        request,
        (request_path, resolved_provider),
        _Limits(out=out, max_cost_usd=None, propagate=True),
    )


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
