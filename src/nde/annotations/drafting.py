"""Gateway-backed annotation with a single bounded syntactic/semantic repair loop."""

import hashlib
import json
import os
import secrets
from pathlib import Path

import yaml
from pydantic import ValidationError

from nde.corpus import provenance
from nde.files import write_text_lf
from nde.gitinfo import git
from nde.llm import structured
from nde.llm.cache import ResponseCache
from nde.llm.errors import ProviderError
from nde.llm.gateway import Gateway, Prompt
from nde.llm.ledger import BudgetExceededError, CostLedger, Pricing
from nde.narrative.models import NarrativePlan, Source
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import PlanError, render, schema_json
from nde.runs.prompts import load_template
from nde.runs.state import Attribution, call_record

from .records import (
    AnnotationConfig,
    Attempt,
    DraftOptions,
    Session,
    digest,
    read_session,
    save_session,
)


def checked_source(manifest: Path, story_id: str) -> tuple[Source, dict[str, object]]:
    stories = provenance.read_manifest(manifest).stories
    matches = [story for story in stories if story.id == story_id]
    if len(matches) != 1:
        raise PlanError(f"expected one source for {story_id}")
    story = matches[0]
    raw = provenance.source_path(manifest, story_id).read_bytes()
    if story.sha256 is None or hashlib.sha256(raw).hexdigest() != story.sha256:
        raise PlanError(f"{story_id}: source checksum mismatch or unrecorded source")
    text = provenance.extract(raw.decode("utf-8-sig"), story)
    source = Source(
        id=f"source-{story_id}",
        title=story.title,
        text=text,
        sha256=digest(text),
        provenance=f"{story.source}; {story.edition}; {story.licence_basis}",
    )
    return source, story.model_dump(mode="json")


def _prepare(config_file: Path, manifest: Path, story_id: str, out: Path) -> Session:
    config = AnnotationConfig.model_validate(
        yaml.safe_load(config_file.read_text(encoding="utf-8"))
    )
    source, source_record = checked_source(manifest, story_id)
    template = load_template(
        (config_file.parent / config.prompts_dir).resolve(),
        config.prompt.template,
        config.prompt.version,
    )
    config_sha = digest(config.model_dump_json())
    schema_sha = digest(schema_json())
    if (out / "session.json").exists():
        session = read_session(out)
        if (
            session.config_sha256 != config_sha
            or session.source != source
            or session.prompt_sha256 != template.sha256
            or session.schema_sha256 != schema_sha
        ):
            raise PlanError("resume inputs differ from saved source/config/prompt/schema")
        return session
    if out.exists() and any(out.iterdir()):
        raise PlanError("new annotation output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    session = Session(
        id=secrets.token_hex(8),
        story_id=story_id,
        source=source,
        provenance=source_record,
        config=config,
        config_sha256=config_sha,
        schema_sha256=schema_sha,
        prompt_sha256=template.sha256,
        git_sha=git(["rev-parse", "HEAD"], config_file.parent) or "unknown",
    )
    write_text_lf(
        out / "prompt.json",
        json.dumps(template.render({"source": source.model_dump_json()}), indent=2) + "\n",
    )
    save_session(out, session)
    return session


def validate_annotation(text: str, source: Source) -> tuple[NarrativePlan | None, list[str]]:
    try:
        plan = structured.parse(text, NarrativePlan)
    except ValidationError as error:
        return None, structured.describe_errors(error)
    if plan.tier != "L":
        return None, ["annotation requires tier L"]
    if plan.sources and plan.sources != [source]:
        return None, ["source_mismatch: use the exact supplied source"]
    plan.sources = [source]
    result = validate_plan(plan)
    return plan, [
        f"{d.code} [{d.location}]: {d.message}" for d in result.diagnostics if d.severity == "error"
    ]


def _generate(session: Session, out: Path, gateway: Gateway, messages: Prompt) -> None:
    config = session.config
    binding = config.annotator
    conversation = messages.messages
    response_format = structured.response_format(NarrativePlan)
    if not gateway.supports_json_schema(binding):
        conversation = structured.with_schema_instruction(conversation, NarrativePlan)
        response_format = None
    for previous in session.attempts:
        if not previous.errors:
            plan, errors = validate_annotation(previous.text, session.source)
            if plan is not None and not errors:
                render(plan, out / "draft")
                session.status = "drafted"
                return
        conversation = (*conversation, *structured.repair_messages(previous.text, previous.errors))
    for _ in range(len(session.attempts), config.max_repairs + 1):
        completion = gateway.complete(
            binding,
            Prompt(messages.template, messages.version, conversation),
            seed=config.seed,
            response_format=response_format,
        )
        plan, errors = validate_annotation(completion.text, session.source)
        record = call_record(
            gateway, Attribution("annotation", 0), len(session.attempts), completion
        )
        session.attempts.append(
            Attempt(text=completion.text, errors=errors, call=record.model_dump(mode="json"))
        )
        session.spent_usd += completion.cost_usd
        if len(session.attempts) == 1:
            write_text_lf(out / "original.txt", completion.text)
        save_session(out, session)
        if plan is not None and not errors:
            render(plan, out / "draft")
            session.status = "drafted"
            return
        conversation = (*conversation, *structured.repair_messages(completion.text, errors))
    session.status = "exhausted"
    session.failure = "repair limit exhausted"


def draft(config_file: Path, manifest: Path, story_id: str, options: DraftOptions) -> Session:
    out = options.out
    session = _prepare(config_file, manifest, story_id, out)
    if session.status == "drafted":
        return session
    config = session.config
    binding = config.annotator
    if binding.provider not in config.providers or binding.label not in config.prices:
        raise PlanError("annotator requires a configured provider and explicit price")
    provider = config.providers[binding.provider]
    if provider.api_key_env and not os.environ.get(provider.api_key_env):
        raise PlanError(f"missing environment variable {provider.api_key_env}")
    template = load_template(
        (config_file.parent / config.prompts_dir).resolve(),
        config.prompt.template,
        config.prompt.version,
    )
    ledger = CostLedger(
        options.ledger, per_run_usd=config.budget.per_run_usd, per_day_usd=config.budget.per_day_usd
    )
    ledger.run_spent = session.spent_usd
    gateway = Gateway(
        config.providers,
        dict(os.environ),
        cache=ResponseCache(options.cache),
        pricing=Pricing(config.prices, config.budget.assumed_output_tokens),
        ledger=ledger,
    )
    try:
        session.status = "running"
        session.failure = None
        prompt = Prompt(
            template.name,
            template.version,
            template.render({"source": session.source.model_dump_json()}),
        )
        _generate(session, out, gateway, prompt)
    except BudgetExceededError as error:
        session.status, session.failure = "budget_aborted", str(error)
    except ProviderError as error:
        session.status, session.failure = "provider_failed", str(error)
    finally:
        save_session(out, session)
        gateway.close()
    return session
