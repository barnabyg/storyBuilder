"""Independent family extraction and source-aligned agreement, never consensus labels."""

import html
import itertools
import json
import os
import secrets
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Self, cast

import yaml
from pydantic import ConfigDict, Field, model_validator

from nde.corpus.acquisition import load_work
from nde.corpus.channels import CHANNEL_FIELDS
from nde.corpus.extraction import INSTRUCTION, digest, extract, verify_extraction
from nde.corpus.extraction_records import (
    INTERPRETIVE_CHANNELS,
    PASS_NAMES,
    ExtractionConfig,
    ExtractionOptions,
    ExtractionSettings,
)
from nde.corpus.segmentation import load_segmentation
from nde.files import write_text_lf
from nde.llm.config import ModelBinding, Strict
from nde.llm.gateway import Gateway
from nde.narrative.models import NarrativePlan
from nde.narrative.workflow import PlanError, load_plan, schema_json


class FamilyConfig(ExtractionSettings):
    annotators: list[ModelBinding] = Field(min_length=2)

    @model_validator(mode="after")
    def independent_families(self) -> Self:
        families = [binding.family for binding in self.annotators]
        endpoints = [
            (
                self.providers[binding.provider].resolved_base_url(os.environ).rstrip("/"),
                binding.model,
            )
            for binding in self.annotators
            if binding.provider in self.providers
        ]
        if None in families or len(set(families)) != len(families):
            raise ValueError("annotators require distinct declared model families")
        if len(set(endpoints)) != len(self.annotators):
            raise ValueError("annotators require distinct configured provider/model bindings")
        if any(binding.label not in self.prices for binding in self.annotators):
            raise ValueError("each independent annotator requires an explicit price")
        return self


class FamilyRow(Strict):
    model_config = ConfigDict(extra="forbid", frozen=False)
    family: str
    binding: ModelBinding
    directory: str
    status: str = "planned"
    annotation_sha256: str | None = None


class FamilySession(Strict):
    model_config = ConfigDict(extra="forbid", frozen=False)
    version: Literal["1"] = "1"
    id: str
    work_id: str
    work_dir: str
    segmentation_file: str
    source_sha256: str
    work_record_sha256: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict[str, Any])
    resolved_providers: dict[str, str] = Field(default_factory=dict[str, str])
    segmentation_sha256: str
    config_file: str
    config: FamilyConfig
    config_sha256: str
    prompt_sha256: str
    schema_sha256: str
    cohort_run_id: str
    run_budget_usd: float | None
    cache_path: str
    ledger_path: str
    families: list[FamilyRow]
    status: str = "running"
    owner_approved: Literal[False] = False


def _save(directory: Path, session: FamilySession) -> None:
    temporary = directory / "families.tmp"
    write_text_lf(temporary, session.model_dump_json(indent=2) + "\n")
    temporary.replace(directory / "families.json")


def read_families(directory: Path) -> FamilySession:
    return FamilySession.model_validate_json(
        (directory / "families.json").read_text(encoding="utf-8")
    )


def _prepare(
    config_file: Path, work_dir: Path, segments_file: Path, options: ExtractionOptions
) -> FamilySession:
    config = FamilyConfig.model_validate(yaml.safe_load(config_file.read_text(encoding="utf-8")))
    record, _ = load_work(work_dir)
    segments = load_segmentation(segments_file, work_dir)
    cap = options.run_budget_usd if options.cohort_run_id else config.budget.per_run_usd
    identity = {
        "config_sha256": digest(config.model_dump_json()),
        "source_sha256": record.text_sha256,
        "work_record_sha256": digest(record.model_dump_json()),
        "provenance": record.model_dump(mode="json"),
        "resolved_providers": config.provider_endpoints(os.environ),
        "segmentation_sha256": digest(segments.model_dump_json()),
        "prompt_sha256": digest(INSTRUCTION),
        "schema_sha256": digest(schema_json()),
        "cache_path": str(options.cache.resolve()),
        "ledger_path": str(options.ledger.resolve()),
        "run_budget_usd": cap,
    }
    if (options.out / "families.json").exists():
        session = read_families(options.out)
        if any(
            session.model_dump()[key] != value for key, value in identity.items()
        ) or options.cohort_run_id not in {None, session.cohort_run_id}:
            raise PlanError("family resume inputs changed")
        return session
    if options.out.exists() and any(options.out.iterdir()):
        raise PlanError("new family output must be empty")
    options.out.mkdir(parents=True, exist_ok=True)
    identifier = secrets.token_hex(12)
    session = FamilySession.model_validate(
        {
            "id": identifier,
            "work_id": record.work_id,
            "work_dir": str(work_dir.resolve()),
            "segmentation_file": str(segments_file.resolve()),
            "config_file": str(config_file.resolve()),
            "config": config,
            "cohort_run_id": options.cohort_run_id or identifier,
            "families": [
                FamilyRow(family=binding.family or "", binding=binding, directory=f"family-{index}")
                for index, binding in enumerate(config.annotators)
            ],
            **identity,
        }
    )
    for row in session.families:
        single = config.model_dump(exclude={"annotators", "annotator"})
        single["annotator"] = row.binding.model_dump(mode="json")
        write_text_lf(options.out / f"{row.directory}.json", json.dumps(single, indent=2) + "\n")
    _save(options.out, session)
    return session


def compare_families(
    config_file: Path,
    work_dir: Path,
    segmentation_file: Path,
    options: ExtractionOptions,
    *,
    gateway: Gateway | None = None,
) -> FamilySession:
    session = _prepare(config_file, work_dir, segmentation_file, options)
    for row in session.families:
        extracted = extract(
            options.out / f"{row.directory}.json",
            work_dir,
            segmentation_file,
            ExtractionOptions(
                options.out / row.directory,
                options.cache,
                options.ledger,
                session.cohort_run_id,
                session.run_budget_usd,
                session.id,
            ),
            gateway=gateway,
        )
        row.status, row.annotation_sha256 = extracted.status, extracted.annotation_sha256
        _save(options.out, session)
    session.status = (
        "complete" if all(row.status == "complete" for row in session.families) else "incomplete"
    )
    _save(options.out, session)
    report = family_report(options.out)
    lines = [
        f"# Independent families: {session.status}",
        "Agreement is not literary correctness, consensus labels or owner approval.",
        "Exact normalized claims; scene alignment by source spans, entities by unique names.",
        "Denominator: multiset union; unmatched retains contradictory or missing claims.",
    ]
    lines.extend(f"{row.family}: {row.status}" for row in session.families)
    lines.extend(json.dumps(row, ensure_ascii=False) for row in report["agreement"])
    write_text_lf(
        options.out / "report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    write_text_lf(options.out / "report.md", "\n\n".join(lines) + "\n")
    write_text_lf(
        options.out / "report.html",
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Independent families</title>"
        + "".join(f"<p>{html.escape(line)}</p>" for line in lines)
        + "</html>\n",
    )
    return session


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _canonical(value: Any, identities: dict[str, str], key: str = "") -> Any:
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        return {
            name: _canonical(item, identities, name)
            for name, item in mapping.items()
            if name != "id"
        }
    if isinstance(value, list):
        return [_canonical(item, identities, key) for item in cast(list[Any], value)]
    if isinstance(value, str):
        if key.endswith(("_id", "_ids")):
            return identities.get(value, f"unresolved:{value}")
        return _normalized(value)
    return value


def aligned_claims(plan: NarrativePlan) -> dict[str, Counter[str]]:
    """Public deterministic matching representation; original plans remain untouched."""
    counts = Counter(_normalized(character.name) for character in plan.characters)
    identities = {source.id: f"source:{source.sha256}" for source in plan.sources}
    identities.update(
        {
            character.id: f"character:{_normalized(character.name)}"
            if counts[_normalized(character.name)] == 1
            else f"ambiguous:{character.id}"
            for character in plan.characters
        }
    )
    for scene in plan.scenes:
        spans = [(span.source_sha256, span.start, span.end) for span in scene.evidence]
        identities[scene.id] = "scene:" + json.dumps(spans, sort_keys=True)
        for beat in scene.beats:
            identities[beat.id] = f"beat:{identities[scene.id]}:{_normalized(beat.summary)}"
    for records in (plan.knowledge_facts, plan.setups, plan.motifs):
        for record in records:
            identities[record.id] = json.dumps(
                _canonical(record.model_dump(mode="json"), identities),
                sort_keys=True,
                ensure_ascii=False,
            )
    channels: dict[str, list[Any]] = {
        "goal": [item.model_dump(mode="json") for item in plan.states if item.kind == "goal"],
        "belief": [item.model_dump(mode="json") for item in plan.states if item.kind == "belief"],
        "values": [
            item.model_dump(mode="json") for item in plan.states if item.kind in {"value", "canon"}
        ],
        "constraints": [item.model_dump(mode="json") for item in plan.constraints],
        "setups_payoffs": [
            item.model_dump(mode="json") for item in (*plan.setups, *plan.payoff_links)
        ],
    }
    channels.update(
        {
            channel: [
                {"channel": field, **item.model_dump(mode="json")}
                for field in fields
                for item in getattr(plan, field)
            ]
            for channel, fields in CHANNEL_FIELDS.items()
        }
    )
    channels.update(
        {
            name: [item.model_dump(mode="json") for item in getattr(plan, name)]
            for name in INTERPRETIVE_CHANNELS
        }
    )
    return {
        channel: Counter(
            json.dumps(_canonical(item, identities), sort_keys=True, ensure_ascii=False)
            for item in records
        )
        for channel, records in channels.items()
    }


def family_report(directory: Path) -> dict[str, Any]:
    """Recompute evidence from verified independent annotations, never trust summary metrics."""
    session = read_families(directory)
    record, _ = load_work(Path(session.work_dir))
    segments = load_segmentation(Path(session.segmentation_file), Path(session.work_dir))
    current = FamilyConfig.model_validate(
        yaml.safe_load(Path(session.config_file).read_text(encoding="utf-8"))
    )
    if (
        digest(current.model_dump_json()) != session.config_sha256
        or current != session.config
        or session.provenance != record.model_dump(mode="json")
        or session.work_record_sha256 != digest(record.model_dump_json())
        or session.resolved_providers != current.provider_endpoints(os.environ)
        or record.text_sha256 != session.source_sha256
        or digest(segments.model_dump_json()) != session.segmentation_sha256
        or digest(INSTRUCTION) != session.prompt_sha256
        or digest(schema_json()) != session.schema_sha256
        or [row.binding for row in session.families] != current.annotators
        or any(row.family != row.binding.family for row in session.families)
    ):
        raise PlanError("stale family source/segmentation/config/schema/prompt")
    family_rows: list[dict[str, Any]] = []
    plans: dict[str, NarrativePlan] = {}
    for row in session.families:
        path = directory / row.directory
        if not (path / "extraction.json").is_file():
            if row.status != "planned":
                raise PlanError("recorded family extraction checkpoint is missing")
            family_rows.append(
                {
                    **row.model_dump(mode="json"),
                    "directory": str(path.resolve()),
                    "source_sha256": session.source_sha256,
                    "segmentation_sha256": session.segmentation_sha256,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "retries": 0,
                    "spent_usd": 0,
                    "token_cost_usd": 0,
                    "accounting_run_spent_usd": 0,
                    "accounting_story_spent_usd": 0,
                    "pending_pass": None,
                    "passes": [
                        {"name": name, "status": "planned", "attempts": []} for name in PASS_NAMES
                    ],
                }
            )
            continue
        extracted = verify_extraction(path)
        expected = current.model_dump(exclude={"annotators", "annotator"})
        expected["annotator"] = row.binding.model_dump(mode="json")
        if extracted.config != ExtractionConfig.model_validate(expected):
            raise PlanError("independent family config does not match registered binding")
        calls = [attempt.call for part in extracted.passes for attempt in part.attempts]
        family_rows.append(
            {
                **row.model_dump(mode="json"),
                "directory": str(path.resolve()),
                "source_sha256": extracted.source_sha256,
                "segmentation_sha256": extracted.segmentation_sha256,
                "input_tokens": sum(call.input_tokens for call in calls),
                "output_tokens": sum(call.output_tokens for call in calls),
                "retries": sum(call.retries for call in calls),
                "spent_usd": extracted.spent_usd,
                "token_cost_usd": sum(call.token_cost_usd for call in calls),
                "accounting_run_spent_usd": extracted.accounting_run_spent_usd,
                "accounting_story_spent_usd": extracted.accounting_story_spent_usd,
                "pending_pass": extracted.pending_pass,
                "passes": [part.model_dump(mode="json") for part in extracted.passes],
            }
        )
        if extracted.status == "complete":
            if (
                extracted.annotation_sha256 != row.annotation_sha256
                or row.status != extracted.status
            ):
                raise PlanError("family completed annotation identity changed")
            plans[row.family] = load_plan(path / "annotation/plan.json")
    agreement: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for left, right in itertools.combinations(plans, 2):
        first, second = aligned_claims(plans[left]), aligned_claims(plans[right])
        for channel, values in first.items():
            other = second[channel]
            agreed, total = sum((values & other).values()), sum((values | other).values())
            unmatched = sum((values - other).values()) + sum((other - values).values())
            agreement.append(
                {
                    "channel": channel,
                    "left_family": left,
                    "right_family": right,
                    "agreed": agreed,
                    "compared": min(sum(values.values()), sum(other.values())),
                    "unmatched": unmatched,
                    "total": total,
                    "agreement": agreed / total if total else None,
                }
            )
            if unmatched:
                disagreements.append(
                    {
                        "channel": channel,
                        "left_family": left,
                        "right_family": right,
                        "left_only": list((values - other).elements()),
                        "right_only": list((other - values).elements()),
                    }
                )
    return {
        "version": "1",
        "work_id": session.work_id,
        "work_record_sha256": session.work_record_sha256,
        "source_sha256": session.source_sha256,
        "segmentation_sha256": session.segmentation_sha256,
        "status": session.status,
        "families": family_rows,
        "agreement": agreement,
        "disagreements": disagreements,
        "owner_approved": False,
        "accounting_story_spent_usd": max(
            (row["accounting_story_spent_usd"] for row in family_rows),
            default=0,
        ),
        "unattributed_new_spend_usd": max(
            0,
            max((row["accounting_story_spent_usd"] for row in family_rows), default=0)
            - sum(row["spent_usd"] for row in family_rows),
        ),
        "matching": "normalized exact claims; multiset union denominator",
    }
