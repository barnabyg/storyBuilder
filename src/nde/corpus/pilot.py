"""A frozen pilot delegates every annotation to the per-story family engine."""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

import yaml
from pydantic import Field, model_validator

from nde import g1_report
from nde.corpus.acquisition import load_work
from nde.corpus.extraction import INSTRUCTION, read_extraction
from nde.corpus.extraction_records import INTERPRETIVE_CHANNELS, ExtractionOptions
from nde.corpus.families import FamilyConfig, compare_families, family_report
from nde.corpus.features import extract_features, load_features
from nde.corpus.records_io import (
    check_snapshot,
    cohort_file,
    create_record,
    evidence_files,
    identity,
    read_json,
    read_sealed,
    report_files,
    save_json,
    snapshot,
)
from nde.corpus.segmentation import load_segmentation
from nde.corpus.splits import load_cohort
from nde.eval import g1
from nde.eval.batch import BatchError, read_key
from nde.llm.config import Strict
from nde.llm.ledger import CostLedger
from nde.narrative.workflow import schema_json
from nde.runs.config import ConfigError
from nde.runs.records import read_manifest

PILOT_STORY_LIMIT = 50


class PilotWork(Strict):
    work_id: str
    work_dir: Path
    segmentation: Path


class PilotConfig(Strict):
    cohort: Path
    extraction_config: Path
    works: list[PilotWork] = Field(min_length=1)
    test_only: bool = False
    g1_batch: Path | None = None
    g1_decision: Path | None = None
    per_run_usd: float = Field(gt=0, allow_inf_nan=False)
    per_day_usd: float = Field(gt=0, allow_inf_nan=False)
    max_stories: int = Field(default=50, ge=1, le=1000)

    @model_validator(mode="after")
    def _unique_works(self) -> Self:
        if len({w.work_id for w in self.works}) != len(self.works):
            raise ValueError("duplicate pilot works")
        return self


def require_g1(batch: Path | None, decision: Path | None) -> None:
    if batch is None or decision is None:
        raise ValueError("real scientific extraction requires recorded, passed G1 evidence")
    saved = read_json(decision)
    replay = g1_report.decision(batch, [])
    fields = (
        "outcome",
        "criteria_outcome",
        "test_only",
        "protocol_sha256",
        "readiness",
        "coherence",
        "originality",
        "judgements_sha256",
    )
    if (
        saved.get("id") != identity(saved)
        or any(saved.get(k) != replay.get(k) for k in fields)
        or saved.get("outcome") != "passed"
        or saved.get("test_only")
        or replay["pending"]
    ):
        raise ValueError("G1 decision is stale, synthetic, pending or has not passed")
    binding = g1.read_g1_binding(batch)
    if binding.test_only:
        raise ValueError("synthetic G1 cannot authorise a real pilot")
    key = read_key(batch)
    config = Path(read_manifest(Path(key.sources[0].run)).config_path)
    g1.require_g1(
        g1.G1Inputs(
            Path(binding.protocol_path),
            config,
            Path(binding.selection_path),
            Path(binding.schema_evidence_path),
        )
    )


def _prepare(config_path: Path, out: Path, *, expansion: bool = False) -> dict[str, Any]:

    config = PilotConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    if not config.test_only:
        require_g1(config.g1_batch, config.g1_decision)
    if len(config.works) > config.max_stories or (
        not expansion and config.max_stories > PILOT_STORY_LIMIT
    ):
        raise ValueError("pilot exceeds registered cohort limit; larger runs require G2 scaling")
    cohort = load_cohort(config.cohort)
    assignments = {a.work_id: a for a in cohort.assignments}
    family = FamilyConfig.model_validate(
        yaml.safe_load(config.extraction_config.read_text(encoding="utf-8"))
    )
    if family.budget.per_day_usd is None or not 0 < family.budget.per_day_usd <= config.per_day_usd:
        raise ValueError("family configuration must enforce the registered pilot daily cap")
    if config.test_only and any(
        urlparse(p.resolved_base_url(os.environ)).hostname not in {"localhost", "127.0.0.1", "::1"}
        for p in family.providers.values()
    ):
        raise ValueError("synthetic pilot requires local fixture providers")
    files = [config_path, cohort_file(config.cohort), config.extraction_config]
    rows: list[dict[str, Any]] = []
    for work in config.works:
        record, _ = load_work(work.work_dir)
        assignment = assignments.get(work.work_id)
        if (
            record.work_id != work.work_id
            or assignment is None
            or assignment.text_sha256 != record.text_sha256
        ):
            raise ValueError("pilot source differs from frozen cohort")
        load_segmentation(work.segmentation, work.work_dir)
        files += [work.segmentation, *evidence_files(work.work_dir)]
        for binding in family.annotators:
            for name in ("structural", "state", "interpretive"):
                rows.append(
                    {
                        "work_id": work.work_id,
                        "family": binding.family,
                        "pass": name,
                        "status": "planned",
                    }
                )
    if config.g1_decision:
        files.append(config.g1_decision)
    design = {
        "config": config.model_dump(mode="json"),
        "inputs": snapshot(files),
        "schema_sha256": hashlib_text(schema_json()),
        "prompt_sha256": hashlib_text(INSTRUCTION),
        "passes": ["structural", "state", "interpretive"],
        "interpretation_channels": list(INTERPRETIVE_CHANNELS),
        "test_only": config.test_only,
        "expansion": expansion,
    }
    if (out / "pilot.json").exists():
        saved = read_sealed(out / "pilot.json")
        if any(saved.get(k) != value for k, value in design.items()):
            raise ValueError("pilot inputs changed; freeze a new extraction version")
        check_snapshot(saved["inputs"])
        return saved
    if out.exists():
        raise ValueError("pilot output exists without a complete registration")
    packet = create_record(out / "pilot.json", design)
    save_json(out / "progress.json", {"rows": rows, "status": "planned"})
    return packet


def hashlib_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def execute(config_path: Path, out: Path, *, expansion: bool = False) -> dict[str, Any]:

    packet = _prepare(config_path, out, expansion=expansion)
    config = PilotConfig.model_validate(packet["config"])
    for work in config.works:
        feature_out = out / "features" / work.work_id
        if not feature_out.exists():
            extract_features(work.work_dir, work.segmentation, feature_out)
        load_features(feature_out / "features.json", work.work_dir, work.segmentation)
        options = ExtractionOptions(
            out=out / "stories" / work.work_id,
            cache=out / "cache.sqlite",
            ledger=out / "ledger.sqlite",
            cohort_run_id=f"pilot-{packet['id']}",
            run_budget_usd=config.per_run_usd,
        )
        compare_families(config.extraction_config, work.work_dir, work.segmentation, options)
        report(out)
    return report(out)


def report(out: Path) -> dict[str, Any]:

    packet = read_sealed(out / "pilot.json")
    check_snapshot(packet["inputs"])
    config = PilotConfig.model_validate(packet["config"])
    if not config.test_only:
        require_g1(config.g1_batch, config.g1_decision)
    rows: list[dict[str, Any]] = []
    complete = 0
    failures: list[str] = []
    features: dict[str, dict[str, str]] = {}
    for work in config.works:
        feature = out / "features" / work.work_id / "features.json"
        feature_status = {"status": "current"}
        try:
            load_features(feature, work.work_dir, work.segmentation)
        except (OSError, ValueError) as error:
            feature_status = {
                "status": "stale" if feature.is_file() else "missing",
                "diagnostic": str(error),
            }
            failures.append(f"{work.work_id}: features {feature_status['status']}: {error}")
        features[work.work_id] = feature_status
        folder = out / "stories" / work.work_id
        families: dict[str, dict[str, Any]] = {}
        if (folder / "families.json").exists():
            comparison = family_report(folder)
            complete += comparison["status"] == "complete" and feature_status["status"] == "current"
            families = {row["family"]: row for row in comparison["families"]}
        family_config = yaml.safe_load(config.extraction_config.read_text(encoding="utf-8"))
        for binding in family_config["annotators"]:
            family = binding["family"].strip().lower()
            entry = families.get(family)
            session = (
                read_extraction(Path(entry["directory"]))
                if entry and (Path(entry["directory"]) / "extraction.json").is_file()
                else None
            )
            for name in ("structural", "state", "interpretive"):
                current = (
                    next((p for p in session.passes if p.name == name), None) if session else None
                )
                calls = [a.call for a in current.attempts] if current else []
                row = {
                    "work_id": work.work_id,
                    "family": family,
                    "pass": name,
                    "status": current.status if current else "planned",
                    "attempts": len(calls),
                    "input_tokens": sum(c.input_tokens for c in calls),
                    "output_tokens": sum(c.output_tokens for c in calls),
                    "retries": sum(c.retries for c in calls),
                    "new_spend_usd": sum(c.cost_usd for c in calls),
                    "list_price_usd": sum(c.token_cost_usd for c in calls),
                    "pending": session.pending_pass == name if session else False,
                }
                rows.append(row)
                if current and current.failure:
                    failures.append(current.failure)
    ledger = CostLedger(out / "ledger.sqlite", run_id=f"pilot-{packet['id']}")
    try:
        reserved_and_spent = ledger.run_spent
    finally:
        ledger.close()
    accounted = sum(r["new_spend_usd"] for r in rows)
    cost = sum(r["list_price_usd"] for r in rows)
    result: dict[str, Any] = {
        "version": "1",
        "pilot_id": packet["id"],
        "test_only": config.test_only,
        "status": "complete" if complete == len(config.works) else "incomplete",
        "planned_stories": len(config.works),
        "completed_stories": complete,
        "rows": rows,
        "coverage": complete / len(config.works),
        "new_spend_usd": accounted,
        "list_price_usd": cost,
        "durable_reserved_and_spent_usd": reserved_and_spent,
        "unresolved_accounting_usd": max(0, reserved_and_spent - accounted),
        "failures": failures,
        "features": features,
        "scientific_status": "pending" if config.test_only else "requires G2 review",
        "scaling": {
            "denominator_completed_stories": complete,
            "cost_basis": "all attempted list-price calls, including repairs and failures",
            "assumption": (
                "same length, family, pass and retry distribution; no throughput estimate"
            ),
            "per_completed_story_usd": cost / complete if complete else None,
            "300_stories_usd": cost / complete * 300 if complete else None,
            "1000_stories_usd": cost / complete * 1000 if complete else None,
            "incomplete_data": complete != len(config.works),
        },
    }
    result["id"] = identity(result)
    save_json(out / "progress.json", {"rows": rows, "status": result["status"]})
    report_files(out, "Extraction pilot coverage and costs", result)
    return result


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="pilot_command", required=True)
    generate = commands.add_parser("run")
    generate.add_argument("config", type=Path)
    generate.add_argument("--out", type=Path, required=True)
    render = commands.add_parser("report")
    render.add_argument("directory", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        result = (
            execute(args.config, args.out)
            if args.pilot_command == "run"
            else report(args.directory)
        )
        return 0 if result["status"] == "complete" else 1
    except (OSError, ValueError, yaml.YAMLError, BatchError, ConfigError) as error:
        print(f"nde corpus-pilot: {error}", file=sys.stderr)
        return 2
