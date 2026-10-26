"""Optional development-only, equal-budget comparisons with approved corpus examples."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import Field

from nde import briefs, realisation
from nde.corpus.exemplar_records import (
    ExemplarSelection,
    dataset_file,
    file_digest,
    load_exemplars,
)
from nde.files import write_text_lf
from nde.narrative.base import Strict
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan
from nde.runs.config import ConfigError, ExperimentConfig, LoadedConfig, load_config
from nde.runs.records import read_calls, read_manifest, read_stories
from nde.runs.runner import RunOptions, run_experiment
from nde.stories.comparison import compare as compare_final


class DevJudgement(Strict):
    brief: str
    repetition: int = Field(ge=0)
    baseline_sha256: str
    exemplars_sha256: str
    baseline_coherence: int = Field(ge=1, le=5)
    exemplars_coherence: int = Field(ge=1, le=5)
    baseline_originality: int = Field(ge=1, le=5)
    exemplars_originality: int = Field(ge=1, le=5)
    notes: str = Field(min_length=1)


class DevReview(Strict):
    version: Literal["1"] = "1"
    comparison_sha256: str
    reviewer: str = Field(min_length=1)
    test_only: bool = False
    judgements: list[DevJudgement] = Field(min_length=1)


def _write(path: Path, data: dict[str, Any]) -> None:
    write_text_lf(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def _dev_inputs(loaded: LoadedConfig) -> dict[str, str]:
    config, stories, folder = loaded.config, loaded.config.stories, loaded.briefs_dir
    if stories is None or folder is None or stories.equal_budget is None:
        raise ValueError("exemplar comparison requires shared-prose equal-budget stories")
    if config.calls or config.plans is not None or config.judges is not None:
        raise ValueError("exemplar comparison cannot include unrelated calls or judges")
    conditions = {item.name: item for item in stories.conditions}
    if set(conditions) != {"baseline", "exemplars"}:
        raise ValueError("conditions must be named baseline and exemplars")
    baseline, augmented = conditions["baseline"], conditions["exemplars"]
    if baseline.pipeline not in {"baseline_c", "baseline_d"} or (
        baseline.pipeline != augmented.pipeline or baseline.params != augmented.params
    ):
        raise ValueError("comparison arms must use the same C or D pipeline and parameters")
    if any(
        item.params.get(name)
        for item in conditions.values()
        for name in ("saved_plan", "provenance", "exemplars")
    ):
        raise ValueError("comparison inputs cannot already include plans or exemplars")
    if set(stories.equal_budget.conditions) != set(conditions):
        raise ValueError("equal-budget matching must cover baseline and exemplars")
    available = briefs.load_briefs(folder)
    ids = stories.briefs.ids or sorted(
        brief.id for brief in available.values() if brief.split == stories.briefs.split
    )
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(item not in available or available[item].split != "dev" for item in ids)
    ):
        raise ValueError("exemplar comparison requires distinct dev-only briefs")
    if stories.briefs.split == "test" or set(ids) & set(briefs.read_frozen(folder).get("test", {})):
        raise ValueError("held-out benchmark works cannot become development inputs")
    return {item: file_digest(folder / "dev" / f"{item}.yaml") for item in ids}


def compare(
    config_path: Path, export: Path, work_ids: list[str], out: Path, *, family: str | None = None
) -> dict[str, Any]:
    """Run both arms through the ordinary accounting and final-story selection path."""
    loaded = load_config(config_path)
    dev = _dev_inputs(loaded)
    selection = ExemplarSelection(
        dataset=str(dataset_file(export).resolve()),
        dataset_sha256=file_digest(dataset_file(export)),
        work_ids=work_ids,
        family=family,
    )
    examples = load_exemplars(selection)
    if examples["test_only"] and any(
        urlparse(provider.resolved_base_url(dict(os.environ))).hostname
        not in {"localhost", "127.0.0.1", "::1"}
        for provider in loaded.config.providers.values()
    ):
        raise ValueError("synthetic exemplar exports are restricted to local fake providers")
    if out.exists():
        raise ValueError("comparison already exists; use a new directory to preserve history")
    config = loaded.config.model_dump(mode="json")
    config["prompts_dir"] = str(loaded.prompts_dir)
    config["stories"]["briefs_dir"] = str(loaded.briefs_dir)
    for condition in config["stories"]["conditions"]:
        if condition["name"] == "exemplars":
            condition["params"]["exemplars"] = selection.model_dump(mode="json")
    ExperimentConfig.model_validate(config)
    out.mkdir(parents=True)
    generated = out / "config.yaml"
    write_text_lf(generated, yaml.safe_dump(config))
    packet: dict[str, Any] = {
        "version": "1",
        "config_path": str(config_path.resolve()),
        "config_sha256": loaded.hash,
        "briefs": dev,
        "selection": selection.model_dump(mode="json"),
        "examples": examples,
        "generated_sha256": file_digest(generated),
        "run": None,
        "artifacts": {},
        "literary_improvement": "pending_owner_evidence",
    }
    _write(out / "comparison.json", packet)
    try:
        run = run_experiment(
            generated,
            RunOptions(
                runs_dir=out / "runs",
                environ=dict(os.environ),
                cache_path=Path(os.environ.get("NDE_CACHE", out / "cache.sqlite")),
                ledger_path=Path(os.environ.get("NDE_LEDGER", out / "ledger.sqlite")),
            ),
        )
    finally:
        runs = sorted((out / "runs").glob("*"))
        if len(runs) == 1:
            packet["run"] = str(runs[0].resolve())
            packet["artifacts"] = {
                str(path.resolve()): file_digest(path)
                for path in sorted(runs[0].rglob("*"))
                if path.is_file()
            }
        _write(out / "comparison.json", packet)
    for story in read_stories(run):
        if story.status == "ok":
            realisation.create(
                run,
                story.condition,
                story.brief,
                story.repetition,
                out / "realisation" / story.condition / f"{story.brief}-r{story.repetition}",
            )
    return report(out)


def _reviews(
    folder: Path, packet: dict[str, Any], pairs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted((folder / "reviews").glob("*.json")):
        value = DevReview.model_validate_json(path.read_text(encoding="utf-8"))
        _check_review(value, folder, pairs)
        result.append(value.model_dump(mode="json"))
    if packet["examples"]["test_only"] and any(not value["test_only"] for value in result):
        raise ValueError("synthetic exemplar results cannot acquire scientific owner evidence")
    return result


def _check_review(review: DevReview, folder: Path, pairs: list[dict[str, Any]]) -> None:
    if review.comparison_sha256 != file_digest(folder / "comparison.json"):
        raise ValueError("review belongs to a different or changed comparison")
    seen: set[tuple[str, int]] = set()
    for judgement in review.judgements:
        key = judgement.brief, judgement.repetition
        if key in seen:
            raise ValueError("duplicate owner judgement for a development pair")
        seen.add(key)
        found = [pair for pair in pairs if (pair["brief"], pair["repetition"]) == key]
        if len(found) != 1 or not found[0]["comparable"]:
            raise ValueError("owner judgement requires a complete comparable development pair")
        conditions = found[0]["conditions"]
        if (
            judgement.baseline_sha256 != conditions["baseline"]["final_sha256"]
            or judgement.exemplars_sha256 != conditions["exemplars"]["final_sha256"]
        ):
            raise ValueError("owner judgement is stale for the selected final prose")


def review(folder: Path, value: DevReview) -> dict[str, Any]:
    current = report(folder)
    if current["problems"]:
        raise ValueError("comparison inputs changed; owner review is stale")
    _check_review(value, folder, current["pairs"])
    if current["test_only"] and not value.test_only:
        raise ValueError("synthetic comparison requires an explicitly test-only review")
    directory = folder / "reviews"
    directory.mkdir(exist_ok=True)
    text = value.model_dump_json(indent=2) + "\n"
    path = directory / f"{hashlib.sha256(text.encode()).hexdigest()}.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return report(folder)


def report(folder: Path) -> dict[str, Any]:
    packet = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    try:
        if (
            load_exemplars(ExemplarSelection.model_validate(packet["selection"]))
            != packet["examples"]
        ):
            problems.append("approved exemplar selection changed")
        source = load_config(Path(packet["config_path"]))
        if source.hash != packet["config_sha256"] or _dev_inputs(source) != packet["briefs"]:
            problems.append("development configuration or briefs changed")
        if file_digest(folder / "config.yaml") != packet["generated_sha256"]:
            problems.append("generated comparison configuration changed")
        for name, checksum in packet["artifacts"].items():
            if file_digest(Path(name)) != checksum:
                problems.append(f"run artifact changed: {name}")
    except (OSError, ValueError, ConfigError, briefs.BriefError) as error:
        problems.append(str(error))
    result: dict[str, Any] = {"pairs": []}
    validity: list[dict[str, Any]] = []
    if packet["run"]:
        run = Path(packet["run"])
        try:
            stories = read_stories(run)
            result = compare_final(run, read_manifest(run), stories, read_calls(run))
            for story in stories:
                plan = run / story.path / "plan.json"
                checked = validate_plan(NarrativePlan.model_validate_json(plan.read_text()))
                validity.append(
                    {
                        "brief": story.brief,
                        "condition": story.condition,
                        "status": story.status,
                        "valid": checked.valid,
                        **checked.model_dump(mode="json"),
                    }
                )
        except (OSError, ValueError) as error:
            problems.append(f"final run evidence unavailable: {error}")
    else:
        problems.append("generation did not publish a run")
    result.update(
        test_only=packet["examples"]["test_only"],
        problems=problems,
        examples=packet["examples"],
        literature_estimates="diagnostic_only",
        literary_improvement="pending_owner_evidence",
        realisation=[
            realisation.report(path)
            for path in sorted((folder / "realisation").glob("*/*"))
            if path.is_dir()
        ],
        validity=validity,
    )
    try:
        result["owner_reviews"] = _reviews(folder, packet, result["pairs"])
    except ValueError as error:
        problems.append(str(error))
        result["owner_reviews"] = []
    if (
        any(not item["test_only"] for item in result["owner_reviews"])
        and not result["test_only"]
        and not problems
    ):
        result["literary_improvement"] = "owner_assessments_available_no_gate_decision"
    result["status"] = (
        "complete"
        if not problems
        and result.get("run_status") == "succeeded"
        and result["pairs"]
        and all(pair["comparable"] for pair in result["pairs"])
        else "incomplete"
    )
    _write(folder / "report.json", result)
    write_text_lf(
        folder / "report.md",
        "\n\n".join(
            [
                "# Development corpus exemplar comparison",
                f"Status: {result['status']}",
                "All exemplar input, planner, critic, repair and revision tokens count. "
                "Cache hits retain experimental tokens; new spend is reported separately.",
                "Reduced target examples retain their hard/soft policy and are not full L plans.",
                f"Literary improvement: {result['literary_improvement']}. "
                "LLM literary estimates are diagnostic only; "
                "this workflow never starts a held-out gate.",
                json.dumps(result, indent=2),
            ]
        )
        + "\n",
    )
    return result


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="exemplars_command", required=True)
    generate = commands.add_parser(
        "run", help="compare baseline and corpus-augmented C or D on dev"
    )
    generate.add_argument("config", type=Path)
    generate.add_argument("--export", type=Path, required=True)
    generate.add_argument("--works", nargs="+", required=True)
    generate.add_argument(
        "--family", help="select a declared annotation family; default first sorted"
    )
    generate.add_argument("--out", type=Path, required=True)
    render = commands.add_parser("report", help="revalidate development evidence")
    render.add_argument("folder", type=Path)
    assess = commands.add_parser("review", help="record source-bound owner development scores")
    assess.add_argument("folder", type=Path)
    assess.add_argument("review", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        if args.exemplars_command == "run":
            result = compare(args.config, args.export, args.works, args.out, family=args.family)
        elif args.exemplars_command == "review":
            result = review(args.folder, DevReview.model_validate_json(args.review.read_text()))
        else:
            result = report(args.folder)
        return 0 if result["status"] == "complete" else 1
    except (OSError, ValueError, ConfigError, briefs.BriefError) as error:
        print(f"nde exemplars: {error}", file=sys.stderr)
        return 2
