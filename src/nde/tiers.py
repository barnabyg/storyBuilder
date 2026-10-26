"""Development-only representation comparisons and immutable owner tier selections."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from nde import briefs, realisation
from nde.files import checksum_lf, write_text_lf
from nde.narrative.base import Strict
from nde.narrative.projection import Tier
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import load_plan, schema_json
from nde.runs.config import ConfigError, LoadedConfig, load_config
from nde.runs.prompts import PromptTemplate
from nde.runs.records import STORY_FILE, read_calls, read_manifest, read_stories
from nde.runs.runner import RunOptions, run_experiment
from nde.stories.budget import story_budget, within
from nde.stories.execute import prepare_stories
from nde.stories.normalise import word_count
from nde.stories.selected import selected_text

TIERS: tuple[Tier, ...] = ("S", "M", "L")


class Settings(Strict):
    settings_sha256: str
    schema_sha256: str
    prompt_sha256: dict[str, str]
    models_sha256: str


class Comparison(Strict):
    version: Literal["1"] = "1"
    id: str
    config_path: str
    config: dict[str, Any]
    briefs: dict[str, str]
    settings: Settings
    runs: dict[Tier, str]
    artifacts: dict[str, str]


class DevJudgement(Strict):
    brief: str
    tier: Tier
    repetition: int = Field(default=0, ge=0)
    coherence: int = Field(ge=1, le=5)
    originality: int = Field(ge=1, le=5)
    note: str = Field(min_length=1)


class DevReview(Strict):
    comparison_sha256: str
    reviewer: str = Field(min_length=1)
    test_only: bool = False
    judgements: list[DevJudgement] = Field(min_length=1)


class Selection(Strict):
    version: Literal["1"] = "1"
    id: str
    tier: Tier
    test_only: bool = False
    owner: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    comparison: str
    comparison_sha256: str
    briefs: dict[str, str]
    owner_sample: list[str]
    settings_sha256: str
    schema_sha256: str
    prompt_sha256: dict[str, str]
    models_sha256: str
    config: dict[str, Any]
    review_sha256: dict[str, str]


class Choice(Strict):
    tier: Tier
    owner: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    test_only: bool = False


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _identity(model: Strict) -> str:
    return _digest(model.model_dump(mode="json", exclude={"id"}))


def settings_identity(config_path: Path) -> Settings:
    """Gate-relevant settings; brief selection changes from dev to test independently."""
    loaded = load_config(config_path)
    templates: dict[str, PromptTemplate] = {}
    prepare_stories(loaded, templates)
    raw = loaded.config.model_dump(mode="json")
    raw.pop("name")
    raw.pop("prompts_dir")
    if raw["stories"] is not None:
        raw["stories"].pop("briefs")
        raw["stories"].pop("briefs_dir")
        for condition in raw["stories"]["conditions"]:
            if condition["pipeline"] == "baseline_c":
                condition["params"].pop("tier", None)
        conditions: list[dict[str, Any]] = raw["stories"]["conditions"]
        conditions.sort(key=lambda condition: condition["name"])
        if raw["stories"]["equal_budget"] is not None:
            raw["stories"]["equal_budget"]["conditions"].sort()
    return Settings(
        settings_sha256=_digest(raw),
        schema_sha256=hashlib.sha256(schema_json().encode()).hexdigest(),
        prompt_sha256={key: value.sha256 for key, value in sorted(templates.items())},
        models_sha256=_digest(
            {name: binding.model_dump(mode="json") for name, binding in loaded.config.roles.items()}
        ),
    )


def _dev_briefs(loaded: LoadedConfig) -> dict[str, str]:
    stories, root = loaded.config.stories, loaded.briefs_dir
    if stories is None or root is None:
        raise ValueError("tier comparison requires a story configuration")
    available = briefs.load_briefs(root)
    selection = stories.briefs
    ids = selection.ids or sorted(
        brief.id for brief in available.values() if brief.split == selection.split
    )
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("dev-only comparison requires distinct nonempty brief IDs")
    if selection.split == "test" or any(
        brief_id not in available or available[brief_id].split != "dev" for brief_id in ids
    ):
        raise ValueError("tier comparisons are dev-only; held-out inputs cannot guide selection")
    frozen_ids = set(briefs.read_frozen(root).get("test", {}))
    if frozen_ids & set(ids):
        raise ValueError("dev-only inputs cannot include frozen held-out brief identities")
    return {
        brief_id: checksum_lf((root / "dev" / f"{brief_id}.yaml").read_bytes()) for brief_id in ids
    }


def _snapshot(loaded: LoadedConfig) -> dict[str, Any]:
    raw = loaded.config.model_dump(mode="json", exclude_none=True)
    raw["prompts_dir"] = str(loaded.prompts_dir)
    raw["stories"]["briefs_dir"] = str(loaded.briefs_dir)
    return raw


def _check_conditions(loaded: LoadedConfig) -> None:
    stories = loaded.config.stories
    if stories is None or {item.name: item.pipeline for item in stories.conditions} != {
        "B": "baseline_b",
        "C": "baseline_c",
    }:
        raise ValueError("tier comparison requires exactly B baseline_b and C baseline_c")
    if loaded.config.calls or loaded.config.judges is not None or loaded.config.plans is not None:
        raise ValueError("dev comparisons cannot include arbitrary calls or judge tuning inputs")
    if stories.equal_budget is None or set(stories.equal_budget.conditions) != {"B", "C"}:
        raise ValueError("tier comparison requires shared B/C equal-budget matching")


def compare(config_path: Path, out: Path) -> Comparison:
    loaded = load_config(config_path)
    dev = _dev_briefs(loaded)
    _check_conditions(loaded)
    if out.exists():
        raise ValueError("comparison already exists; choose a new folder to preserve history")
    settings = settings_identity(config_path)
    raw = _snapshot(loaded)
    out.mkdir(parents=True)
    configs = out / "configs"
    configs.mkdir()
    source = out / "source.yaml"
    write_text_lf(source, yaml.safe_dump(raw, sort_keys=False))
    runs: dict[Tier, str] = {}
    for tier in TIERS:
        variant = json.loads(json.dumps(raw))
        variant["name"] = f"{loaded.config.name}-tier-{tier}"
        for condition in variant["stories"]["conditions"]:
            if condition["pipeline"] == "baseline_c":
                condition.setdefault("params", {})["tier"] = tier
        config = configs / f"{tier}.yaml"
        write_text_lf(config, yaml.safe_dump(variant, sort_keys=False))
        options = RunOptions(
            runs_dir=out / "runs",
            environ=dict(os.environ),
            cache_path=Path(os.environ.get("NDE_CACHE", out / "cache.sqlite")),
            ledger_path=Path(os.environ.get("NDE_LEDGER", out / "ledger.sqlite")),
        )
        run_dir = run_experiment(config, options)
        runs[tier] = str(run_dir.resolve())
        for story in read_stories(run_dir):
            if story.condition == "C" and story.status == "ok":
                realisation.create(
                    run_dir,
                    "C",
                    story.brief,
                    story.repetition,
                    out / "realisation" / tier / f"{story.brief}-r{story.repetition}",
                )
    paths = [source, *configs.glob("*.yaml")]
    paths.extend(path for run in runs.values() for path in Path(run).rglob("*") if path.is_file())
    packet = Comparison(
        id="",
        config_path=str(source.resolve()),
        config=raw,
        briefs=dev,
        settings=settings,
        runs=runs,
        artifacts={str(path.resolve()): checksum_lf(path.read_bytes()) for path in paths},
    )
    packet.id = _identity(packet)
    write_text_lf(out / "comparison.json", packet.model_dump_json(indent=2) + "\n")
    report(out)
    return packet


def load_comparison(folder: Path) -> Comparison:
    packet = Comparison.model_validate_json(
        (folder / "comparison.json").read_text(encoding="utf-8")
    )
    if packet.id != _identity(packet):
        raise ValueError("comparison identity changed")
    return packet


def _comparison_problems(packet: Comparison) -> list[str]:
    problems: list[str] = []
    for name, expected in packet.artifacts.items():
        path = Path(name)
        if not path.is_file() or checksum_lf(path.read_bytes()) != expected:
            problems.append(f"comparison artifact missing or changed: {name}")
    try:
        if _dev_briefs(load_config(Path(packet.config_path))) != packet.briefs:
            problems.append("development brief checksums changed")
    except (OSError, ValueError, ConfigError, briefs.BriefError) as error:
        problems.append(f"development inputs unavailable: {error}")
    return problems


def _run_rows(run: Path, packet: Comparison) -> tuple[list[dict[str, Any]], list[str]]:
    manifest = read_manifest(run)
    summary = manifest.stories
    if summary is None:
        return [], ["missing story run summary"]
    records = read_stories(run)
    calls = read_calls(run)
    rows: list[dict[str, Any]] = []
    problems = [] if manifest.status == "succeeded" else [f"run {manifest.status}"]
    expected = {
        (condition, brief, repetition)
        for condition in ("B", "C")
        for brief in packet.briefs
        for repetition in range(summary.repeat)
    }
    identities = [(item.condition, item.brief, item.repetition) for item in records]
    if set(identities) != expected or len(identities) != len(expected):
        problems.append("incomplete or duplicate dev story cohort")
    for story in records:
        row: dict[str, Any] = story.model_dump(mode="json")
        row["new_spend_usd"] = sum(
            call.cost_usd
            for call in calls
            if (call.condition, call.brief, call.repetition)
            == (story.condition, story.brief, story.repetition)
        )
        path = run / story.path / STORY_FILE
        budget = story_budget(calls, story.condition, story.brief, story.repetition)
        row["attributed_budget"] = budget.model_dump(mode="json")
        try:
            text = selected_text(run, story)
        except (OSError, ValueError) as error:
            text = ""
            problems.append(f"{story.condition}/{story.brief}: selected prose unavailable: {error}")
        row["final_story_sha256"] = checksum_lf(text.encode("utf-8")) if text else None
        words = word_count(text)
        row["final_words"] = words
        row["length_valid"] = bool(
            path.is_file()
            and story.status == "ok"
            and (abs(words - story.target_words) <= summary.length_tolerance * story.target_words)
        )
        if not row["length_valid"] or budget != story.budget:
            problems.append(f"{story.condition}/{story.brief}: invalid length or incomplete budget")
        if story.condition == "C":
            plan_file = run / story.path / "plan.json"
            row["plan_valid"] = plan_file.is_file() and validate_plan(load_plan(plan_file)).valid
            if not row["plan_valid"]:
                problems.append(f"{story.brief}: plan invalid or missing")
        rows.append(row)
    matching = summary.equal_budget
    if matching is None or len(matching.matches) != len(packet.briefs) * summary.repeat:
        problems.append("incomplete budget comparisons")
    else:
        for match in matching.matches:
            a = story_budget(calls, "B", match.brief, match.repetition).tokens
            b = story_budget(calls, "C", match.brief, match.repetition).tokens
            if not match.matched or not within(a, b, matching.tolerance):
                problems.append(f"{match.brief}: unmatched experimental budgets")
    return rows, problems


def report(folder: Path) -> dict[str, Any]:
    packet = load_comparison(folder)
    problems = _comparison_problems(packet)
    result: dict[str, Any] = {
        "comparison_sha256": packet.id,
        "briefs": packet.briefs,
        "tiers": {},
        "owner_judgements": [],
        "synthetic_test_only_judgements": [],
        "problems": problems,
    }
    for tier, run in packet.runs.items():
        try:
            rows, pending = _run_rows(Path(run), packet)
        except (OSError, ValueError) as error:
            rows, pending = [], [f"run evidence unreadable: {error}"]
        realisations: list[dict[str, Any]] = []
        for path in sorted((folder / "realisation" / tier).glob("*")):
            if path.is_dir():
                try:
                    realisations.append(realisation.report(path))
                except (OSError, ValueError) as error:
                    realisations.append({"owner_status": "pending", "missing_evidence": str(error)})
        result["tiers"][tier] = {"stories": rows, "problems": pending, "realisation": realisations}
        problems.extend(f"{tier}: {item}" for item in pending)
    for path in sorted((folder / "reviews").glob("*.json")):
        review = DevReview.model_validate_json(path.read_text(encoding="utf-8"))
        if review.comparison_sha256 != packet.id:
            problems.append("owner review belongs to a different comparison")
            continue
        key = "synthetic_test_only_judgements" if review.test_only else "owner_judgements"
        result[key].extend(
            item.model_dump(mode="json") | {"reviewer": review.reviewer}
            for item in review.judgements
        )
    result["status"] = "incomplete" if problems else "complete"
    result["owner_status"] = "available" if result["owner_judgements"] else "pending"
    write_text_lf(folder / "report.json", json.dumps(result, indent=2) + "\n")
    lines = [
        "# Development tier comparison",
        f"Comparison {packet.id}",
        f"Generation/budget status: {result['status']}",
        f"Owner dev judgement status: {result['owner_status']}",
        "LLM diagnostics cannot choose a tier or establish literary realisation.",
    ]
    for tier, value in result["tiers"].items():
        lines.append(f"## Tier {tier}")
        table = [
            "| Brief | Condition | Calls | Tokens | New USD | Words | Length | Plan |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for row in value["stories"]:
            budget = row["attributed_budget"]
            tokens = budget["input_tokens"] + budget["output_tokens"]
            table.append(
                f"| {row['brief']} r{row['repetition']} | {row['condition']} | {budget['calls']} | "
                f"{tokens} | {row['new_spend_usd']:.8f} | {row['final_words']} | "
                f"{row['length_valid']} | {row.get('plan_valid', 'beat outline')} |"
            )
        lines.append("\n".join(table))
        for evidence in value["realisation"]:
            lines.append(
                f"Realisation {evidence.get('sample', [])}: owner {evidence['owner_status']}; "
                f"assessed {evidence.get('assessed', 0)}/{evidence.get('denominator', 'unknown')}; "
                f"missing {evidence.get('missing', 'unknown')}; "
                f"unknown {evidence.get('unknown', 'unknown')}; "
                f"alignment {evidence.get('alignment', 'unresolved')}; "
                f"stale {evidence.get('stale', evidence.get('missing_evidence', []))}."
            )
    lines.extend(
        [
            "## Owner development judgements",
            json.dumps(result["owner_judgements"]),
            "## Synthetic test-only judgements",
            json.dumps(result["synthetic_test_only_judgements"]),
        ]
    )
    lines.extend(["## Pending/problems", *problems])
    write_text_lf(folder / "report.md", "\n\n".join(lines) + "\n")
    return result


def review(folder: Path, assessment: DevReview) -> None:
    packet = load_comparison(folder)
    if assessment.comparison_sha256 != packet.id or _comparison_problems(packet):
        raise ValueError("review comparison identity is mismatched or stale")
    seen: set[tuple[str, Tier, int]] = set()
    for item in assessment.judgements:
        key = item.brief, item.tier, item.repetition
        records = read_stories(Path(packet.runs[item.tier]))
        if (
            key in seen
            or item.brief not in packet.briefs
            or not any(
                (story.brief, story.condition, story.repetition, story.status)
                == (item.brief, "C", item.repetition, "ok")
                for story in records
            )
        ):
            raise ValueError("review must identify distinct successful dev C stories")
        seen.add(key)
    history = folder / "reviews"
    history.mkdir(exist_ok=True)
    path = (
        history
        / f"{len(list(history.glob('*.json'))) + 1:06d}-{_digest(assessment.model_dump())}.json"
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(assessment.model_dump_json(indent=2) + "\n")


def load_selection(path: Path) -> Selection:
    selection = Selection.model_validate_json(path.read_text(encoding="utf-8"))
    if selection.id != _identity(selection):
        raise ValueError("selection identity changed; create a new immutable selection")
    return selection


def _selected_sample(selection: Selection, packet: Comparison) -> list[str]:
    sample: set[str] = set()
    for name, expected in selection.review_sha256.items():
        path = Path(name)
        if not path.is_file() or checksum_lf(path.read_bytes()) != expected:
            raise ValueError("selection owner review missing or changed")
        assessment = DevReview.model_validate_json(path.read_text(encoding="utf-8"))
        if (assessment.comparison_sha256, assessment.reviewer, assessment.test_only) != (
            packet.id,
            selection.owner,
            selection.test_only,
        ):
            raise ValueError("selection owner review attribution mismatch")
        sample.update(item.brief for item in assessment.judgements if item.tier == selection.tier)
    return sorted(sample)


def selection_problems(selection: Selection, config_path: Path) -> list[str]:
    """Revalidate content, dev evidence and live settings; test-only status is caller-visible."""
    problems: list[str] = []
    if selection.id != _identity(selection):
        problems.append("selection identity changed")
    try:
        packet = load_comparison(Path(selection.comparison))
        problems.extend(_comparison_problems(packet))
        if packet.id != selection.comparison_sha256 or packet.briefs != selection.briefs:
            problems.append("selection comparison/dev brief identity mismatch")
        if selection.config != packet.config or any(
            getattr(selection, name) != getattr(packet.settings, name)
            for name in Settings.model_fields
        ):
            problems.append("selection settings do not match the reviewed comparison")
        if (
            _selected_sample(selection, packet) != selection.owner_sample
            or not selection.owner_sample
        ):
            problems.append("owner dev sample missing or inconsistent")
        for tier, run in packet.runs.items():
            _, pending = _run_rows(Path(run), packet)
            problems.extend(f"{tier}: {item}" for item in pending)
        current = settings_identity(config_path)
        for name in Settings.model_fields:
            if getattr(selection, name) != getattr(current, name):
                problems.append(f"selected {name} does not match current configuration")
        loaded = load_config(config_path)
        stories = loaded.config.stories
        candidates = (
            []
            if stories is None
            else [item for item in stories.conditions if item.pipeline == "baseline_c"]
        )
        if len(candidates) != 1 or candidates[0].params.get("tier", "L") != selection.tier:
            problems.append("selected tier does not match C configuration")
    except (OSError, ValueError, ConfigError, briefs.BriefError) as error:
        problems.append(f"selection evidence unavailable: {error}")
    return problems


def select(folder: Path, choice: Choice, out: Path) -> Selection:
    packet = load_comparison(folder)
    if report(folder)["status"] != "complete":
        raise ValueError("cannot select from incomplete dev comparisons")
    review_files: dict[str, str] = {}
    for path in sorted((folder / "reviews").glob("*.json")):
        assessment = DevReview.model_validate_json(path.read_text(encoding="utf-8"))
        if assessment.reviewer == choice.owner and assessment.test_only == choice.test_only:
            review_files[str(path.resolve())] = checksum_lf(path.read_bytes())
    selection = Selection(
        id="",
        tier=choice.tier,
        test_only=choice.test_only,
        owner=choice.owner,
        rationale=choice.rationale,
        comparison=str(folder.resolve()),
        comparison_sha256=packet.id,
        briefs=packet.briefs,
        owner_sample=[],
        config=packet.config,
        review_sha256=review_files,
        **packet.settings.model_dump(),
    )
    selection.owner_sample = _selected_sample(selection, packet)
    if not selection.owner_sample:
        raise ValueError("explicit owner dev judgements for the selected tier are required")
    selection.id = _identity(selection)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(selection.model_dump_json(indent=2) + "\n")
    return selection


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="tiers_command", required=True)
    comparison = commands.add_parser("compare", help="generate B/C comparisons for S/M/L on dev")
    comparison.add_argument("config", type=Path)
    comparison.add_argument("--out", type=Path, required=True)
    render = commands.add_parser("report", help="rebuild a comparison report from saved evidence")
    render.add_argument("folder", type=Path)
    assessment = commands.add_parser("review", help="append explicit owner dev assessments")
    assessment.add_argument("folder", type=Path)
    assessment.add_argument("file", type=Path)
    choice = commands.add_parser("select", help="freeze an owner tier/configuration choice")
    choice.add_argument("folder", type=Path)
    choice.add_argument("--tier", choices=TIERS, required=True)
    choice.add_argument("--owner", required=True)
    choice.add_argument("--rationale", required=True)
    choice.add_argument("--test-only", action="store_true")
    choice.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("check", help="check a saved selection against current settings")
    check.add_argument("selection", type=Path)
    check.add_argument("config", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        if args.tiers_command == "check":
            problems = selection_problems(load_selection(args.selection), args.config)
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1 if problems else 0
        if args.tiers_command == "compare":
            compare(args.config, args.out)
            return 0 if report(args.out)["status"] == "complete" else 1
        if args.tiers_command == "report":
            return 0 if report(args.folder)["status"] == "complete" else 1
        if args.tiers_command == "review":
            review(
                args.folder, DevReview.model_validate_json(args.file.read_text(encoding="utf-8"))
            )
        else:
            select(
                args.folder,
                Choice(
                    tier=args.tier,
                    owner=args.owner,
                    rationale=args.rationale,
                    test_only=args.test_only,
                ),
                args.out,
            )
        return 0
    except (OSError, ValueError, ConfigError, briefs.BriefError) as error:
        print(f"nde tiers: {error}", file=sys.stderr)
        return 2
