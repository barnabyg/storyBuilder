"""Development C/D experiments reuse the existing final-story and budget pipeline."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from nde import briefs, realisation, tiers
from nde.files import checksum_lf, write_text_lf
from nde.narrative.workflow import schema_json
from nde.runs.config import ConfigError, ExperimentConfig, load_config
from nde.runs.prompts import PromptTemplate
from nde.runs.records import read_calls, read_manifest, read_stories
from nde.runs.runner import RunOptions, run_experiment
from nde.stories.baseline_c import BaselineCParams
from nde.stories.budget import story_budget
from nde.stories.comparison import compare as compare_final
from nde.stories.execute import prepare_stories


def _c_params(config: ExperimentConfig) -> BaselineCParams:
    found = (
        []
        if config.stories is None
        else [item for item in config.stories.conditions if item.pipeline == "baseline_c"]
    )
    if len(found) != 1:
        raise ValueError("comparison requires exactly one selected C condition")
    return BaselineCParams.model_validate(found[0].params)


def selected_c_problems(selection: tiers.Selection, config_path: Path) -> list[str]:
    """Check historical selection evidence, then compare only the selected C contract.

    D's extra critic/prompts and the changed comparison arm do not stale C's
    already reviewed settings; its planner, prose, schema and prompts still must match.
    """
    historical = Path(selection.comparison) / "configs" / f"{selection.tier}.yaml"
    problems = tiers.selection_problems(selection, historical)
    try:
        loaded = load_config(config_path)
        current = _c_params(loaded.config)
        original = _c_params(ExperimentConfig.model_validate(selection.config))
        expected = original.model_dump(exclude={"tier", "saved_plan", "provenance"})
        actual = current.model_dump(exclude={"tier", "saved_plan", "provenance"})
        if current.tier != selection.tier or actual != expected:
            problems.append("selected C tier/planning parameters differ from reviewed baseline")
        original_config = ExperimentConfig.model_validate(selection.config)
        if any(
            loaded.config.roles.get(role) != original_config.roles.get(role)
            for role in ("planner", "prose")
        ):
            problems.append("selected C planner/prose model bindings differ from reviewed baseline")
        if checksum_lf(schema_json().encode()) != selection.schema_sha256:
            problems.append("selected C schema differs from reviewed baseline")
        templates: dict[str, PromptTemplate] = {}
        prepare_stories(loaded, templates)
        for label in (
            current.skeleton_prompt.label,
            current.channels_prompt.label,
            "scene_prose@1",
        ):
            if label not in templates or templates[label].sha256 != selection.prompt_sha256.get(
                label
            ):
                problems.append(f"selected C/prose prompt differs from reviewed baseline: {label}")
    except (OSError, ValueError, ConfigError, briefs.BriefError) as failure:
        problems.append(f"selected C configuration unavailable: {failure}")
    return problems


def _dev_inputs(config_path: Path) -> dict[str, str]:
    loaded = load_config(config_path)
    stories, folder = loaded.config.stories, loaded.briefs_dir
    if stories is None or folder is None:
        raise ValueError("C/D comparison requires a stories configuration")
    if {c.name: c.pipeline for c in stories.conditions} != {"C": "baseline_c", "D": "baseline_d"}:
        raise ValueError("C/D comparison requires exactly C baseline_c and D baseline_d")
    if loaded.config.calls or loaded.config.judges is not None or loaded.config.plans is not None:
        raise ValueError("C/D development comparison cannot include unrelated calls or judges")
    if stories.equal_budget is None or set(stories.equal_budget.conditions) != {"C", "D"}:
        raise ValueError("C/D comparison requires shared equal-budget matching")
    available = briefs.load_briefs(folder)
    ids = stories.briefs.ids or sorted(
        brief.id for brief in available.values() if brief.split == stories.briefs.split
    )
    if (
        not ids
        or len(set(ids)) != len(ids)
        or stories.briefs.split == "test"
        or any(brief_id not in available or available[brief_id].split != "dev" for brief_id in ids)
    ):
        raise ValueError("C/D development comparison requires distinct dev-only briefs")
    if set(briefs.read_frozen(folder).get("test", {})) & set(ids):
        raise ValueError("C/D development inputs cannot reuse frozen held-out identities")
    return {
        brief_id: checksum_lf((folder / "dev" / f"{brief_id}.yaml").read_bytes())
        for brief_id in ids
    }


def compare(config_path: Path, selection_path: Path, out: Path) -> dict[str, Any]:
    dev = _dev_inputs(config_path)
    selection = tiers.load_selection(selection_path)
    problems = selected_c_problems(selection, config_path)
    if problems:
        raise ValueError("; ".join(problems))
    if out.exists():
        raise ValueError("comparison already exists; preserve history with a new folder")
    loaded = load_config(config_path)
    out.mkdir(parents=True)
    run = run_experiment(
        config_path,
        RunOptions(
            runs_dir=out / "runs",
            environ=dict(os.environ),
            cache_path=Path(os.environ.get("NDE_CACHE", out / "cache.sqlite")),
            ledger_path=Path(os.environ.get("NDE_LEDGER", out / "ledger.sqlite")),
        ),
    )
    for story in read_stories(run):
        if story.status == "ok":
            realisation.create(
                run,
                story.condition,
                story.brief,
                story.repetition,
                out / "realisation" / story.condition / f"{story.brief}-r{story.repetition}",
            )
    # Snapshot every run artifact; subsequent reports verify these identities.
    artifacts = {
        str(path.resolve()): checksum_lf(path.read_bytes())
        for path in sorted(run.rglob("*"))
        if path.is_file()
    }
    packet = {
        "version": "1",
        "config_path": str(config_path.resolve()),
        "config_sha256": loaded.hash,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": checksum_lf(selection_path.read_bytes()),
        "selected_c_id": selection.id,
        "selected_c_tier": selection.tier,
        "test_only_selection": selection.test_only,
        "briefs": dev,
        "run": str(run.resolve()),
        "artifacts": artifacts,
    }
    with (out / "cd-comparison.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(packet, indent=2) + "\n")
    return report(out)


def report(folder: Path) -> dict[str, Any]:
    packet = json.loads((folder / "cd-comparison.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    for name, checksum in packet["artifacts"].items():
        path = Path(name)
        if not path.is_file() or checksum_lf(path.read_bytes()) != checksum:
            problems.append(f"run artifact missing or changed: {name}")
    config = Path(packet["config_path"])
    selection_path = Path(packet["selection_path"])
    try:
        if (
            load_config(config).hash != packet["config_sha256"]
            or _dev_inputs(config) != packet["briefs"]
        ):
            problems.append("comparison configuration or dev briefs changed")
        if checksum_lf(selection_path.read_bytes()) != packet["selection_sha256"]:
            problems.append("selected C evidence changed")
        problems.extend(selected_c_problems(tiers.load_selection(selection_path), config))
    except (OSError, ValueError, ConfigError, briefs.BriefError) as failure:
        problems.append(f"comparison inputs unavailable: {failure}")
    run = Path(packet["run"])
    manifest = read_manifest(run)
    calls = read_calls(run)
    result = compare_final(run, manifest, read_stories(run), calls)
    for pair in result["pairs"]:
        match = pair["matching"]
        pair["matching_outcome"] = (
            "incomplete"
            if match is None
            else "no_revision_needed"
            if match["matched"] and not match["passes"]
            else "matched"
            if match["matched"]
            else "overshot"
            if "overshot" in (match["reason"] or "")
            else "exhausted_limit"
            if match["passes"] >= result["max_revision_passes"]
            else "incomplete"
        )
        for condition in ("C", "D"):
            if condition not in pair["conditions"]:
                budget = story_budget(calls, condition, pair["brief"], pair["repetition"])
                pair["conditions"][condition] = {
                    "status": "failed",
                    "missing": True,
                    "final_sha256": None,
                    "length_valid": False,
                    **budget.model_dump(mode="json"),
                    "tokens": budget.tokens,
                    "new_spend_usd": sum(
                        call.cost_usd
                        for call in calls
                        if (call.condition, call.brief, call.repetition)
                        == (condition, pair["brief"], pair["repetition"])
                    ),
                    "token_cost_usd": sum(
                        call.token_cost_usd
                        for call in calls
                        if (call.condition, call.brief, call.repetition)
                        == (condition, pair["brief"], pair["repetition"])
                    ),
                }
    result.update(
        selected_c_id=packet["selected_c_id"],
        selected_c_tier=packet["selected_c_tier"],
        test_only_selection=packet["test_only_selection"],
        briefs=packet["briefs"],
        input_identities=packet["artifacts"],
        problems=problems,
    )
    result["status"] = (
        "complete"
        if manifest.status == "succeeded"
        and not problems
        and all(pair["comparable"] for pair in result["pairs"])
        else "incomplete"
    )
    result["realisation"] = [
        realisation.report(path)
        for path in sorted((folder / "realisation").glob("*/*"))
        if path.is_dir()
    ]
    write_text_lf(folder / "report.json", json.dumps(result, indent=2) + "\n")
    lines = [
        "# C/D development comparison",
        f"Status: {result['status']}",
        f"Selected C: {packet['selected_c_id']} tier {packet['selected_c_tier']}",
        "All planning, critic, rejected regeneration, repairs, length and matching calls "
        "count as experimental tokens. New spend and list-price costs are separate.",
        "Owner realisation and literary decisions remain separate from structural compliance.",
    ]
    for pair in result["pairs"]:
        lines.append(
            f"## {pair['brief']} r{pair['repetition']}: "
            + ("comparable" if pair["comparable"] else "unresolved")
        )
        lines.append(json.dumps(pair, sort_keys=True))
    lines.extend(problems)
    write_text_lf(folder / "report.md", "\n\n".join(lines) + "\n")
    return result


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="cd_command", required=True)
    generate = commands.add_parser("run", help="compare selected C and D on dev briefs")
    generate.add_argument("config", type=Path)
    generate.add_argument("--selection", type=Path, required=True)
    generate.add_argument("--out", type=Path, required=True)
    render = commands.add_parser("report", help="revalidate saved comparison evidence")
    render.add_argument("folder", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        result = (
            compare(args.config, args.selection, args.out)
            if args.cd_command == "run"
            else report(args.folder)
        )
        return 0 if result["status"] == "complete" else 1
    except (OSError, ValueError, ConfigError, briefs.BriefError) as failure:
        print(f"nde compare-cd: {failure}", file=sys.stderr)
        return 2
