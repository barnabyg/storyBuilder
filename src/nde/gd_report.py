"""Immutable G-D decisions; fixture criteria can never promote a real baseline."""

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from nde import realisation
from nde.annotations.records import digest
from nde.controllability import replay_report
from nde.eval.batch import (
    BatchError,
    Judgement,
    Key,
    StoryScores,
    View,
    read_closed,
    read_judgements,
    read_key,
    read_view,
)
from nde.eval.g1 import file_checksum
from nde.eval.gd_artifacts import GDBinding, binding_problems, read_gd_binding
from nde.files import write_text_lf
from nde.g1_report import reading_caveats, validate_required_scores
from nde.runs.config import load_config
from nde.runs.records import read_calls

PHASE_6 = (
    "A completed G-D failure retains C and does not itself block Phase 6 when the separate "
    "G1 and G2 prerequisites hold. This report authorises no automatic later experiment."
)
BLINDING_THRESHOLD = 0.7


def _cohort_problems(binding: GDBinding, view: View, key: Key) -> list[str]:
    expected = {(brief, 0) for brief in binding.protocol.cohort.brief_checksums}
    actual = [(pair.brief, pair.repetition) for pair in key.pairs]
    admitted = [(pair.brief, pair.repetition) for pair in binding.readiness.pairs]
    problems: list[str] = []
    if set(actual) != expected or len(actual) != len(expected) or view.size != len(expected):
        problems.append("closed batch differs from the fixed distinct G-D cohort")
    if set(admitted) != expected or len(admitted) != len(expected):
        problems.append("admission evidence does not cover every registered pair")
    if tuple(source.condition for source in key.sources) != ("D", "C"):
        problems.append("candidate direction differs from registered D-versus-C order")
    for pair in binding.readiness.pairs:
        if pair.problems or {story.condition for story in pair.stories} != {"D", "C"}:
            problems.append(f"{pair.brief}: incomplete or invalid final-story admission")
    for pair in key.pairs:
        if {pair.a.condition, pair.b.condition} != {"D", "C"}:
            problems.append(f"{pair.brief}: sealed pair does not compare D with C")
        admitted_pair = next(
            (
                row
                for row in binding.readiness.pairs
                if (row.brief, row.repetition) == (pair.brief, pair.repetition)
            ),
            None,
        )
        if admitted_pair is not None:
            expected_stories = {
                (s.condition, Path(s.run).resolve(), Path(s.path)) for s in admitted_pair.stories
            }
            sealed_stories = {
                (s.condition, Path(s.run).resolve(), Path(s.story)) for s in (pair.a, pair.b)
            }
            if sealed_stories != expected_stories:
                problems.append(f"{pair.brief}: sealed sources differ from admitted final stories")
    return problems


def _marks(key: Key, judgements: dict[int, Judgement]) -> list[tuple[StoryScores, StoryScores]]:
    rows: list[tuple[StoryScores, StoryScores]] = []
    for number, pair in enumerate(key.pairs, 1):
        judgement = judgements[number]
        if judgement.a is None or judgement.b is None:
            raise ValueError("required owner scores missing")
        rows.append(
            (judgement.a, judgement.b) if pair.a.condition == "D" else (judgement.b, judgement.a)
        )
    return rows


def _comparison(
    marks: list[tuple[StoryScores, StoryScores]], dimension: str, margin: float
) -> dict[str, Any]:
    deltas = [d.scores[dimension] - c.scores[dimension] for d, c in marks]
    mean = sum(deltas) / len(deltas)
    return {
        "analysis": "mean_paired_difference",
        "claim": "descriptive ordinal-score comparison",
        "candidate": "D",
        "comparator": "C",
        "denominator": len(deltas),
        "paired_differences": deltas,
        "mean_candidate_minus_comparator": mean,
        "margin": margin,
        "lower_bound": -margin,
        "meets_criterion": mean >= -margin,
        "candidate_wins": sum(delta > 0 for delta in deltas),
        "comparator_wins": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "ties_are": "zero paired differences",
        "inferential_noninferiority": False,
    }


def _costs(binding: GDBinding) -> dict[str, Any]:
    costs: dict[str, Any] = {}
    for condition in ("D", "C"):
        stories = [
            story
            for pair in binding.readiness.pairs
            for story in pair.stories
            if story.condition == condition
        ]
        try:
            calls = [
                call
                for root in sorted({story.run for story in stories})
                for call in read_calls(Path(root))
                if call.condition == condition
            ]
        except (OSError, ValueError) as error:
            costs[condition] = {"status": "unavailable", "reason": str(error)}
            continue
        costs[condition] = {
            "calls": sum(s.budget.calls for s in stories),
            "generation_tokens": sum(s.budget.tokens for s in stories),
            "new_spend_usd": sum(s.new_spend_usd for s in stories),
            "list_price_usd": sum(call.token_cost_usd for call in calls),
        }
    return costs


def _histories(binding: GDBinding) -> dict[str, Any]:
    histories: dict[str, Any] = {}
    for name, checksum in sorted(binding.artifact_checksums.items()):
        if not (
            name.endswith(".json")
            and any(
                marker in Path(name).parts[-1]
                for marker in ("provenance", "refinement", "selection")
            )
        ):
            continue
        try:
            record = json.loads(Path(name).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            record = {"status": "unavailable", "reason": str(error)}
        histories[name] = {"sha256": checksum, "record": record}
    return histories


def _realisation(binding: GDBinding, packets: list[Path]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for folder in packets:
        try:
            packet, row = realisation.read_packet(folder), realisation.report(folder)
        except (OSError, ValueError) as error:
            evidence.append(
                {"owner_status": "pending", "source": str(folder), "reason": str(error)}
            )
            continue
        matches = [
            story
            for pair in binding.readiness.pairs
            for story in pair.stories
            if (
                Path(story.run).resolve(),
                story.condition,
                pair.brief,
                pair.repetition,
                Path(story.path),
                story.sha256,
            )
            == (
                Path(packet.run).resolve(),
                packet.story.condition,
                packet.story.brief,
                packet.story.repetition,
                Path(packet.story.path) / "story.md",
                packet.story_sha256,
            )
        ]
        plan = (Path(packet.run) / packet.story.path / "plan.json").resolve()
        if (
            len(matches) != 1
            or packet.plan_kind != "narrative"
            or not plan.is_file()
            or binding.artifact_checksums.get(str(plan)) != file_checksum(plan)
        ):
            row["stale"] = [*row["stale"], "packet does not bind the admitted final plan/story"]
            row["owner_status"] = "pending"
        row["source"] = {
            "run": packet.run,
            "condition": packet.story.condition,
            "brief": packet.story.brief,
            "repetition": packet.story.repetition,
        }
        evidence.append(row)
    return evidence or [{"owner_status": "pending", "note": "No realisation packets supplied"}]


def _reference(binding: GDBinding, condition: str) -> dict[str, Any]:
    loaded = load_config(Path(binding.config_path))
    stories = loaded.config.stories
    if stories is None:
        raise ValueError("registered story configuration missing")
    chosen = next(item for item in stories.conditions if item.name == condition)
    return {
        "condition": condition,
        "pipeline": chosen.pipeline,
        "params": chosen.params,
        "tier": binding.protocol.tier if condition == "C" else "L",
        "config_sha256": binding.protocol.config_sha256,
        "selection_sha256": binding.protocol.selection_sha256,
        "schema_sha256": binding.protocol.schema_sha256,
        "models": {
            name: model.model_dump(mode="json") for name, model in loaded.config.roles.items()
        },
        "models_sha256": binding.protocol.models_sha256,
        "prompts": binding.protocol.prompt_sha256,
        "protocol_sha256": binding.protocol_sha256,
        "protocol_commit": binding.protocol_commit,
        "planning_artifacts": {
            name: checksum
            for name, checksum in binding.artifact_checksums.items()
            if f"/{condition}/" in Path(name).as_posix()
            and ("provenance" in name or name.endswith("plan.json"))
        },
    }


def _pending(batch: Path, reason: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "version": "1",
        "outcome": "pending",
        "criteria_outcome": "pending",
        "test_only": None,
        "pending": [reason],
        "reference_baseline": None,
        "batch": str(batch.resolve()),
        "phase_6": PHASE_6,
    }
    result["id"] = digest(json.dumps(result, sort_keys=True))
    return result


def _blinding(binding: GDBinding, key: Key, judgements: dict[int, Judgement]) -> dict[str, Any]:
    assessed = [
        (pair, judgements[number])
        for number, pair in enumerate(key.pairs, 1)
        if number in judgements and judgements[number].guess in binding.guess_conditions
    ]
    correct = sum(
        binding.guess_conditions[j.guess or ""] == pair.a.condition for pair, j in assessed
    )
    accuracy = correct / len(assessed) if assessed else None
    return {
        "guesses": len(assessed),
        "correct": correct,
        "accuracy": accuracy,
        "flag_threshold": BLINDING_THRESHOLD,
        "possibly_unblinded": accuracy is not None and accuracy > BLINDING_THRESHOLD,
    }


def decision(batch: Path, packets: list[Path]) -> dict[str, Any]:
    try:
        binding = read_gd_binding(batch)
        view, key, judgements = read_view(batch), read_key(batch), read_judgements(batch)
    except (OSError, ValueError, BatchError) as error:
        return _pending(batch, str(error))
    protocol = binding.protocol
    problems = binding_problems(batch, binding) + validate_required_scores(view, judgements)
    problems += _cohort_problems(binding, view, key)
    closed = read_closed(batch)
    if closed is None or closed.judged != protocol.n:
        problems.append("closure evidence does not cover the registered cohort")
    coherence: dict[str, Any] = {}
    originality: dict[str, Any] = {}
    secondary: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    criteria = "pending"
    if not problems:
        controls = {
            brief: replay_report(Path(path)).model_dump(mode="json")
            for brief, path in sorted(binding.controls.items())
        }
        marks = _marks(key, judgements)
        coherence = _comparison(marks, "coherence", protocol.coherence.margin)
        originality = _comparison(marks, "originality", protocol.originality.margin)
        control_success = all(report["status"] == "success" for report in controls.values())
        criteria = (
            "passed"
            if (coherence["meets_criterion"] and originality["meets_criterion"] and control_success)
            else "failed"
        )
        preferences = sum(
            (pair.a.condition if judgements[i].preferred == "A" else pair.b.condition) == "D"
            for i, pair in enumerate(key.pairs, 1)
        )
        secondary = {
            "candidate_overall_preferences": preferences,
            "comparator_overall_preferences": protocol.n - preferences,
            "strengths": dict(Counter(j.strength for j in judgements.values())),
        }
    outcome = "pending" if protocol.test_only or problems else criteria
    if protocol.test_only:
        problems.append(
            "Synthetic test-only outcomes cannot promote a scientific reference baseline."
        )
    blinding = _blinding(binding, key, judgements)
    result: dict[str, Any] = {
        "version": "1",
        "outcome": outcome,
        "criteria_outcome": criteria,
        "test_only": protocol.test_only,
        "pending": problems,
        "protocol": protocol.model_dump(mode="json"),
        "protocol_sha256": binding.protocol_sha256,
        "protocol_commit": binding.protocol_commit,
        "coherence": coherence,
        "originality": originality,
        "controllability": controls,
        "secondary_preference": secondary,
        "readiness": binding.readiness.model_dump(mode="json"),
        "artifact_identities": binding.artifact_checksums,
        "costs": _costs(binding),
        "revision_histories": _histories(binding),
        "realisation": _realisation(binding, packets),
        "critic_and_judges": (
            "Diagnostic only; not owner evidence or a substitute for the primary criterion"
        ),
        "reading_caveats": [
            *reading_caveats(batch, judgements, binding.owner_code_work_at),
            protocol.prior_benchmark_exposure,
            "Condition-guess accuracy alone cannot establish successful blinding.",
        ],
        "blinding": blinding,
        "reference_baseline": _reference(binding, "D" if outcome == "passed" else "C")
        if outcome in {"passed", "failed"}
        else None,
        "phase_6": PHASE_6,
        "judgements_sha256": digest(
            json.dumps(
                {str(n): j.model_dump(mode="json") for n, j in judgements.items()}, sort_keys=True
            )
        ),
    }
    result["id"] = digest(json.dumps(result, sort_keys=True))
    return result


def write_decision(batch: Path, out: Path, packets: list[Path]) -> dict[str, Any]:
    if out.exists():
        raise ValueError(
            "G-D decision already exists; preserve prior evidence and choose a new folder"
        )
    result = decision(batch, packets)
    out.mkdir(parents=True)
    text = json.dumps(result, indent=2) + "\n"
    write_text_lf(out / "decision.json", text)
    write_text_lf(
        out / "decision.md", f"# G-D decision: {result['outcome']}\n\n```json\n{text}```\n"
    )
    write_text_lf(
        out / "decision.html",
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>G-D decision</title><body><pre>{html.escape(text)}</pre></body></html>\n",
    )
    if result["reference_baseline"] is not None:
        write_text_lf(
            out / "reference-baseline.json",
            json.dumps(result["reference_baseline"], indent=2) + "\n",
        )
    return result


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("batch", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--realisation", type=Path, action="append", default=[])


def run(args: argparse.Namespace) -> int:
    try:
        write_decision(args.batch, args.out, args.realisation)
        return 0
    except (OSError, ValueError, BatchError) as error:
        print(f"nde gd-report: {error}", file=sys.stderr)
        return 2
