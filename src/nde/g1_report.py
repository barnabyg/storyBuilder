"""Reproducible decisions from closed, protocol-bound G1 owner evaluations."""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from nde import realisation
from nde.annotations.records import digest
from nde.eval.batch import (
    BatchError,
    Judgement,
    Key,
    StoryScores,
    View,
    parse_judgement,
    read_closed,
    read_judgements,
    read_key,
    read_view,
    read_warnings,
)
from nde.eval.g1 import G1Binding, binding_problems, file_checksum, read_g1_binding
from nde.eval.readiness import ReadinessReport
from nde.eval.stats import binomial_two_sided
from nde.files import write_text_lf
from nde.runs.records import STORY_FILE

FOLLOW_UP = (
    "At most one separate, pre-registered fixed-size follow-up on fresh held-out briefs: "
    "either a larger unchanged comparison, or the permitted diagnostic retry with stronger "
    "prose in both conditions. Preserve this result. Do not top up n, tune on revealed test "
    "stories, pool outcomes, or automatically launch a follow-up."
)


def _validate_scores(view: View, judgements: dict[int, Judgement]) -> list[str]:
    problems: list[str] = []
    if set(judgements) != set(range(1, view.size + 1)):
        problems.append("required owner judgements are missing or include extra pair identities")
    for number, judgement in judgements.items():
        fields = {"preferred": [judgement.preferred], "strength": [str(judgement.strength)]}
        if judgement.guess is not None:
            fields["guess"] = [judgement.guess]
        for side, scores in (("a", judgement.a), ("b", judgement.b)):
            if scores is not None:
                fields.update(
                    {f"score-{side}-{name}": [str(value)] for name, value in scores.scores.items()}
                )
                fields[f"tags-{side}"] = scores.tags
        try:
            parse_judgement(view, number, fields)
        except BatchError as error:
            problems.append(f"pair {number}: {error}")
        for scores in (judgement.a, judgement.b):
            if scores is None or any(
                name not in scores.scores for name in ("coherence", "originality")
            ):
                problems.append(f"pair {number}: required coherence/originality evidence missing")
    return problems


def _marks(key: Key, judgements: dict[int, Judgement]) -> list[tuple[StoryScores, StoryScores]]:
    rows: list[tuple[StoryScores, StoryScores]] = []
    for number, pair in enumerate(key.pairs, 1):
        judgement = judgements[number]
        if judgement.a is None or judgement.b is None:
            raise ValueError("required marks missing")
        if {pair.a.condition, pair.b.condition} != {"C", "B"}:
            raise ValueError("sealed mapping does not compare C with B")
        rows.append(
            (judgement.a, judgement.b) if pair.a.condition == "C" else (judgement.b, judgement.a)
        )
    return rows


def _caveats(
    batch: Path, judgements: dict[int, Judgement], owner_work: datetime | None
) -> list[str]:
    caveats = [warning.warning for warning in read_warnings(batch)]
    missing = sum(j.guess is None for j in judgements.values())
    if missing:
        caveats.append(f"Condition guesses absent for {missing} pairs; blinding is unconfirmed.")
    if owner_work is None:
        caveats.append("Owner code-work timing absent; the 48-hour reading delay is unconfirmed.")
    else:
        try:
            first = min(datetime.fromisoformat(j.at) for j in judgements.values())
            if owner_work.tzinfo is None or first.tzinfo is None:
                caveats.append("Owner/read timing lacks a timezone; reading delay is unconfirmed.")
            elif first - owner_work < timedelta(hours=48):
                caveats.append("Reading began less than 48 hours after recorded owner code work.")
        except (ValueError, TypeError):
            caveats.append("Owner/read timing cannot establish the 48-hour delay.")
    return caveats


def _cohort_problems(
    expected: set[tuple[str, int]], readiness: ReadinessReport, view: View, key: Key
) -> list[str]:
    problems: list[str] = []
    actual = [(pair.brief, pair.repetition) for pair in key.pairs]
    if set(actual) != expected or len(actual) != len(expected) or view.size != len(expected):
        problems.append("closed batch is not the exact registered cohort")
    admitted = [(pair.brief, pair.repetition) for pair in readiness.pairs]
    if set(admitted) != expected or len(admitted) != len(expected):
        problems.append("readiness evidence does not cover every registered pair")
    for pair in readiness.pairs:
        if (
            pair.problems
            or len(pair.stories) != len(key.sources)
            or {story.condition for story in pair.stories} != {"C", "B"}
        ):
            problems.append(f"{pair.brief}: incomplete or invalid readiness evidence")
    if [source.condition for source in key.sources] != ["C", "B"]:
        problems.append("candidate/comparator order differs from registration")
    return problems


def _realisation_reports(binding: G1Binding, folders: list[Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for folder in folders:
        packet = realisation.read_packet(folder)
        row = realisation.report(folder)
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
            )
            == (
                Path(packet.run).resolve(),
                packet.story.condition,
                packet.story.brief,
                packet.story.repetition,
                Path(packet.story.path) / STORY_FILE,
            )
            and story.sha256 == packet.story_sha256
        ]
        problem: str | None = None
        if len(matches) != 1:
            problem = "full run/story identity is not admitted to this batch"
        else:
            plan_kind = "narrative" if packet.story.condition == "C" else "beat_outline"
            filename = "plan.json" if plan_kind == "narrative" else "outline.json"
            plan = (Path(packet.run) / packet.story.path / filename).resolve()
            expected = binding.artifact_checksums.get(str(plan))
            if (
                packet.plan_kind != plan_kind
                or expected is None
                or not plan.is_file()
                or (file_checksum(plan) != expected)
            ):
                problem = "packet plan identity is not bound to the admitted story"
        if problem is not None:
            row["stale"] = [*row["stale"], problem]
            row["owner_status"] = "pending"
        row["source"] = {
            "run": packet.run,
            "condition": packet.story.condition,
            "brief": packet.story.brief,
            "repetition": packet.story.repetition,
            "path": packet.story.path,
        }
        reports.append(row)
    return reports or [{"owner_status": "pending", "note": "no packets supplied"}]


def decision(batch: Path, packets: list[Path]) -> dict[str, Any]:
    binding = read_g1_binding(batch)
    protocol = binding.protocol
    view, key = read_view(batch), read_key(batch)
    judgements = read_judgements(batch)
    problems = binding_problems(batch, binding)
    problems += _validate_scores(view, judgements)
    expected = {(brief, 0) for brief in protocol.cohort.brief_checksums}
    problems += _cohort_problems(expected, binding.readiness, view, key)
    closed = read_closed(batch)
    if closed is None or closed.judged != len(expected):
        problems.append("closure evidence does not cover the registered cohort")
    coherence: dict[str, Any] = {
        "candidate": protocol.candidate,
        "denominator": len(expected),
        "threshold": protocol.win_threshold,
    }
    originality: dict[str, Any] = protocol.originality.model_dump(mode="json")
    secondary: dict[str, Any] = {}
    tags: dict[str, dict[str, int]] = {}
    criteria = "pending"
    if not problems:
        try:
            marks = _marks(key, judgements)
        except ValueError as error:
            problems.append(str(error))
        else:
            wins = sum(c.scores["coherence"] > b.scores["coherence"] for c, b in marks)
            ties = sum(c.scores["coherence"] == b.scores["coherence"] for c, b in marks)
            delta = sum(c.scores["originality"] - b.scores["originality"] for c, b in marks) / len(
                expected
            )
            coherence.update(
                candidate_wins=wins,
                comparator_wins=len(expected) - wins - ties,
                ties=ties,
                ties_are="candidate non-wins",
                exact_two_sided_p=binomial_two_sided(wins, len(expected)),
                meets_threshold=wins >= protocol.win_threshold,
            )
            originality.update(
                mean_candidate_minus_comparator=delta,
                lower_bound=-protocol.originality.margin,
                meets_safeguard=delta >= -protocol.originality.margin,
            )
            criteria = (
                "passed"
                if wins >= protocol.win_threshold and (delta >= -protocol.originality.margin)
                else "failed_inconclusive"
            )
            preferences = sum(
                (pair.a.condition if judgements[number].preferred == "A" else pair.b.condition)
                == protocol.candidate
                for number, pair in enumerate(key.pairs, 1)
            )
            secondary = {
                "candidate_overall_preferences": preferences,
                "comparator_overall_preferences": len(expected) - preferences,
                "strengths": dict(Counter(j.strength for j in judgements.values())),
            }
            for condition, index in (("C", 0), ("B", 1)):
                tags[condition] = dict(Counter(tag for row in marks for tag in row[index].tags))
    costs: dict[str, dict[str, float | int]] = {}
    for condition in protocol.cohort.conditions:
        admitted = [
            s for p in binding.readiness.pairs for s in p.stories if s.condition == condition
        ]
        costs[condition] = {
            "calls": sum(s.budget.calls for s in admitted),
            "generation_tokens": sum(s.budget.tokens for s in admitted),
            "new_spend_usd": sum(s.new_spend_usd for s in admitted),
        }
    result: dict[str, Any] = {
        "version": "1",
        "test_only": binding.test_only,
        "outcome": "pending" if binding.test_only or problems else criteria,
        "criteria_outcome": criteria,
        "pending": problems,
        "protocol_id": protocol.id,
        "protocol_sha256": binding.protocol_sha256,
        "protocol_commit": binding.protocol_commit,
        "protocol": protocol.model_dump(mode="json"),
        "readiness": binding.readiness.model_dump(mode="json"),
        "coherence": coherence,
        "originality": originality,
        "secondary_preference": secondary,
        "failure_tags": tags,
        "costs": costs,
        "realisation": _realisation_reports(binding, packets),
        "judges": "diagnostic only; see the ordinary closed-batch report for available agreement",
        "reading_caveats": _caveats(batch, judgements, binding.owner_code_work_at),
        "follow_up": FOLLOW_UP,
        "judgements_sha256": digest(
            json.dumps(
                {str(n): j.model_dump(mode="json") for n, j in judgements.items()}, sort_keys=True
            )
        ),
    }
    if binding.test_only:
        result["pending"].append(
            "Synthetic test-only judgements cannot establish a scientific gate."
        )
    result["id"] = digest(json.dumps(result, sort_keys=True))
    return result


def write_decision(batch: Path, out: Path, packets: list[Path]) -> dict[str, Any]:
    if out.exists():
        raise ValueError(
            "decision already exists; preserve completed results and choose a new folder"
        )
    result = decision(batch, packets)
    out.mkdir(parents=True)
    write_text_lf(out / "decision.json", json.dumps(result, indent=2) + "\n")
    lines = [
        "# G1 decision",
        f"Outcome: {result['outcome']}",
        f"Registered criteria: {result['criteria_outcome']}",
        f"Test only: {result['test_only']}",
        f"Result identity: {result['id']}",
        f"Protocol: {result['protocol_sha256']}",
        "Coherence is primary. Originality is an independent registered safeguard. "
        "Overall preference and judges are diagnostic/secondary.",
        json.dumps(result["coherence"], indent=2),
        json.dumps(result["originality"], indent=2),
    ]
    lines.extend(result["pending"])
    lines.extend(result["reading_caveats"])
    lines.extend(
        [
            "Costs: " + json.dumps(result["costs"]),
            "Realisation: " + json.dumps(result["realisation"]),
            FOLLOW_UP,
        ]
    )
    write_text_lf(out / "decision.md", "\n\n".join(lines) + "\n")
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
        print(f"nde g1-report: {error}", file=sys.stderr)
        return 2
