"""Inspectable paired budgets and identities of the final selected prose."""

from pathlib import Path
from typing import Any

from nde.annotations.records import digest
from nde.runs.records import STORY_FILE, CallRecord, Manifest, StoryRecord
from nde.stories.budget import story_budget, within
from nde.stories.normalise import word_count
from nde.stories.selected import selected_text


def _selected(run: Path, record: StoryRecord, calls: list[CallRecord]) -> dict[str, Any]:
    budget = story_budget(calls, record.condition, record.brief, record.repetition)
    mine = [
        c
        for c in calls
        if (c.condition, c.brief, c.repetition)
        == (record.condition, record.brief, record.repetition)
    ]
    result: dict[str, Any] = {
        "status": record.status,
        "error": record.error,
        "path": f"{record.path}/{STORY_FILE}",
        "revision_passes": record.revision_passes,
        "chosen_attempt": record.chosen_attempt,
        "calls": budget.calls,
        "input_tokens": budget.input_tokens,
        "output_tokens": budget.output_tokens,
        "tokens": budget.tokens,
        "new_spend_usd": sum(c.cost_usd for c in mine),
        "token_cost_usd": sum(c.token_cost_usd for c in mine),
        "budget_record_consistent": budget == record.budget,
    }
    try:
        text = selected_text(run, record)
        result.update(final_sha256=digest(text), words=word_count(text), target=record.target_words)
    except (OSError, ValueError) as error:
        result.update(final_sha256=None, words=None, file_error=str(error))
    return result


def compare(
    run: Path, manifest: Manifest, records: list[StoryRecord], calls: list[CallRecord]
) -> dict[str, Any]:
    summary = manifest.stories
    if summary is None or summary.equal_budget is None:
        return {"pairs": []}
    spec = summary.equal_budget
    pairs: list[dict[str, Any]] = []
    for brief in summary.brief_ids:
        for repetition in range(summary.repeat):
            problems: list[str] = []
            conditions: dict[str, Any] = {}
            for condition in spec.conditions:
                found = [
                    r
                    for r in records
                    if (r.brief, r.repetition, r.condition) == (brief, repetition, condition)
                ]
                if len(found) != 1:
                    problems.append(f"{condition}: missing or duplicate story")
                    continue
                record = found[0]
                row = _selected(run, record, calls)
                conditions[condition] = row
                words = row["words"]
                valid_length = words is not None and abs(words - record.target_words) <= (
                    summary.length_tolerance * record.target_words
                )
                row["length_valid"] = valid_length
                if record.status != "ok" or not row["final_sha256"]:
                    problems.append(f"{condition}: final prose unavailable or failed")
                if not valid_length or words != record.words:
                    problems.append(
                        f"{condition}: final length invalid or differs from saved record"
                    )
                if not row["budget_record_consistent"]:
                    problems.append(f"{condition}: attributable budget differs from saved record")
            matches = [m for m in spec.matches if (m.brief, m.repetition) == (brief, repetition)]
            match = matches[0] if len(matches) == 1 else None
            if match is None or not match.matched:
                problems.append("budget comparison incomplete or unmatched")
            if len(conditions) == len(spec.conditions):
                first, second = (conditions[name]["tokens"] for name in spec.conditions)
                if not within(first, second, spec.tolerance):
                    problems.append("actual attributable tokens outside budget tolerance")
                if match is not None and any(
                    match.after.get(name) != conditions[name]["tokens"] for name in spec.conditions
                ):
                    problems.append("matching outcome differs from attributable calls")
            pairs.append(
                {
                    "brief": brief,
                    "repetition": repetition,
                    "conditions": conditions,
                    "matching": None if match is None else match.model_dump(mode="json"),
                    "comparable": not problems,
                    "problems": problems,
                }
            )
    return {
        "version": "1",
        "run_status": manifest.status,
        "config_hash": manifest.config_hash,
        "budget_measure": "generation_tokens",
        "budget_tolerance": spec.tolerance,
        "length_tolerance": summary.length_tolerance,
        "max_revision_passes": spec.max_revision_passes,
        "pairs": pairs,
    }


def lines(comparison: dict[str, Any]) -> list[str]:
    if not comparison["pairs"]:
        return []
    result = [
        "## Final selected comparison",
        "",
        "Experimental tokens include cache hits and all attributable calls. "
        "New provider spend counts only calls paid in this run.",
        "",
    ]
    for pair in comparison["pairs"]:
        result.append(
            f"{pair['brief']} repetition {pair['repetition'] + 1}: "
            f"{'comparable' if pair['comparable'] else 'unresolved'}"
        )
        for condition, row in pair["conditions"].items():
            result.append(
                f"- {condition}: final `{row['path']}`, SHA-256 `{row['final_sha256']}`; "
                f"{row['words']} words; {row['tokens']} tokens; "
                f"new spend USD {row['new_spend_usd']:.6f}"
            )
        result.extend(f"- Unresolved: {problem}" for problem in pair["problems"])
        result.append("")
    return result
