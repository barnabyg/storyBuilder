"""Inspectable paired budgets and identities of the final selected prose."""

from pathlib import Path
from typing import Any

from nde.annotations.records import digest
from nde.runs.records import (
    STORY_FILE,
    BudgetMatch,
    CallRecord,
    EqualBudgetSummary,
    LocalWorkRecord,
    Manifest,
    StoryRecord,
    read_local_work,
)
from nde.stories.budget import story_budget, within
from nde.stories.normalise import word_count
from nde.stories.resources import story_resources
from nde.stories.selected import selected_text

# Under a registered resource policy, what a matched pair's budget does and does not mean.
BUDGET_BASIS = (
    "provider generation tokens only; local conditional-model work is reported "
    "separately and never equated"
)


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


def _local_compute(
    spec: EqualBudgetSummary,
    pair: dict[str, Any],
    match: BudgetMatch | None,
    usage: tuple[list[CallRecord], list[LocalWorkRecord]],
    problems: list[str],
) -> None:
    """Disclose each side's local model work; it never joins the provider-token budget."""
    if spec.resource_policy is None:
        return
    rows: dict[str, Any] = pair["conditions"]
    for name, row in rows.items():
        row["resources"] = story_resources(*usage, name, pair["brief"], pair["repetition"])
    local = {name: row["resources"]["local"] for name, row in rows.items()}
    units = {
        name: (value["operations"], value["passes"], value["tokens"], value["attention_cells"])
        for name, value in local.items()
    }
    pair["budget_basis"] = BUDGET_BASIS
    pair["local_compute_equal"] = len(set(units.values())) <= 1
    if any(row["tokens"] == 0 for row in rows.values()):
        problems.append("zero attributable provider tokens cannot establish an equal budget")
    recorded = None if match is None else match.local_work
    if recorded is not None and any(
        name in recorded
        and recorded[name].model_dump(exclude={"elapsed_seconds"})
        != {key: value for key, value in local[name].items() if key != "elapsed_seconds"}
        for name in local
    ):
        problems.append("matching local work differs from recorded local work")


def compare(
    run: Path, manifest: Manifest, records: list[StoryRecord], calls: list[CallRecord]
) -> dict[str, Any]:
    summary = manifest.stories
    if summary is None or summary.equal_budget is None:
        return {"pairs": []}
    spec = summary.equal_budget
    work = [] if spec.resource_policy is None else read_local_work(run)
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
            pair: dict[str, Any] = {
                "brief": brief,
                "repetition": repetition,
                "conditions": conditions,
                "matching": None if match is None else match.model_dump(mode="json"),
            }
            _local_compute(spec, pair, match, (calls, work), problems)
            pairs.append(pair | {"comparable": not problems, "problems": problems})
    result: dict[str, Any] = {
        "version": "1",
        "run_status": manifest.status,
        "config_hash": manifest.config_hash,
        "budget_measure": "generation_tokens",
        "budget_tolerance": spec.tolerance,
        "length_tolerance": summary.length_tolerance,
        "max_revision_passes": spec.max_revision_passes,
        "pairs": pairs,
    }
    policy = spec.resource_policy
    return result | (
        {} if policy is None else {"resource_policy": policy, "budget_basis": BUDGET_BASIS}
    )


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
        if "budget_basis" in pair:
            result.append(
                f"- Budget basis: {pair['budget_basis']}; local compute equal: "
                f"{'yes' if pair['local_compute_equal'] else 'no'}"
            )
        result.extend(f"- Unresolved: {problem}" for problem in pair["problems"])
        result.append("")
    return result
