"""Generate a run's Markdown report from its run directory."""

import json
from pathlib import Path

from nde.files import write_text_lf
from nde.runs.records import (
    REPORT,
    CallRecord,
    Manifest,
    OutputRecord,
    StoryRecord,
    read_calls,
    read_manifest,
    read_outputs,
    read_stories,
)


def render_report(run_dir: Path) -> str:
    manifest = read_manifest(run_dir)
    outputs = read_outputs(run_dir)
    calls = read_calls(run_dir)
    lines = [
        f"# Run report: {manifest.name}",
        "",
        f"- **Status:** {_status(manifest)}",
        f"- **Started:** {manifest.started_at}",
        f"- **Finished:** {manifest.finished_at or '-'}",
        f"- **Git SHA:** `{manifest.git_sha}`{' (dirty)' if manifest.git_dirty else ''}",
        f"- **Config:** `{manifest.config_path}` (hash `{manifest.config_hash[:12]}`)",
        f"- **Seed:** {manifest.seed}",
        f"- **Cache:** {manifest.cache.mode}, {manifest.cache.hits} hits, "
        f"{manifest.cache.misses} misses",
        "",
        "## Models",
        "",
        "| Role | Provider | Model | Family |",
        "| --- | --- | --- | --- |",
        *(
            f"| {use.role} | {use.provider} | {use.model} | {use.family or '-'} |"
            for use in manifest.models
        ),
        "",
        "## Prompt templates",
        "",
        *(
            f"- `{use.template}@{use.version}` (sha256 `{use.sha256[:12]}`)"
            for use in manifest.prompt_templates
        ),
        "",
    ]
    if outputs or manifest.stories is None:
        lines += ["## Outputs", ""]
    if not outputs and manifest.stories is None:
        lines += ["No outputs were recorded.", ""]
    for output in outputs:
        lines += [
            f"### {output.call} #{output.repetition + 1} ({output.provider}/{output.model})",
            "",
            *_output_body(output),
            "",
        ]
    stories = read_stories(run_dir)
    lines += _stories_section(manifest, stories, calls)
    lines += _budget_section(manifest, stories)
    lines += _equal_budget_section(manifest)
    lines += _benchmark_section(manifest)
    lines += _structured_section(manifest)
    lines += _cost_section(manifest, calls)
    lines += [
        "## Calls",
        "",
        "| Call | Attempt | Model | Template | Seed | Cached | Input tokens | Output tokens "
        "| Cost | Latency (s) | Retries |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *(
            f"| {call.call} #{call.repetition + 1} | {call.attempt + 1} | {call.reported_model} "
            f"| {call.template}@{call.template_version} | {call.seed} "
            f"| {'yes' if call.cached else 'no'} "
            f"| {call.input_tokens} | {call.output_tokens} | {_usd(call.cost_usd)} "
            f"| {call.latency_s:.2f} | {call.retries} |"
            for call in calls
        ),
        "",
    ]
    return "\n".join(lines)


def _status(manifest: Manifest) -> str:
    if manifest.abort_reason:
        return f"{manifest.status}: {manifest.abort_reason} - {manifest.abort_message}"
    if manifest.error:
        where = f" (call `{manifest.error.call}`)" if manifest.error.call else ""
        return f"{manifest.status}: {manifest.error.type}{where} - {manifest.error.message}"
    return manifest.status


def _usd(value: float | None) -> str:
    return "-" if value is None else f"${value:.4f}"


def _cost_section(manifest: Manifest, calls: list[CallRecord]) -> list[str]:
    cost = manifest.cost
    per_model: dict[str, list[CallRecord]] = {}
    for call in calls:
        per_model.setdefault(f"{call.provider}/{call.model}", []).append(call)
    lines = [
        "## Cost",
        "",
        f"- New spend this run: {_usd(cost.usd)}",
        f"- Token cost at list price, including cache hits: {_usd(cost.token_cost_usd)}",
        f"- Tokens: {cost.input_tokens} input, {cost.output_tokens} output",
        f"- Per-run cap: {_usd(cost.per_run_cap_usd)}; per-day cap: {_usd(cost.per_day_cap_usd)}",
        f"- Spent today (UTC, all runs): {_usd(cost.day_spent_usd)}",
        "",
    ]
    if per_model:
        lines += [
            "| Model | Calls | Cache hits | Input tokens | Output tokens "
            "| New spend | Token cost |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            *(
                f"| {label} | {len(items)} | {sum(c.cached for c in items)} "
                f"| {sum(c.input_tokens for c in items)} | {sum(c.output_tokens for c in items)} "
                f"| {_usd(sum(c.cost_usd for c in items))} "
                f"| {_usd(sum(c.token_cost_usd for c in items))} |"
                for label, items in per_model.items()
            ),
            "",
        ]
    if manifest.warnings:
        lines += ["### Warnings", "", *(f"- {warning}" for warning in manifest.warnings), ""]
    return lines


def _output_body(output: OutputRecord) -> list[str]:
    if output.response_model is None:
        return _quote(output.text)
    if output.parsed is not None:
        return [
            f"Parsed `{output.response_model}`:",
            "",
            "```json",
            json.dumps(output.parsed, indent=2, ensure_ascii=False),
            "```",
        ]
    return [
        f"No valid `{output.response_model}` after repairs: {output.error}",
        "",
        "Final response:",
        "",
        *_quote(output.text),
    ]


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _structured_section(manifest: Manifest) -> list[str]:
    if not manifest.structured:
        return []
    return [
        "## Structured output",
        "",
        "| Model | Calls | First-attempt validity | Validity after repair | Repair rate | Failed |",
        "| --- | --- | --- | --- | --- | --- |",
        *(
            f"| {label} | {m.calls} | {_percent(m.first_attempt_validity)} "
            f"| {_percent(m.validity_after_repair)} | {_percent(m.repair_rate)} "
            f"| {m.failed_calls} |"
            for label, m in manifest.structured.items()
        ),
        "",
    ]


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _benchmark_section(manifest: Manifest) -> list[str]:
    benchmark = manifest.benchmark
    if benchmark is None:
        return []
    return [
        "## Benchmark (gate G0)",
        "",
        f"Candidates for the `{benchmark.role}` role; threshold: "
        f"{benchmark.threshold:.0%} validity after repair. Latency is per call, repairs "
        "included. Costs are at list price, cache hits included.",
        "",
        "| Model | Calls | First-attempt validity | Validity after repair | Repair rate "
        "| Latency median (s) | Latency p95 (s) | Cost per 1k output tokens | Total cost "
        "| Meets G0 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *(
            f"| {row.model} | {row.calls} | {_percent(row.first_attempt_validity)} "
            f"| {_percent(row.validity_after_repair)} | {_percent(row.repair_rate)} "
            f"| {_seconds(row.latency_median_s)} | {_seconds(row.latency_p95_s)} "
            f"| {_usd(row.cost_per_1k_output_usd)} | {_usd(row.token_cost_usd)} "
            f"| {'yes' if row.meets_threshold else 'no'} |"
            for row in benchmark.rows
        ),
        "",
    ]


PLUS_MINUS = chr(0xB1)


def _tolerance(fraction: float) -> str:
    return f"{PLUS_MINUS} {fraction:.0%}"


def _story_calls(calls: list[CallRecord]) -> dict[tuple[str, str, int], list[CallRecord]]:
    """Calls per story, keyed by (condition, brief, repetition)."""
    grouped: dict[tuple[str, str, int], list[CallRecord]] = {}
    for call in calls:
        if call.condition is not None and call.brief is not None:
            grouped.setdefault((call.condition, call.brief, call.repetition), []).append(call)
    return grouped


def _usage(calls: list[CallRecord]) -> str:
    """Calls, input tokens, output tokens, token cost and new spend, as table cells."""
    return (
        f"{len(calls)} | {sum(c.input_tokens for c in calls)} "
        f"| {sum(c.output_tokens for c in calls)} | {_usd(sum(c.token_cost_usd for c in calls))} "
        f"| {_usd(sum(c.cost_usd for c in calls))}"
    )


def _stories_section(
    manifest: Manifest, stories: list[StoryRecord], calls: list[CallRecord]
) -> list[str]:
    summary = manifest.stories
    if summary is None:
        return []
    per_story = _story_calls(calls)
    checksums = ", ".join(
        f"{split} `{value[:12]}`" for split, value in summary.split_checksums.items()
    )
    lines = [
        "## Stories",
        "",
        f"- **Briefs:** {', '.join(summary.brief_ids)} (from `{summary.briefs_dir}`)",
        f"- **Frozen splits:** {checksums or 'none'}",
        "- **Conditions:** " + ", ".join(f"{c.name} (`{c.pipeline}`)" for c in summary.conditions),
        f"- **Repetitions:** {summary.repeat}",
        "",
        "Token cost is at list price, cache hits included; new spend is what this run paid.",
        "",
        f"Words are counted on the normalised story; the length tolerance is "
        f"{_tolerance(summary.length_tolerance)}, with at most {summary.max_attempts} attempts. "
        "Attempts are those of the story's final generation: its last revision pass, if any.",
        "",
        "| Brief | Condition | Rep | Words | Target | Attempts | In range | Revisions | Status "
        "| Calls | Input tokens | Output tokens | Token cost | New spend | Files |",
        "|" + " --- |" * 15,
    ]
    for story in stories:
        # Attempt folders hold every generation; link only the story and its plan files.
        links = " ".join(
            f"[{name}]({story.path}/{name})" for name in story.files if "/" not in name
        )
        story_calls = per_story.get((story.condition, story.brief, story.repetition), [])
        lines.append(
            f"| {story.brief} | {story.condition} | {story.repetition + 1} | {story.words} "
            f"| {story.target_words} | {len(story.final_attempts)} "
            f"| {'yes' if story.in_range else 'no'} | {story.revision_passes} | {story.status} "
            f"| {_usage(story_calls)} | {links} |"
        )
    lines += [
        "",
        "### Per condition",
        "",
        "| Condition | Stories | Failed | Calls | Input tokens | Output tokens | Token cost "
        "| New spend | Token cost per story |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for condition in summary.conditions:
        mine = [story for story in stories if story.condition == condition.name]
        condition_calls = [call for call in calls if call.condition == condition.name]
        token_cost = sum(call.token_cost_usd for call in condition_calls)
        lines.append(
            f"| {condition.name} | {len(mine)} | {sum(s.status == 'failed' for s in mine)} "
            f"| {_usage(condition_calls)} | {_usd(token_cost / len(mine) if mine else None)} |"
        )
    lines.append("")
    out_of_range = [story for story in stories if story.status == "ok" and not story.in_range]
    if out_of_range:
        lines += [
            "### Out of length range",
            "",
            "Kept, but still outside the length range after every attempt:",
            "",
            *(
                f"- {s.condition}/{s.brief} #{s.repetition + 1}: {s.words} words after "
                f"{len(s.final_attempts)} attempts (target {s.target_words} "
                f"{_tolerance(summary.length_tolerance)})"
                for s in out_of_range
            ),
            "",
        ]
    failed = [story for story in stories if story.status == "failed"]
    if failed:
        lines += [
            "### Failed stories",
            "",
            *(f"- {s.condition}/{s.brief} #{s.repetition + 1}: {s.error}" for s in failed),
            "",
        ]
    return lines


def _budget_section(manifest: Manifest, stories: list[StoryRecord]) -> list[str]:
    summary = manifest.stories
    if summary is None:
        return []
    lines = [
        "## Generation budget",
        "",
        "Generation tokens are the input and output tokens of every call a story made: "
        "structured-output repairs, length regenerations and revision passes included.",
        "",
        "| Condition | Stories | Calls | Generation tokens | Tokens per story |",
        "| --- | --- | --- | --- | --- |",
    ]
    for condition in summary.conditions:
        mine = [story for story in stories if story.condition == condition.name]
        tokens = sum(story.budget.tokens for story in mine)
        per_story = str(round(tokens / len(mine))) if mine else "-"
        lines.append(
            f"| {condition.name} | {len(mine)} | {sum(story.budget.calls for story in mine)} "
            f"| {tokens} | {per_story} |"
        )
    return [*lines, ""]


def _equal_budget_section(manifest: Manifest) -> list[str]:
    equal = None if manifest.stories is None else manifest.stories.equal_budget
    if equal is None:
        return []
    first, second = equal.conditions
    lines = [
        "## Equal budget",
        "",
        f"{first} and {second} are matched per brief and repetition: the cheaper story gets "
        f"revision passes until the generation-token budgets are within "
        f"{_tolerance(equal.tolerance)} of the larger, with at most "
        f"{equal.max_revision_passes} passes.",
        "",
        f"| Brief | Rep | {first} before | {second} before | Revised | Passes "
        f"| {first} after | {second} after | Matched | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for match in equal.matches:
        lines.append(
            f"| {match.brief} | {match.repetition + 1} | {match.before.get(first, '-')} "
            f"| {match.before.get(second, '-')} | {match.revised or '-'} | {match.passes} "
            f"| {match.after.get(first, '-')} | {match.after.get(second, '-')} "
            f"| {'yes' if match.matched else 'no'} | {match.reason or ''} |"
        )
    unmatched = sum(not match.matched for match in equal.matches)
    lines += ["", f"{len(equal.matches) - unmatched} matched, {unmatched} not matched.", ""]
    return lines


def _quote(text: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in (text.splitlines() or [""])]


def write_report(run_dir: Path) -> Path:
    path = run_dir / REPORT
    write_text_lf(path, render_report(run_dir))
    return path
