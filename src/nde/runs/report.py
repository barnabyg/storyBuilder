"""Generate a run's Markdown report from its run directory."""

import json
from pathlib import Path

from nde.files import write_text_lf
from nde.runs.records import (
    REPORT,
    CallRecord,
    Manifest,
    OutputRecord,
    read_calls,
    read_manifest,
    read_outputs,
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
        "| Role | Provider | Model |",
        "| --- | --- | --- |",
        *(f"| {use.role} | {use.provider} | {use.model} |" for use in manifest.models),
        "",
        "## Prompt templates",
        "",
        *(
            f"- `{use.template}@{use.version}` (sha256 `{use.sha256[:12]}`)"
            for use in manifest.prompt_templates
        ),
        "",
        "## Outputs",
        "",
    ]
    if not outputs:
        lines += ["No outputs were recorded.", ""]
    for output in outputs:
        lines += [
            f"### {output.call} #{output.repetition + 1} ({output.provider}/{output.model})",
            "",
            *_output_body(output),
            "",
        ]
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


def _quote(text: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in (text.splitlines() or [""])]


def write_report(run_dir: Path) -> Path:
    path = run_dir / REPORT
    write_text_lf(path, render_report(run_dir))
    return path
