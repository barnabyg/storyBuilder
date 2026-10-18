"""Generate a run's Markdown report from its run directory."""

from pathlib import Path

from nde.files import write_text_lf
from nde.runs.records import REPORT, read_calls, read_manifest, read_outputs


def render_report(run_dir: Path) -> str:
    manifest = read_manifest(run_dir)
    outputs = read_outputs(run_dir)
    calls = read_calls(run_dir)
    lines = [
        f"# Run report: {manifest.name}",
        "",
        f"- **Status:** {manifest.status}",
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
            *_quote(output.text),
            "",
        ]
    lines += [
        "## Calls",
        "",
        "| Call | Model | Template | Seed | Cached | Input tokens | Output tokens | Latency (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *(
            f"| {call.call} #{call.repetition + 1} | {call.response_model} "
            f"| {call.template}@{call.template_version} | {call.seed} "
            f"| {'yes' if call.cached else 'no'} "
            f"| {call.input_tokens} | {call.output_tokens} | {call.latency_s:.2f} |"
            for call in calls
        ),
        "",
    ]
    return "\n".join(lines)


def _quote(text: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in (text.splitlines() or [""])]


def write_report(run_dir: Path) -> Path:
    path = run_dir / REPORT
    write_text_lf(path, render_report(run_dir))
    return path
