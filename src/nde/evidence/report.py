"""Gate logic and concise reproducible review packets."""

import base64
import hashlib
import html
import itertools
from pathlib import Path

from nde.annotations.records import digest
from nde.corpus import provenance
from nde.files import write_text_lf
from nde.narrative.models import Source
from nde.narrative.workflow import schema_json

from .annotations import assess_annotation
from .mutations import run_suite
from .records import Comparison, Report, Snapshot, StoryEvidence, StoryInput

MIN_STORIES = 3
MAX_STORIES = 5
MIN_WORDS = 2000
MAX_WORDS = 5000
CORRECTION_TARGET_SECONDS = 7200


def _source(story: StoryInput) -> tuple[StoryEvidence, Source | None]:
    record = story.provenance
    row = StoryEvidence(
        id=record.id,
        title=record.title,
        provenance=record.model_dump(mode="json"),
        source_sha256=record.sha256,
    )
    if story.source_base64 is None:
        row.pending.append(story.source_error or "source missing")
        return row, None
    raw = base64.b64decode(story.source_base64, validate=True)
    if record.sha256 is None or hashlib.sha256(raw).hexdigest() != record.sha256:
        row.mechanical_failures.append("source checksum mismatch or source checksum unrecorded")
        row.source_status = "failed"
        return row, None
    try:
        text = provenance.extract(raw.decode("utf-8-sig"), record)
    except (UnicodeError, provenance.ProvenanceError) as error:
        row.mechanical_failures.append(f"source extraction failed: {error}")
        row.source_status = "failed"
        return row, None
    row.source_status = "verified"
    row.text_sha256 = digest(text)
    row.word_count = len(text.split())
    if not MIN_WORDS <= row.word_count <= MAX_WORDS:
        row.pending.append("source is outside the approximately 2,000-5,000-word selection range")
    source = Source(
        id=f"source-{record.id}",
        title=record.title,
        text=text,
        sha256=digest(text),
        provenance=f"{record.source}; {record.edition}; {record.licence_basis}",
    )
    return row, source


def evaluate(snapshot: Snapshot) -> Report:
    suite = run_suite(snapshot.suite)
    rows: list[StoryEvidence] = []
    for story in snapshot.stories:
        row, source = _source(story)
        assess_annotation(story.annotation, row, source, digest(snapshot.schema_text))
        rows.append(row)
    pending: list[str] = []
    if not MIN_STORIES <= len(rows) <= MAX_STORIES:
        pending.append("select 3-5 distinct public-domain stories")
    comparisons = [
        Comparison(
            story_a=a.id,
            story_b=b.id,
            distinct=a.trajectory_signature != b.trajectory_signature
            if a.trajectory_signature and b.trajectory_signature
            else None,
        )
        for a, b in itertools.combinations(rows, 2)
    ]
    if any(pair.distinct is None for pair in comparisons):
        pending.append("trajectory comparisons are incomplete")
    if not all(row.owner_confirmed_trajectories for row in rows):
        pending.append("owner confirmation of meaningful trajectory differences is pending")
    mechanical_failures = [
        *suite.problems,
        *(f"{row.id}: {failure}" for row in rows for failure in row.mechanical_failures),
    ]
    if snapshot.schema_text != schema_json():
        mechanical_failures.append("replay schema differs; use the recorded code revision")
    if len({row.id for row in rows}) != len(rows):
        mechanical_failures.append("story IDs in saved evidence must be distinct")
    failures = [
        *mechanical_failures,
        *(f"{row.id}: {failure}" for row in rows for failure in row.failures),
    ]
    mechanical = "failed" if mechanical_failures else "passed"
    if comparisons and all(pair.distinct is False for pair in comparisons):
        failures.append("the recorded trajectories do not distinguish any stories")
    gate = (
        "failed"
        if failures
        else "pending"
        if pending or any(row.pending for row in rows)
        else "passed"
    )
    return Report(
        report_id=digest(snapshot.model_dump_json()),
        schema_sha256=digest(snapshot.schema_text),
        git_sha=snapshot.git_sha,
        manifest_sha256=snapshot.manifest_sha256,
        mechanical_status=mechanical,
        gate_status=gate,
        mutations=suite,
        stories=rows,
        comparisons=comparisons,
        pending=pending,
        failures=failures,
    )


def _lines(report: Report) -> list[str]:
    lines = [
        "# G-schema evidence",
        f"Gate: {report.gate_status}",
        f"Automated mechanical evidence: {report.mechanical_status}",
        "Automated consistency and LLM agreement do not establish literary accuracy.",
        f"Report/run ID: {report.report_id}",
        f"Code revision: {report.git_sha}",
        f"Schema SHA-256: {report.schema_sha256}",
        f"Manifest SHA-256: {report.manifest_sha256}",
        "## Validator mutation coverage",
        f"{report.mutations.caught}/{report.mutations.total} intended categories caught "
        f"({report.mutations.catch_rate:.1%}); required ≥95% and ≥20 categories.",
        f"Suite SHA-256: {report.mutations.suite_sha256}",
    ]
    lines.extend(
        f"Control {name}: {'passed' if passed else 'failed'}"
        for name, passed in report.mutations.control_results.items()
    )
    for case in report.mutations.results:
        diagnostics = (
            "; ".join(f"{item.code} [{item.location}]: {item.message}" for item in case.diagnostics)
            or case.parse_error
            or "none"
        )
        lines.append(
            f"{case.category} ({case.family}): {'caught' if case.caught else 'MISSED'}; "
            f"expected {case.expected_severity} {case.expected_code}. "
            f"{case.description} Actual: {diagnostics}"
        )
    lines.append("## Story inventory and owner review")
    for row in report.stories:
        lines.extend(
            [
                f"### {row.id}: {row.title}",
                f"Provenance: {row.provenance['source']}; {row.provenance['edition']}; "
                f"{row.provenance['licence_basis']}",
                f"Source: {row.source_status}; words: {row.word_count}; "
                f"source SHA-256: {row.source_sha256}; text SHA-256: {row.text_sha256}",
                f"Annotation: {row.annotation_status}; correction revision: {row.revision}",
                f"Generation attempts: {row.draft_attempts}; new provider spend USD: "
                f"{row.generation_cost_usd}",
                f"Actual owner correction seconds: {row.correction_seconds}; target <7200",
                f"Owner review: {row.owner_review}; literary evidence: "
                f"{row.literary_evidence or '(absent)'}",
            ]
        )
        lines.extend(f"{name}: {value}" for name, value in sorted(row.identifiers.items()))
        lines.extend(f"pending: {problem}" for problem in row.pending)
        lines.extend(f"failed: {problem}" for problem in row.failures)
        lines.extend(f"mechanically failed: {problem}" for problem in row.mechanical_failures)
        lines.extend(row.trajectories or ["Trajectories: missing"])
    lines.append("## Trajectory comparisons")
    lines.extend(
        f"{pair.story_a} / {pair.story_b}: distinct={pair.distinct} (automated comparison only)"
        for pair in report.comparisons
    )
    lines.extend(f"pending: {problem}" for problem in report.pending)
    lines.extend(f"failed: {problem}" for problem in report.failures)
    lines.extend(
        [
            "## Remaining owner procedure",
            "Select 3-5 provenance-checked stories in range, correct and validate each "
            "tier-L draft with the annotation timer, pause it during idle/generation, "
            "and explicitly review the latest correction with literary notes. Compare "
            "the coarse trajectories and record whether differences are meaningful.",
            "Fail action: cut or merge slow or non-discriminating channels, then repeat "
            "on another story. Routine manual regression testing is not required.",
            "Replay: uv run nde schema-evidence --replay inputs.json --out replayed",
        ]
    )
    return lines


def write_packet(snapshot: Snapshot, out: Path) -> Report:
    report = evaluate(snapshot)
    out.mkdir(parents=True, exist_ok=True)
    write_text_lf(out / "inputs.json", snapshot.model_dump_json(indent=2) + "\n")
    write_text_lf(out / "evidence.json", report.model_dump_json(indent=2) + "\n")
    write_text_lf(out / "mutation-results.json", report.mutations.model_dump_json(indent=2) + "\n")
    lines = _lines(report)
    write_text_lf(out / "evidence.md", "\n\n".join(lines) + "\n")
    body = "\n".join(f"<p>{html.escape(line)}</p>" for line in lines)
    write_text_lf(
        out / "evidence.html",
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>G-schema evidence</title><body>{body}</body></html>\n",
    )
    return report
