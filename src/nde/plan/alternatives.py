"""Independent revision alternatives from one recorded original, with honest duplicates.

Every alternative of a `plan revise` run starts from the same saved source plan.
The report counts only distinct accepted changes as diversity: no-ops, exact
duplicates and structural duplicates (identical apart from descriptive text) are
listed, never counted as additional alternatives.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from nde.files import write_text_lf
from nde.narrative.models import NarrativePlan
from nde.narrative.workflow import PlanError, load_plan
from nde.plan.revision import identity

REPORT = "alternatives.json"
DESCRIPTIVE = frozenset({"name", "title", "summary", "description", "note"})
INTERPRETIVE = frozenset(
    {"arcs", "reader_affect", "tension", "themes", "motifs", "discourse", "subtext"}
)
DIVERSITY = (
    "Only distinct accepted changed plans count as alternatives. No-ops, exact duplicates "
    "and structural duplicates are reported, never counted as diversity. Literary "
    "adherence and quality are unassessed."
)


def structure_identity(plan: NarrativePlan) -> str:
    """Plan identity with descriptive and interpretive reading text replaced by null."""

    def strip(value: Any, collection: str, key: str) -> Any:
        if isinstance(value, dict):
            row = cast(dict[str, Any], value)
            return {name: strip(child, collection or name, name) for name, child in row.items()}
        if isinstance(value, list):
            return [strip(child, collection, key) for child in cast(list[Any], value)]
        reading = collection in INTERPRETIVE and key in {"value", "alternatives"}
        return None if isinstance(value, str) and (key in DESCRIPTIVE or reading) else value

    return identity(strip(plan.model_dump(mode="json"), "", ""))


class AlternativesError(Exception):
    """A run's saved alternatives report or plan records cannot be read."""


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _costs(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "calls": len(calls),
        "input_tokens": sum(call["input_tokens"] for call in calls),
        "output_tokens": sum(call["output_tokens"] for call in calls),
        "cost_usd": sum(call["cost_usd"] for call in calls),
        "token_cost_usd": sum(call["token_cost_usd"] for call in calls),
    }


def _alternative(run_dir: Path, record: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    folder = run_dir / record["path"]
    provenance = _read(folder / "revisions/revision/provenance.json")
    row: dict[str, Any] = {
        "brief": record["brief"],
        "repetition": record["repetition"],
        "path": record["path"],
        "status": "missing" if provenance is None else provenance["status"],
        "diagnostics": [],
    }
    if provenance is None:
        row["diagnostics"].append("alternative provenance is missing or unreadable")
        return row
    try:
        backend: dict[str, Any] = provenance.get("backend") or {"version": "1", "kind": "d"}
        row.update(
            backend=backend["kind"],
            seed=provenance["seed"],
            checkpoint_id=backend.get("checkpoint_id"),
            guidance_scale=backend.get("guidance_scale"),
            test_only=backend.get("test_only", False),
            local_work=backend.get("work"),
            before_sha256=provenance["before_sha256"],
            after_sha256=provenance["after_sha256"],
            changed_units=provenance["changed_units"],
            **_costs(provenance["calls"]),
        )
    except (AttributeError, KeyError, TypeError) as error:
        row["diagnostics"].append(
            f"alternative provenance is malformed: {type(error).__name__}: {error}"
        )
        return row
    if provenance["before_sha256"] != source_sha256:
        row["diagnostics"].append("alternative was not derived from the recorded original")
    selected = folder / ("revisions/revision/last-accepted/plan.json")
    if row["status"] in {"accepted", "noop"}:
        selected = folder / "plan.json"
    elif (folder / "plan.json").exists():
        row["diagnostics"].append("failed alternative exposes a selected plan")
    try:
        plan = load_plan(selected)
    except (OSError, PlanError) as error:
        row["diagnostics"].append(f"selected plan unreadable: {error}")
        return row
    expected = source_sha256 if row["status"] == "failed" else provenance["after_sha256"]
    if identity(plan) != expected:
        row["diagnostics"].append("selected plan identity differs from its provenance")
    if row["status"] != "failed" and (row["status"] == "noop") != (
        provenance["after_sha256"] == source_sha256
    ):
        row["diagnostics"].append("no-op status contradicts its before/after identities")
    row["structure_sha256"] = structure_identity(plan)
    return row


def _duplicates(rows: list[dict[str, Any]]) -> dict[str, int]:
    exact: dict[str, str] = {}
    structural: dict[str, str] = {}
    for row in rows:
        row["duplicate_of"] = row["structural_duplicate_of"] = None
        if row["status"] != "accepted" or row["diagnostics"]:
            continue
        row["duplicate_of"] = exact.setdefault(row["after_sha256"], row["path"])
        row["structural_duplicate_of"] = structural.setdefault(row["structure_sha256"], row["path"])
        for key in ("duplicate_of", "structural_duplicate_of"):
            if row[key] == row["path"]:
                row[key] = None
    accepted = [row for row in rows if row["status"] == "accepted" and not row["diagnostics"]]
    return {
        "distinct_changed": len(exact),
        "exact_duplicates": len(accepted) - len(exact),
        "structurally_distinct": len(structural),
        "structural_duplicates": len(accepted) - len(structural),
    }


def build_report(
    run_dir: Path, source: Path, recorded: dict[str, str], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Recompute alternative identities from the published artifacts of a revision run."""
    current = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
    rows = [_alternative(run_dir, record, recorded["plan_sha256"]) for record in records]
    counts = {
        "expected": len(records),
        **{
            status: sum(row["status"] == status for row in rows)
            for status in ("accepted", "noop", "failed", "missing")
        },
        "inconsistent": sum(bool(row["diagnostics"]) for row in rows),
        **_duplicates(rows),
    }
    return {
        "kind": "revision-alternatives",
        "version": "1",
        "source": {
            "path": str(source),
            "file_sha256": recorded["file_sha256"],
            "plan_sha256": recorded["plan_sha256"],
            "current_file_sha256": current,
            "unchanged": current == recorded["file_sha256"],
        },
        "backends": sorted({row["backend"] for row in rows if "backend" in row}),
        "test_only": any(row.get("test_only") for row in rows),
        "counts": counts,
        "consistent": counts["inconsistent"] == 0 and current == recorded["file_sha256"],
        "alternatives": rows,
        "diversity": DIVERSITY,
    }


def _markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Revision alternatives",
        "",
        f"Source: `{report['source']['plan_sha256']}` "
        f"({'unchanged' if report['source']['unchanged'] else 'CHANGED OR MISSING'})",
        f"Backends: {', '.join(report['backends']) or 'none recorded'}"
        + (" (test-only fixture checkpoint)" if report["test_only"] else ""),
        "",
        f"- Expected alternatives: {counts['expected']}",
        f"- Accepted: {counts['accepted']}; no-op: {counts['noop']}; "
        f"failed: {counts['failed']}; missing: {counts['missing']}",
        f"- Distinct changed plans: {counts['distinct_changed']} "
        f"(exact duplicates: {counts['exact_duplicates']})",
        f"- Structurally distinct: {counts['structurally_distinct']} "
        f"(structural duplicates: {counts['structural_duplicates']})",
        f"- Inconsistent records: {counts['inconsistent']}",
        "",
        "| Alternative | Backend | Seed | Status | Duplicate of | Calls | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {row['path']} | {row.get('backend', '-')} | {row.get('seed', '-')} | "
        f"{row['status']} | {row.get('duplicate_of') or '-'} | {row.get('calls', 0)} | "
        f"{row.get('cost_usd', 0):.6f} |"
        for row in report["alternatives"]
    ]
    return "\n".join([*lines, "", report["diversity"], ""])


def write_report(run_dir: Path, report: dict[str, Any]) -> None:
    write_text_lf(run_dir / REPORT, json.dumps(report, indent=2) + "\n")
    write_text_lf(run_dir / "alternatives.md", _markdown(report))


def check_report(run_dir: Path) -> dict[str, Any]:
    """Rebuild a saved report from current artifacts; stale or changed evidence is inconsistent.

    Raises AlternativesError when the saved report or plan records cannot be read.
    """
    try:
        saved = json.loads((run_dir / REPORT).read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in (run_dir / "plans.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        source = saved["source"]
        path = Path(source["path"])
        recorded = {"file_sha256": source["file_sha256"], "plan_sha256": source["plan_sha256"]}
        if any(not {"path", "brief", "repetition"} <= set(record) for record in records):
            raise KeyError("plan record without path, brief or repetition")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise AlternativesError(
            f"saved alternatives evidence in {run_dir} is unreadable or malformed: "
            f"{type(error).__name__}: {error}"
        ) from error
    report = build_report(run_dir, path, recorded, records)
    if report != saved:
        report["consistent"] = False
        report["stale"] = "saved alternatives report differs from current artifacts"
    return report
