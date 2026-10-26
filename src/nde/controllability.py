"""Deterministic plan-only edit evidence; no claim about literary or prose success."""

import argparse
import html
import sys
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from nde.files import checksum_lf, write_text_lf
from nde.narrative.base import Strict
from nde.narrative.ledger import plan_identity
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import load_plan
from nde.plan.revision import (
    RevisionSpec,
    boundary_errors,
    canonical,
    changed_units,
    identity,
    selector_value,
    validate_selection,
)

type Status = Literal["success", "failure", "unknown"]


class ControlTarget(Strict):
    id: str = Field(min_length=1)
    kind: Literal["changed", "unchanged", "renewed_hope_middle", "literary"]
    selector: str | None = None
    annotation_id: str | None = None
    min_episodes: int = Field(default=1, ge=1)
    threshold: int = Field(default=2, ge=1, le=4)

    @model_validator(mode="after")
    def _explicit_target(self) -> Self:
        if self.kind in {"changed", "unchanged"} and not self.selector:
            raise ValueError("change predicates require a stable selector")
        if self.kind == "renewed_hope_middle" and not self.annotation_id:
            raise ValueError("trajectory predicates require an explicit reader_affect identity")
        return self


class TargetResult(Strict):
    id: str
    kind: str
    status: Status
    reason: str
    request: ControlTarget
    request_sha256: str


class ControlInputs(Strict):
    before: str
    after: str
    revision: RevisionSpec
    targets: list[ControlTarget] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_targets(self) -> Self:
        ids = [target.id for target in self.targets]
        if len(ids) != len(set(ids)):
            raise ValueError("target identities must be distinct")
        return self


class ControlReport(Strict):
    version: Literal["1"] = "1"
    scope: Literal["plan-only"] = "plan-only"
    status: Status
    before_sha256: str
    after_sha256: str
    inputs: dict[str, str]
    request_path: str
    revision: RevisionSpec
    targets: list[TargetResult]
    lock_stable: bool
    authorised_scope: bool
    validator_passed: bool
    actual_changes: list[str]
    diagnostics: list[str]
    prose_realisation: Literal["unassessed"] = "unassessed"


def _renewals(plan: NarrativePlan, target: ControlTarget) -> tuple[Status, str]:
    annotation = next((a for a in plan.reader_affect if a.id == target.annotation_id), None)
    if annotation is None or annotation.reading.status != "known" or annotation.trajectory is None:
        return "unknown", "Missing or uncertain explicitly selected affect trajectory"
    positions = {scene.id: index for index, scene in enumerate(plan.scenes)}
    points = {
        positions[p.scene_id]: p.value
        for p in annotation.trajectory.points
        if p.scene_id in positions
    }
    count = len(plan.scenes)
    start, stop = count // 3, (2 * count + 2) // 3
    # No interpolation: a renewal needs an explicitly low preceding adjacent scene.
    required = range(max(0, start - 1), stop)
    if start == 0 or any(points.get(i) is None for i in required):
        return "unknown", "Middle-third or immediately preceding ordinal observations are missing"
    episodes = 0
    for i in range(start, stop):
        current, previous = points[i], points[i - 1]
        if current is not None and previous is not None:
            episodes += current >= target.threshold and previous < target.threshold
    status = "success" if episodes >= target.min_episodes else "failure"
    return status, f"{episodes} separate low-to-high renewals in [{start}, {stop})"


def _target(before: NarrativePlan, after: NarrativePlan, target: ControlTarget) -> TargetResult:
    if target.kind == "literary":
        status, reason = "unknown", "Requires an actual literary assessment"
    elif target.kind == "renewed_hope_middle":
        status, reason = _renewals(after, target)
    else:
        try:
            changed = canonical(selector_value(before, target.selector or "")) != canonical(
                selector_value(after, target.selector or "")
            )
        except ValueError as error:
            status, reason = "unknown", str(error)
        else:
            achieved = changed if target.kind == "changed" else not changed
            status, reason = ("success" if achieved else "failure"), f"Changed: {changed}"
    return TargetResult(
        id=target.id,
        kind=target.kind,
        status=status,
        reason=reason,
        request=target,
        request_sha256=identity(target.model_dump(mode="json")),
    )


def control_report(path: Path) -> ControlReport:
    inputs = ControlInputs.model_validate_json(path.read_text(encoding="utf-8"))
    before_path, after_path = path.parent / inputs.before, path.parent / inputs.after
    before, after = load_plan(before_path), load_plan(after_path)
    validate_selection(before, inputs.revision)
    validation = validate_plan(after)
    valid = not any(d.severity == "error" for d in validation.diagnostics)
    errors = boundary_errors(before, after, inputs.revision)
    locks = True
    for selector in inputs.revision.locks:
        try:
            locks &= canonical(selector_value(before, selector)) == canonical(
                selector_value(after, selector)
            )
        except ValueError:
            locks = False
    targets = [_target(before, after, target) for target in inputs.targets]
    status: Status = "success"
    if not locks or errors or not valid or any(t.status == "failure" for t in targets):
        status = "failure"
    elif any(t.status == "unknown" for t in targets):
        status = "unknown"
    return ControlReport(
        status=status,
        before_sha256=plan_identity(before),
        after_sha256=plan_identity(after),
        inputs={
            str(file.resolve()): checksum_lf(file.read_bytes())
            for file in (path, before_path, after_path)
        },
        request_path=str(path.resolve()),
        revision=inputs.revision,
        targets=targets,
        lock_stable=locks,
        authorised_scope=not errors,
        validator_passed=valid,
        actual_changes=changed_units(before, after),
        diagnostics=[*errors, *[d.model_dump_json() for d in validation.diagnostics]],
    )


def replay_report(path: Path) -> ControlReport:
    """Recompute bound evidence; a saved success flag is insufficient for admission."""
    saved = ControlReport.model_validate_json(path.read_text(encoding="utf-8"))
    for filename, checksum in saved.inputs.items():
        if checksum_lf(Path(filename).read_bytes()) != checksum:
            raise ValueError(f"control evidence input changed: {filename}")
    recomputed = control_report(Path(saved.request_path))
    if recomputed != saved:
        raise ValueError("control report changed or differs from recomputed evidence")
    return recomputed


def write_evidence(report: Strict, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump_json(indent=2) + "\n"
    write_text_lf(out / f"{name}.json", payload)
    write_text_lf(out / f"{name}.md", f"# {name}: plan-only\n\n```json\n{payload}```\n")
    write_text_lf(
        out / f"{name}.html",
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>{name}: plan-only</title><body><pre>{html.escape(payload)}</pre></body></html>\n",
    )


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="JSON request, paths relative to this file")
    parser.add_argument("--out", type=Path, required=True)


def run(args: argparse.Namespace) -> int:
    try:
        result = control_report(args.input)
        write_evidence(result, args.out, "control-report")
    except (OSError, ValueError) as error:
        print(f"nde control-report: {error}", file=sys.stderr)
        return 2
    print(f"control-report {result.status}: {args.out}")
    return 0
