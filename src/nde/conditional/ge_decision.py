"""The immutable G-E decision: registered criteria applied to current dev evidence.

`decide` revalidates a committed G-E protocol (genuine G3 authority, checkpoint,
data and reference lineage), its one claimed run and evidence report, and the
owner's structural review packet. It then applies the registered validator
comparison and structural-competitiveness criterion in the candidate direction,
E against the identified C/D reference, and seals the result. Outcomes are a
complete pass, a complete failure, pending evidence, or inconsistent evidence.
Synthetic (test-only) criteria results stay scientifically pending.

The single development iteration is enforced across protocols from the
repository's G-E history: every G-E protocol, run claim, decision claim and
review packet in any commit of any local ref, plus the claims kept beside those
protocols. A comparison is every protocol with the same test-only status and
canonical reference identity, whatever its dev briefs; it may have one run first
attempt and one run development iteration, which is decided only against the
first attempt's settled, replayable failure.

The first decision that applies the structural criterion freezes the owner
ratings it used in a sealed review claim beside the protocol; every later
decision uses exactly those reviews, and later reviews are reported, never
applied. Inconsistent evidence never unblinds owner ratings. A complete decision
settles its protocol: a sealed decision claim beside the protocol names it, and
no other decision of that protocol is ever written.
Nothing here trains, generates, spends, launches another gate or starts Phase 8.
"""

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from nde.conditional import ge, ge_review
from nde.corpus.records_io import create_record, file_sha, identity, read_sealed
from nde.files import write_text_lf
from nde.gitinfo import git

type Record = dict[str, Any]

DECISION_KIND = "ge-decision"
DECISION_VERSION = "1"
DECISION = "decision.json"
SETTLEMENT_KIND = "ge-decision-claim"
RATINGS_KIND = "ge-review-claim"
WITHHELD = "withheld: the evidence is inconsistent, so no owner rating is unblinded"
COMPLETE = ("passed", "failed")
FIRST, ITERATION = 1, 2
# Boundaries are inclusive; a floating-point rate this close to a threshold meets it.
TOLERANCE = 1e-12
# Every G-E record kind starts with this; `git log -G` finds the commits that add one.
HISTORY_PATTERN = '"kind": "ge-'
STATEMENT = (
    "G-E decision on the registered development comparison of Candidate E against the "
    "identified reference planner. A pass records eligibility to prepare Phase 8; it is not "
    "evidence that E improves final stories. Nothing is launched: no training, generation, "
    "spending, later gate or Phase 8."
)
SYNTHETIC = (
    "synthetic (test-only) evidence: the registered criteria were applied as software "
    "evidence only and cannot establish a scientific G-E result"
)
NOTHING_LAUNCHED = (
    "Reporting starts nothing: no training, paid generation, another gate or Phase 8."
)
ACTIONS = {
    "eligible_to_prepare_phase_8": (
        "E met both registered criteria on the dev comparison. Phase 8 may be prepared: "
        "freeze a fresh held-out benchmark and pre-register its comparison separately. This "
        "is not proof that E improves final stories; the {reference} reference planner stays "
        "the planner until Phase 8 decides."
    ),
    "iterate_once_or_stop": (
        "E fell below the registered criteria. Keep the identified {reference} reference "
        "planner. At most one registered development iteration (attempt 2), decided against "
        "this decision, is permitted; otherwise stop."
    ),
    "stop": (
        "The single development iteration also fell below the registered criteria. Stop: "
        "keep the identified {reference} reference planner and do not prepare Phase 8 with E."
    ),
    "none": (
        "No G-E result: resolve the pending or inconsistent evidence and decide again. The "
        "identified {reference} reference planner remains the planner."
    ),
}
LIMITATIONS = [
    "Development evidence on frozen dev briefs only; not the Phase 8 held-out experiment.",
    "The validator comparison is a descriptive pass-rate difference, not an inferential "
    "non-inferiority test.",
    "Owner reviewer identity is self-declared and blinding is procedural: the key and run "
    "stay on disk and plan structure may reveal the condition.",
    "Model judges are diagnostic only; their agreement never replaces the owner's review.",
    "Condition-adherence judgement, realisation review and overlap findings are reported "
    "evidence, not G-E criteria.",
    "History covers G-E records committed to this repository on any local ref and the "
    "claims kept beside their protocols; records never committed here are not visible.",
]


def at_least(value: float, bound: float) -> bool:
    """Inclusive comparison, tolerant of floating-point rates that equal the bound."""
    return value >= bound or math.isclose(value, bound, rel_tol=0, abs_tol=TOLERANCE)


def validator_criterion(validator: Record | None, rule: Record) -> Record:
    """E's measured plan validity against the reference's, in the candidate direction.

    E's rate minus the reference's must be at least minus the margin and E's rate at
    least the registered minimum. Every expected slot is in each denominator, a slot
    without a plan counts as not valid, and any unknown unit leaves it pending.
    """
    result: Record = {
        "rule": "E's rate minus the reference's is at least minus the margin, and E's "
        "rate is at least the minimum; every expected slot counts and boundaries are inclusive",
        "aggregation": rule["aggregation"],
        "margin": rule["margin"],
        "lower_bound": -rule["margin"],
        "minimum_candidate_rate": rule["minimum_candidate_rate"],
        "candidate_rate": None,
        "reference_rate": None,
        "difference": None,
        "counts": None,
        "unknown_units": [],
        "meets_margin": None,
        "meets_minimum": None,
        "status": "pending",
        "reason": None,
    }
    if validator is None:
        return result | {"reason": "no current measured run"}
    candidate, reference = validator["candidate"], validator["reference"]
    result.update(
        candidate_rate=candidate["rate"],
        reference_rate=reference["rate"],
        counts={"E": candidate["counts"], "reference": reference["counts"]},
        unknown_units=validator["unknown_units"],
    )
    if validator["aggregation"] != rule["aggregation"]:
        return result | {"reason": "measured aggregation differs from the registered one"}
    if not validator["complete"] or candidate["rate"] is None or reference["rate"] is None:
        unknown = len(validator["unknown_units"])
        return result | {"reason": f"{unknown} units have unknown validity"}
    difference = candidate["rate"] - reference["rate"]
    margin = at_least(difference, -rule["margin"])
    minimum = at_least(candidate["rate"], rule["minimum_candidate_rate"])
    return result | {
        "difference": difference,
        "meets_margin": margin,
        "meets_minimum": minimum,
        "status": "passed" if margin and minimum else "failed",
    }


def structural_criterion(outcomes: list[Record] | None, rule: Record, reason: str) -> Record:
    """The share of owner-comparable sampled pairs in which E's structure is competitive.

    `outcomes` are the packet's unblinded item outcomes, or None (with `reason`) while
    no usable owner review exists. Ties are competitive; unknown ratings and slots where
    E has no plan are not; slots where only the reference lacks a plan are excluded and
    reported; pending items, or no owner-comparable pair at all, leave it pending.
    """
    rows = outcomes or []
    counts = Counter(str(row["outcome"]) for row in rows)
    denominator = counts["competitive"] + counts["not_competitive"]
    result: Record = {
        "rule": "competitive pairs (E better or equivalent) over the owner denominator reach "
        "the minimum share; boundaries are inclusive",
        "question": rule["question"],
        "minimum_share": rule["minimum_share"],
        "items": rows,
        "competitive": counts["competitive"],
        "not_competitive": counts["not_competitive"],
        "excluded": [row["slot"] for row in rows if row["outcome"] == "excluded"],
        "pending_items": [row["slot"] for row in rows if row["outcome"] == "pending"],
        "denominator": denominator,
        "share": None,
        "status": "pending",
        "reason": None,
    }
    if outcomes is None:
        return result | {"reason": reason}
    if counts["pending"]:
        return result | {"reason": f"{counts['pending']} sampled items are pending"}
    if denominator == 0:
        return result | {"reason": "no owner-comparable pair: every sampled slot is excluded"}
    share = counts["competitive"] / denominator
    return result | {
        "share": share,
        "status": "passed" if at_least(share, rule["minimum_share"]) else "failed",
    }


def action(outcome: str, attempt: int, reference: Record) -> Record:
    """What a decision permits next. It never starts anything itself."""
    result = {
        "passed": "eligible_to_prepare_phase_8",
        "failed": "iterate_once_or_stop" if attempt == 1 else "stop",
    }.get(outcome, "none")
    condition = reference.get("condition")
    return {
        "result": result,
        "phase_8": "eligible_to_prepare" if outcome == "passed" else "not_eligible",
        "reference_planner": {
            "condition": condition,
            "identification": reference.get("identification"),
            "status": "retained",
        },
        "statement": ACTIONS[result].format(reference=condition),
        "automatic": NOTHING_LAUNCHED,
    }


def conclude(content: Record) -> Record:
    """The criteria outcome, outcome, pending reasons and action of assessed content.

    Inconsistent evidence decides nothing. Otherwise any pending criterion leaves the
    criteria pending, both criteria must pass, and synthetic evidence keeps a complete
    criteria result scientifically pending.
    """
    validator, structural = content["validator"], content["structural"]
    problems: list[str] = content["problems"]
    pending = [f"validator: {validator['reason']}"] if validator["status"] == "pending" else []
    if structural["status"] == "pending":
        pending.append(f"structural: {structural['reason']}")
    statuses = {validator["status"], structural["status"]}
    if problems or "pending" in statuses:
        criteria = "pending"
    else:
        criteria = "passed" if statuses == {"passed"} else "failed"
    if content["test_only"] or content["owner_review"]["evidence"] == "synthetic":
        pending.append(SYNTHETIC)
    outcome = "inconsistent" if problems else "pending" if pending else criteria
    return {
        "criteria_outcome": criteria,
        "outcome": outcome,
        "pending": pending,
        "action": action(outcome, content["protocol"]["attempt"], content["reference"]),
    }


def _applies_ratings(record: Record) -> bool:
    """Whether a decision applied the structural criterion to owner ratings."""
    return not record["problems"] and record["structural"]["status"] in COMPLETE


def settles(record: Record) -> bool:
    """Whether a decision is complete enough to settle its protocol for good."""
    return not record["problems"] and record["criteria_outcome"] in COMPLETE


# History: the repository's G-E records, for the one-iteration rule across protocols.


# A frozen rating set: the packet ID and its saved review files with their IDs.
type Frozen = tuple[str, tuple[tuple[str, str], ...]]


def _frozen(record: Record) -> Frozen:
    reviews: dict[str, str] = record["reviews"]
    return str(record["packet_id"]), tuple(sorted(reviews.items()))


@dataclass
class History:
    protocols: dict[str, Record] = field(default_factory=dict[str, Record])
    # Protocol IDs with a run claim, committed or kept beside a protocol.
    claims: set[str] = field(default_factory=set[str])
    # Protocol ID to the decision IDs its decision claims name.
    settlements: dict[str, set[str]] = field(default_factory=dict[str, set[str]])
    # Protocol ID to the IDs of committed review packets.
    packets: dict[str, set[str]] = field(default_factory=dict[str, set[str]])
    # Protocol ID to the frozen rating sets (packet ID and review files) review claims name.
    ratings: dict[str, set[Frozen]] = field(default_factory=dict[str, set[Frozen]])

    def add(self, record: Record) -> None:
        kind = record.get("kind")
        if kind == ge.PROTOCOL_KIND:
            self.protocols[record["id"]] = record
        elif kind == ge.CLAIM_KIND:
            self.claims.add(str(record["protocol_id"]))
        elif kind == SETTLEMENT_KIND:
            self.settlements.setdefault(str(record["protocol_id"]), set()).add(
                str(record["decision_id"])
            )
        elif kind == ge_review.PACKET_KIND:
            self.packets.setdefault(str(record["protocol_id"]), set()).add(str(record["id"]))
        elif kind == RATINGS_KIND:
            self.ratings.setdefault(str(record["protocol_id"]), set()).add(_frozen(record))


def _sealed(text: str | None) -> Record | None:
    """A sealed record from file text, or None when it is not one."""
    if not text:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    record = cast(Record, value)
    if record.get("version") != "1" or record.get("id") != identity(record):
        return None
    return record


def settlement_path(protocol: Path) -> Path:
    """Where a protocol's settled decision is claimed: beside the protocol."""
    return protocol.with_name(f"{protocol.stem}.decision-claim.json")


def ratings_path(protocol: Path) -> Path:
    """Where the owner ratings a decision first applied are frozen: beside the protocol."""
    return protocol.with_name(f"{protocol.stem}.review-claim.json")


def history(protocol: Path) -> History:
    """Every G-E record the protocol's repository has committed, plus claims beside protocols.

    Deleted and superseded versions count: a protocol or claim removed to reset the
    history is still found in the commit that added it.
    """
    found = History()
    texts: list[str | None] = []
    protocols = [protocol.resolve()]
    top = git(["rev-parse", "--show-toplevel"], protocol.parent)
    if top is not None:
        root = Path(top)
        arguments = ["log", "--all", "--format=", "--name-only", "-G", HISTORY_PATTERN]
        listed = git([*arguments, "--", "*.json"], root) or ""
        for name in sorted({line for line in listed.splitlines() if line}):
            commits = (git(["log", "--all", "--format=%H", "--", name], root) or "").split()
            texts += [git(["show", f"{commit}:{name}"], root) for commit in commits]
            protocols.append(root / name)
    # Claims are kept beside protocols and need not be committed; the files there now count.
    for path in protocols:
        for name in (path, ge.claim_path(path), settlement_path(path), ratings_path(path)):
            if name.is_file():
                texts.append(name.read_text(encoding="utf-8"))
    for text in texts:
        record = _sealed(text)
        if record is not None:
            found.add(record)
    return found


def reference_identity(reference: Record) -> Record:
    """A reference's canonical identity: its planner contract and the evidence naming it.

    The owner's C selection ID or the G-D decision ID identifies it, never the bytes or
    path of the file holding that record.
    """
    selection, decision = reference.get("c_selection"), reference.get("gd_decision")
    return {
        "condition": reference["condition"],
        "pipeline": reference["pipeline"],
        "params": reference["params"],
        "c_selection": None if selection is None else selection["id"],
        "gd_decision": None if decision is None else decision["id"],
    }


def _comparison(protocol: Record) -> tuple[bool, str]:
    """What a protocol is compared against: its test-only status and reference identity."""
    reference = reference_identity(protocol["frozen"]["reference"])
    return bool(protocol["test_only"]), json.dumps(reference, sort_keys=True)


def _attempt(protocol: Record) -> tuple[int, str | None]:
    iteration = protocol["frozen"]["iteration"]
    previous = iteration.get("previous")
    return int(iteration["attempt"]), None if previous is None else str(previous["id"])


def history_problems(protocol: Record, first: str, found: History) -> list[str]:
    """Violations of the one-iteration rule in the protocol's comparison.

    `first` is the comparison's first attempt: the protocol itself, or the one its
    iteration names. Every protocol with the same test-only status and canonical
    reference identity belongs to the comparison, whatever its dev briefs; only those
    that were run (claimed) count, and a run attempt counts even if never decided.
    """
    comparison = _comparison(protocol)
    family: dict[str, Record] = {}
    for other in [protocol, *found.protocols.values()]:
        try:
            key = _comparison(other)
            _attempt(other)
        except (KeyError, TypeError, ValueError):
            continue
        if key == comparison:
            family.setdefault(str(other["id"]), other)
    run = [other for name, other in sorted(family.items()) if name in found.claims]
    resets = [str(p["id"]) for p in run if _attempt(p)[0] == FIRST and p["id"] != first]
    iterations = [str(p["id"]) for p in run if _attempt(p) == (ITERATION, first)]
    strays = [str(p["id"]) for p in run if _attempt(p)[0] == ITERATION and _attempt(p)[1] != first]
    problems: list[str] = []
    if resets:
        problems.append(
            "history reset: another first attempt against the same reference was run, "
            f"decided or not: {', '.join(resets)}"
        )
    if len(iterations) > 1:
        problems.append(
            f"more than one development iteration of first attempt {first} was run: "
            + ", ".join(iterations)
        )
    if strays:
        problems.append(
            "a development iteration of another first attempt in this comparison was run: "
            + ", ".join(strays)
        )
    return problems


def _settled(protocol: Path, protocol_id: str, found: History) -> set[str]:
    """The decision IDs settled for a protocol, beside it now or anywhere in history."""
    names = set(found.settlements.get(protocol_id, set()))
    path = settlement_path(protocol)
    if path.is_file():
        claim = read_sealed(path)
        if claim.get("kind") != SETTLEMENT_KIND or claim.get("protocol_id") != protocol_id:
            raise ValueError(f"not this protocol's G-E decision claim: {path}")
        names.add(str(claim["decision_id"]))
    return names


def _first_attempt(path: Path, protocol: Record, out: Path) -> tuple[Record | None, list[str]]:
    """The development iteration's binding to its first attempt's settled failure."""
    previous: Record = protocol["frozen"]["iteration"]["previous"]
    first = (path.parent / previous["path"]).resolve()
    problems: list[str] = []
    try:
        if ge.read_protocol(first)["id"] != previous["id"]:
            problems.append("the first attempt's protocol changed since the iteration was bound")
        claim = ge.read_claim(first)
    except (OSError, ValueError) as error:
        return None, [f"first attempt protocol unavailable: {error}"]
    if claim is None or claim["id"] != previous.get("claim_id"):
        problems.append("the first attempt's run claim changed or is missing")
    settlement = settlement_path(first)
    if not settlement.is_file():
        return None, [
            *problems,
            "the first attempt has no settled decision: decide it before its iteration",
        ]
    try:
        ge.committed(settlement)
    except ValueError:
        problems.append("the first attempt's decision claim is not committed")
    try:
        claimed = read_sealed(settlement)
        folder = (first.parent / str(claimed["decision"])).resolve()
        saved, reasons = replay(folder)
    except (OSError, ValueError, KeyError) as error:
        return None, [*problems, f"the first attempt's decision cannot be replayed: {error}"]
    problems += [f"first attempt decision: {reason}" for reason in reasons]
    if saved["criteria_outcome"] == "passed":
        problems.append("the first attempt passed; no development iteration follows a pass")
    elif saved["criteria_outcome"] != "failed":
        problems.append("the first attempt's decision is not a complete failure")
    if not protocol["test_only"] and (saved["test_only"] or saved["outcome"] != "failed"):
        problems.append("a real development iteration follows only an actual failed first attempt")
    return {
        "protocol_id": previous["id"],
        "claim_id": previous.get("claim_id"),
        "decision": ge.relative(folder, out),
        "decision_id": saved["id"],
        "criteria_outcome": saved["criteria_outcome"],
        "outcome": saved["outcome"],
        "test_only": saved["test_only"],
    }, problems


def _history_section(
    path: Path, protocol: Record, out: Path, found: History
) -> tuple[Record, list[str]]:
    attempt, previous = _attempt(protocol)
    problems: list[str] = []
    try:
        ge.committed(ge.claim_path(path))
    except ValueError:
        problems.append(
            "the run claim is not committed: commit it beside the protocol so the "
            "repository's G-E history records this run"
        )
    first: Record | None = None
    if attempt == ITERATION:
        first, binding = _first_attempt(path, protocol, out)
        problems += binding
    problems += history_problems(protocol, previous or protocol["id"], found)
    section: Record = {
        "attempt": attempt,
        "policy": protocol["frozen"]["iteration"]["policy"],
        "change": protocol["frozen"]["iteration"]["change"],
        "first_attempt": first,
        "rule": "One run first attempt per comparison (test-only status and canonical "
        "reference identity, whatever the dev briefs) and at most one run development "
        "iteration, decided against the first attempt's settled failure.",
    }
    return section, problems


# Owner review: the packet must show exactly this protocol's sample of this run.


def _packet_problems(  # noqa: PLR0913 - the packet, its folder and what it must bind
    folder: Path,
    packet: Record,
    *,
    protocol: Record,
    path: Path,
    run: Path,
    record: Record | None,
) -> list[str]:
    owner = protocol["owner_review"]
    reference = protocol["frozen"]["reference"]["condition"]
    problems: list[str] = []
    if packet["protocol_id"] != protocol["id"] or packet["protocol_sha256"] != file_sha(path):
        problems.append("the review packet belongs to another protocol")
    if (folder / str(packet["run"])).resolve() != run.resolve():
        problems.append("the review packet shows another run folder")
    if packet["reviewer"] != owner["reviewer"]:
        problems.append("the review packet names another reviewer")
    if packet["criterion"] != protocol["criteria"]["structural_criterion"]:
        problems.append("the review packet states another structural criterion")
    if [item["slot"] for item in packet["items"]] != owner["sample"]:
        problems.append("the review packet's items differ from the registered owner sample")
    if record is None:
        return problems
    if packet["run_record_id"] != record["id"]:
        problems.append("the review packet belongs to another run record")
    if packet["test_only"] != record["test_only"]:
        problems.append("the review packet's test-only status differs from its run")
    key_file = folder / ge_review.KEY
    key: Record = read_sealed(key_file) if key_file.is_file() else {"items": {}, "paths": {}}
    labels: dict[str, dict[str, str]] = key["items"]
    paths: dict[str, dict[str, str]] = key["paths"]
    for item in packet["items"]:
        candidate = ge_review.unit(record, ge.CANDIDATE, item["slot"])
        baseline = ge_review.unit(record, reference, item["slot"])
        if item["kind"] != ge_review.item_kind(candidate, baseline):
            problems.append(f"{item['item']} is not settled as the run's outputs require")
        if item["kind"] != ge_review.OWNER:
            continue
        shown = labels.get(item["item"], {})
        units = {ge.CANDIDATE: candidate, reference: baseline}
        if sorted(shown.values()) != sorted(units) or any(
            paths.get(item["item"], {}).get(label) != f"{units[condition]['path']}/plan.json"
            for label, condition in shown.items()
        ):
            problems.append(f"{item['item']} does not show this slot's E and reference plans")
    return problems


def _reviews(folder: Path, used: dict[str, str]) -> list[Record]:
    """The identities of the saved reviews a decision applies."""
    rows: list[Record] = []
    for name in sorted(used):
        review = read_sealed(folder / ge_review.REVIEWS / name)
        rows.append(
            {
                "file": name,
                "id": review["id"],
                "classification": review["classification"],
                "reviewer": review["review"]["reviewer"],
                "source_sha256": review["source_sha256"],
                "items": len(review["review"]["ratings"]),
            }
        )
    return rows


def _owner_review(  # noqa: PLR0913 - the packet folder and everything it must bind
    folder: Path | None,
    protocol: Record,
    *,
    path: Path,
    run: Path,
    record: Record | None,
    out: Path,
    found: History,
) -> tuple[Record, list[Record] | None, str, list[str]]:
    """The owner review section, usable item outcomes (or why not) and problems.

    Once ratings are frozen for the protocol, only those reviews are applied.
    """
    section: Record = {
        "packet": None,
        "status": "pending",
        "evidence": None,
        "reviewer": protocol["owner_review"]["reviewer"],
        "judges": "diagnostic only: model estimates never replace the owner's review",
    }
    if folder is None:
        return section, None, "no owner review packet was supplied", []
    frozen, problems = _frozen_ratings(path, protocol, found)
    try:
        packet = ge_review.read_packet(folder)
        problems += _packet_problems(
            folder, packet, protocol=protocol, path=path, run=run, record=record
        )
        if frozen is not None and frozen[0] != packet["id"]:
            problems.append("the owner ratings were frozen for another review packet")
        status = ge_review.status(folder, None if frozen is None else dict(frozen[1]))
        reviews = _reviews(folder, status["review_files"])
    except (OSError, ValueError, KeyError, StopIteration) as error:
        section["packet"] = {"path": ge.relative(folder.resolve(), out)}
        reason = "the review packet cannot be read"
        return section, None, reason, [*problems, f"review packet: {error}"]
    problems += [f"review packet stale: {reason}" for reason in status["stale"]]
    others = found.packets.get(str(protocol["id"]), set()) - {packet["id"]}
    if others:
        problems.append(
            "another review packet for this protocol is committed: " + ", ".join(sorted(others))
        )
    section.update(
        packet={
            "path": ge.relative(folder.resolve(), out),
            "id": packet["id"],
            "key_id": packet["key_id"],
            "test_only": packet["test_only"],
            "budget": packet["budget"],
        },
        status=status["status"],
        evidence=status["evidence"],
        rated=status["rated"],
        awaiting_owner=status["awaiting_owner"],
        missing=status["items"]["missing"],
        reviews=reviews,
        reviewer_identity=status["reviewer_identity"],
    )
    usable = status["evidence"] == "actual" or (
        status["evidence"] == "synthetic" and packet["test_only"]
    )
    if problems or not usable:
        reason = (
            "the review packet is inconsistent"
            if problems
            else "synthetic ratings of a real packet cannot establish the structural criterion"
            if status["evidence"] == "synthetic"
            else f"{len(status['awaiting_owner'])} owner items await the registered reviewer's "
            "actual ratings"
            if status["awaiting_owner"]
            else "sampled evidence is missing"
        )
        return section, None, reason, problems
    return section, status["outcomes"], "", problems


def _frozen_ratings(
    path: Path, protocol: Record, found: History
) -> tuple[Frozen | None, list[str]]:
    """The protocol's frozen rating set, if any, and why it cannot be relied on."""
    claim = ratings_path(path)
    sets = set(found.ratings.get(str(protocol["id"]), set()))
    if claim.is_file():
        record = read_sealed(claim)
        if record.get("kind") != RATINGS_KIND or record.get("protocol_id") != protocol["id"]:
            raise ValueError(f"not this protocol's G-E review claim: {claim}")
        sets.add(_frozen(record))
    if not sets:
        return None, []
    problems: list[str] = []
    if len(sets) > 1:
        problems.append("more than one owner rating set was frozen for this protocol")
    try:
        ge.committed(claim)
    except ValueError:
        problems.append(
            "the frozen owner ratings are not committed: commit the review claim beside "
            "the protocol"
        )
    return min(sets), problems


def _run_section(run: Path, record: Record | None, out: Path) -> Record:
    if record is None:
        return {"path": ge.relative(run.resolve(), out), "record_id": None}
    return {
        "path": ge.relative(run.resolve(), out),
        "record_id": record["id"],
        "claim_id": record["claim"]["id"],
        "protocol_commit": record["protocol"]["commit"],
        "status": record["run"]["status"],
        "config_sha256": record["run"]["config_sha256"],
        "manifest_sha256": record["run"]["manifest_sha256"],
        "artifacts": len(record["artifacts"]),
    }


def assess(path: Path, run: Path, packet: Path | None, out: Path) -> Record:
    """The decision content for a protocol, run and packet; recorded paths are relative to `out`.

    Raises ValueError only when `path` is not a readable G-E protocol.
    """
    protocol = ge.read_protocol(path)
    checked = ge.check(path)
    problems = ge.verify(path, run, checked)
    try:
        record: Record | None = ge.read_run(run)
    except (OSError, ValueError):
        record = None
    if record is not None:
        problems += [f"run: {problem}" for problem in record["problems"]]
        if (
            record["evidence"]["status"] == "available"
            and not (record["evidence"]["reference"]["identified"])
        ):
            problems.append("evidence report: the reference baseline is not identified")
    found = history(path)
    review, outcomes, reason, review_problems = _owner_review(
        packet, protocol, path=path, run=run, record=record, out=out, found=found
    )
    problems += review_problems
    iteration, history_issues = _history_section(path, protocol, out, found)
    problems += history_issues
    if problems and outcomes is not None:
        # Owner ratings are never unblinded on inconsistent evidence.
        outcomes, reason = None, WITHHELD
    criteria = protocol["criteria"]
    frozen = protocol["frozen"]
    content: Record = {
        "kind": DECISION_KIND,
        "decision_version": DECISION_VERSION,
        "name": protocol["name"],
        "test_only": bool(protocol["test_only"] or (record is not None and record["test_only"])),
        "statement": STATEMENT,
        "protocol": {
            "path": ge.relative(path.resolve(), out),
            "id": protocol["id"],
            "sha256": file_sha(path),
            "attempt": iteration["attempt"],
        },
        "g3": checked["authority"],
        "candidate": frozen["candidate"],
        "lineage": frozen["lineage"],
        "reference": frozen["reference"],
        "cohort": frozen["cohort"],
        "run": _run_section(run, record, out),
        "validator": validator_criterion(
            None if record is None else record["validator"], criteria["validator_comparison"]
        ),
        "structural": structural_criterion(outcomes, criteria["structural_criterion"], reason),
        "owner_review": review,
        "history": iteration,
        "units": None if record is None else record["units"],
        "failures": None if record is None else record["failures"],
        "costs": None if record is None else record["costs"],
        "evidence": None if record is None else record["evidence"],
        "judges": criteria["judges"],
        "prior_exposure": protocol["registration"]["prior_exposure"],
        "limitations": LIMITATIONS,
        "problems": problems,
    }
    return content | conclude(content)


def read_decision(folder: Path) -> Record:
    record = read_sealed(folder / DECISION)
    if record.get("kind") != DECISION_KIND or record.get("decision_version") != DECISION_VERSION:
        raise ValueError(f"not a version {DECISION_VERSION} G-E decision: {folder}")
    return record


def _inputs(folder: Path, saved: Record) -> tuple[Path, Path, Path | None]:
    packet = saved["owner_review"]["packet"]
    return (
        (folder / saved["protocol"]["path"]).resolve(),
        (folder / saved["run"]["path"]).resolve(),
        None if packet is None else (folder / packet["path"]).resolve(),
    )


def replay(folder: Path) -> tuple[Record, list[str]]:
    """A saved decision and why it no longer follows from current evidence, if at all."""
    saved = read_decision(folder)
    protocol, run, packet = _inputs(folder, saved)
    current = json.loads(json.dumps(assess(protocol, run, packet, folder.resolve())))
    reasons = [
        f"decision {section} differs from current evidence"
        for section in sorted((saved.keys() | current.keys()) - {"id", "version"})
        if saved.get(section) != current.get(section)
    ]
    reasons += [
        f"new problem: {problem}"
        for problem in current["problems"]
        if problem not in saved["problems"]
    ]

    if settles(saved):
        settled = _settled(protocol, saved["protocol"]["id"], history(protocol))
        if settled != {saved["id"]}:
            reasons.append("this decision is not its protocol's only settled decision")
    return saved, reasons


def decide(path: Path, run: Path, packet: Path | None, out: Path) -> Record:
    """Write a new sealed decision; a complete one settles its protocol for good."""
    if out.exists():
        raise ValueError("decision folder already exists; choose a new folder to keep history")
    protocol = ge.read_protocol(path)
    settled = _settled(path, protocol["id"], history(path))
    if settled:
        raise ValueError(
            f"protocol already has a settled G-E decision ({', '.join(sorted(settled))}): "
            "replay it with `nde ge replay`; a decision is never replaced"
        )
    content = assess(path, run, packet, out.resolve())
    out.mkdir(parents=True)
    record = create_record(out / DECISION, content)
    write_text_lf(out / "decision.md", markdown(record))
    if _applies_ratings(record) and not ratings_path(path).exists():
        create_record(
            ratings_path(path),
            {
                "kind": RATINGS_KIND,
                "protocol_id": protocol["id"],
                "packet_id": record["owner_review"]["packet"]["id"],
                "reviews": {row["file"]: row["id"] for row in record["owner_review"]["reviews"]},
                "first_decision_id": record["id"],
            },
        )
    if settles(record):
        create_record(
            settlement_path(path),
            {
                "kind": SETTLEMENT_KIND,
                "protocol_id": protocol["id"],
                "decision_id": record["id"],
                "decision": ge.relative(out.resolve(), path.parent.resolve()),
                "attempt": record["protocol"]["attempt"],
                "criteria_outcome": record["criteria_outcome"],
                "outcome": record["outcome"],
            },
        )
    return record


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _evidence_lines(evidence: Record | None) -> list[str]:
    if evidence is None:
        return ["No run record."]
    if evidence["status"] != "available":
        return [f"Evidence report unavailable: {evidence['error']}"]
    deterministic = evidence["adherence"]["deterministic"]["total"]
    judgement = evidence["adherence"]["judgement"]["total"]
    overlap = evidence["overlap"]
    return [
        f"Evidence report `{evidence['report']['id']}`; test-only "
        f"{evidence['report']['test_only']}.",
        "",
        f"- Deterministic condition adherence: {json.dumps(deterministic, sort_keys=True)}",
        f"- Literary adherence judgements: {json.dumps(judgement, sort_keys=True)}",
        *(
            f"- Diversity ({group['group']}): {json.dumps(group['summary'], sort_keys=True)}"
            for group in evidence["diversity"]
        ),
        f"- Training lineage complete: {overlap['lineage']['complete']}; G3 population "
        f"{overlap['g3_population']}; protected overlap: {overlap['protected_overlap']}",
        f"- Reference {evidence['reference']['permitted']} identified: "
        f"{evidence['reference']['identified']}",
        *(f"- Reported, not a criterion: {item}" for item in evidence["pending"]),
    ]


def _cost_lines(costs: Record | None) -> list[str]:
    if costs is None:
        return ["No run record."]
    return [
        costs["rule"],
        "",
        f"New spend ${costs['run']['usd']:.6f}; list price ${costs['run']['token_cost_usd']:.6f}.",
        *(
            f"- {name}: {sum(c['calls'] for c in cost['provider'].values())} provider calls, "
            f"{cost['local']['passes']} local passes, {cost['local']['tokens']} local tokens"
            for name, cost in costs["conditions"].items()
        ),
    ]


def markdown(record: Record) -> str:
    """The rendered decision; the sealed JSON is authoritative."""
    validator, structural = record["validator"], record["structural"]
    first = record["history"]["first_attempt"]
    g3: Record = record["g3"] or {}
    failures: list[Record] = record["failures"] or []
    lines = [
        f"# G-E decision: {record['outcome']}" + (" (test-only)" if record["test_only"] else ""),
        "",
        record["statement"],
        "",
        f"Decision `{record['id']}`; protocol `{record['protocol']['id']}` "
        f"({record['name']}, attempt {record['protocol']['attempt']}); run record "
        f"`{record['run']['record_id']}`; G3 authority {g3.get('status')}.",
        "",
        f"Criteria outcome: **{record['criteria_outcome']}**. Outcome: **{record['outcome']}**. "
        f"Owner review evidence: {record['owner_review']['evidence']}.",
        "",
        "## Action",
        "",
        record["action"]["statement"],
        "",
        f"Reference planner: {record['action']['reference_planner']['condition']} "
        f"({record['action']['reference_planner']['identification']}), retained. "
        f"{record['action']['automatic']}",
        "",
        "## Validator comparison (E minus reference)",
        "",
        validator["rule"] + ".",
        "",
        f"E rate {_rate(validator['candidate_rate'])}; reference rate "
        f"{_rate(validator['reference_rate'])}; difference {_rate(validator['difference'])} "
        f"against lower bound {_rate(validator['lower_bound'])}; minimum E rate "
        f"{_rate(validator['minimum_candidate_rate'])}. Status: {validator['status']}"
        + (f" ({validator['reason']})" if validator["reason"] else "")
        + ".",
        "",
        "## Structural competitiveness (owner)",
        "",
        f"{structural['rule']}. Question: {structural['question']}",
        "",
        f"Competitive {structural['competitive']} of {structural['denominator']}; share "
        f"{_rate(structural['share'])} against minimum {_rate(structural['minimum_share'])}; "
        f"excluded {structural['excluded'] or 'none'}. Status: {structural['status']}"
        + (f" ({structural['reason']})" if structural["reason"] else "")
        + ".",
        "",
        *(f"- {item['slot']}: {item['kind']}, {item['outcome']}" for item in structural["items"]),
        "",
        "## Development iteration history",
        "",
        record["history"]["rule"],
        "",
        "First attempt."
        if first is None
        else (
            f"Development iteration of first attempt `{first['protocol_id']}`, bound to its "
            f"settled decision `{first['decision_id']}` ({first['criteria_outcome']})."
        ),
        "",
        "## Evidence (reported)",
        "",
        *_evidence_lines(record["evidence"]),
        "",
        "## Costs",
        "",
        *_cost_lines(record["costs"]),
        "",
        "## Failures",
        "",
        *(
            f"- {f['condition']} {f['slot']}: {f['status']} {f['error'] or ''}".rstrip()
            for f in failures
        ),
        *([] if failures else ["None"]),
        "",
        "## Pending",
        "",
        *(f"- {item}" for item in record["pending"] or ["None"]),
        "",
        "## Problems",
        "",
        *(f"- {problem}" for problem in record["problems"] or ["None"]),
        "",
        "## Limitations",
        "",
        f"Prior exposure: {record['prior_exposure']}",
        "",
        *(f"- {item}" for item in record["limitations"]),
        "",
    ]
    return "\n".join(lines)
