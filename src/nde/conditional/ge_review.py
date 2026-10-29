"""The owner's fixed, budgeted structural review packet for a registered G-E run.

The packet holds exactly the protocol's frozen dev sample: for each sampled slot,
E's plan and the reference's plan under opaque labels A and B, the registered
structural question and the owner time budget. Slots where a plan is absent are
settled by the registered rules without owner reading. Reviews bind the packet
and each plan's salted token, so a changed output makes the packet stale.
Blinding is procedural: the coin and tokens use a random salt kept only in the
sealed key, so nothing the owner reads links a label to a condition, but the key
and the run stay on disk. Only the
registered reviewer's own, non-synthetic assessments complete it; model judges
and synthetic fixtures are recorded and counted but never complete it, and
missing review stays pending. The reviewer's identity is self-declared, as for
every owner review in this repository. Applying the criterion is the G-E decision's job.
"""

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from nde.conditional import ge
from nde.corpus.records_io import create_record, file_sha, read_sealed
from nde.files import write_text_lf
from nde.llm.config import Strict
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import card_lines, load_plan

type Record = dict[str, Any]

PACKET_KIND = "ge-review-packet"
PACKET_VERSION = "1"
PACKET = "packet.json"
KEY = "key.json"
REVIEWS = "reviews"
REVIEW_KIND = "ge-structural-review"
OWNER, CANDIDATE_ABSENT, REFERENCE_ABSENT, MISSING = (
    "owner",
    "candidate-no-output",
    "reference-no-output",
    "missing",
)
RULES = {
    OWNER: "the owner compares the two plans under the registered question",
    CANDIDATE_ABSENT: "E produced no plan: not competitive under the registered rule",
    REFERENCE_ABSENT: "the reference produced no plan: excluded from the owner denominator "
    "and reported",
    MISSING: "a plan's evidence is missing or unreadable: pending until the run is complete",
}


class Rating(Strict):
    item: str = Field(min_length=1)
    # The tokens of the two plans the owner read, exactly as the packet lists them.
    plans: dict[Literal["A", "B"], str]
    rating: Literal["A", "B", "equivalent", "unknown"]
    note: str = Field(min_length=1)


class StructuralReview(Strict):
    packet_id: str = Field(min_length=64, max_length=64)
    reviewer: str = Field(min_length=1)
    test_only: bool = False
    # A model's estimate names its model and prompt; it never counts as owner review.
    diagnostic_provenance: dict[str, str] | None = None
    ratings: list[Rating] = Field(min_length=1)

    @model_validator(mode="after")
    def _identified(self) -> Self:
        provenance = self.diagnostic_provenance
        if provenance is not None and not (provenance.get("model") and provenance.get("prompt")):
            raise ValueError("diagnostic estimates must identify their model and prompt")
        if len({rating.item for rating in self.ratings}) != len(self.ratings):
            raise ValueError("rate each packet item at most once per review")
        return self


def unit(record: Record, condition: str, slot: str) -> Record:
    """A run record's unit for one condition and cohort slot."""
    return next(u for u in record["units"] if u["condition"] == condition and u["slot"] == slot)


def _plan(run: Path, row: Record) -> tuple[str, str]:
    """A unit's saved plan file, relative to the run, and its checksum."""
    path = Path(str(row["path"])) / "plan.json"
    return path.as_posix(), file_sha(run / path)


def item_kind(candidate: Record, reference: Record) -> str:
    """How the registered rules settle a sampled slot, from the run's two units."""
    if "unknown" in {candidate["validity"], reference["validity"]}:
        return MISSING
    if candidate["plan_sha256"] is None:
        return CANDIDATE_ABSENT
    return REFERENCE_ABSENT if reference["plan_sha256"] is None else OWNER


def _order(salt: str, slot: str) -> bool:
    """Whether E is shown as plan A: a per-slot coin from the key's secret salt."""
    return int(hashlib.sha256(f"order:{salt}:{slot}".encode()).hexdigest(), 16) % 2 == 0


def _token(salt: str, sha256: str) -> str:
    """A plan's identity in the packet, unlinkable to the run's condition-named checksums."""
    return hashlib.sha256(f"plan:{salt}:{sha256}".encode()).hexdigest()


def _cards(plan_file: Path) -> list[str]:
    plan = load_plan(plan_file)
    # The title and plan-identity line can name the condition; headings nest under the label.
    lines = card_lines(plan, validate_plan(plan))[2:]
    return [f"####{line[2:]}" if line.startswith("## ") else line for line in lines]


def create(protocol_path: Path, run: Path, out: Path) -> Record:
    """Assemble the packet for a current registered run; raises ValueError when stale."""
    if out.exists():
        raise ValueError("packet folder already exists; choose a new folder to keep history")
    reasons = ge.verify(protocol_path, run)
    if reasons:
        raise ge.StaleEvidenceError(reasons)
    protocol, record = ge.read_protocol(protocol_path), ge.read_run(run)
    owner = protocol["owner_review"]
    criterion = protocol["criteria"]["structural_criterion"]
    reference = protocol["frozen"]["reference"]["condition"]
    items: list[Record] = []
    # Conditions, run paths and checksums name the condition, so they stay in the key.
    salt = secrets.token_hex(16)
    key: dict[str, dict[str, str]] = {}
    paths: dict[str, dict[str, str]] = {}
    checksums: dict[str, dict[str, str]] = {}
    for index, slot in enumerate(owner["sample"], start=1):
        name = f"item-{index:02d}"
        candidate, baseline = unit(record, ge.CANDIDATE, slot), unit(record, reference, slot)
        kind = item_kind(candidate, baseline)
        item: Record = {
            "item": name,
            "slot": slot,
            "kind": kind,
            "rule": RULES[kind],
            "plans": None,
        }
        if kind == OWNER:
            first, second = (candidate, baseline) if _order(salt, slot) else (baseline, candidate)
            shown = {"A": _plan(run, first), "B": _plan(run, second)}
            item["plans"] = {
                label: {"token": _token(salt, sha)} for label, (_, sha) in shown.items()
            }
            paths[name] = {label: path for label, (path, _) in shown.items()}
            checksums[name] = {label: sha for label, (_, sha) in shown.items()}
            key[name] = {"A": first["condition"], "B": second["condition"]}
        items.append(item)
    out.mkdir(parents=True)
    sealed_key = create_record(
        out / KEY,
        {
            "kind": "ge-review-key",
            "salt": salt,
            "items": key,
            "paths": paths,
            "sha256": checksums,
        },
    )
    owner_items = sum(item["kind"] == OWNER for item in items)
    packet = create_record(
        out / PACKET,
        {
            "kind": PACKET_KIND,
            "packet_version": PACKET_VERSION,
            "test_only": record["test_only"],
            "protocol_id": protocol["id"],
            "protocol_sha256": file_sha(protocol_path),
            "run": ge.relative(run.resolve(), out.resolve()),
            "run_record_id": record["id"],
            "reviewer": owner["reviewer"],
            "question": criterion["question"],
            "criterion": criterion,
            "ratings": {
                "A": "plan A has the better structure",
                "B": "plan B has the better structure",
                "equivalent": "neither is better",
                "unknown": "cannot judge",
            },
            "budget": {
                "sample": len(items),
                "owner_items": owner_items,
                "minutes_per_item": owner["minutes_per_item"],
                "max_minutes": owner["max_minutes"],
                "estimated_minutes": owner_items * owner["minutes_per_item"],
            },
            "judges": "diagnostic only: model estimates never replace the owner's review",
            "blinding": "Procedural: plans are labelled A and B by a coin from a secret salt "
            "kept only in the sealed key, and identified by salted tokens. Do not open the key "
            "or the run folder while reviewing. Structure itself may still reveal the condition.",
            "items": items,
            "key_sha256": file_sha(out / KEY),
            "key_id": sealed_key["id"],
        },
    )
    write_text_lf(out / "packet.md", _markdown(packet, run, paths))
    template = {
        "packet_id": packet["id"],
        "reviewer": owner["reviewer"],
        "test_only": packet["test_only"],
        "ratings": [
            {
                "item": item["item"],
                "plans": {label: plan["token"] for label, plan in item["plans"].items()},
                "rating": "A | B | equivalent | unknown",
                "note": "Why, in a sentence.",
            }
            for item in items
            if item["kind"] == OWNER
        ],
    }
    write_text_lf(out / "review-template.json", json.dumps(template, indent=2) + "\n")
    return packet


def _markdown(packet: Record, run: Path, paths: dict[str, dict[str, str]]) -> str:
    budget = packet["budget"]
    lines = [
        "# G-E owner structural review" + (" (test-only)" if packet["test_only"] else ""),
        "",
        f"Packet `{packet['id']}` for reviewer {packet['reviewer']}.",
        "",
        f"**Question:** {packet['question']}",
        "",
        f"Rate each pair A, B, equivalent or unknown, with a one-sentence note. "
        f"{budget['owner_items']} pairs to read, about {budget['estimated_minutes']} of "
        f"{budget['max_minutes']} budgeted minutes. Do not open `key.json`.",
        "",
        packet["blinding"],
        "",
    ]
    for item in packet["items"]:
        lines += [f"## {item['item']}", ""]
        if item["kind"] != OWNER:
            lines += [f"No owner reading: {item['rule']}.", ""]
            continue
        for label in ("A", "B"):
            lines += [f"### Plan {label}", "", *_cards(run / paths[item["item"]][label]), ""]
    return "\n\n".join(line for line in lines if line) + "\n"


def read_packet(folder: Path) -> Record:
    packet = read_sealed(folder / PACKET)
    if packet.get("kind") != PACKET_KIND or packet.get("packet_version") != PACKET_VERSION:
        raise ValueError(f"not a version {PACKET_VERSION} G-E review packet: {folder}")
    return packet


def stale_reasons(folder: Path, packet: Record) -> list[str]:
    """Why the packet no longer shows the run's current outputs, if at all."""
    run = (folder / packet["run"]).resolve()
    try:
        record = ge.read_run(run)
    except (OSError, ValueError) as error:
        return [f"G-E run record missing or changed: {error}"]
    reasons = [] if record["id"] == packet["run_record_id"] else ["the G-E run record changed"]
    reasons += ge.artifact_problems(run, record)
    if not (folder / KEY).is_file() or file_sha(folder / KEY) != packet["key_sha256"]:
        return [*reasons, "the sealed blinding key changed or is missing"]
    key = read_sealed(folder / KEY)
    paths: dict[str, dict[str, str]] = key["paths"]
    for item in packet["items"]:
        plans: dict[str, Record] = item["plans"] or {}
        for label, plan in plans.items():
            path = run / paths[item["item"]][label]
            if not path.is_file() or _token(key["salt"], file_sha(path)) != plan["token"]:
                reasons.append(f"{item['item']} plan {label} changed or missing")
    return reasons


def _classification(review: StructuralReview, packet: Record) -> str:
    if review.diagnostic_provenance is not None:
        return "diagnostic"
    if review.test_only or packet["test_only"]:
        return "synthetic"
    return "actual" if review.reviewer == packet["reviewer"] else "other-reviewer"


def record_review(folder: Path, path: Path) -> Record:
    """Save one review immutably beside the packet; reject it unless it binds this packet."""
    packet = read_packet(folder)
    reasons = stale_reasons(folder, packet)
    if reasons:
        raise ge.StaleEvidenceError(reasons)
    review = StructuralReview.model_validate_json(path.read_text(encoding="utf-8"))
    if review.packet_id != packet["id"]:
        raise ValueError("review names another packet")
    shown = {
        item["item"]: {label: plan["token"] for label, plan in item["plans"].items()}
        for item in packet["items"]
        if item["kind"] == OWNER
    }
    for rating in review.ratings:
        if rating.item not in shown:
            raise ValueError(f"{rating.item} is not an owner item in this packet")
        if rating.plans != shown[rating.item]:
            raise ValueError(f"{rating.item} rates plans other than the packet's current plans")
    existing = sorted((folder / REVIEWS).glob("review-*.json"))
    return create_record(
        folder / REVIEWS / f"review-{len(existing) + 1:03d}.json",
        {
            "kind": REVIEW_KIND,
            "packet_id": packet["id"],
            "classification": _classification(review, packet),
            "source_sha256": file_sha(path),
            "review": review.model_dump(mode="json"),
        },
    )


def _outcome(item: Record, rating: Record | None, key: dict[str, dict[str, str]]) -> str:
    if item["kind"] == CANDIDATE_ABSENT:
        return "not_competitive"
    if item["kind"] == REFERENCE_ABSENT:
        return "excluded"
    if item["kind"] == MISSING or rating is None:
        return "pending"
    if rating["rating"] == "equivalent":
        return "competitive"
    if rating["rating"] == "unknown":
        return "not_competitive"
    return (
        "competitive" if key[item["item"]][rating["rating"]] == ge.CANDIDATE else "not_competitive"
    )


def status(folder: Path) -> Record:
    """The packet's review completeness, and for #85 the unblinded item outcomes.

    `complete` requires the registered reviewer's actual ratings for every owner
    item of a current, non-synthetic packet with no missing evidence. A synthetic
    review can finish the mechanics (`evidence: synthetic`) but never completes it.
    """
    packet = read_packet(folder)
    stale = stale_reasons(folder, packet)
    reviews = [read_sealed(path) for path in sorted((folder / REVIEWS).glob("review-*.json"))]
    latest: dict[str, dict[str, Record]] = {
        name: {} for name in ("actual", "synthetic", "diagnostic", "other-reviewer")
    }
    for review in reviews:
        if review.get("kind") != REVIEW_KIND or review.get("packet_id") != packet["id"]:
            raise ValueError("a saved review does not belong to this packet")
        for rating in review["review"]["ratings"]:
            # Later reviews supersede earlier ratings of the same kind; history is kept.
            latest[review["classification"]][rating["item"]] = rating
    owner = [item["item"] for item in packet["items"] if item["kind"] == OWNER]
    missing = [item["item"] for item in packet["items"] if item["kind"] == MISSING]
    actual = not packet["test_only"] and all(name in latest["actual"] for name in owner)
    synthetic = all(name in latest["actual"] or name in latest["synthetic"] for name in owner)
    evidence = (
        None if stale or missing else "actual" if actual else "synthetic" if synthetic else None
    )
    source = latest["actual"] if evidence == "actual" else latest["synthetic"] | latest["actual"]
    key: dict[str, dict[str, str]] = {} if stale else read_sealed(folder / KEY)["items"]
    outcomes = [
        {
            "item": item["item"],
            "slot": item["slot"],
            "kind": item["kind"],
            "outcome": _outcome(item, source.get(item["item"]), key) if evidence else "pending",
        }
        for item in packet["items"]
    ]
    state = "stale" if stale else "complete" if evidence == "actual" else "pending"
    pending = [name for name in owner if name not in latest["actual"]]
    return {
        "kind": "ge-review-status",
        "packet_id": packet["id"],
        "status": state,
        "evidence": evidence,
        "test_only": packet["test_only"],
        "reviewer": packet["reviewer"],
        "stale": stale,
        "items": {
            "sample": len(packet["items"]),
            "owner": len(owner),
            "automatic": len(packet["items"]) - len(owner) - len(missing),
            "missing": missing,
        },
        "rated": {name: len(rows) for name, rows in latest.items()},
        "awaiting_owner": pending,
        "reviews": len(reviews),
        "outcomes": outcomes,
        "rule": "Only the registered reviewer's actual ratings complete review. Model "
        "judges are diagnostic and synthetic reviews are test-only; their agreement "
        "never replaces the owner.",
        "reviewer_identity": "self-declared: the review names the registered reviewer; "
        "nothing proves who rated the plans or that the key stayed closed",
    }
