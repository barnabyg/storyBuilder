"""Freeze stratified review work before labels; retain source-linked owner history."""

import argparse
import hashlib
import html
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from nde.annotations.records import digest, read_session
from nde.corpus.acquisition import load_work
from nde.corpus.families import aligned_claims, family_report
from nde.corpus.features import load_features
from nde.corpus.records_io import (
    check_snapshot,
    cohort_file,
    create_record,
    evidence_files,
    file_sha,
    identity,
    read_sealed,
    report_files,
    snapshot,
)
from nde.corpus.segmentation import load_segmentation
from nde.corpus.splits import load_cohort
from nde.files import write_text_lf
from nde.llm.config import Strict
from nde.narrative.models import NarrativePlan

CHANNELS = ("plot", "causality", "information", "goal", "belief", "relationships", "interpretation")
CORE = ("plot", "causality", "information", "goal", "belief")


class ReviewInput(Strict):
    work_id: str
    work_dir: Path
    segmentation: Path
    features: Path
    families: Path
    gold_session: Path | None = None


class ReviewPolicy(Strict):
    cohort: Path
    inputs: list[ReviewInput] = Field(min_length=1)
    test_only: bool = False
    seed: int = 7
    per_stratum: int = Field(default=1, ge=1)
    max_samples: int = Field(default=350, ge=1)
    channels: list[str] = Field(default_factory=lambda: list(CHANNELS))
    min_owner_per_channel: int = Field(default=3, ge=1)
    min_gold_per_channel: int = Field(default=1, ge=0)
    reliable_agreement: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    reliable_owner: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    reliable_gold: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    weak_owner: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)


class Judgement(Strict):
    sample_id: str
    owner: str = Field(min_length=1)
    judgement: Literal["accurate", "defensible", "not_defensible", "unknown"]
    notes: str = Field(min_length=1)
    test_only: bool = False


def channel_claims(plan: NarrativePlan, channel: str) -> list[dict[str, Any]]:
    """Gold comparison ignores generated IDs but retains values and exact source spans."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel: {channel}")
    return [json.loads(value) for value in aligned_claims(plan)[channel].elements()]


def _gold(item: ReviewInput, source_sha256: str) -> tuple[NarrativePlan | None, str]:
    if item.gold_session is None:
        return None, "missing"
    session = read_session(item.gold_session)
    if session.source.sha256 != source_sha256:
        return None, "stale source"
    if not session.revisions or not session.owner_reviews:
        return None, "unapproved"
    revision = session.revisions[-1]
    reviews = [r for r in session.owner_reviews if r.revision == revision.number]
    if not reviews or not reviews[-1].accepted or reviews[-1].sha256 != revision.sha256:
        return None, "latest revision not approved"
    path = item.gold_session / revision.file

    if digest(path.read_text(encoding="utf-8")) != revision.sha256:
        return None, "stale revision"
    return NarrativePlan.model_validate_json(path.read_text(encoding="utf-8")), "current approved"


def _inputs(policy: ReviewPolicy) -> tuple[list[dict[str, Any]], list[Path]]:

    cohort = load_cohort(policy.cohort)
    assignments = {row.work_id: row for row in cohort.assignments}
    if len({item.work_id for item in policy.inputs}) != len(policy.inputs):
        raise ValueError("duplicate review work")
    candidates: list[dict[str, Any]] = []
    files = [cohort_file(policy.cohort)]
    for item in policy.inputs:
        work, _ = load_work(item.work_dir)
        if work.work_id != item.work_id or item.work_id not in assignments:
            raise ValueError("review work not in frozen cohort")
        assignment = assignments[item.work_id]
        if assignment.text_sha256 != work.text_sha256:
            raise ValueError("review source differs from frozen cohort")
        load_segmentation(item.segmentation, item.work_dir)
        feature = load_features(item.features, item.work_dir, item.segmentation)
        comparison = family_report(item.families)
        if comparison.get("status") != "complete":
            raise ValueError("review requires complete independent-family annotations")
        families: list[dict[str, Any]] = comparison["families"]
        plans = [Path(row["directory"]) / "annotation" / "plan.json" for row in families]
        plan = NarrativePlan.model_validate_json(plans[0].read_text(encoding="utf-8"))
        gold, gold_status = _gold(item, work.text_sha256)
        agreements: dict[str, dict[str, int]] = {}
        for pair in comparison["agreement"]:
            counts = agreements.setdefault(
                pair["channel"], {"agreed": 0, "total": 0, "unmatched": 0}
            )
            for key in ("agreed", "total", "unmatched"):
                counts[key] += pair[key]
        files += [item.segmentation, item.features, *evidence_files(item.work_dir)]
        files += evidence_files(item.families)
        if item.gold_session is not None:
            files += evidence_files(item.gold_session)
        for channel in policy.channels:
            claims = channel_claims(plan, channel)
            gold_claims = channel_claims(gold, channel) if gold is not None else None
            expected = Counter(json.dumps(c, sort_keys=True) for c in gold_claims or [])
            actual = Counter(json.dumps(c, sort_keys=True) for c in claims)
            agreement = agreements.get(channel, {"agreed": 0, "total": 0, "unmatched": 0})
            uncertain = any(
                "unknown" in json.dumps(c) or "ambiguous" in json.dumps(c) for c in claims
            )
            row: dict[str, Any] = {
                "work_id": work.work_id,
                "author_id": assignment.author_id,
                "channel": channel,
                "source_sha256": work.text_sha256,
                "annotation_files": snapshot(plans),
                "segmentation_sha256": file_sha(item.segmentation),
                "feature_path": str(item.features.resolve()),
                "feature_sha256": file_sha(item.features),
                "features": feature.work.model_dump(mode="json"),
                "stratum": [
                    work.period,
                    work.genre,
                    assignment.split,
                    "disagreement" if agreement["agreed"] != agreement["total"] else "agreement",
                    "uncertain" if uncertain else "no_explicit_uncertainty",
                ],
                "claims": claims,
                "agreement": agreement,
                "gold_status": gold_status,
                "gold_matches": sum((actual & expected).values()) if gold is not None else None,
                "gold_union": sum((actual | expected).values()) if gold is not None else None,
                "gold_annotation": str(item.gold_session) if item.gold_session else None,
            }
            row["id"] = identity(row)
            candidates.append(row)
    return candidates, files


def freeze(config: Path, out: Path) -> dict[str, Any]:
    policy = ReviewPolicy.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    if not policy.channels or len(set(policy.channels)) != len(policy.channels):
        raise ValueError("review needs distinct registered channels")
    if set(policy.channels) - set(CHANNELS):
        raise ValueError("unknown registered channel")
    if policy.weak_owner > policy.reliable_owner:
        raise ValueError("weak threshold cannot exceed reliable threshold")
    if not policy.test_only and policy.min_gold_per_channel < 1:
        raise ValueError("real channel reliability requires current approved gold evidence")
    candidates, files = _inputs(policy)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        strata[json.dumps([row["channel"], row["stratum"]])].append(row)
    samples: list[dict[str, Any]] = []
    inclusion: dict[str, Any] = {}
    for stratum, rows in sorted(strata.items()):
        ordered = sorted(
            rows, key=lambda row: hashlib.sha256(f"{policy.seed}:{row['id']}".encode()).hexdigest()
        )
        selected = ordered[: policy.per_stratum]
        samples.extend(selected)
        inclusion[stratum] = {"available": len(rows), "included": len(selected)}
    if len(samples) > policy.max_samples:
        raise ValueError(
            "registered sample cap cannot cover required strata; register a new policy"
        )
    if out.exists():
        raise ValueError("review packet exists; preserve labels and freeze a new version")
    packet = create_record(
        out / "packet.json",
        {
            "policy": policy.model_dump(mode="json"),
            "inputs": snapshot([config, *files]),
            "test_only": policy.test_only,
            "samples": samples,
            "inclusion": inclusion,
            "population": len(candidates),
            "matching": (
                "exact source/scene order, named entities, values and spans; ID spelling ignored"
            ),
        },
    )
    cards = "".join(
        f"<h2>{html.escape(s['work_id'])}: {html.escape(s['channel'])}</h2>"
        f"<p>Sample {s['id']}; source {s['source_sha256']}; "
        f"gold {html.escape(s['gold_status'])}</p>"
        f"<pre>{html.escape(json.dumps(s, indent=2))}</pre>"
        for s in samples
    )
    write_text_lf(
        out / "packet.html",
        "<!doctype html><meta charset='utf-8'><h1>Frozen corpus review sample</h1>" + cards,
    )
    report(out)
    return packet


def record_judgement(directory: Path, file: Path) -> dict[str, Any]:
    packet = read_sealed(directory / "packet.json")
    check_snapshot(packet["inputs"])
    judgement = Judgement.model_validate_json(file.read_text(encoding="utf-8"))
    if judgement.test_only != packet["test_only"]:
        raise ValueError("synthetic and actual owner evidence cannot be interchanged")
    sample = next((s for s in packet["samples"] if s["id"] == judgement.sample_id), None)
    if sample is None:
        raise ValueError("judgement sample is outside frozen packet")
    labels = directory / "judgements"
    count = len(list(labels.glob("*.json"))) if labels.exists() else 0
    return create_record(
        labels / f"{count + 1:06d}.json",
        {
            **judgement.model_dump(mode="json"),
            "packet_id": packet["id"],
            "annotation_files": sample["annotation_files"],
            "source_sha256": sample["source_sha256"],
            "recorded_at": datetime.now(UTC).isoformat(),
        },
    )


def report(directory: Path) -> dict[str, Any]:
    packet = read_sealed(directory / "packet.json")
    problems: list[str] = []
    try:
        check_snapshot(packet["inputs"])
        _inputs(ReviewPolicy.model_validate(packet["policy"]))
    except (OSError, ValueError) as error:
        problems.append(str(error))
    labels: dict[str, dict[str, Any]] = {}
    for path in sorted((directory / "judgements").glob("*.json")):
        label = read_sealed(path)
        if label["packet_id"] != packet["id"] or label["test_only"] != packet["test_only"]:
            raise ValueError("label belongs to different review evidence")
        labels[label["sample_id"]] = label
    policy = ReviewPolicy.model_validate(packet["policy"])
    channels: dict[str, Any] = {}
    for channel in policy.channels:
        samples = [s for s in packet["samples"] if s["channel"] == channel]
        reviewed = [labels[s["id"]] for s in samples if s["id"] in labels]
        known = [r for r in reviewed if r["judgement"] != "unknown"]
        defensible = sum(r["judgement"] in {"accurate", "defensible"} for r in known)
        agreed = sum(s["agreement"]["agreed"] for s in samples)
        total = sum(s["agreement"]["total"] for s in samples)
        gold = [s for s in samples if s["gold_union"] is not None and s["gold_union"] > 0]
        gold_matches = sum(s["gold_matches"] for s in gold)
        gold_total = sum(s["gold_union"] for s in gold)
        owner_rate = defensible / len(known) if known else None
        agreement_rate = agreed / total if total else None
        gold_rate = gold_matches / gold_total if gold_total else None
        enough = (
            not problems
            and len(known) >= policy.min_owner_per_channel
            and len(reviewed) == len(samples)
            and total > 0
            and len(gold) >= policy.min_gold_per_channel
        )
        classification = "pending"
        if enough:
            reliable = (
                owner_rate is not None
                and owner_rate >= policy.reliable_owner
                and agreement_rate is not None
                and agreement_rate >= policy.reliable_agreement
                and (
                    policy.min_gold_per_channel == 0
                    or (gold_rate is not None and gold_rate >= policy.reliable_gold)
                )
            )
            classification = (
                "reliable"
                if reliable
                else "weak"
                if owner_rate is not None and owner_rate >= policy.weak_owner
                else "unmeasurable"
            )
        channels[channel] = {
            "classification": classification,
            "target_policy": {
                "reliable": "hard",
                "weak": "soft",
                "unmeasurable": "excluded",
                "pending": "excluded",
            }[classification],
            "included": len(samples),
            "reviewed": len(reviewed),
            "known_owner": len(known),
            "unknown_owner": len(reviewed) - len(known),
            "defensible": defensible,
            "owner_rate": owner_rate,
            "agreement_rate": agreement_rate,
            "agreement_numerator": agreed,
            "agreement_denominator": total,
            "gold_works": len(gold),
            "gold_matches": gold_matches,
            "gold_denominator": gold_total,
            "gold_rate": gold_rate,
            "missing_gold": [s["work_id"] for s in samples if s["gold_union"] is None],
            "unmatched": sum(s["agreement"]["unmatched"] for s in samples),
        }
    result: dict[str, Any] = {
        "version": "1",
        "packet_id": packet["id"],
        "test_only": packet["test_only"],
        "channels": channels,
        "problems": problems,
        "inclusion": packet["inclusion"],
        "judgements": snapshot(list((directory / "judgements").glob("*.json"))),
        "status": "pending"
        if problems or any(c["classification"] == "pending" for c in channels.values())
        else "classified",
        "scientific_status": "pending"
        if packet["test_only"]
        else "owner evidence required/current as above",
        "core_failures": [
            c for c in CORE if channels.get(c, {}).get("classification") in {"weak", "unmeasurable"}
        ],
    }
    result["id"] = identity(result)
    report_files(directory, "Corpus channel reliability", result)
    return result


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="review_command", required=True)
    create = commands.add_parser("freeze")
    create.add_argument("config", type=Path)
    create.add_argument("--out", type=Path, required=True)
    for name in ("report", "label"):
        child = commands.add_parser(name)
        child.add_argument("directory", type=Path)
        if name == "label":
            child.add_argument("judgement", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        if args.review_command == "freeze":
            freeze(args.config, args.out)
        elif args.review_command == "label":
            record_judgement(args.directory, args.judgement)
        else:
            report(args.directory)
        return 0
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"nde corpus-review: {error}", file=sys.stderr)
        return 2
