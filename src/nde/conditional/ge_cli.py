"""`nde ge`: readiness, registration, the registered dev run, owner review and the decision.

Exit codes: 0 ready, current or complete; 1 blocked, stale, pending, inconsistent or
a run that did not succeed; 2 malformed or unsupported input. Only `decide` applies
the registered criteria, and it launches nothing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from nde.conditional import ge, ge_decision, ge_review
from nde.llm.cache import DEFAULT_CACHE_PATH
from nde.llm.ledger import DEFAULT_LEDGER_PATH
from nde.runs.config import ConfigError
from nde.runs.records import read_manifest
from nde.runs.runner import RunOptions


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="ge_command", required=True)
    readiness = commands.add_parser("readiness", help="no-spend G3 authority and lineage check")
    readiness.add_argument("registration", type=Path)
    readiness.add_argument("--out", type=Path, required=True)
    register = commands.add_parser("register", help="freeze a ready registration as a protocol")
    register.add_argument("registration", type=Path)
    register.add_argument("--out", type=Path, required=True, help="new protocol JSON file")
    check = commands.add_parser("check", help="revalidate a committed protocol, no spend")
    check.add_argument("protocol", type=Path)
    check.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run", help="execute the registered dev comparison")
    run.add_argument("protocol", type=Path)
    run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    run.add_argument("--cache", type=Path, default=Path(DEFAULT_CACHE_PATH))
    run.add_argument("--ledger", type=Path, default=Path(DEFAULT_LEDGER_PATH))
    verify = commands.add_parser("verify", help="replay a recorded run against its protocol")
    verify.add_argument("protocol", type=Path)
    verify.add_argument("run", type=Path)
    packet = commands.add_parser("packet", help="assemble the owner's structural review packet")
    packet.add_argument("protocol", type=Path)
    packet.add_argument("run", type=Path)
    packet.add_argument("--out", type=Path, required=True)
    review = commands.add_parser("review", help="record an owner review; report completeness")
    review.add_argument("packet", type=Path)
    review.add_argument("review", type=Path, nargs="?")
    decide = commands.add_parser("decide", help="seal the G-E decision; starts nothing")
    decide.add_argument("protocol", type=Path)
    decide.add_argument("run", type=Path)
    decide.add_argument("--packet", type=Path, help="the owner's review packet folder")
    decide.add_argument("--out", type=Path, required=True, help="new decision folder")
    replay = commands.add_parser("replay", help="replay a sealed G-E decision, no spend")
    replay.add_argument("decision", type=Path, help="decision folder")


def _stale(reasons: list[str], label: str) -> int:
    for reason in reasons:
        print(reason, file=sys.stderr)
    print(f"{label} STALE")
    return 1


def _review(args: argparse.Namespace) -> int:
    if args.review is not None:
        saved = ge_review.record_review(args.packet, args.review)
        print(f"recorded {saved['classification']} review {saved['id']}")
    result = ge_review.status(args.packet)
    if result["stale"]:
        return _stale(result["stale"], "G-E review packet")
    rated = ", ".join(f"{name} {count}" for name, count in result["rated"].items())
    print(
        f"G-E owner review {result['status']} (evidence {result['evidence']}); "
        f"{result['items']['owner']} owner items; rated: {rated}"
    )
    return 0 if result["status"] == "complete" else 1


def _decide(args: argparse.Namespace) -> int:
    record = ge_decision.decide(args.protocol, args.run, args.packet, args.out)
    for problem in record["problems"]:
        print(f"inconsistent: {problem}", file=sys.stderr)
    for reason in record["pending"]:
        print(f"pending: {reason}", file=sys.stderr)
    print(
        f"G-E decision {record['outcome']} (criteria {record['criteria_outcome']}), action "
        f"{record['action']['result']}: {args.out / 'decision.md'}"
    )
    ratings = ge_decision.ratings_path(args.protocol)
    if ratings.is_file():
        print(f"owner ratings frozen: commit {ratings} with the protocol")
    if ge_decision.settles(record):
        claim = ge_decision.settlement_path(args.protocol)
        print(f"settled: commit {claim} and {args.out} with the protocol")

    return 0 if record["outcome"] in ge_decision.COMPLETE else 1


def _replay(args: argparse.Namespace) -> int:
    saved, reasons = ge_decision.replay(args.decision)
    if reasons:
        return _stale(reasons, "G-E decision")
    for name in ge_decision.unapplied_reviews(args.decision):
        print(f"later review not applied: {name} (the owner ratings are frozen)")
    print(f"G-E decision {saved['outcome']} current: {args.decision}")
    return 0


def _dispatch(args: argparse.Namespace) -> int:  # noqa: PLR0911 - one exit per command
    command = args.ge_command
    if command == "decide":
        return _decide(args)
    if command == "replay":
        return _replay(args)
    if command == "readiness":
        result = ge.readiness(args.registration)
        ge.write_status(result, args.out, "ge-readiness", "G-E readiness")
        print(f"G-E readiness {result['status']}: {args.out / 'ge-readiness.md'}")
        return 0 if result["status"] == "ready" else 1
    if command == "register":
        protocol = ge.register(args.registration, args.out)
        print(f"registered G-E protocol {protocol['id']}: commit {args.out} before running")
        return 0
    if command == "check":
        result = ge.check(args.protocol)
        ge.write_status(result, args.out, "ge-check", "G-E prerequisites")
        print(f"G-E prerequisites {result['status']}: {args.out / 'ge-check.md'}")
        return 0 if result["status"] == "ready" else 1
    if command == "run":
        options = RunOptions(args.runs_dir, dict(os.environ), args.cache, ledger_path=args.ledger)
        folder = ge.run(args.protocol, options)
        status = read_manifest(folder).status
        evidence = ge.read_run(folder)["evidence"]
        print(f"registered G-E run {status}, evidence {evidence['status']}: {folder}")
        if evidence["status"] != "available":
            print(f"evidence report unavailable: {evidence['error']}", file=sys.stderr)
        return 0 if status == "succeeded" and evidence["status"] == "available" else 1
    if command == "verify":
        reasons = ge.verify(args.protocol, args.run)
        if reasons:
            return _stale(reasons, "G-E run")
        print(f"G-E run current: {args.run}")
        return 0
    if command == "packet":
        packet = ge_review.create(args.protocol, args.run, args.out)
        print(f"G-E review packet {packet['id']}: {args.out / 'packet.md'}")
        return 0
    return _review(args)


def run(args: argparse.Namespace) -> int:
    try:
        return _dispatch(args)
    except ge.StaleEvidenceError as error:
        return _stale(error.reasons, f"nde ge {args.ge_command}:")
    except ValidationError as error:
        details = json.dumps(error.errors(include_url=False), default=str)
        print(f"nde ge {args.ge_command}: invalid input: {details}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, ConfigError) as error:
        print(f"nde ge {args.ge_command}: {error}", file=sys.stderr)
        return 2
