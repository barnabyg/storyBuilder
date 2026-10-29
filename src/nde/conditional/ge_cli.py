"""`nde ge`: no-spend readiness, registration, registered dev run and owner review packet.

Exit codes: 0 ready, current or complete; 1 blocked, stale, pending or a run that
did not succeed; 2 malformed or unsupported input. No command decides G-E.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from nde.conditional import ge, ge_review
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


def _dispatch(args: argparse.Namespace) -> int:  # noqa: PLR0911 - one exit per command
    command = args.ge_command
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
        print(f"registered G-E run {status}: {folder}")
        return 0 if status == "succeeded" else 1
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
