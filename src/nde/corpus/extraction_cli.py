"""Public pass extraction and independent-family comparison commands."""

import argparse
import sys
from pathlib import Path

from nde.corpus.extraction import extract, verify_extraction
from nde.corpus.extraction_records import ExtractionOptions
from nde.corpus.families import compare_families, family_report


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="extraction_command", required=True)
    for name in ("run", "families"):
        command = commands.add_parser(name, help="extract source-linked channel passes")
        command.add_argument("work", type=Path)
        command.add_argument("--segments", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--cache", type=Path, default=Path(".nde/cache.sqlite"))
        command.add_argument("--ledger", type=Path, default=Path(".nde/ledger.sqlite"))
    report = commands.add_parser("report")
    report.add_argument("directory", type=Path)
    families_report = commands.add_parser("families-report")
    families_report.add_argument("directory", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        if args.extraction_command == "families-report":
            result = family_report(args.directory)
            print(f"{result['work_id']}: {result['status']}; independent unreviewed families")
            return 0 if result["status"] == "complete" else 1
        if args.extraction_command == "families":
            family_session = compare_families(
                args.config,
                args.work,
                args.segments,
                ExtractionOptions(args.out, args.cache, args.ledger),
            )
            print(f"{family_session.work_id}: {family_session.status}; unreviewed families")
            return 0 if family_session.status == "complete" else 1
        if args.extraction_command == "report":
            session = verify_extraction(args.directory)
        else:
            session = extract(
                args.config,
                args.work,
                args.segments,
                ExtractionOptions(args.out, args.cache, args.ledger),
            )
        print(f"{session.work_id}: {session.status}; unreviewed model annotation")
        return 0 if session.status == "complete" else 1
    except (OSError, ValueError) as error:
        print(f"nde extract: {error}", file=sys.stderr)
        return 2
