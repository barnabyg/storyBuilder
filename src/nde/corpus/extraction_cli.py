"""Public pass extraction and independent-family comparison commands."""

import argparse
import sys
from pathlib import Path

from nde.corpus.extraction import extract, verify_extraction
from nde.corpus.extraction_records import ExtractionOptions


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="extraction_command", required=True)
    for name in ("run",):
        command = commands.add_parser(name, help="extract source-linked channel passes")
        command.add_argument("work", type=Path)
        command.add_argument("--segments", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--cache", type=Path, default=Path(".nde/cache.sqlite"))
        command.add_argument("--ledger", type=Path, default=Path(".nde/ledger.sqlite"))
    report = commands.add_parser("report")
    report.add_argument("directory", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
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
