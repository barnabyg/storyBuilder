"""Generate or replay a saved G-schema review packet."""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from nde.corpus import provenance

from .collect import collect
from .mutations import DEFAULT_SUITE
from .records import Snapshot
from .report import write_packet


def configure(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path)
    source.add_argument("--replay", type=Path)
    parser.add_argument("--annotations", type=Path, default=Path(".nde/annotations"))
    parser.add_argument("--mutations", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--out", type=Path, required=True)


def run(args: argparse.Namespace) -> int:
    try:
        snapshot = (
            Snapshot.model_validate_json(args.replay.read_text(encoding="utf-8"))
            if args.replay
            else collect(
                args.manifest or provenance.DEFAULT_MANIFEST, args.annotations, args.mutations
            )
        )
        report = write_packet(snapshot, args.out)
    except (OSError, ValueError, KeyError, ValidationError, provenance.ProvenanceError) as error:
        print(f"nde schema-evidence: {error}", file=sys.stderr)
        return 2
    print(f"G-schema {report.gate_status}; mechanical {report.mechanical_status}: {args.out}")
    return {"passed": 0, "failed": 1, "pending": 3}[report.gate_status]
