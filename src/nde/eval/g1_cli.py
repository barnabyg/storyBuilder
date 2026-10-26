"""Public commands for the registered G1 workflow."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from nde.eval import g1
from nde.eval.batch import BatchError, Source
from nde.eval.rubric import DEFAULT_RUBRIC_DIR, RubricError, load_rubric
from nde.llm.cache import DEFAULT_CACHE_PATH
from nde.llm.ledger import DEFAULT_LEDGER_PATH
from nde.runs.config import ConfigError
from nde.runs.records import read_manifest
from nde.runs.runner import RunOptions


def configure(parser: argparse.ArgumentParser, command: str) -> None:
    parser.add_argument("protocol", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--schema-evidence", type=Path, required=True)
    if command == "g1-check":
        parser.add_argument("--out", type=Path, required=True)
    elif command == "g1-run":
        parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
        parser.add_argument("--cache", type=Path, default=Path(DEFAULT_CACHE_PATH))
        parser.add_argument("--ledger", type=Path, default=Path(DEFAULT_LEDGER_PATH))
    else:
        parser.add_argument("--name", required=True)
        parser.add_argument("--from", dest="sources", nargs=2, action="append", required=True)
        parser.add_argument("--batches-dir", type=Path, default=Path("batches"))
        parser.add_argument("--rubric-dir", type=Path, default=DEFAULT_RUBRIC_DIR)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--owner-code-work-at", type=datetime.fromisoformat)


def run(args: argparse.Namespace) -> int:
    inputs = g1.G1Inputs(args.protocol, args.config, args.selection, args.schema_evidence)
    try:
        if args.eval_command == "g1-check":
            result = g1.check_g1(inputs)
            g1.write_check(result, args.out)
            print(f"G1 prerequisites {result.status}: {args.out / 'g1-check.md'}")
            return 0 if result.status == "ready" else 2
        if args.eval_command == "g1-run":
            path = g1.generate_g1(
                inputs,
                RunOptions(args.runs_dir, dict(os.environ), args.cache, ledger_path=args.ledger),
            )
            status = read_manifest(path).status
            print(f"registered G1 run {status}: {path}")
            return 0 if status == "succeeded" else 1
        sources: list[list[str]] = args.sources
        if len(sources) != 2:  # noqa: PLR2004 - a G1 batch compares C and B
            raise BatchError("give --from exactly twice, C then B")
        first, second = (Source(Path(path), condition) for path, condition in sources)
        checked = g1.require_g1(inputs)
        if checked.protocol is None:
            raise BatchError("registered G1 protocol is missing")
        rubric = load_rubric(args.rubric_dir, checked.protocol.rubric_version)
        path = g1.build_g1_batch(
            inputs,
            (first, second),
            g1.G1BatchSettings(
                args.batches_dir, args.name, rubric, args.seed, args.owner_code_work_at
            ),
        )
        print(f"sealed G1 batch: {path}")
        return 0
    except (OSError, ValueError, ConfigError, BatchError, RubricError) as error:
        print(f"nde eval {args.eval_command}: {error}", file=sys.stderr)
        return 2
