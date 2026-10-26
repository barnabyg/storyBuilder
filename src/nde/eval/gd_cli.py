"""Public no-spend check, registered generation and sealed-batch G-D commands."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from nde.eval import gd, gd_artifacts
from nde.eval.batch import SIDES, BatchError, Source
from nde.eval.rubric import DEFAULT_RUBRIC_DIR, RubricError, load_rubric
from nde.llm.cache import DEFAULT_CACHE_PATH
from nde.llm.ledger import DEFAULT_LEDGER_PATH
from nde.runs.config import ConfigError
from nde.runs.records import read_manifest
from nde.runs.runner import RunOptions


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="gd_command", required=True)
    for command in ("check", "run", "batch"):
        sub = commands.add_parser(command)
        sub.add_argument("protocol", type=Path)
        sub.add_argument("config", type=Path)
        sub.add_argument("--selection", type=Path, required=True)
        sub.add_argument("--g1-batch", type=Path, required=True)
        sub.add_argument("--g1-decision", type=Path, required=True)
        if command == "check":
            sub.add_argument("--out", type=Path, required=True)
        elif command == "run":
            sub.add_argument("--runs-dir", type=Path, default=Path("runs"))
            sub.add_argument("--cache", type=Path, default=Path(DEFAULT_CACHE_PATH))
            sub.add_argument("--ledger", type=Path, default=Path(DEFAULT_LEDGER_PATH))
        else:
            sub.add_argument("--name", required=True)
            sub.add_argument("--from", dest="sources", nargs=2, action="append", required=True)
            sub.add_argument(
                "--control", nargs=2, action="append", required=True, metavar=("BRIEF", "REPORT")
            )
            sub.add_argument("--batches-dir", type=Path, default=Path("batches"))
            sub.add_argument("--rubric-dir", type=Path, default=DEFAULT_RUBRIC_DIR)
            sub.add_argument("--seed", type=int)
            sub.add_argument("--owner-code-work-at", type=datetime.fromisoformat)


def _batch(args: argparse.Namespace, inputs: gd.GDInputs) -> Path:
    sources: list[list[str]] = args.sources
    controls: list[list[str]] = args.control
    if len(sources) != len(SIDES):
        raise BatchError("give --from exactly twice, D then C")
    control_paths = {brief: Path(path) for brief, path in controls}
    if len(control_paths) != len(controls):
        raise BatchError("give each brief's control report exactly once")
    first, second = (Source(Path(path), condition) for path, condition in sources)
    checked = gd.require_gd(inputs)
    if checked.protocol is None:
        raise BatchError("registered G-D protocol is missing")
    rubric = load_rubric(args.rubric_dir, checked.protocol.rubric_version)
    return gd_artifacts.build_gd_batch(
        inputs,
        (first, second),
        gd_artifacts.GDBatchSettings(
            args.batches_dir, args.name, rubric, control_paths, args.seed, args.owner_code_work_at
        ),
    )


def run(args: argparse.Namespace) -> int:
    inputs = gd.GDInputs(
        args.protocol, args.config, args.selection, args.g1_batch, args.g1_decision
    )
    try:
        if args.gd_command == "check":
            checked = gd.check_gd(inputs)
            gd.write_check(checked, args.out)
            print(f"G-D prerequisites {checked.status}: {args.out / 'gd-check.md'}")
            return 0 if checked.status == "ready" else 2
        if args.gd_command == "run":
            path = gd_artifacts.generate_gd(
                inputs,
                RunOptions(args.runs_dir, dict(os.environ), args.cache, ledger_path=args.ledger),
            )
            status = read_manifest(path).status
            print(f"registered G-D run {status}: {path}")
            return 0 if status == "succeeded" else 1
        print(f"sealed G-D batch: {_batch(args, inputs)}")
        return 0
    except (OSError, ValueError, ConfigError, BatchError, RubricError) as error:
        print(f"nde gd {args.gd_command}: {error}", file=sys.stderr)
        return 2
