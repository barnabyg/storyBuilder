"""The `nde` command-line interface."""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from nde import __version__
from nde.env import load_dotenv
from nde.runs import prompts
from nde.runs.config import ConfigError
from nde.runs.records import read_manifest
from nde.runs.report import write_report
from nde.runs.runner import RunOptions, run_experiment

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nde", description="Narrative Diffusion Engine")
    parser.add_argument("--version", action="version", version=f"nde {__version__}")
    commands = parser.add_subparsers(dest="command")

    run = commands.add_parser("run", help="run an experiment config and record a run directory")
    run.add_argument("config", type=Path, help="experiment config (YAML)")
    run.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="where run directories go (./runs)"
    )

    report = commands.add_parser("report", help="regenerate a run directory's report")
    report.add_argument("run_dir", type=Path)

    prompt = commands.add_parser("prompts", help="manage versioned prompt templates")
    prompt_commands = prompt.add_subparsers(dest="prompts_command", required=True)
    lock = prompt_commands.add_parser("lock", help="record checksums of new template versions")
    lock.add_argument("--dir", type=Path, default=Path("prompts"), help="templates (./prompts)")
    return parser


def _run(args: argparse.Namespace) -> int:
    load_dotenv(Path(".env"))
    options = RunOptions(runs_dir=args.runs_dir, environ=dict(os.environ))
    try:
        run_dir = run_experiment(args.config, options)
    except ConfigError as error:
        print(f"nde run: {error}", file=sys.stderr)
        return EXIT_USAGE
    manifest = read_manifest(run_dir)
    print(f"run {manifest.status}: {run_dir}")
    return EXIT_OK if manifest.status == "succeeded" else EXIT_RUN_FAILED


def _report(args: argparse.Namespace) -> int:
    print(write_report(args.run_dir))
    return EXIT_OK


def _prompts_lock(args: argparse.Namespace) -> int:
    added, changed = prompts.lock(args.dir)
    for label in added:
        print(f"recorded {label}")
    for label in changed:
        print(f"CHANGED {label}: restore its text or save it as a new version", file=sys.stderr)
    return EXIT_USAGE if changed else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "report":
        return _report(args)
    if args.command == "prompts":
        return _prompts_lock(args)
    parser.print_help()
    return EXIT_OK
