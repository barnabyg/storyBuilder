"""The `nde` command-line interface."""

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from nde import __version__, briefs
from nde.env import load_dotenv
from nde.eval import cli as eval_cli
from nde.eval import rubric
from nde.llm.cache import DEFAULT_CACHE_PATH
from nde.llm.ledger import DEFAULT_LEDGER_PATH
from nde.runs import prompts
from nde.runs.config import ConfigError
from nde.runs.records import read_manifest
from nde.runs.report import write_report
from nde.runs.runner import RunOptions, run_experiment
from nde.schema.brief import SPLITS

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

    run.add_argument(
        "--cache",
        type=Path,
        default=Path(os.environ.get("NDE_CACHE", DEFAULT_CACHE_PATH)),
        help="response cache file (env NDE_CACHE, default .nde/cache.sqlite)",
    )
    run.add_argument(
        "--ledger",
        type=Path,
        default=Path(os.environ.get("NDE_LEDGER", DEFAULT_LEDGER_PATH)),
        help="cost ledger with the daily totals (env NDE_LEDGER, default .nde/ledger.sqlite)",
    )
    cache_mode = run.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--no-cache",
        dest="cache_mode",
        action="store_const",
        const="bypass",
        default="use",
        help="neither read nor write the response cache",
    )
    cache_mode.add_argument(
        "--refresh-cache",
        dest="cache_mode",
        action="store_const",
        const="refresh",
        help="call the provider for every request and replace the stored responses",
    )

    report = commands.add_parser("report", help="regenerate a run directory's report")
    report.add_argument("run_dir", type=Path)

    prompt = commands.add_parser("prompts", help="manage versioned prompt templates")
    prompt_commands = prompt.add_subparsers(dest="prompts_command", required=True)
    lock = prompt_commands.add_parser("lock", help="record checksums of new template versions")
    lock.add_argument("--dir", type=Path, default=Path("prompts"), help="templates (./prompts)")

    eval_cli.configure(
        commands.add_parser("eval", help="blind evaluation: batches, reader, reports")
    )

    rubric_parser = commands.add_parser("rubric", help="manage versions of the rubric")
    rubric_commands = rubric_parser.add_subparsers(dest="rubric_command", required=True)
    rubric_lock = rubric_commands.add_parser("lock", help="record checksums of new versions")
    rubric_lock.add_argument(
        "--dir", type=Path, default=rubric.DEFAULT_RUBRIC_DIR, help="(./benchmark/rubric)"
    )
    rubric_lock.add_argument(
        "--check", action="store_true", help="report unrecorded or changed versions; write nothing"
    )

    brief = commands.add_parser("briefs", help="check, freeze and draft benchmark briefs")
    brief_commands = brief.add_subparsers(dest="briefs_command", required=True)
    brief_check = brief_commands.add_parser("check", help="validate every brief and frozen split")
    brief_freeze = brief_commands.add_parser("freeze", help="record the checksums of a split")
    brief_freeze.add_argument("split", choices=SPLITS)
    brief_drafts = brief_commands.add_parser("drafts", help="write a run's drafted briefs out")
    brief_drafts.add_argument("run_dir", type=Path, help="a run of configs/draft-briefs.yaml")
    brief_drafts.add_argument(
        "--out", type=Path, default=Path("benchmark/drafts"), help="(./benchmark/drafts)"
    )
    for sub in (brief_check, brief_freeze):
        sub.add_argument(
            "--dir",
            type=Path,
            default=briefs.DEFAULT_BENCHMARK_DIR,
            help="benchmark folder (./benchmark)",
        )
    return parser


def _run(args: argparse.Namespace) -> int:
    options = RunOptions(
        runs_dir=args.runs_dir,
        environ=dict(os.environ),
        cache_path=args.cache,
        cache_mode=args.cache_mode,
        ledger_path=args.ledger,
    )
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


def _rubric_lock(args: argparse.Namespace) -> int:
    if args.check:
        unrecorded, changed = rubric.check(args.dir)
        for label in unrecorded:
            print(f"UNRECORDED {label}: run `nde rubric lock`", file=sys.stderr)
    else:
        added, changed = rubric.lock(args.dir)
        unrecorded = []
        for label in added:
            print(f"recorded {label}")
    for label in changed:
        print(f"CHANGED {label}: restore its text or save it as a new version", file=sys.stderr)
    return EXIT_USAGE if changed or unrecorded else EXIT_OK


def _briefs(args: argparse.Namespace) -> int:
    if args.briefs_command == "drafts":
        try:
            written = briefs.write_drafts(args.run_dir, args.out)
        except briefs.BriefError as error:
            print(f"nde briefs drafts: {error}", file=sys.stderr)
            return EXIT_RUN_FAILED
        print(f"wrote {len(written)} draft briefs to {args.out} for review")
        return EXIT_OK
    if args.briefs_command == "freeze":
        try:
            frozen = briefs.freeze(args.dir, args.split)
        except briefs.BriefError as error:
            print(f"nde briefs freeze: {error}", file=sys.stderr)
            return EXIT_RUN_FAILED
        print(f"froze the {args.split} split: {len(frozen)} briefs")
        return EXIT_OK
    result = briefs.check(args.dir)
    for problem in result.problems:
        print(problem, file=sys.stderr)
    if not result.passed:
        print(f"brief check FAILED: {len(result.problems)} problems", file=sys.stderr)
        return EXIT_RUN_FAILED
    counts = ", ".join(f"{split}: {len(files)}" for split, files in result.briefs.items())
    total = sum(len(files) for files in result.briefs.values())
    print(f"{total} briefs valid ({counts})")
    return EXIT_OK


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "run": _run,
    "report": _report,
    "eval": eval_cli.run,
    "rubric": _rubric_lock,
    "briefs": _briefs,
    "prompts": _prompts_lock,
}


def main(argv: Sequence[str] | None = None) -> int:
    # Before parsing, so `.env` can also supply option defaults such as NDE_CACHE.
    load_dotenv(Path(".env"))
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return EXIT_OK
    return handler(args)
