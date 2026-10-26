"""The `nde` command-line interface."""

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from nde import (
    __version__,
    briefs,
    cd_comparison,
    controllability,
    diversity,
    g1_report,
    gd_report,
    realisation,
    tiers,
)
from nde.annotations import cli as annotation_cli
from nde.corpus import cli as corpus_cli
from nde.corpus import extraction_cli, provenance
from nde.corpus import reliability as corpus_review
from nde.env import load_dotenv
from nde.eval import cli as eval_cli
from nde.eval import gd_cli, rubric
from nde.evidence import cli as evidence_cli
from nde.llm.cache import DEFAULT_CACHE_PATH
from nde.llm.ledger import DEFAULT_LEDGER_PATH
from nde.narrative import cli as narrative_cli
from nde.runs import prompts
from nde.runs.config import ConfigError
from nde.runs.records import read_manifest
from nde.runs.report import write_report
from nde.runs.runner import RunOptions, run_experiment
from nde.schema.brief import SPLITS

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_USAGE = 2


def _check_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--check", action="store_true", help="report unrecorded or changed versions; write nothing"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nde", description="Narrative Diffusion Engine")
    parser.add_argument("--version", action="version", version=f"nde {__version__}")
    commands = parser.add_subparsers(dest="command")
    corpus_cli.configure(commands.add_parser("corpus", help="acquire and inspect corpus works"))
    for name, configure, description in (
        (
            "extract",
            extraction_cli.configure,
            "extract independent evidence-linked narrative channels",
        ),
        ("corpus-review", corpus_review.configure, "freeze and assess corpus channel reviews"),
        ("gd-report", gd_report.configure, "report a closed registered G-D batch"),
        ("gd", gd_cli.configure, "registered D-versus-C gate workflow"),
        ("control-report", controllability.configure, "report plan-only structural edit evidence"),
        ("diversity-report", diversity.configure, "compare structures across seeds"),
        ("compare-cd", cd_comparison.configure, "compare selected C and D on dev"),
        ("g1-report", g1_report.configure, "report a closed registered G1 batch"),
        ("tiers", tiers.configure, "compare dev tiers and freeze a selection"),
        ("realisation", realisation.configure, "review final prose intentions"),
        ("schema-evidence", evidence_cli.configure, "G-schema review packet"),
    ):
        configure(commands.add_parser(name, help=description))

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
    _check_option(lock)

    eval_cli.configure(
        commands.add_parser("eval", help="blind evaluation: batches, reader, reports")
    )

    corpus = commands.add_parser("calibration", help="fetch the human calibration stories")
    corpus_commands = corpus.add_subparsers(dest="calibration_command", required=True)
    corpus_fetch = corpus_commands.add_parser(
        "fetch", help="fetch, or reuse and verify, every story in the provenance manifest"
    )
    corpus_import = corpus_commands.add_parser(
        "import", help="use a local copy of a story's source instead of downloading it"
    )
    corpus_import.add_argument("id", help="the story's id in the manifest")
    corpus_import.add_argument("file", type=Path, help="the source file")
    for sub in (corpus_fetch, corpus_import):
        sub.add_argument(
            "--manifest",
            type=Path,
            default=provenance.DEFAULT_MANIFEST,
            help="(./data/manifest.calibration.yaml)",
        )

    rubric_parser = commands.add_parser("rubric", help="manage versions of the rubric")
    rubric_commands = rubric_parser.add_subparsers(dest="rubric_command", required=True)
    rubric_lock = rubric_commands.add_parser("lock", help="record checksums of new versions")
    rubric_lock.add_argument(
        "--dir", type=Path, default=rubric.DEFAULT_RUBRIC_DIR, help="(./benchmark/rubric)"
    )
    _check_option(rubric_lock)

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
    narrative_cli.configure(commands.add_parser("plan", help="inspect versioned narrative plans"))
    annotation_cli.configure(commands.add_parser("annotate", help="draft and correct annotations"))
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


type Versions = Callable[[Path], tuple[list[str], list[str]]]


def _lock(args: argparse.Namespace, command: str, check: Versions, lock: Versions) -> int:
    """`nde <command> lock`: record new versions, or with --check only report them.

    Exits 2 if a released version changed, or, with --check, if one is unrecorded.
    """
    if args.check:
        unrecorded, changed = check(args.dir)
        for label in unrecorded:
            print(f"UNRECORDED {label}: run `nde {command} lock`", file=sys.stderr)
    else:
        added, changed = lock(args.dir)
        unrecorded = []
        for label in added:
            print(f"recorded {label}")
    for label in changed:
        print(f"CHANGED {label}: restore its text or save it as a new version", file=sys.stderr)
    return EXIT_USAGE if changed or unrecorded else EXIT_OK


def _prompts_lock(args: argparse.Namespace) -> int:
    return _lock(args, "prompts", prompts.check, prompts.lock)


def _calibration(args: argparse.Namespace) -> int:
    try:
        if args.calibration_command == "import":
            stories = [provenance.import_story(args.manifest, args.id, args.file)]
        else:
            stories = provenance.fetch(args.manifest)
    except provenance.ProvenanceError as error:
        print(f"nde calibration {args.calibration_command}: {error}", file=sys.stderr)
        return EXIT_RUN_FAILED
    for story in stories:
        print(f"{story.id}: {story.title} ({story.author}), sha256 {story.sha256}")
    print(f"texts are in {provenance.texts_dir(args.manifest)}; commit {args.manifest}")
    return EXIT_OK


def _rubric_lock(args: argparse.Namespace) -> int:
    return _lock(args, "rubric", rubric.check, rubric.lock)


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
    "extract": extraction_cli.run,
    "corpus-review": corpus_review.run,
    "corpus": corpus_cli.run,
    "gd-report": gd_report.run,
    "gd": gd_cli.run,
    "control-report": controllability.run,
    "diversity-report": diversity.run,
    "compare-cd": cd_comparison.run,
    "g1-report": g1_report.run,
    "tiers": tiers.run,
    "realisation": realisation.run,
    "schema-evidence": evidence_cli.run,
    "annotate": annotation_cli.run,
    "plan": narrative_cli.run,
    "run": _run,
    "report": _report,
    "eval": eval_cli.run,
    "rubric": _rubric_lock,
    "calibration": _calibration,
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
