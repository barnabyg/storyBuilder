"""The `nde eval` commands: build a blind batch, read it, and report it."""

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from nde.corpus.provenance import DEFAULT_MANIFEST, ProvenanceError
from nde.eval import batch as batches
from nde.eval import g1_cli, readiness, report
from nde.eval.calibration import CalibrationSpec, build_calibration_batch
from nde.eval.leakage import generating_code_warning
from nde.eval.reader import DEFAULT_PORT, HOST
from nde.eval.rubric import DEFAULT_RUBRIC_DIR, RubricError, load_rubric

EXIT_OK = 0
EXIT_USAGE = 2


def _batches_dir_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batches-dir",
        type=Path,
        default=Path(os.environ.get("NDE_BATCHES", batches.DEFAULT_BATCHES_DIR)),
        help="where batches live (env NDE_BATCHES, default ./batches)",
    )


def configure(evaluation: argparse.ArgumentParser) -> None:
    """Add the `nde eval` subcommands to its parser."""
    sub = evaluation.add_subparsers(dest="eval_command", required=True)
    for command in ("g1-check", "g1-run", "g1-batch"):
        g1_cli.configure(sub.add_parser(command, help="protocol-bound G1 workflow"), command)

    ready = sub.add_parser("readiness", help="check exact registered gate cohort before reading")
    ready.add_argument("cohort", type=Path)
    ready.add_argument("--benchmark", type=Path, required=True)
    ready.add_argument("--from", dest="sources", nargs=2, action="append", required=True)
    ready.add_argument("--out", type=Path, required=True)

    build = sub.add_parser("batch", help="build a blind batch of story pairs from two runs")
    build.add_argument("name", help="batch name, e.g. p1-instrument")
    build.add_argument(
        "--from",
        dest="sources",
        nargs=2,
        action="append",
        metavar=("RUN_DIR", "CONDITION"),
        required=True,
        help="a run directory and one of its conditions; give exactly two",
    )
    build.add_argument("--seed", type=int, help="shuffle seed (default: random, kept sealed)")
    build.add_argument(
        "--rubric-dir", type=Path, default=DEFAULT_RUBRIC_DIR, help="(./benchmark/rubric)"
    )
    build.add_argument("--rubric-version", type=int, default=1, help="rubric version (1)")
    build.add_argument(
        "--optional-scores",
        action="store_true",
        help="require only the preference and strength; rubric scores, tags and the guess "
        "may be partial (for skim reading)",
    )
    build.add_argument(
        "--primary-score",
        metavar="DIMENSION",
        help="a rubric dimension that is always required and leads the report, e.g. coherence",
    )
    _batches_dir_option(build)

    serve = sub.add_parser("serve", help="open the blind reader on localhost")
    serve.add_argument("--batch", required=True, help="batch name")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"(default {DEFAULT_PORT})")
    _batches_dir_option(serve)

    calibration = sub.add_parser(
        "calibration", help="build a single-story calibration batch of human and generated stories"
    )
    calibration.add_argument("name", help="batch name, e.g. p1-calibration")
    calibration.add_argument(
        "--humans", type=Path, default=DEFAULT_MANIFEST, help="(./data/manifest.calibration.yaml)"
    )
    calibration.add_argument(
        "--from",
        dest="sources",
        nargs=2,
        action="append",
        metavar=("RUN_DIR", "CONDITION"),
        default=[],
        help="a run directory and one of its conditions; repeat for each condition",
    )
    calibration.add_argument(
        "--per-source", type=int, default=2, help="stories from each --from (default 2)"
    )
    calibration.add_argument("--seed", type=int, help="shuffle seed (default: random)")
    calibration.add_argument(
        "--rubric-dir", type=Path, default=DEFAULT_RUBRIC_DIR, help="(./benchmark/rubric)"
    )
    calibration.add_argument("--rubric-version", type=int, default=1, help="rubric version (1)")
    _batches_dir_option(calibration)

    reveal = sub.add_parser("report", help="reveal the key of a closed batch and report it")
    reveal.add_argument("name", help="batch name")
    _batches_dir_option(reveal)


def _build(args: argparse.Namespace) -> int:
    sources: list[list[str]] = args.sources
    if len(sources) != 2:  # noqa: PLR2004 - a batch compares exactly two sources
        print("nde eval batch: give --from exactly twice", file=sys.stderr)
        return EXIT_USAGE
    first, second = (batches.Source(Path(run), condition) for run, condition in sources)
    rubric = load_rubric(args.rubric_dir, args.rubric_version)
    path = batches.build_batch(
        args.batches_dir,
        args.name,
        (first, second),
        scoring=batches.Scoring(rubric, optional=args.optional_scores, primary=args.primary_score),
        seed=args.seed,
    )
    pairs = len(batches.read_view(path).pairs)
    print(f"built batch {args.name!r}: {pairs} pairs in {path}")
    print(f"read it with: uv run nde eval serve --batch {args.name}")
    return EXIT_OK


def _readiness(args: argparse.Namespace) -> int:
    sources: list[list[str]] = args.sources
    try:
        if len(sources) != 2:  # noqa: PLR2004 - gate admission compares exactly two conditions
            raise ValueError("give --from exactly twice")
        cohort = readiness.GateCohort.model_validate_json(args.cohort.read_text(encoding="utf-8"))
        first, second = (batches.Source(Path(run), condition) for run, condition in sources)
        result = readiness.check_readiness(cohort, args.benchmark, (first, second))
    except (OSError, ValueError) as error:
        result = readiness.ReadinessReport(
            status="malformed", benchmark=str(args.benchmark), sources=[], problems=[str(error)]
        )
    readiness.write_readiness(result, args.out)
    print(f"gate {result.status}: {args.out / 'readiness.md'}")
    return {"ready": 0, "incomplete": 1, "malformed": 2}[result.status]


def _calibration(args: argparse.Namespace) -> int:
    sources = [batches.Source(Path(run), condition) for run, condition in args.sources]
    spec = CalibrationSpec(
        name=args.name,
        humans=args.humans,
        sources=sources,
        per_source=args.per_source,
        rubric=load_rubric(args.rubric_dir, args.rubric_version),
        seed=args.seed,
    )
    path = build_calibration_batch(args.batches_dir, spec)
    stories = len(batches.read_view(path).stories)
    print(f"built calibration batch {args.name!r}: {stories} stories in {path}")
    print(f"read it with: uv run nde eval serve --batch {args.name}")
    return EXIT_OK


def _serve(args: argparse.Namespace) -> int:
    # Imported here so other commands do not pay for loading the web stack.
    import uvicorn  # noqa: PLC0415

    from nde.eval.reader import create_app  # noqa: PLC0415

    path = batches.batch_dir(args.batches_dir, args.batch)
    warning = generating_code_warning(Path.cwd(), datetime.now(UTC))
    if warning is not None:
        batches.record_warning(path, warning)
        print(f"WARNING: {warning}; the batch report will record this", file=sys.stderr)
    print(f"blind reader for {args.batch!r}: http://{HOST}:{args.port}/  (Ctrl+C to stop)")
    uvicorn.run(create_app(path), host=HOST, port=args.port, log_level="warning")
    return EXIT_OK


def _report(args: argparse.Namespace) -> int:
    path = batches.batch_dir(args.batches_dir, args.name)
    print(report.write_report(path), end="")
    return EXIT_OK


def run(args: argparse.Namespace) -> int:
    handlers = {
        "g1-check": g1_cli.run,
        "g1-run": g1_cli.run,
        "g1-batch": g1_cli.run,
        "readiness": _readiness,
        "batch": _build,
        "calibration": _calibration,
        "serve": _serve,
        "report": _report,
    }
    try:
        return handlers[args.eval_command](args)
    except (batches.BatchError, RubricError, ProvenanceError) as error:
        print(f"nde eval {args.eval_command}: {error}", file=sys.stderr)
        return EXIT_USAGE
