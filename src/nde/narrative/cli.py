"""Narrative plan commands."""

import argparse
import json
import os
import sys
from pathlib import Path

from nde.files import write_text_lf
from nde.narrative.ledger import render_ledger
from nde.narrative.projection import project
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import PlanError, load_plan, render, schema_json
from nde.plan.alternatives import check_report
from nde.runs.config import ConfigError, load_config
from nde.runs.records import read_manifest
from nde.runs.runner import RunOptions, run_experiment


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="plan_command", required=True)
    for name in ("generate", "revise"):
        generate = commands.add_parser(name, help=f"{name} dev plans in a recorded experiment")
        generate.add_argument("config", type=Path)
        generate.add_argument("--runs-dir", type=Path, default=Path("runs"))
        generate.add_argument("--cache", type=Path, default=Path(".nde/cache.sqlite"))
        generate.add_argument("--ledger", type=Path, default=Path(".nde/ledger.sqlite"))
    alternatives = commands.add_parser(
        "alternatives", help="recheck a revision run's independent alternatives report"
    )
    alternatives.add_argument("run_dir", type=Path)
    schema = commands.add_parser("schema", help="export the deterministic JSON Schema")
    schema.add_argument("--out", type=Path, required=True)
    for name in ("inspect", "validate", "project", "replay"):
        command = commands.add_parser(name)
        command.add_argument("file", type=Path)
        if name in {"inspect", "project", "replay"}:
            command.add_argument("--out", type=Path, required=True)
        if name == "project":
            command.add_argument("--tier", choices=("S", "M", "L"), required=True)


def run(args: argparse.Namespace) -> int:
    try:
        if args.plan_command in {"generate", "revise"}:
            plans = load_config(args.config).config.plans
            if plans is None:
                raise ConfigError("plan generate requires a plans configuration")
            if args.plan_command == "revise" and "revision" not in plans.params:
                raise ConfigError("plan revise requires plans.params.revision and saved_plan")
            run_dir = run_experiment(
                args.config,
                RunOptions(
                    runs_dir=args.runs_dir,
                    environ=dict(os.environ),
                    cache_path=args.cache,
                    ledger_path=args.ledger,
                ),
            )
            manifest = read_manifest(run_dir)
            print(f"run {manifest.status}: {run_dir}")
            return 0 if manifest.status == "succeeded" else 1
        if args.plan_command == "alternatives":
            report = check_report(args.run_dir)
            print(json.dumps(report, indent=2))
            return 0 if report["consistent"] else 1
        if args.plan_command == "schema":
            write_text_lf(args.out, schema_json())
            return 0
        plan = load_plan(args.file)
        result = validate_plan(plan)
        if args.plan_command == "replay":
            args.out.mkdir(parents=True, exist_ok=True)
            write_text_lf(args.out / "validation.json", result.model_dump_json(indent=2) + "\n")
            if result.valid:
                render_ledger(plan, args.out)
        if args.plan_command == "project" and result.valid:
            plan = project(plan, args.tier)
            result = render(plan, args.out)
        elif args.plan_command == "inspect":
            result = render(plan, args.out)
        for diagnostic in result.diagnostics:
            print(
                f"{diagnostic.severity} {diagnostic.code} [{diagnostic.location}]: "
                f"{diagnostic.message}",
                file=sys.stderr,
            )
        return 0 if result.valid else 1
    except (ConfigError, PlanError, OSError, KeyError, ValueError) as error:
        print(f"nde plan: {error}", file=sys.stderr)
        return 2
