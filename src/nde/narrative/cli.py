"""Narrative plan commands."""

import argparse
import sys
from pathlib import Path

from nde.files import write_text_lf
from nde.narrative.projection import project
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import PlanError, load_plan, render, schema_json


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="plan_command", required=True)
    schema = commands.add_parser("schema", help="export the deterministic JSON Schema")
    schema.add_argument("--out", type=Path, required=True)
    for name in ("inspect", "validate", "project"):
        command = commands.add_parser(name)
        command.add_argument("file", type=Path)
        if name in {"inspect", "project"}:
            command.add_argument("--out", type=Path, required=True)
        if name == "project":
            command.add_argument("--tier", choices=("S", "M", "L"), required=True)


def run(args: argparse.Namespace) -> int:
    try:
        if args.plan_command == "schema":
            write_text_lf(args.out, schema_json())
            return 0
        plan = load_plan(args.file)
        result = validate_plan(plan)
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
    except (PlanError, OSError) as error:
        print(f"nde plan: {error}", file=sys.stderr)
        return 2
