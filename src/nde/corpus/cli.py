"""Public corpus acquisition, split and measurement commands."""

import argparse
import sys
from pathlib import Path

import httpx
import yaml
from pydantic import ValidationError

from nde.corpus.acquisition import CorpusError, acquire, load_work


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="corpus_command", required=True)
    for name in ("import", "fetch"):
        command = commands.add_parser(name)
        command.add_argument("spec", type=Path)
        if name == "import":
            command.add_argument("file", type=Path)
        command.add_argument("--out", type=Path, required=True)
    commands.add_parser("inspect").add_argument("work", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        if args.corpus_command == "inspect":
            record, _ = load_work(args.work)
        else:
            record = acquire(
                args.spec, args.out, args.file if args.corpus_command == "import" else None
            )
        print(record.model_dump_json(indent=2))
    except (
        CorpusError,
        OSError,
        ValueError,
        ValidationError,
        yaml.YAMLError,
        httpx.HTTPError,
    ) as error:
        print(f"nde corpus: {error}", file=sys.stderr)
        return 1
    return 0
