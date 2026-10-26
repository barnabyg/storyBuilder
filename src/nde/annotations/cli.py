"""Annotation commands."""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from nde.corpus.provenance import ProvenanceError
from nde.runs.config import ConfigError

from .corrections import correction_seconds, review, revise, timer
from .drafting import draft
from .records import DraftOptions, ReviewDecision, read_session


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="annotation_command", required=True)
    command = commands.add_parser("draft")
    command.add_argument("story_id")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--cache", type=Path, default=Path(".nde/cache.sqlite"))
    command.add_argument("--ledger", type=Path, default=Path(".nde/ledger.sqlite"))
    for name in ("revise", "review", "timer", "status"):
        sub = commands.add_parser(name)
        sub.add_argument("directory", type=Path)
        if name == "revise":
            sub.add_argument("file", type=Path)
        elif name == "review":
            sub.add_argument("revision", type=int)
            sub.add_argument("--owner", required=True)
            sub.add_argument("--notes", required=True)
            sub.add_argument("--accept", action="store_true")
            sub.add_argument("--meaningful-trajectories", action="store_true")
        elif name == "timer":
            sub.add_argument("action", choices=("start", "resume", "pause", "recover"))


def _correction(args: argparse.Namespace) -> int:
    if args.annotation_command == "timer":
        timer(args.directory, "start" if args.action == "resume" else args.action)
    elif args.annotation_command == "revise":
        revision = revise(args.directory, args.file)
        print(f"revision {revision.number}: valid={revision.valid}; unreviewed")
        return 0 if revision.valid else 1
    elif args.annotation_command == "review":
        review(
            args.directory,
            args.revision,
            decision=ReviewDecision(
                owner=args.owner,
                notes=args.notes,
                accepted=args.accept,
                meaningful=args.meaningful_trajectories,
            ),
        )
    else:
        session = read_session(args.directory)
        print(
            f"{session.id}: {session.status}; revisions={len(session.revisions)}; "
            f"explicit owner reviews={len(session.owner_reviews)}; "
            f"correction seconds={correction_seconds(args.directory)}"
        )
    return 0


def run(args: argparse.Namespace) -> int:
    try:
        if args.annotation_command != "draft":
            return _correction(args)
        session = draft(
            args.config,
            args.manifest,
            args.story_id,
            DraftOptions(args.out, args.cache, args.ledger),
        )
        print(f"{session.id}: {session.status}; unreviewed LLM draft; ${session.spent_usd:.6f}")
        return 0 if session.status == "drafted" else 1
    except (OSError, ValueError, ValidationError, ConfigError, ProvenanceError) as error:
        print(f"nde annotate: {error}", file=sys.stderr)
        return 2
