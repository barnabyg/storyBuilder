"""Public corpus acquisition, split and measurement commands."""

import argparse
import sys
from pathlib import Path

import httpx
import yaml
from pydantic import ValidationError

from nde.corpus.acquisition import CorpusError, acquire, load_work
from nde.corpus.features import extract_features, load_features
from nde.corpus.segmentation import compare_segmentation, load_segmentation, segment_work
from nde.corpus.splits import freeze_cohort, load_cohort


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="corpus_command", required=True)
    for name in ("import", "fetch"):
        command = commands.add_parser(name)
        command.add_argument("spec", type=Path)
        if name == "import":
            command.add_argument("file", type=Path)
        command.add_argument("--out", type=Path, required=True)
    commands.add_parser("inspect").add_argument("work", type=Path)
    split = commands.add_parser("split").add_subparsers(dest="split_command", required=True)
    freeze = split.add_parser("freeze")
    freeze.add_argument("config", type=Path)
    freeze.add_argument("--out", type=Path, required=True)
    split.add_parser("inspect").add_argument("cohort", type=Path)
    segment = commands.add_parser("segment")
    segment.add_argument("work", type=Path)
    segment.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("segment-check")
    compare = commands.add_parser("segment-compare")
    for command in (check, compare):
        command.add_argument("work", type=Path)
        command.add_argument("segmentation", type=Path)
    compare.add_argument("--gold", type=Path)
    compare.add_argument("--out", type=Path, required=True)
    features = commands.add_parser("features")
    feature_check = commands.add_parser("features-check")
    for command in (features, feature_check):
        command.add_argument("work", type=Path)
        command.add_argument("segmentation", type=Path)
    features.add_argument("--out", type=Path, required=True)
    features.add_argument("--aliases", type=Path)
    feature_check.add_argument("features", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        if args.corpus_command == "features":
            record = extract_features(args.work, args.segmentation, args.out, args.aliases)
        elif args.corpus_command == "features-check":
            record = load_features(args.features, args.work, args.segmentation)
        elif args.corpus_command == "segment":
            record = segment_work(args.work, args.out)
        elif args.corpus_command == "segment-check":
            record = load_segmentation(args.segmentation, args.work)
        elif args.corpus_command == "segment-compare":
            record = compare_segmentation(args.work, args.segmentation, args.gold, args.out)
        elif args.corpus_command == "split":
            record = (
                freeze_cohort(args.config, args.out)
                if args.split_command == "freeze"
                else load_cohort(args.cohort)
            )
        elif args.corpus_command == "inspect":
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
