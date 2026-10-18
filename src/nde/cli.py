"""The `nde` command-line interface."""

import argparse
from collections.abc import Sequence

from nde import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nde", description="Narrative Diffusion Engine")
    parser.add_argument("--version", action="version", version=f"nde {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
