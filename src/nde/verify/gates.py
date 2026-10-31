"""The canonical verification profile: seven gates in the standard order."""

import os
import sys

from nde.verify.runner import Command, Gate

TESTS_GATE = "tests"
GATE_ORDER = (
    "format",
    "lint",
    "types",
    "static-analysis",
    TESTS_GATE,
    "dependencies-and-secrets",
    "build",
)

# Ruff rule families that make up gate 4 (static bug analysis): flake8-bugbear
# (B) and flake8-bandit security rules (S). Gate 2 runs the full committed rule
# selection, which also includes them: ignoring them there would make RUF100
# report every justified bandit suppression as unused.
_STATIC_ANALYSIS_RULES = "B,S"


def _python_module(module: str, *args: str) -> Command:
    return Command((sys.executable, "-m", module, *args))


def _uv(*args: str) -> Command:
    # `uv run` exports the path of the running uv binary as UV.
    return Command((os.environ.get("UV", "uv"), *args))


def canonical_profile() -> list[Gate]:
    gates = [
        Gate("format", (_python_module("ruff", "format", "--check", "--no-cache", "."),)),
        Gate("lint", (_python_module("ruff", "check", "--no-cache", "."),)),
        Gate("types", (_python_module("pyright", "--warnings"),)),
        Gate(
            "static-analysis",
            (
                _python_module(
                    "ruff", "check", "--no-cache", "--select", _STATIC_ANALYSIS_RULES, "."
                ),
            ),
        ),
        Gate(
            TESTS_GATE,
            (
                _python_module(
                    "pytest", "-p", "no:cacheprovider", "-p", "nde.verify.pytest_reporter"
                ),
            ),
        ),
        Gate(
            "dependencies-and-secrets",
            (
                # First because it needs no uv: a wrong uv version must not hide a leaked secret.
                _python_module("nde.verify.steps", "secrets"),
                _uv("lock", "--check"),
                _python_module("nde.verify.steps", "audit"),
            ),
        ),
        Gate("build", (_python_module("nde.verify.steps", "build"),)),
    ]
    if tuple(gate.name for gate in gates) != GATE_ORDER:  # pragma: no cover - guards edits
        raise RuntimeError("canonical profile is out of the standard gate order")
    return gates


def quick_profile() -> list[Gate]:
    """Every gate except the slow tests, for a check before pushing. Not full verification."""
    return [gate for gate in canonical_profile() if gate.name != TESTS_GATE]
