"""Leakage checks for blind reading.

The 48-hour rule (§4 of the implementation plan): reading should start at
least 48 hours after the owner last worked on the code that generates the
stories, so that fresh memory of the conditions cannot unblind the reader.
"""

import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# The code and prompts that decide what a story looks like.
GENERATING_PATHS = (
    "src/nde/stories",
    "src/nde/plan",
    "src/nde/prose",
    "src/nde/llm",
    "prompts",
)
WAIT = timedelta(hours=48)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        # S603/S607: fixed git arguments; git is resolved from PATH as a developer prerequisite.
        completed = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def generating_code_warning(cwd: Path, now: datetime) -> str | None:
    """A warning when the generating code changed less than 48 hours ago, or may have."""
    root = _git(["rev-parse", "--show-toplevel"], cwd)
    if not root:
        return (
            f"cannot tell when the generating code last changed: {cwd} is not in a git "
            "repository, so the 48-hour rule cannot be checked"
        )
    repo = Path(root)
    if _git(["status", "--porcelain", "--", *GENERATING_PATHS], repo):
        return (
            "the generating code has uncommitted changes, so it changed less than 48 hours "
            "before reading"
        )
    last = _git(["log", "-1", "--format=%cI", "--", *GENERATING_PATHS], repo)
    if not last:
        return None
    changed = datetime.fromisoformat(last)
    age = now - changed
    if age >= WAIT:
        return None
    hours = age.total_seconds() / 3600
    return (
        f"the generating code last changed {hours:.0f} hours ago ({changed.isoformat()}), "
        "less than 48 hours before reading"
    )
