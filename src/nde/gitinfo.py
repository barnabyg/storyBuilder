"""Read-only git queries: the commit a run came from, or when code last changed."""

import subprocess
from pathlib import Path


def git(args: list[str], cwd: Path) -> str | None:
    """A git command's stripped output, or None when git fails or is unavailable."""
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
