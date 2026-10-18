"""File helpers shared across the package."""

from pathlib import Path


def write_text_lf(path: Path, text: str) -> None:
    """Write UTF-8 text with LF line endings on every platform.

    `Path.write_text` translates "\\n" to the platform separator, which would make
    run files and committed files differ between Windows and Linux.
    """
    path.write_text(text, encoding="utf-8", newline="\n")
