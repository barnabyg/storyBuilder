"""File helpers shared across the package."""

import hashlib
import os
from pathlib import Path


def write_text_lf(path: Path, text: str) -> None:
    """Write UTF-8 text with LF line endings on every platform.

    `Path.write_text` translates "\\n" to the platform separator, which would make
    run files and committed files differ between Windows and Linux.
    """
    path.write_text(text, encoding="utf-8", newline="\n")


def checksum_lf(data: bytes) -> str:
    """SHA-256 of LF-normalised bytes, so a CRLF checkout cannot change a file's identity."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def append_line(path: Path, line: str) -> None:
    """Append one LF-terminated line and force it to disk before returning.

    For records that must survive a crash or a killed process, such as a
    reader's judgements, each saved the moment it is given.
    """
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
