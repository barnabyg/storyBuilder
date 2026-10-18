"""Load `KEY=VALUE` lines from a `.env` file into the environment.

Variables already set in the environment win, so CI and shells can override
the file. Lines may be blank, comments (`#`), or `export KEY=VALUE`; values may
be wrapped in single or double quotes.
"""

import os
from pathlib import Path

_QUOTES = ("'", '"')


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in _QUOTES:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for key, value in parse_dotenv(path.read_text(encoding="utf-8")).items():
        os.environ.setdefault(key, value)
