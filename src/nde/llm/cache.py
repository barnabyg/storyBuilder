"""Content-addressed response cache in SQLite.

A response is keyed by everything that can change it: provider, model,
parameters, prompt template version, rendered messages, seed and response
format. Re-running an experiment is then free and reproduces its outputs.

Several runs may share one cache file: SQLite's write-ahead log lets readers
and a writer work at once, and writers wait for each other.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from nde.llm import sqlite
from nde.llm.client import ChatRequest, ChatResult

type CacheMode = Literal["use", "bypass", "refresh"]

DEFAULT_CACHE_PATH = Path(".nde") / "cache.sqlite"


def cache_key(provider: str, template: str, version: int, request: ChatRequest) -> str:
    material = {
        "provider": provider,
        "template": template,
        "template_version": version,
        "request": request.body(),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0


class ResponseCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._db = sqlite.connect(
            path,
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY,"
            " request TEXT NOT NULL,"
            " response TEXT NOT NULL,"
            " stored_at TEXT NOT NULL)",
        )

    def close(self) -> None:
        self._db.close()

    def get(self, key: str) -> ChatResult | None:
        row: tuple[str] | None = self._db.execute(
            "SELECT response FROM responses WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return ChatResult(**json.loads(row[0]))

    def put(self, key: str, request: ChatRequest, result: ChatResult) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO responses (key, request, response, stored_at)"
            " VALUES (?, ?, ?, ?)",
            (
                key,
                json.dumps(request.body(), sort_keys=True, ensure_ascii=False),
                json.dumps(asdict(result), sort_keys=True, ensure_ascii=False),
                datetime.now(UTC).isoformat(),
            ),
        )
