"""SQLite connections shared safely by concurrent runs."""

import sqlite3
import time
from pathlib import Path

# Seconds a connection waits for another writer before failing.
BUSY_TIMEOUT_S = 60.0
_SETUP_RETRY_S = 0.05


def connect(path: Path, schema: str) -> sqlite3.Connection:
    """Open `path` in write-ahead-log mode and create `schema` if needed.

    Statements run in autocommit mode; callers manage transactions explicitly.
    Switching to WAL needs an exclusive lock that SQLite's busy timeout does not
    cover, so setup is retried while another connection holds the database.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(
        path, timeout=BUSY_TIMEOUT_S, isolation_level=None, check_same_thread=False
    )
    deadline = time.monotonic() + BUSY_TIMEOUT_S
    while True:
        try:
            db.execute("PRAGMA journal_mode=WAL")
            # In WAL mode NORMAL never corrupts the database; a power cut can
            # lose only the last commits. It avoids an fsync on every commit.
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute(schema)
        except sqlite3.OperationalError as error:
            if "locked" not in str(error) or time.monotonic() > deadline:
                db.close()
                raise
            time.sleep(_SETUP_RETRY_S)
        else:
            return db
