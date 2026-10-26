"""Durable phase deadlines; unknown interrupted inference retains its full allowance."""

import time
from dataclasses import dataclass, field
from pathlib import Path

from nde.corpus.records_io import read_sealed, save_json
from nde.reconstruction.codec import seal


@dataclass
class TimeAllowance:
    path: Path
    binding_id: str
    max_seconds: float
    started: float
    elapsed_floor: float = 0.0
    charge_interruption: bool = True
    tokens_before: int = field(init=False, default=0)
    cells_before: int = field(init=False, default=0)
    interrupted: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            previous = read_sealed(self.path)
            if previous["binding_id"] != self.binding_id:
                raise ValueError("phase time reservation binding changed")
            self.elapsed_floor = max(self.elapsed_floor, previous["elapsed_seconds"])
            self.tokens_before = previous.get("tokens", 0)
            self.cells_before = previous.get("attention_cells", 0)
            self.interrupted = previous["status"] == "running"
            if self.interrupted and self.charge_interruption:
                self.elapsed_floor = max(self.elapsed_floor, self.max_seconds)
        self.save(tokens=self.tokens_before, cells=self.cells_before)

    @property
    def deadline(self) -> float:
        return self.started + max(0.0, self.max_seconds - self.elapsed_floor)

    def elapsed(self) -> float:
        return self.elapsed_floor + max(0.0, time.monotonic() - self.started)

    def expired(self) -> bool:
        return self.elapsed() >= self.max_seconds

    def save(self, *, tokens: int, cells: int, status: str = "running") -> None:
        save_json(
            self.path,
            seal(
                {
                    "binding_id": self.binding_id,
                    "elapsed_seconds": self.elapsed(),
                    "tokens": tokens,
                    "attention_cells": cells,
                    "status": status,
                    "max_seconds": self.max_seconds,
                    "interrupted_allowance_retained": self.interrupted and self.charge_interruption,
                }
            ),
        )
