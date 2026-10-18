"""Cost accounting and hard budget caps.

Prices are supplied by the user in each config, per `provider/model`, in USD
per million tokens, because they vary by provider and change over time.

Before every provider call the gateway reserves a conservative estimate of its
cost against the per-run and per-day caps; if the reservation would exceed a
cap the call is not made. After the call the reservation is settled at the
actual cost. The output estimate is bounded by max_tokens, but the input
estimate is a heuristic, so a call can cost slightly more than it reserved.
The per-day total is kept in SQLite so it persists across runs and is shared
by concurrent runs. Days are UTC calendar days.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from nde.llm import sqlite
from nde.llm.client import ChatRequest
from nde.llm.config import Strict

DEFAULT_LEDGER_PATH = Path(".nde") / "ledger.sqlite"

# Output tokens assumed for the estimate when a call sets no max_tokens.
DEFAULT_ASSUMED_OUTPUT_TOKENS = 1024

# Conservative input estimate: real tokenisers average about 4 characters per
# token for English; assume 3, plus overhead for each message's framing.
_CHARS_PER_TOKEN = 3
_TOKENS_PER_MESSAGE = 8

_PER_MILLION = 1_000_000

type Clock = Callable[[], datetime]
type CapScope = Literal["run", "day"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Price(Strict):
    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / _PER_MILLION


@dataclass(frozen=True)
class Pricing:
    """The price table. Unpriced models cost zero; configs reject them unless allowed."""

    prices: Mapping[str, Price]
    assumed_output_tokens: int = DEFAULT_ASSUMED_OUTPUT_TOKENS

    def price(self, label: str) -> Price | None:
        return self.prices.get(label)

    def cost(self, label: str, input_tokens: int, output_tokens: int) -> float:
        price = self.price(label)
        return price.cost(input_tokens, output_tokens) if price else 0.0

    def estimate(self, label: str, request: ChatRequest) -> float:
        """An upper estimate of a call's cost, made before the call."""
        price = self.price(label)
        if price is None:
            return 0.0
        characters = sum(len(message["content"]) for message in request.messages)
        input_tokens = math.ceil(characters / _CHARS_PER_TOKEN) + _TOKENS_PER_MESSAGE * len(
            request.messages
        )
        max_tokens = request.params.get("max_tokens", request.params.get("max_completion_tokens"))
        output_tokens = max_tokens if isinstance(max_tokens, int) else self.assumed_output_tokens
        return price.cost(input_tokens, output_tokens)


class BudgetExceededError(Exception):
    def __init__(self, scope: CapScope, cap: float, spent: float, estimate: float) -> None:
        self.scope: CapScope = scope
        self.cap = cap
        self.spent = spent
        self.estimate = estimate
        label = "per-run" if scope == "run" else "per-day"
        super().__init__(
            f"the next call (estimated ${estimate:.4f}) would exceed the {label} cap of "
            f"${cap:.4f} (${spent:.4f} already spent)"
        )


@dataclass(frozen=True)
class Reservation:
    day: str
    estimate: float


class CostLedger:
    def __init__(
        self,
        path: Path,
        *,
        per_run_usd: float | None = None,
        per_day_usd: float | None = None,
        clock: Clock = _utc_now,
    ) -> None:
        self.per_run_usd = per_run_usd
        self.per_day_usd = per_day_usd
        self.run_spent = 0.0
        self._clock = clock
        self._db = sqlite.connect(
            path, "CREATE TABLE IF NOT EXISTS daily_spend (day TEXT PRIMARY KEY, usd REAL NOT NULL)"
        )

    def close(self) -> None:
        self._db.close()

    def _today(self) -> str:
        return self._clock().astimezone(UTC).date().isoformat()

    def day_spent(self, day: str | None = None) -> float:
        row: tuple[float] | None = self._db.execute(
            "SELECT usd FROM daily_spend WHERE day = ?", (day or self._today(),)
        ).fetchone()
        return row[0] if row else 0.0

    def reserve(self, estimate: float) -> Reservation:
        """Hold `estimate` against both caps, or raise BudgetExceededError."""
        if self.per_run_usd is not None and self.run_spent + estimate > self.per_run_usd:
            raise BudgetExceededError("run", self.per_run_usd, self.run_spent, estimate)
        day = self._today()
        self._db.execute("BEGIN IMMEDIATE")
        try:
            spent = self.day_spent(day)
            if self.per_day_usd is not None and spent + estimate > self.per_day_usd:
                raise BudgetExceededError("day", self.per_day_usd, spent, estimate)
            self._add(day, estimate)
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self.run_spent += estimate
        return Reservation(day=day, estimate=estimate)

    def settle(self, reservation: Reservation, actual: float) -> None:
        """Replace a reservation's estimate with the call's actual cost."""
        delta = actual - reservation.estimate
        self.run_spent += delta
        self._add(reservation.day, delta)

    def _add(self, day: str, usd: float) -> None:
        self._db.execute(
            "INSERT INTO daily_spend (day, usd) VALUES (?, ?)"
            " ON CONFLICT(day) DO UPDATE SET usd = usd + excluded.usd",
            (day, usd),
        )
