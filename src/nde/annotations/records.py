"""Persisted annotation provenance, progress and owner evidence."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field

from nde.files import checksum_lf, write_text_lf
from nde.llm.config import ModelBinding, ProviderConfig, Strict
from nde.llm.ledger import Price
from nde.narrative.models import Source
from nde.runs.config import BudgetConfig, PromptRef


class AnnotationConfig(Strict):
    seed: int = 7
    providers: dict[str, ProviderConfig]
    annotator: ModelBinding
    prices: dict[str, Price]
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    max_repairs: int = Field(default=2, ge=0, le=10)
    prompts_dir: str = "../prompts"
    prompt: PromptRef = Field(default_factory=lambda: PromptRef(template="annotate_l", version=1))


class Attempt(Strict):
    text: str
    errors: list[str]
    call: dict[str, Any]


class Revision(Strict):
    number: int
    sha256: str
    valid: bool
    file: str


class OwnerReview(Strict):
    revision: int
    sha256: str
    owner: str = Field(min_length=1)
    literary_notes: str = Field(min_length=1)
    meaningful_trajectories: bool
    accepted: bool
    recorded_at: str


class ReviewDecision(Strict):
    owner: str = Field(min_length=1)
    notes: str = Field(min_length=1)
    accepted: bool
    meaningful: bool


@dataclass(frozen=True)
class DraftOptions:
    out: Path
    cache: Path
    ledger: Path


class TimerEvent(Strict):
    action: Literal["start", "pause", "recover"]
    at: float
    seconds: float = Field(default=0, ge=0)


class Session(Strict):
    model_config = ConfigDict(extra="forbid", frozen=False)
    version: Literal["1"] = "1"
    id: str
    story_id: str
    status: Literal[
        "running", "drafted", "exhausted", "budget_aborted", "provider_failed", "recovery_required"
    ] = "running"
    source: Source
    provenance: dict[str, Any]
    config: AnnotationConfig
    config_sha256: str
    schema_sha256: str
    prompt_sha256: str
    git_sha: str
    attempts: list[Attempt] = Field(default_factory=list[Attempt])
    spent_usd: float = 0
    pending_call: bool = False
    ledger_path: str | None = None
    cache_path: str | None = None
    failure: str | None = None
    revisions: list[Revision] = Field(default_factory=list[Revision])
    owner_reviews: list[OwnerReview] = Field(default_factory=list[OwnerReview])


def digest(text: str) -> str:
    return checksum_lf(text.encode("utf-8"))


def save_session(directory: Path, session: Session) -> None:
    temporary = directory / "session.tmp"
    write_text_lf(temporary, session.model_dump_json(indent=2) + "\n")
    temporary.replace(directory / "session.json")


def read_session(directory: Path) -> Session:
    return Session.model_validate_json((directory / "session.json").read_text(encoding="utf-8"))


def timer_events(directory: Path) -> list[TimerEvent]:
    path = directory / "timer.jsonl"
    if not path.exists():
        return []
    return [
        TimerEvent.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
