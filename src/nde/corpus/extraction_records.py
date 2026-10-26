"""Versioned extraction checkpoints; Phase 2 annotation sessions remain unchanged."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from nde.llm.config import ModelBinding, ProviderConfig, Strict
from nde.llm.ledger import Price
from nde.narrative.models import NarrativePlan, Source
from nde.runs.config import BudgetConfig
from nde.runs.records import CallRecord

PASS_NAMES = ("structural", "state", "interpretive")
STATE_CHANNELS = ("states", "relationships", "constraints")
INTERPRETIVE_CHANNELS = (
    "arcs",
    "reader_affect",
    "tension",
    "themes",
    "motifs",
    "motif_echoes",
    "discourse",
    "subtext",
)


class ExtractionSettings(Strict):
    version: Literal["1"] = "1"
    seed: int = 7
    providers: dict[str, ProviderConfig]
    prices: dict[str, Price]
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    max_repairs: int = Field(default=1, ge=0, le=10)


class ExtractionConfig(ExtractionSettings):
    annotator: ModelBinding


@dataclass(frozen=True)
class ExtractionOptions:
    out: Path
    cache: Path
    ledger: Path
    cohort_run_id: str | None = None
    run_budget_usd: float | None = None
    story_run_id: str | None = None


class ExtractionAttempt(Strict):
    text: str
    errors: list[str]
    call: CallRecord


class ExtractionPass(Strict):
    model_config = ConfigDict(extra="forbid", frozen=False)
    name: str
    status: str = "planned"
    attempts: list[ExtractionAttempt] = Field(default_factory=list[ExtractionAttempt])
    accepted_plan: NarrativePlan | None = None
    failure: str | None = None


class ExtractionSession(Strict):
    model_config = ConfigDict(extra="forbid", frozen=False)
    version: Literal["1"] = "1"
    id: str
    work_id: str
    source: Source
    source_sha256: str
    segmentation_sha256: str
    work_dir: str
    segmentation_file: str
    config_file: str
    config: ExtractionConfig
    config_sha256: str
    schema_sha256: str
    prompt_sha256: str
    cache_path: str
    ledger_path: str
    cohort_run_id: str | None = None
    run_budget_usd: float | None = None
    story_run_id: str | None = None
    passes: list[ExtractionPass]
    status: str = "running"
    pending_pass: str | None = None
    spent_usd: float = 0
    accounting_run_spent_usd: float = 0
    accounting_story_spent_usd: float = 0
    annotation_sha256: str | None = None
    failure: str | None = None
    owner_approved: Literal[False] = False
