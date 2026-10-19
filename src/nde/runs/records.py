"""The files in a run directory and their schemas.

- `manifest.json`: what ran, from what, and how it ended (`Manifest`).
- `outputs.jsonl`: one `OutputRecord` per call. Deterministic: re-running a run
  from cache produces a byte-identical file.
- `calls.jsonl`: one `CallRecord` per provider call, with timings and usage.
- `stories.jsonl` and `stories/`: for story runs, one `StoryRecord` per story,
  and each story's files under `stories/<condition>/<brief>/r<repetition>/`.
  Deterministic, like `outputs.jsonl`.
- `report.md`: generated from the three files above, never edited by hand.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from nde.files import write_text_lf

MANIFEST = "manifest.json"
OUTPUTS = "outputs.jsonl"
CALLS = "calls.jsonl"
REPORT = "report.md"
STORIES = "stories.jsonl"
STORY_DIR = "stories"

type RunStatus = Literal["running", "succeeded", "failed", "aborted"]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelUse(_Record):
    role: str
    provider: str
    model: str


class TemplateUse(_Record):
    template: str
    version: int
    sha256: str


class CacheSummary(_Record):
    mode: str
    path: str | None
    hits: int = 0
    misses: int = 0


class CostSummary(_Record):
    input_tokens: int = 0
    output_tokens: int = 0
    # New spend in this run; cache hits cost nothing.
    usd: float = 0.0
    # What every call's tokens cost at list price, cached or not.
    token_cost_usd: float = 0.0
    per_run_cap_usd: float | None = None
    per_day_cap_usd: float | None = None
    # Spent today (UTC) across every run sharing the ledger, as of the end of this run.
    day_spent_usd: float | None = None


class StructuredMetrics(_Record):
    """Schema-validity counts and rates for one model's structured calls."""

    calls: int = 0
    valid_first_attempt: int = 0
    valid_after_repair: int = 0
    # Calls that needed at least one repair request, whether or not it succeeded.
    repaired_calls: int = 0
    failed_calls: int = 0
    first_attempt_validity: float = 0.0
    validity_after_repair: float = 0.0
    repair_rate: float = 0.0

    def record(self, *, attempts: int, valid: bool) -> None:
        self.calls += 1
        self.valid_first_attempt += valid and attempts == 1
        self.valid_after_repair += valid
        self.repaired_calls += attempts > 1
        self.failed_calls += not valid
        self.first_attempt_validity = self.valid_first_attempt / self.calls
        self.validity_after_repair = self.valid_after_repair / self.calls
        self.repair_rate = self.repaired_calls / self.calls


class BenchmarkRow(_Record):
    """One candidate model's G0 benchmark results."""

    model: str
    # Logical calls: a call and its repair attempts count once.
    calls: int
    first_attempt_validity: float
    validity_after_repair: float
    repair_rate: float
    # Per logical call, all attempts included; None when the model made no calls.
    latency_median_s: float | None
    latency_p95_s: float | None
    # At list price, cache hits included, so a cached re-run reports the same numbers.
    cost_per_1k_output_usd: float | None
    token_cost_usd: float
    # Actually spent in this run.
    new_spend_usd: float
    meets_threshold: bool


class BenchmarkSummary(_Record):
    role: str
    threshold: float
    rows: list[BenchmarkRow]


class ConditionUse(_Record):
    name: str
    pipeline: str
    params: dict[str, JsonValue]


class StoriesSummary(_Record):
    """What a story run generated from."""

    briefs_dir: str
    brief_ids: list[str]
    # Checksum of each frozen split the briefs come from (see nde.briefs).
    split_checksums: dict[str, str]
    conditions: list[ConditionUse]
    repeat: int
    stories: int = 0
    failed: int = 0


class RunError(_Record):
    """Why a run failed: a common provider error type (see nde.llm.errors) or "internal"."""

    type: str
    message: str
    # The call that was running when the run failed.
    call: str | None = None
    # The provider's HTTP status, when it answered.
    status: int | None = None


class Manifest(_Record):
    name: str
    status: RunStatus
    git_sha: str
    git_dirty: bool
    config_path: str
    config_hash: str
    seed: int
    models: list[ModelUse]
    prompt_templates: list[TemplateUse]
    cache: CacheSummary
    cost: CostSummary
    started_at: str
    finished_at: str | None = None
    # Why an aborted run stopped, e.g. "budget".
    abort_reason: str | None = None
    abort_message: str | None = None
    # Set when status is "failed".
    error: RunError | None = None
    warnings: list[str] = Field(default_factory=list[str])
    # Per "provider/model": validity of structured (response-model) calls.
    structured: dict[str, StructuredMetrics] = Field(default_factory=dict[str, StructuredMetrics])
    # Set for benchmark configs: the per-candidate comparison for gate G0.
    benchmark: BenchmarkSummary | None = None
    # Set for story runs.
    stories: StoriesSummary | None = None
    files: dict[str, str] = Field(
        default_factory=lambda: {"outputs": OUTPUTS, "calls": CALLS, "report": REPORT}
    )


class OutputRecord(_Record):
    call: str
    repetition: int
    role: str
    provider: str
    model: str
    seed: int
    # The final attempt's raw text.
    text: str
    # For structured calls: the response model, the validated object, or why it failed.
    response_model: str | None = None
    parsed: JsonValue = None
    error: str | None = None


class CallRecord(_Record):
    call: str
    repetition: int
    # For story runs: the brief, condition and pipeline step the call served.
    brief: str | None = None
    condition: str | None = None
    step: str | None = None
    # 0 for the first request; 1, 2, ... for structured-output repair requests.
    attempt: int = 0
    provider: str
    model: str
    # The model ID the provider reported.
    reported_model: str
    template: str
    template_version: int
    seed: int
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool
    # Retries after rate-limited or transient failures before this call succeeded.
    retries: int = 0
    # New spend: zero for a cache hit.
    cost_usd: float
    # What the call's tokens cost at list price, cached or not.
    token_cost_usd: float


type StoryStatus = Literal["ok", "failed"]


class StoryRecord(_Record):
    """One generated story. Deterministic: no timings or spend, so re-runs match byte for byte."""

    brief: str
    condition: str
    pipeline: str
    repetition: int
    seed: int
    status: StoryStatus
    # Folder of the story's files, relative to the run directory.
    path: str
    words: int = 0
    # Why the story could not be produced, when status is "failed".
    error: str | None = None
    # The story's files (the story and every intermediate output), relative to `path`.
    files: list[str] = Field(default_factory=list[str])


def write_jsonl(
    path: Path, records: list[OutputRecord] | list[CallRecord] | list[StoryRecord]
) -> None:
    write_text_lf(path, "".join(record.model_dump_json() + "\n" for record in records))


def read_outputs(run_dir: Path) -> list[OutputRecord]:
    return [OutputRecord.model_validate_json(line) for line in _lines(run_dir / OUTPUTS)]


def read_calls(run_dir: Path) -> list[CallRecord]:
    return [CallRecord.model_validate_json(line) for line in _lines(run_dir / CALLS)]


def read_stories(run_dir: Path) -> list[StoryRecord]:
    return [StoryRecord.model_validate_json(line) for line in _lines(run_dir / STORIES)]


def read_manifest(run_dir: Path) -> Manifest:
    return Manifest.model_validate_json((run_dir / MANIFEST).read_text(encoding="utf-8"))


def write_manifest(run_dir: Path, manifest: Manifest) -> None:
    write_text_lf(run_dir / MANIFEST, manifest.model_dump_json(indent=2) + "\n")


def _lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]
