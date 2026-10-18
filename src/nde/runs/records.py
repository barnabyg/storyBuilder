"""The files in a run directory and their schemas.

- `manifest.json`: what ran, from what, and how it ended (`Manifest`).
- `outputs.jsonl`: one `OutputRecord` per call. Deterministic: re-running a run
  from cache produces a byte-identical file.
- `calls.jsonl`: one `CallRecord` per call, with timings and usage.
- `report.md`: generated from the three files above, never edited by hand.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nde.files import write_text_lf

MANIFEST = "manifest.json"
OUTPUTS = "outputs.jsonl"
CALLS = "calls.jsonl"
REPORT = "report.md"

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
    started_at: str
    finished_at: str | None = None
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
    text: str


class CallRecord(_Record):
    call: str
    repetition: int
    provider: str
    model: str
    # The model ID the provider reported.
    response_model: str
    template: str
    template_version: int
    seed: int
    input_tokens: int
    output_tokens: int
    latency_s: float


def write_jsonl(path: Path, records: list[OutputRecord] | list[CallRecord]) -> None:
    write_text_lf(path, "".join(record.model_dump_json() + "\n" for record in records))


def read_outputs(run_dir: Path) -> list[OutputRecord]:
    return [OutputRecord.model_validate_json(line) for line in _lines(run_dir / OUTPUTS)]


def read_calls(run_dir: Path) -> list[CallRecord]:
    return [CallRecord.model_validate_json(line) for line in _lines(run_dir / CALLS)]


def read_manifest(run_dir: Path) -> Manifest:
    return Manifest.model_validate_json((run_dir / MANIFEST).read_text(encoding="utf-8"))


def write_manifest(run_dir: Path, manifest: Manifest) -> None:
    write_text_lf(run_dir / MANIFEST, manifest.model_dump_json(indent=2) + "\n")


def _lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]
