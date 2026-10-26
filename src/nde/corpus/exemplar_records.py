"""Immutable selection of approved corpus targets supplied to development planners."""

import hashlib
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from pydantic import Field

from nde.corpus.channels import CHANNEL_FIELDS
from nde.narrative.base import Strict
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import schema_json
from nde.runs.config import ConfigError


class ExemplarSelection(Strict):
    dataset: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    work_ids: list[str] = Field(min_length=1)
    family: str | None = Field(default=None, min_length=1)


def dataset_file(path: Path) -> Path:
    return path / "dataset.json" if path.is_dir() else path


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _targets(plan: NarrativePlan, targets: dict[str, str]) -> dict[str, Any]:
    data = plan.model_dump(mode="json")
    result: dict[str, Any] = {"hard": {}, "soft_only": {}}
    for channel, policy in targets.items():
        if policy not in {"hard", "soft"}:
            raise ValueError(f"unsupported exemplar target policy: {channel}={policy}")
        if channel in {"goal", "belief"}:
            value = {"states": [state for state in data["states"] if state["kind"] == channel]}
        elif channel in CHANNEL_FIELDS:
            value = {field: data[field] for field in CHANNEL_FIELDS[channel]}
        else:
            raise ValueError(f"unsupported exemplar channel: {channel}")
        result["hard" if policy == "hard" else "soft_only"][channel] = value
    return result


def _example(record: dict[str, Any], policy: dict[str, str]) -> dict[str, Any]:
    work_id = record["work_id"]
    annotation = Path(record["annotation_path"])
    if file_digest(annotation) != record["annotation_sha256"]:
        raise ValueError(f"exemplar annotation changed: {work_id}")
    if record["schema_sha256"] != hashlib.sha256(schema_json().encode()).hexdigest():
        raise ValueError(f"exemplar schema is incompatible: {work_id}")
    plan = NarrativePlan.model_validate_json(annotation.read_text(encoding="utf-8"))
    if not validate_plan(plan).valid:
        raise ValueError(f"exemplar annotation is invalid: {work_id}")
    if record["source_sha256"] not in {source.sha256 for source in plan.sources}:
        raise ValueError(f"exemplar source identity differs: {work_id}")
    targets = cast(dict[str, str], record["targets"])
    if not targets or any(policy.get(channel) != target for channel, target in targets.items()):
        raise ValueError(f"exemplar target policy differs from approval: {work_id}")
    return {
        "work_id": work_id,
        "author_id": record["author_id"],
        "family": record["family"],
        "annotation_sha256": record["annotation_sha256"],
        "source_sha256": record["source_sha256"],
        "schema_sha256": record["schema_sha256"],
        "target_policy": targets,
        "context": {
            "characters": [item.model_dump(mode="json") for item in plan.characters],
            "scene_ids": [scene.id for scene in plan.scenes],
            "sources": [
                {"id": source.id, "sha256": source.sha256, "span_basis": source.span_basis}
                for source in plan.sources
            ],
        },
        "targets": _targets(plan, targets),
    }


def load_exemplars(selection: ExemplarSelection) -> dict[str, Any]:
    """Replay export eligibility before using any of its content in a prompt.

    The export validator is loaded lazily to keep the unconfigured planning path
    independent of the corpus/report dependency graph.
    """
    path = dataset_file(Path(selection.dataset))
    if file_digest(path) != selection.dataset_sha256:
        raise ValueError("exemplar export changed from the configured checksum")
    if len(set(selection.work_ids)) != len(selection.work_ids):
        raise ValueError("exemplar work identities must be distinct")
    validator = import_module("nde.corpus.export").validate_export
    dataset = cast(dict[str, Any], validator(path))
    if dataset["version"] != "1":
        raise ValueError("incompatible exemplar export version")
    available: dict[str, list[dict[str, Any]]] = {}
    identities: set[tuple[str, str]] = set()
    for record in dataset["records"]:
        key = record["work_id"], record["family"]
        if key in identities:
            raise ValueError("duplicate exemplar work/family identity in export")
        identities.add(key)
        available.setdefault(record["work_id"], []).append(record)
    examples: list[dict[str, Any]] = []
    for work_id in selection.work_ids:
        candidates = sorted(available.get(work_id, []), key=lambda row: row["family"])
        if selection.family is not None:
            candidates = [row for row in candidates if row["family"] == selection.family]
        record = candidates[0] if candidates else None
        if record is None or record["split"] != "train" or not record["eligible"]:
            raise ValueError(f"exemplar is missing, protected or ineligible: {work_id}")
        examples.append(_example(record, dataset["target_policy"]))
    return {
        "version": "1",
        "dataset_id": dataset["id"],
        "dataset_sha256": selection.dataset_sha256,
        "selection": selection.model_dump(mode="json"),
        "test_only": dataset["test_only"],
        "examples": examples,
        "note": "Selected approved targets; this is not a complete tier-L narrative plan. "
        "Soft targets are uncertain examples, never hard constraints. "
        "Without an explicit family choice, the first sorted family is selected independently "
        "for each work; independent annotations are never merged.",
    }


def check_exemplars(selection: ExemplarSelection | None) -> None:
    if selection is not None:
        try:
            load_exemplars(selection)
        except (OSError, ValueError) as error:
            raise ConfigError(f"exemplar export: {error}") from error
