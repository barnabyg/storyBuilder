"""Public file, version dispatch, validation and card rendering workflow."""

import html
import json
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from nde.files import write_text_lf
from nde.narrative.models import NarrativePlan
from nde.narrative.structural_render import structural_lines
from nde.narrative.validation import Validation, validate_plan


class PlanError(ValueError):
    """An unreadable, unsupported or invalid plan."""


def migrate(data: dict[str, Any], target: str = "1") -> dict[str, Any]:
    """Version dispatch entry point; no older narrative formats have been released."""
    version = data.get("schema_version")
    if version != "1" or target != "1":
        raise PlanError(f"unsupported schema version {version!r}, target {target!r}")
    return data


def load_plan(path: Path) -> NarrativePlan:
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise PlanError("a plan must be a JSON object")
        return NarrativePlan.model_validate(migrate(cast(dict[str, Any], data)))
    except (OSError, ValueError, ValidationError) as error:
        raise PlanError(f"{path}: {error}") from error


def save_plan(plan: NarrativePlan, path: Path) -> None:
    write_text_lf(path, plan.model_dump_json(indent=2) + "\n")


def schema_json() -> str:
    return json.dumps(NarrativePlan.model_json_schema(), indent=2, sort_keys=True) + "\n"


def card_lines(plan: NarrativePlan, result: Validation) -> list[str]:
    lines = [f"# {plan.title}", f"Plan {plan.id}; schema {plan.schema_version}; tier {plan.tier}"]
    lines.extend(f"Character {character.id}: {character.name}" for character in plan.characters)
    for scene in plan.scenes:
        lines.extend(
            [
                f"## {scene.id}: {scene.title}",
                scene.summary,
                f"Present: {', '.join(scene.character_ids) or '(none recorded)'}",
            ]
        )
        lines.extend(f"Beat {beat.id}: {beat.summary}" for beat in scene.beats)
        lines.extend(structural_lines(plan, scene.id))
        lines.extend(f"Evidence: {span.model_dump_json()}" for span in scene.evidence)
        lines.extend(
            f"Beat {beat.id} evidence: {span.model_dump_json()}"
            for beat in scene.beats
            for span in beat.evidence
        )
    lines.extend(
        f"{item.severity} {item.code} [{item.location}]: {item.message}"
        for item in result.diagnostics
    )
    return lines


def render(plan: NarrativePlan, out: Path) -> Validation:
    result = validate_plan(plan)
    lines = card_lines(plan, result)
    out.mkdir(parents=True, exist_ok=True)
    save_plan(plan, out / "plan.json")
    write_text_lf(out / "validation.json", result.model_dump_json(indent=2) + "\n")
    write_text_lf(out / "cards.md", "\n\n".join(lines) + "\n")
    body = "\n".join(f"<p>{html.escape(line)}</p>" for line in lines)
    write_text_lf(
        out / "cards.html",
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>{html.escape(plan.title)}</title><body>{body}</body></html>\n",
    )
    return result
