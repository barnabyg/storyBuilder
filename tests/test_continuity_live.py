"""Opt-in checks that the real model catches real G1 defects: `uv run pytest -m live`.

The fake-provider tests show what the code does with a reading of a story. Only these
show that a real reading finds the defect. They spend a few cents and skip themselves
without `OPENAI_API_KEY`.
"""

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from nde.continuity.audit import Audit, run_audit
from nde.continuity.flags import Flag
from nde.continuity.pipeline import AUDITS
from nde.continuity.plan import ContinuityPlan
from nde.continuity.story import NumberedStory
from nde.llm.gateway import Gateway
from nde.llm.ledger import CostLedger, Pricing
from nde.runs.config import ExperimentConfig, StructuredConfig
from nde.runs.prompts import load_template
from nde.runs.state import RunState
from nde.schema.brief import Brief
from nde.stories.pipeline import StoryContext
from tests.conftest import REPO
from tests.continuity_evidence import (
    MRS_ALCOTT_AIDE,
    MRS_ALCOTT_TELLS_HER,
    POWER_CUT,
    PRIYA_EARLIER,
    PRIYA_KNOWS_THE_LEDGER,
    SOUR_AFTER_FOUR_MINUTES,
)

pytestmark = pytest.mark.live

BUDGET_USD = 1.0


def _config() -> ExperimentConfig:
    raw = yaml.safe_load((REPO / "configs" / "continuity-dev.yaml").read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(raw)


def _plan(name: str, knows: list[str]) -> ContinuityPlan:
    return ContinuityPlan.model_validate(
        {
            "title": "Fixture",
            "characters": [
                {
                    "name": name,
                    "appearance": "Tired.",
                    "back_story": "Works nights.",
                    "enters_scene": 1,
                    "introduction": "At work.",
                    "knows_at_start": knows,
                }
            ],
            "scenes": [
                {
                    "summary": "It happens.",
                    "beats": ["It happens."],
                    "elapsed_minutes": 60,
                    "key_events": [],
                }
            ],
        }
    )


def _audit(name: str) -> Audit[Any]:
    return next(audit for audit in AUDITS if audit.name == name)


def _read(tmp_path: Path, name: str, plan: ContinuityPlan, story: NumberedStory) -> list[Flag]:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")
    config = _config()
    audit = _audit(name)
    gateway = Gateway(
        config.providers,
        environ=dict(os.environ),
        pricing=Pricing(config.prices, config.budget.assumed_output_tokens),
        ledger=CostLedger(tmp_path / "ledger.sqlite", per_run_usd=BUDGET_USD),
    )
    brief = Brief.model_validate(
        yaml.safe_load((REPO / "benchmark" / "dev" / "brief-01.yaml").read_text(encoding="utf-8"))
    )
    ctx = StoryContext(
        gateway=gateway,
        state=RunState(),
        templates={
            audit.prompt.label: load_template(
                REPO / "prompts", audit.prompt.template, audit.prompt.version
            )
        },
        roles=config.roles,
        structured_config=StructuredConfig(),
        brief=brief,
        condition="continuity",
        repetition=0,
        seed=1,
    )
    try:
        return run_audit(ctx, audit, plan, story, round_number=0)
    finally:
        gateway.close()


def test_the_real_model_catches_the_real_power_cut(tmp_path: Path) -> None:
    story = NumberedStory((POWER_CUT, SOUR_AFTER_FOUR_MINUTES), (1, 1))

    flags = _read(tmp_path, "clock", _plan("Lucía", []), story)

    assert SOUR_AFTER_FOUR_MINUTES in {flag.passage for flag in flags}


def test_the_real_model_does_not_flag_what_people_know_about_their_own_lives(
    tmp_path: Path,
) -> None:
    # Passages the first version of the check wrongly flagged on brief-01.
    story = NumberedStory(
        (
            "Martin laid the transfer papers on Ada's kitchen table. Ada's husband built the boat.",
            '"The rent\'s overdue," he said. "There\'s another notice at home. Debt letters '
            "beside the bread bin. I leave them there so I don't open them with the children "
            'watching."',
            "The cabin light dimmed, then brightened. Her husband had taken the fitting apart "
            "twice. He had left it for the next wet afternoon.",
        ),
        (1, 1, 1),
    )

    flags = _read(tmp_path, "knowledge", _plan("Martin Rook", []), story)

    assert flags == []


def test_the_real_model_accepts_a_secret_learned_on_the_page_first(tmp_path: Path) -> None:
    story = NumberedStory(
        (
            'Mrs. Alcott leaned over the counter. "I sell the pills," she told Priya. "I get a '
            "hundred a month. It pays my aide three days a week. I stretch my insulin past its "
            'label."',
            PRIYA_KNOWS_THE_LEDGER,
        ),
        (1, 1),
    )

    flags = _read(tmp_path, "knowledge", _plan("Priya Raman", []), story)

    assert [flag.reason for flag in flags if flag.passage == PRIYA_KNOWS_THE_LEDGER] == []


def test_the_real_model_accepts_plausible_timing(tmp_path: Path) -> None:
    later = "Two days later the food in her refrigerators had begun to smell sour."
    story = NumberedStory((POWER_CUT, later), (1, 1))

    flags = _read(tmp_path, "clock", _plan("Lucía", []), story)

    assert later not in {flag.passage for flag in flags}


def test_the_real_model_reports_a_stranger_with_no_sheet_or_introduction(tmp_path: Path) -> None:
    story = NumberedStory(
        (
            "Priya Raman, a tired pharmacist in a white coat, counted pills behind the counter.",
            'Viktor Hale took the bag from her hand. "Not tonight," he said.',
        ),
        (1, 1),
    )

    flags = _read(tmp_path, "introductions", _plan("Priya Raman", []), story)

    assert any("Viktor Hale" in flag.reason for flag in flags)


def test_the_real_model_accepts_a_character_introduced_as_planned(tmp_path: Path) -> None:
    story = NumberedStory(
        (
            "Priya Raman, a night-shift pharmacist of forty with cropped grey hair and a thin "
            "face, stood at work behind the counter.",
        ),
        (1,),
    )

    flags = _read(tmp_path, "introductions", _plan("Priya Raman", []), story)

    assert flags == []


def test_the_real_model_does_not_count_clothes_and_belongings_as_a_description(
    tmp_path: Path,
) -> None:
    # The owner read this opening and found no physical description of the sisters.
    story = NumberedStory(
        (
            "Priya Raman, a night-shift pharmacist, stood at work behind the counter in a "
            "patched orange jacket, a brass compass hanging from her neck.",
        ),
        (1,),
    )

    flags = _read(tmp_path, "introductions", _plan("Priya Raman", []), story)

    assert [flag.problem_type for flag in flags] == ["missing_description"]


def test_the_real_model_catches_priya_doing_the_sums_too_early(tmp_path: Path) -> None:
    story = NumberedStory(
        (
            f"An old woman came to the counter to {PRIYA_EARLIER}.",
            PRIYA_KNOWS_THE_LEDGER,
            MRS_ALCOTT_TELLS_HER,
            MRS_ALCOTT_AIDE,
        ),
        (1, 1, 1, 1),
    )

    flags = _read(tmp_path, "knowledge", _plan("Priya Raman", []), story)

    assert PRIYA_KNOWS_THE_LEDGER in {flag.passage for flag in flags}
