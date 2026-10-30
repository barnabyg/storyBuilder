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
