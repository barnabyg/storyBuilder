"""Pilot registration must fail before any paid work on invalid cohorts or limits."""

import json
from pathlib import Path

import pytest

from nde.cli import main
from tests.corpus_support import corpus_inputs, queue_annotation
from tests.fake_openai import FakeOpenAI
from tests.pilot_support import pilot_config


def test_real_pilot_cannot_start_without_g1_evidence(tmp_path: Path) -> None:
    config = tmp_path / "pilot.json"
    config.write_text(
        json.dumps({"cohort": "missing", "extraction_config": "missing", "works": []})
    )
    out = tmp_path / "pilot"
    assert main(["corpus-pilot", "run", str(config), "--out", str(out)]) == 2
    assert not out.exists()


def test_pilot_reports_all_passes_and_resumes_without_resetting_costs(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    inputs = corpus_inputs(tmp_path, fake_llm)
    queue_annotation(inputs, fake_llm)
    config = pilot_config(tmp_path, inputs)
    out = tmp_path / "pilot"
    args = ["corpus-pilot", "run", str(config), "--out", str(out)]
    assert main(args) == 0
    first = json.loads((out / "report.json").read_text())
    assert first["completed_stories"] == 1
    assert len(first["rows"]) == 6
    assert sum(r["input_tokens"] + r["output_tokens"] for r in first["rows"]) == 90
    assert first["new_spend_usd"] == pytest.approx(0.00009)
    assert first["scaling"]["300_stories_usd"] == pytest.approx(0.027)
    assert first["unresolved_accounting_usd"] == pytest.approx(0)
    assert main(args) == 0
    second = json.loads((out / "report.json").read_text())
    assert second["rows"] == first["rows"]
    assert fake_llm.call_count == 6
    changed = json.loads(config.read_text())
    changed["per_run_usd"] = 2
    config.write_text(json.dumps(changed))
    assert main(args) == 2
    assert fake_llm.call_count == 6


def test_pilot_cap_preserves_failed_and_unattempted_rows(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    inputs = corpus_inputs(tmp_path, fake_llm)
    config = pilot_config(tmp_path, inputs, cap=0.000001)
    out = tmp_path / "pilot"
    args = ["corpus-pilot", "run", str(config), "--out", str(out)]
    assert main(args) == 1
    saved = json.loads((out / "report.json").read_text())
    assert saved["status"] == "incomplete"
    assert len(saved["rows"]) == 6
    assert saved["completed_stories"] == 0
    assert saved["scaling"]["per_completed_story_usd"] is None
    assert any(row["status"] == "budget_aborted" for row in saved["rows"])
    assert main(args) == 1
    assert fake_llm.call_count == 0


@pytest.mark.parametrize("change", ["missing", "stale"])
def test_pilot_reports_incomplete_coverage_when_features_are_missing_or_stale(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    change: str,
) -> None:
    inputs = corpus_inputs(tmp_path, fake_llm)
    queue_annotation(inputs, fake_llm)
    config = pilot_config(tmp_path, inputs)
    out = tmp_path / "pilot"
    assert main(["corpus-pilot", "run", str(config), "--out", str(out)]) == 0
    feature = out / "features/lamp/features.json"
    if change == "missing":
        feature.unlink()
    else:
        feature.write_text("{}")
    assert main(["corpus-pilot", "report", str(out)]) == 1
    result = json.loads((out / "report.json").read_text())
    assert result["completed_stories"] == 0
    assert result["coverage"] == 0
    assert result["features"]["lamp"]["status"] == change
    assert len(result["rows"]) == 6
    assert result["list_price_usd"] == pytest.approx(0.00009)
