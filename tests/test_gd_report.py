"""Closed G-D software fixtures remain explicitly synthetic scientific evidence."""

import json
import shutil
from pathlib import Path

import pytest

from nde.cli import main
from nde.controllability import ControlTarget
from nde.plan.revision import identity
from tests.eval_helpers import reader
from tests.fake_openai import FakeOpenAI
from tests.gd_fixtures import gd_batch, prepare_gd
from tests.test_g1 import git_fixture

pytest_plugins = ("tests.test_g1",)


def test_closed_synthetic_gate_reports_criteria_but_cannot_promote_d(
    registered_inputs: dict[str, Path],
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common, _ = prepare_gd(registered_inputs, tmp_path, fake_llm, monkeypatch)
    batch, _, _ = gd_batch(common, tmp_path)
    client = reader(batch)
    fields = {"preferred": "A", "strength": "2"}
    fields.update(
        {
            f"score-{side}-{dimension}": "4"
            for side in ("a", "b")
            for dimension in ("coherence", "originality")
        }
    )
    assert client.post("/pairs/1", data=fields).status_code == 200
    assert client.post("/close").status_code == 200
    out = tmp_path / "gd-result"
    assert main(["gd-report", str(batch), "--out", str(out)]) == 0
    result = json.loads((out / "decision.json").read_text())
    assert result["criteria_outcome"] == "passed"
    assert result["outcome"] == "pending"
    assert result["reference_baseline"] is None
    assert result["coherence"]["ties"] == 1
    assert (out / "decision.md").is_file()
    assert result["costs"]["D"]["list_price_usd"] >= result["costs"]["D"]["new_spend_usd"]
    assert result["revision_histories"]
    replay = tmp_path / "replayed-result"
    assert main(["gd-report", str(batch), "--out", str(replay)]) == 0
    assert (out / "decision.json").read_bytes() == (replay / "decision.json").read_bytes()
    assert main(["gd-report", str(batch), "--out", str(out)]) == 2


def score_and_close(batch: Path, d_score: int, c_score: int, d_originality: int = 4) -> None:
    key = json.loads((batch / "sealed/key.json").read_text())
    d_side = "a" if key["pairs"][0]["a"]["condition"] == "D" else "b"
    c_side = "b" if d_side == "a" else "a"
    fields = {
        "preferred": "A",
        "strength": "3",
        f"score-{d_side}-coherence": str(d_score),
        f"score-{c_side}-coherence": str(c_score),
        f"score-{d_side}-originality": str(d_originality),
        f"score-{c_side}-originality": "4",
    }
    client = reader(batch)
    assert client.post("/pairs/1", data=fields).status_code == 200
    assert client.post("/close").status_code == 200


def test_margin_boundaries_ties_and_originality_have_registered_candidate_direction(
    registered_inputs: dict[str, Path],
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common, _ = prepare_gd(registered_inputs, tmp_path, fake_llm, monkeypatch)
    protocol = Path(common[0])
    raw = json.loads(protocol.read_text())
    raw["coherence"]["margin"] = 1
    protocol.write_text(json.dumps(raw))
    git_fixture(tmp_path, "add", "gd.json")
    git_fixture(tmp_path, "commit", "-q", "-m", "synthetic margin boundary registration")
    template, _, _ = gd_batch(common, tmp_path)
    for name, d_score, c_score, originality, expected in (
        ("boundary", 3, 4, 4, "passed"),
        ("below", 2, 4, 4, "failed"),
        ("ties", 4, 4, 4, "passed"),
        ("originality", 5, 4, 2, "failed"),
    ):
        batch = tmp_path / name
        shutil.copytree(template, batch)
        score_and_close(batch, d_score, c_score, originality)
        out = tmp_path / f"{name}-result"
        assert main(["gd-report", str(batch), "--out", str(out)]) == 0
        result = json.loads((out / "decision.json").read_text())
        assert result["criteria_outcome"] == expected
        assert result["coherence"]["mean_candidate_minus_comparator"] == d_score - c_score
        assert result["outcome"] == "pending"
        assert result["reference_baseline"] is None
        assert not result["coherence"]["inferential_noninferiority"]


def test_stale_missing_reversed_and_open_evidence_remains_pending(
    registered_inputs: dict[str, Path],
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common, _ = prepare_gd(registered_inputs, tmp_path, fake_llm, monkeypatch)
    template, run, _ = gd_batch(common, tmp_path)
    plan = run / "stories/D/brief-01/r1/plan.json"
    original_plan = plan.read_bytes()
    for mutation in ("open", "reversed", "missing_scores", "artifact", "calls"):
        batch = tmp_path / mutation
        shutil.copytree(template, batch)
        if mutation != "open":
            score_and_close(batch, 5, 4)
        if mutation == "reversed":
            path = batch / "sealed/key.json"
            data = json.loads(path.read_text())
            data["sources"].reverse()
            path.write_text(json.dumps(data))
        elif mutation == "missing_scores":
            (batch / "judgements.jsonl").unlink()
        elif mutation == "artifact":
            plan.write_text(plan.read_text() + " ")
        elif mutation == "calls":
            (run / "calls.jsonl").unlink()
        out = tmp_path / f"{mutation}-result"
        assert main(["gd-report", str(batch), "--out", str(out)]) == 0
        result = json.loads((out / "decision.json").read_text())
        assert result["criteria_outcome"] == "pending"
        assert result["outcome"] == "pending"
        assert result["pending"]
        if mutation == "artifact":
            plan.write_bytes(original_plan)


def test_known_failed_control_cannot_be_overridden_by_favourable_owner_scores(
    registered_inputs: dict[str, Path],
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common, _ = prepare_gd(registered_inputs, tmp_path, fake_llm, monkeypatch)
    protocol = Path(common[0])
    raw = json.loads(protocol.read_text())
    raw["control_targets"]["preserve"] = identity(
        ControlTarget(id="preserve", kind="changed", selector="scenes/scene-0")
    )
    protocol.write_text(json.dumps(raw))
    git_fixture(tmp_path, "add", "gd.json")
    git_fixture(tmp_path, "commit", "-q", "-m", "synthetic failed-control registration")
    batch, _, _ = gd_batch(common, tmp_path, target_kind="changed")
    score_and_close(batch, 5, 4)
    out = tmp_path / "control-failure"
    assert main(["gd-report", str(batch), "--out", str(out)]) == 0
    result = json.loads((out / "decision.json").read_text())
    assert result["criteria_outcome"] == "failed"
    assert result["coherence"]["meets_criterion"]
    assert result["controllability"]["brief-01"]["status"] == "failure"
    assert result["reference_baseline"] is None
