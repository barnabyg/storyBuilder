"""Offline G-D acceptance; synthetic fixtures cannot establish a scientific pass."""

import json
from pathlib import Path
from typing import Any

import pytest

from nde import g1_report
from nde.cli import main
from nde.eval.gd import GDInputs, check_gd, write_check
from nde.files import checksum_lf
from nde.narrative.workflow import load_plan
from nde.plan.revision import identity
from tests.conftest import REPO
from tests.eval_helpers import reader
from tests.fake_openai import FakeOpenAI
from tests.gd_fixtures import gd_batch, prepare_gd
from tests.test_g1 import git_fixture
from tests.test_g1_report import synthetic_batch

pytest_plugins = ("tests.test_g1",)


def test_missing_scientific_evidence_blocks_without_generation(tmp_path: Path) -> None:
    inputs = GDInputs(
        protocol=tmp_path / "uncommitted.json",
        config=tmp_path / "config.yaml",
        selection=tmp_path / "selection.json",
        g1_batch=tmp_path / "g1-batch",
        g1_decision=tmp_path / "decision.json",
    )
    result = check_gd(inputs)
    write_check(result, tmp_path / "check")
    assert result.status == "blocked"
    assert result.problems
    assert (tmp_path / "check/gd-check.md").is_file()
    assert not (tmp_path / "runs").exists()


def protocol_data(root: Path) -> dict[str, Any]:
    return {
        "id": "synthetic-test-only-G-D",
        "test_only": True,
        "cohort": {
            "id": "synthetic-test-only",
            "brief_checksums": {"brief-01": "a" * 64},
            "split_checksums": {"dev": "b" * 64},
            "conditions": ["D", "C"],
            "length_tolerance": 0.1,
            "budget_tolerance": 0.1,
        },
        "n": 1,
        "tier": "L",
        "coherence": {"analysis": "mean_paired_difference", "margin": 0.5},
        "originality": {"analysis": "mean_paired_difference", "margin": 0.25},
        "reading_mode": "required_only",
        "g1_decision_sha256": "c" * 64,
        "selection_sha256": checksum_lf((root / "selection.json").read_bytes()),
        "config_sha256": "d" * 64,
        "schema_sha256": "e" * 64,
        "models_sha256": "f" * 64,
        "prompt_sha256": {"plan@1": "a" * 64},
        "control_targets": {"edit": "b" * 64},
        "critic_iterations": 1,
        "generation_limits": {
            "max_repairs": 2,
            "max_length_attempts": 3,
            "max_revision_passes": 4,
            "per_run_usd": 1,
            "per_day_usd": 5,
        },
        "prior_benchmark_exposure": "Synthetic fixture; no scientific owner reading",
    }


@pytest.mark.parametrize("change", ["synthetic", "fake-pass", "failed"])
def test_g1_outcomes_are_recomputed_and_synthetic_cannot_enable_real_generation(
    tmp_path: Path, change: str
) -> None:
    batch = synthetic_batch(tmp_path, 0 if change == "failed" else 30)
    decision = tmp_path / "g1-result"
    g1_report.write_decision(batch, decision, [])
    raw = protocol_data(tmp_path)
    raw["test_only"] = False
    if change == "fake-pass":
        path = decision / "decision.json"
        saved = json.loads(path.read_text())
        saved["outcome"] = "passed"
        path.write_text(json.dumps(saved))
    raw["g1_decision_sha256"] = checksum_lf((decision / "decision.json").read_bytes())
    protocol = tmp_path / "gd.json"
    protocol.write_text(json.dumps(raw))
    git_fixture(tmp_path, "init", "-q")
    git_fixture(tmp_path, "add", "gd.json")
    git_fixture(tmp_path, "commit", "-q", "-m", "synthetic G-D registration")
    result = check_gd(
        GDInputs(
            protocol,
            tmp_path / "missing-config",
            tmp_path / "selection.json",
            batch,
            decision / "decision.json",
        )
    )
    assert result.status == "blocked"
    assert "G1" in " ".join(result.problems)


def test_registered_fake_workflow_seals_controls_and_requires_persisted_scores(
    registered_inputs: dict[str, Path],
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common, _ = prepare_gd(registered_inputs, tmp_path, fake_llm, monkeypatch)
    assert main(["gd", "check", *common, "--out", str(tmp_path / "checked")]) == 0
    batch, run, _ = gd_batch(common, tmp_path)
    client = reader(batch)
    page = client.get("/pairs/1")
    assert page.status_code == 200
    assert all(
        secret not in page.text
        for secret in ("baseline_d", "critic", "planning-provenance", str(run))
    )
    assert client.get("/sealed/gd.json").status_code == 404
    assert client.post("/pairs/1", data={"preferred": "A", "strength": "1"}).status_code == 422
    assert client.post("/close").status_code == 409
    fields = {"preferred": "A", "strength": "2"}
    fields.update(
        {
            f"score-{side}-{name}": "4"
            for side in ("a", "b")
            for name in ("coherence", "originality")
        }
    )
    assert client.post("/pairs/1", data=fields).status_code == 200
    saved = client.get("/api/pairs/1").json()
    assert saved["judgement"]["a"]["scores"] == {"coherence": 4, "originality": 4}
    assert all(
        secret not in str(saved) for secret in ("baseline_d", "critic", "condition", str(run))
    )
    assert client.post("/close").status_code == 200


def test_unrelated_before_plan_cannot_fabricate_gate_controllability(
    registered_inputs: dict[str, Path],
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    common, _ = prepare_gd(registered_inputs, tmp_path, fake_llm, monkeypatch)
    _, run, _ = gd_batch(common, tmp_path)
    request = json.loads((tmp_path / "controls.json").read_text())
    before = json.loads(Path(request["before"]).read_text())
    before["scenes"][2]["summary"] = "An unrelated invented starting intention"
    fabricated = tmp_path / "fabricated-before.json"
    fabricated.write_text(json.dumps(before))
    proof = json.loads(Path(request["revision_provenance"]).read_text())
    proof["before_sha256"] = identity(load_plan(fabricated))
    invented_proof = tmp_path / "invented-provenance.json"
    invented_proof.write_text(json.dumps(proof))
    request.update(before=str(fabricated), revision_provenance=str(invented_proof))
    forged = tmp_path / "fabricated-request.json"
    forged.write_text(json.dumps(request))
    out = tmp_path / "fabricated-control"
    assert main(["control-report", str(forged), "--out", str(out)]) == 0
    saved = out / "control-report.json"
    assert json.loads(saved.read_text())["status"] == "success"
    assert (
        main(
            [
                "gd",
                "batch",
                *common,
                "--name",
                "forged",
                "--batches-dir",
                str(tmp_path / "gd-batches"),
                "--from",
                str(run),
                "D",
                "--from",
                str(run),
                "C",
                "--control",
                "brief-01",
                str(saved),
                "--rubric-dir",
                str(REPO / "benchmark/rubric"),
            ]
        )
        == 2
    )
    assert "saved D operator revision" in capsys.readouterr().err
    assert not (tmp_path / "gd-batches/forged").exists()


@pytest.mark.parametrize(
    "mutation", ["unspecified_margin", "unsupported_analysis", "reversed", "dirty"]
)
def test_registration_rejects_unsupported_or_changed_scientific_choices(
    tmp_path: Path, mutation: str
) -> None:
    (tmp_path / "selection.json").write_text("{}")
    raw = protocol_data(tmp_path)
    if mutation == "unspecified_margin":
        raw["coherence"].pop("margin")
    elif mutation == "unsupported_analysis":
        raw["coherence"]["analysis"] = "nonsignificant_means_equal"
    elif mutation == "reversed":
        raw["cohort"]["conditions"] = ["C", "D"]
    path = tmp_path / "gd.json"
    path.write_text(json.dumps(raw))
    git_fixture(tmp_path, "init", "-q")
    git_fixture(tmp_path, "add", "gd.json")
    git_fixture(tmp_path, "commit", "-q", "-m", "synthetic G-D protocol")
    if mutation == "dirty":
        path.write_text(path.read_text() + "\n")
    out = tmp_path / "check"
    assert (
        main(
            [
                "gd",
                "check",
                str(path),
                str(tmp_path / "config.yaml"),
                "--selection",
                str(tmp_path / "selection.json"),
                "--g1-batch",
                str(tmp_path / "batch"),
                "--g1-decision",
                str(tmp_path / "decision.json"),
                "--out",
                str(out),
            ]
        )
        == 2
    )
    assert json.loads((out / "gd-check.json").read_text())["status"] == "blocked"
    assert not (tmp_path / "runs").exists()
