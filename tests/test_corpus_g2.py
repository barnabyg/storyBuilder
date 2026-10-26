import json
from pathlib import Path

from nde.cli import main
from tests.corpus_support import label_packet
from tests.fake_openai import FakeOpenAI
from tests.g2_support import approved_export, g2_packet


def test_synthetic_criteria_pass_roundtrips_export_but_cannot_authorise_real_scaling(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    dataset = approved_export(tmp_path, fake_llm)
    data = json.loads(dataset.read_text())
    assert data["test_only"] is True
    assert len(data["records"]) == 2
    assert data["target_policy"]["goal"] == "hard"
    assert data["target_policy"]["interpretation"] == "excluded"
    assert all(row["split"] == "train" for row in data["records"])
    assert main(["corpus-export", "inspect", str(dataset)]) == 0
    gate = tmp_path / "gate"
    decision = json.loads((gate / "decision.json").read_text())
    assert decision["criteria_outcome"] == "passed"
    assert decision["outcome"] == "pending"
    assert (
        main(
            [
                "corpus-g2",
                "scale",
                str(tmp_path / "pilot-config.json"),
                "--gate",
                str(gate),
                "--out",
                str(tmp_path / "expansion"),
            ]
        )
        == 2
    )
    assert fake_llm.call_count == 6
    plan = Path(data["records"][0]["annotation_path"])
    plan.write_text("{}")
    assert main(["corpus-export", "inspect", str(dataset)]) == 2


def test_g2_pending_and_core_failure_are_explicit(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    _, review, protocol = g2_packet(tmp_path, fake_llm)
    pending = tmp_path / "pending"
    assert main(["corpus-g2", "report", str(protocol), "--out", str(pending)]) == 0
    assert json.loads((pending / "decision.json").read_text())["criteria_outcome"] == "pending"
    label_packet(tmp_path, review, judgement="not_defensible")
    failed = tmp_path / "failed"
    assert main(["corpus-g2", "report", str(protocol), "--out", str(failed)]) == 0
    decision = json.loads((failed / "decision.json").read_text())
    assert decision["criteria_outcome"] == "failed"
    assert "stop before training" in decision["action"]
    assert (
        main(
            [
                "corpus-export",
                "create",
                str(failed),
                "--out",
                str(tmp_path / "rejected"),
                "--test-only",
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "corpus-g2",
                "register",
                str(tmp_path / "g2-config.json"),
                "--out",
                str(tmp_path / "late-protocol.json"),
            ]
        )
        == 2
    )
