import json
from pathlib import Path

from nde.cli import main
from nde.corpus.extraction_records import ExtractionOptions
from nde.corpus.families import compare_families
from tests.corpus_support import label_packet, queue_annotation, review_config
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


def test_g2_cannot_approve_pilot_using_a_separate_annotation_run(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    pilot, _, _ = g2_packet(tmp_path, fake_llm)
    saved = json.loads((pilot / "pilot.json").read_text())["config"]
    work = saved["works"][0]
    inputs = {
        "work": Path(work["work_dir"]),
        "segments": Path(work["segmentation"]),
        "cohort": Path(saved["cohort"]),
        "family_config": Path(saved["extraction_config"]),
    }
    queue_annotation(inputs, fake_llm)
    families = tmp_path / "separate-families"
    assert (
        compare_families(
            inputs["family_config"],
            inputs["work"],
            inputs["segments"],
            ExtractionOptions(
                families, tmp_path / "separate-cache.sqlite", tmp_path / "separate-ledger.sqlite"
            ),
        ).status
        == "complete"
    )
    config = review_config(tmp_path, inputs, families, pilot / "features/lamp/features.json")
    review = tmp_path / "separate-review"
    assert main(["corpus-review", "freeze", str(config), "--out", str(review)]) == 0
    config = tmp_path / "separate-g2.json"
    config.write_text(
        json.dumps(
            {
                "pilot": str(pilot),
                "review": str(review),
                "min_stories": 1,
                "min_owner_per_channel": 1,
                "scaling_budget_usd": 1,
            }
        )
    )
    assert (
        main(["corpus-g2", "register", str(config), "--out", str(tmp_path / "rejected.json")]) == 2
    )
