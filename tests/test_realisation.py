import json
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from nde.files import write_text_lf
from nde.runs.records import StoryRecord, write_jsonl
from tests.conftest import REPO


def packet_source(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    folder = run / "stories/C/brief-01/r1"
    folder.mkdir(parents=True)
    (folder / "attempts/1").mkdir(parents=True)
    write_text_lf(folder / "story.md", "The harbour was quiet.\n")
    write_text_lf(folder / "story.raw.md", "The harbour was quiet.\n")
    write_text_lf(folder / "attempts/1/scene-01.md", "The harbour was quiet.\n")
    write_text_lf(folder / "plan.json", (REPO / "examples/harbour-state.plan.json").read_text())
    write_jsonl(
        run / "stories.jsonl",
        [
            StoryRecord(
                brief="brief-01",
                condition="C",
                pipeline="baseline_c",
                repetition=0,
                seed=7,
                status="ok",
                path="stories/C/brief-01/r1",
                chosen_attempt=0,
            )
        ],
    )
    return run


def create_packet(run: Path, out: Path) -> int:
    return main(
        [
            "realisation",
            "create",
            str(run),
            "--condition",
            "C",
            "--brief",
            "brief-01",
            "--out",
            str(out),
        ]
    )


def test_realisation_packet_binds_final_prose_and_reports_missing_owner_evidence(
    tmp_path: Path,
) -> None:
    run = packet_source(tmp_path)
    out = tmp_path / "packet"
    assert create_packet(run, out) == 0
    packet = json.loads((out / "packet.json").read_text())
    assert packet["story_sha256"]
    assert packet["items"]
    assert main(["realisation", "report", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["owner_status"] == "pending"
    assert report["assessed"] == 0
    assert report["denominator"] == len(packet["items"])
    assert report["missing"] == report["denominator"]


def test_changed_selected_prose_invalidates_assessments_without_losing_history(
    tmp_path: Path,
) -> None:
    run = packet_source(tmp_path)
    out = tmp_path / "packet"
    assert create_packet(run, out) == 0
    packet = json.loads((out / "packet.json").read_text())
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "packet_sha256": packet["packet_sha256"],
                "reviewer": "synthetic fixture",
                "test_only": True,
                "assessments": [
                    {
                        "item_id": packet["items"][0]["id"],
                        "rating": "unknown",
                        "note": "test-only assessment",
                    }
                ],
            }
        )
    )
    assert main(["realisation", "assess", str(out), str(review)]) == 0
    write_text_lf(run / "stories/C/brief-01/r1/story.md", "A different final story.\n")
    assert main(["realisation", "report", str(out)]) == 1
    report = json.loads((out / "report.json").read_text())
    assert report["stale"]
    assert report["assessed"] == 0
    assert len(list((out / "history").glob("*.json"))) == 1


@pytest.mark.parametrize("change", ["revision", "plan", "span", "identity"])
def test_evidence_rejects_stale_sources_and_spans(tmp_path: Path, change: str) -> None:
    run = packet_source(tmp_path)
    out = tmp_path / "packet"
    if change == "revision":
        rows = json.loads((run / "stories.jsonl").read_text())
        rows["revision_passes"] = 1
        (run / "stories.jsonl").write_text(json.dumps(rows) + "\n")
    assert create_packet(run, out) == 0
    packet = json.loads((out / "packet.json").read_text())
    assert packet["alignment"] == "unresolved"
    review: dict[str, Any] = {
        "packet_sha256": packet["packet_sha256"],
        "reviewer": "synthetic fixture",
        "test_only": True,
        "assessments": [
            {"item_id": packet["items"][0]["id"], "rating": "realised", "note": "test-only"}
        ],
    }
    if change == "plan":
        plan_file = run / "stories/C/brief-01/r1/plan.json"
        plan = json.loads(plan_file.read_text())
        plan["title"] = "Changed plan"
        plan_file.write_text(json.dumps(plan))
    elif change == "span":
        review["assessments"][0].update(start=0, end=5, quote="stale")
    elif change == "identity":
        review["packet_sha256"] = "wrong"
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    expected = 0 if change == "revision" else 2
    assert main(["realisation", "assess", str(out), str(path)]) == expected
    assert main(["realisation", "report", str(out)]) == (1 if change == "plan" else 0)
    report = json.loads((out / "report.json").read_text())
    assert report["owner_status"] == "pending"
    assert report["assessed"] == 0


def test_baseline_b_packet_records_absent_channels_without_inventing_deltas(tmp_path: Path) -> None:
    run = packet_source(tmp_path)
    folder = run / "stories/C/brief-01/r1"
    (folder / "plan.json").unlink()
    record = json.loads((run / "stories.jsonl").read_text())
    record["pipeline"] = "baseline_b"
    (run / "stories.jsonl").write_text(json.dumps(record) + "\n")
    (folder / "outline.json").write_text(
        json.dumps(
            {
                "title": "Synthetic beat-only plan",
                "scenes": [
                    {
                        "summary": "The keeper waits",
                        "beats": ["A quiet harbour"],
                    }
                ],
            }
        )
    )
    out = tmp_path / "packet"
    assert create_packet(run, out) == 0
    packet = json.loads((out / "packet.json").read_text())
    assert packet["plan_kind"] == "beat_outline"
    assert packet["items"] == []
    assert packet["alignment"] == "scene-aligned"
    assert main(["realisation", "report", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["denominator"] == 0
    assert report["owner_status"] == "pending"


@pytest.mark.parametrize("identity", ["recorded_checksum", "legacy_raw"])
def test_packet_creation_refuses_prose_that_the_run_did_not_select(
    tmp_path: Path, identity: str
) -> None:
    run = packet_source(tmp_path)
    if identity == "recorded_checksum":
        record = json.loads((run / "stories.jsonl").read_text())
        record["sha256"] = "0" * 64
        (run / "stories.jsonl").write_text(json.dumps(record) + "\n")
    else:
        write_text_lf(
            run / "stories/C/brief-01/r1/story.md", "A superseded draft was substituted.\n"
        )
    out = tmp_path / "packet"
    assert create_packet(run, out) == 2
    assert not out.exists()


@pytest.mark.parametrize(
    "provenance", [{}, {"model": "test-model"}, {"model": " ", "prompt": "diagnostic@1"}]
)
def test_diagnostic_estimates_require_model_and_prompt_identities(
    tmp_path: Path, provenance: dict[str, str]
) -> None:
    run = packet_source(tmp_path)
    out = tmp_path / "packet"
    assert create_packet(run, out) == 0
    packet = json.loads((out / "packet.json").read_text())
    review = tmp_path / "diagnostic.json"
    review.write_text(
        json.dumps(
            {
                "packet_sha256": packet["packet_sha256"],
                "reviewer": "synthetic model",
                "test_only": True,
                "diagnostic_provenance": provenance,
                "assessments": [
                    {
                        "item_id": packet["items"][0]["id"],
                        "rating": "unknown",
                        "note": "Synthetic diagnostic only",
                    }
                ],
            }
        )
    )
    assert main(["realisation", "assess", str(out), str(review)]) == 2
    assert not (out / "history").exists()


def test_identified_diagnostics_stay_separate_and_legacy_estimates_remain_readable(
    tmp_path: Path,
) -> None:
    run = packet_source(tmp_path)
    out = tmp_path / "packet"
    assert create_packet(run, out) == 0
    packet = json.loads((out / "packet.json").read_text())
    data = {
        "packet_sha256": packet["packet_sha256"],
        "reviewer": "synthetic model",
        "test_only": True,
        "diagnostic_provenance": {"model": "fake/model-v1", "prompt": "review@1"},
        "assessments": [
            {
                "item_id": packet["items"][0]["id"],
                "rating": "realised",
                "note": "Synthetic diagnostic only",
            }
        ],
    }
    file = tmp_path / "diagnostic.json"
    file.write_text(json.dumps(data))
    assert main(["realisation", "assess", str(out), str(file)]) == 0
    data["diagnostic_provenance"] = {}
    (out / "history/000002-legacy.json").write_text(json.dumps(data))
    assert main(["realisation", "report", str(out)]) == 0
    result = json.loads((out / "report.json").read_text())
    assert result["diagnostic_only_estimates"] == 1
    assert result["unresolved_diagnostic_estimates"] == 1
    assert result["assessed"] == 0
    assert result["owner_status"] == "pending"
