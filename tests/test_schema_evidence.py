import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nde.annotations.drafting import checked_source
from nde.annotations.records import AnnotationConfig, Attempt, Session, digest
from nde.cli import main
from nde.narrative.models import NarrativePlan
from nde.narrative.workflow import schema_json
from tests.conftest import REPO


def test_public_report_replays_validator_evidence_without_inventing_owner_review(
    tmp_path: Path,
) -> None:
    out = tmp_path / "evidence"
    code = main(
        [
            "schema-evidence",
            "--manifest",
            str(REPO / "data/manifest.calibration.yaml"),
            "--annotations",
            str(tmp_path / "annotations"),
            "--out",
            str(out),
        ]
    )
    assert code == 3
    report = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert report["mechanical_status"] == "passed"
    assert report["gate_status"] == "pending"
    assert report["mutations"]["total"] >= 20
    assert report["mutations"]["catch_rate"] >= 0.95
    assert len(report["stories"]) == 3
    for filename in ("evidence.md", "evidence.html"):
        content = (out / filename).read_text(encoding="utf-8")
        assert "pending" in content
        assert "Owl Creek" in content
        assert "owner" in content
    replay = tmp_path / "replayed"
    assert (
        main(["schema-evidence", "--replay", str(out / "inputs.json"), "--out", str(replay)]) == 3
    )
    assert (replay / "evidence.json").read_bytes() == (out / "evidence.json").read_bytes()
    assert (replay / "evidence.md").read_bytes() == (out / "evidence.md").read_bytes()


def test_missed_expected_diagnostics_and_unrelated_parse_failures_fail_mechanical_gate(
    tmp_path: Path,
) -> None:
    out = tmp_path / "evidence"
    main(
        [
            "schema-evidence",
            "--manifest",
            str(REPO / "data/manifest.calibration.yaml"),
            "--out",
            str(out),
        ]
    )
    inputs = json.loads((out / "inputs.json").read_text(encoding="utf-8"))
    for case in inputs["suite"]["mutations"][:3]:
        case["expected_code"] = "wrong_validator"
    inputs["suite"]["mutations"][0]["value"] = 123
    replay = tmp_path / "missed.json"
    replay.write_text(json.dumps(inputs), encoding="utf-8")
    assert main(["schema-evidence", "--replay", str(replay), "--out", str(out)]) == 1
    report = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert report["mechanical_status"] == "failed"
    assert report["gate_status"] == "failed"
    assert report["mutations"]["results"][0]["caught"] is False
    assert report["mutations"]["results"][0]["parse_error"]
    assert report["mutations"]["catch_rate"] < 0.95


def test_malformed_verified_source_still_produces_failure_packet(tmp_path: Path) -> None:
    manifest = _sources(tmp_path)
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw = b"Checksum matches, but extraction markers are missing."
    (tmp_path / "calibration/synthetic-0.source.txt").write_bytes(raw)
    data["stories"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
    out = tmp_path / "evidence"
    assert main(["schema-evidence", "--manifest", str(manifest), "--out", str(out)]) == 1
    report = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert report["stories"][0]["source_status"] == "failed"
    assert "source extraction failed" in (out / "evidence.md").read_text(encoding="utf-8")


def test_inflight_saved_call_cannot_pass_even_with_prior_test_only_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _sources(tmp_path)
    annotations = _sessions(tmp_path, manifest)
    _record_test_only_review(annotations, monkeypatch)
    file = annotations / "synthetic-0/session.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    data["pending_call"] = True
    file.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "evidence"
    code = main(
        [
            "schema-evidence",
            "--manifest",
            str(manifest),
            "--annotations",
            str(annotations),
            "--out",
            str(out),
        ]
    )
    assert code in (1, 3)
    report = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert report["gate_status"] != "passed"


@pytest.mark.parametrize(
    ("scenario", "expected_code", "expected_status"),
    [
        ("new_unreviewed_revision", 3, "pending"),
        ("missing_time", 3, "pending"),
        ("active_time", 3, "pending"),
        ("human_diagnostic", 3, "pending"),
        ("changed_revision", 1, "failed"),
        ("changed_source", 1, "failed"),
        ("corrupt_timer", 1, "failed"),
        ("owner_rejects_trajectories", 1, "failed"),
    ],
)
def test_incomplete_or_corrupted_saved_evidence_never_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    expected_code: int,
    expected_status: str,
) -> None:
    manifest = _sources(tmp_path)
    annotations = _sessions(tmp_path, manifest)
    _record_test_only_review(annotations, monkeypatch)
    directory = annotations / "synthetic-0"
    if scenario in {"new_unreviewed_revision", "human_diagnostic"}:
        file = directory / "draft/plan.json"
        data = json.loads(file.read_text(encoding="utf-8"))
        if scenario == "human_diagnostic":
            data["constraints"] = [{"id": "hc1", "kind": "literary", "key": "earned ending"}]
        revision = directory / "next.json"
        revision.write_text(json.dumps(data), encoding="utf-8")
        assert main(["annotate", "revise", str(directory), str(revision)]) == 0
        if scenario == "human_diagnostic":
            assert (
                main(
                    [
                        "annotate",
                        "review",
                        str(directory),
                        "2",
                        "--owner",
                        "TEST-ONLY",
                        "--notes",
                        "TEST ONLY: synthetic review.",
                        "--accept",
                        "--meaningful-trajectories",
                    ]
                )
                == 0
            )
    elif scenario == "missing_time":
        (directory / "timer.jsonl").unlink()
    elif scenario == "active_time":
        assert main(["annotate", "timer", str(directory), "start"]) == 0
    elif scenario == "changed_revision":
        (directory / "revisions/0001/submitted.json").write_text("{}", encoding="utf-8")
    elif scenario == "changed_source":
        (tmp_path / "calibration/synthetic-0.source.txt").write_bytes(b"changed")
    elif scenario == "corrupt_timer":
        (directory / "timer.jsonl").write_text(
            '{"action":"start","at":0}\n{"action":"pause","at":10,"seconds":999}\n',
            encoding="utf-8",
        )
    else:
        assert (
            main(
                [
                    "annotate",
                    "review",
                    str(directory),
                    "1",
                    "--owner",
                    "TEST-ONLY",
                    "--notes",
                    "TEST ONLY: not meaningful.",
                    "--accept",
                ]
            )
            == 0
        )
    out = tmp_path / "evidence"
    assert (
        main(
            [
                "schema-evidence",
                "--manifest",
                str(manifest),
                "--annotations",
                str(annotations),
                "--out",
                str(out),
            ]
        )
        == expected_code
    )
    report = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert report["gate_status"] == expected_status


def test_measured_correction_above_target_fails_owner_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _sources(tmp_path)
    annotations = _sessions(tmp_path, manifest)
    _record_test_only_review(annotations, monkeypatch, seconds=7201)
    out = tmp_path / "evidence"
    assert (
        main(
            [
                "schema-evidence",
                "--manifest",
                str(manifest),
                "--annotations",
                str(annotations),
                "--out",
                str(out),
            ]
        )
        == 1
    )
    report = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert report["mechanical_status"] == "passed"
    assert report["stories"][0]["correction_seconds"] == 7201
    assert any("two-hour target" in failure for failure in report["failures"])


def _sources(directory: Path) -> Path:
    records: list[dict[str, Any]] = []
    sources = directory / "calibration"
    sources.mkdir()
    for index in range(3):
        name = f"synthetic-{index}"
        raw = ("START\n\n" + "The keeper waited by the shore. " * 400 + "\n\nEND\n").encode()
        (sources / f"{name}.source.txt").write_bytes(raw)
        records.append(
            {
                "id": name,
                "title": f"Synthetic test story {index}",
                "author": "Test fixture",
                "source": f"fixture://{name}",
                "edition": "Synthetic test-only edition",
                "licence_basis": "Synthetic test-only source, no owner evidence",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "start": "START",
                "end": "END",
            }
        )
    manifest = directory / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"stories": records}), encoding="utf-8")
    return manifest


def _sessions(directory: Path, manifest: Path) -> Path:
    annotations = directory / "annotations"
    for index in range(3):
        story_id = f"synthetic-{index}"
        source, provenance = checked_source(manifest, story_id)
        config = AnnotationConfig.model_validate(
            {
                "providers": {"fake": {"base_url": "http://127.0.0.1"}},
                "annotator": {"provider": "fake", "model": "test-only"},
                "prices": {"fake/test-only": {"input_per_mtok": 0, "output_per_mtok": 0}},
            }
        )
        points = [0, 2, 4] if index == 0 else [4, 2, 0]
        plan = NarrativePlan.model_validate(
            {
                "schema_version": "1",
                "id": story_id,
                "title": "Synthetic test-only annotation",
                "sources": [source.model_dump()],
                "scenes": [{"id": f"s{n}", "title": f"Scene {n}"} for n in range(3)],
                "tension": [
                    {
                        "id": "t1",
                        "reading": {"value": "Synthetic reading"},
                        "trajectory": {
                            "shape": "rising" if index == 0 else "falling",
                            "points": [{"scene_id": f"s{n}", "value": points[n]} for n in range(3)],
                        },
                    }
                ],
            }
        )
        text = plan.model_dump_json(indent=2)
        session = Session(
            id=f"test-only-session-{index}",
            story_id=story_id,
            status="drafted",
            source=source,
            provenance=provenance,
            config=config,
            config_sha256=digest(config.model_dump_json()),
            schema_sha256=digest(schema_json()),
            prompt_sha256="test-only-prompt",
            git_sha="test-only-code",
            attempts=[Attempt(text=text, errors=[], call={"cost_usd": 0})],
        )
        folder = annotations / story_id
        (folder / "draft").mkdir(parents=True)
        (folder / "session.json").write_text(session.model_dump_json(), encoding="utf-8")
        (folder / "original.txt").write_text(text, encoding="utf-8")
        (folder / "draft/plan.json").write_text(text, encoding="utf-8")
        assert main(["annotate", "revise", str(folder), str(folder / "draft/plan.json")]) == 0
    return annotations


def test_saved_corrections_show_distinct_trajectories_but_need_actual_time_and_owner_review(
    tmp_path: Path,
) -> None:
    manifest = _sources(tmp_path)
    annotations = _sessions(tmp_path, manifest)
    out = tmp_path / "evidence"
    assert (
        main(
            [
                "schema-evidence",
                "--manifest",
                str(manifest),
                "--annotations",
                str(annotations),
                "--out",
                str(out),
            ]
        )
        == 3
    )
    report = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert report["mechanical_status"] == "passed"
    assert report["comparisons"][0]["distinct"] is True
    for row in report["stories"]:
        assert row["word_count"] == 2400
        assert row["revision"] == 1
        assert row["owner_review"] == "pending"
        assert row["correction_seconds"] is None
    for name in ("evidence.md", "evidence.html"):
        content = (out / name).read_text(encoding="utf-8")
        assert "0 → 2 → 4" in content
        assert "4 → 2 → 0" in content
        assert "explicit owner literary review missing" in content


def _record_test_only_review(
    annotations: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    accept: bool = True,
    seconds: float = 120,
) -> None:
    """Exercise explicit review commands with synthetic evidence only, in a temporary tree."""
    for directory in sorted(annotations.iterdir()):
        monkeypatch.setattr("nde.annotations.corrections.time.time", lambda: 1000.0)
        assert main(["annotate", "timer", str(directory), "start"]) == 0
        monkeypatch.setattr("nde.annotations.corrections.time.time", lambda: 1000.0 + seconds)
        assert main(["annotate", "timer", str(directory), "pause"]) == 0
        options = ["--accept"] if accept else []
        assert (
            main(
                [
                    "annotate",
                    "review",
                    str(directory),
                    "1",
                    "--owner",
                    "TEST-FIXTURE-ONLY",
                    "--notes",
                    "TEST ONLY: synthetic review; no actual owner or real story was reviewed.",
                    "--meaningful-trajectories",
                    *options,
                ]
            )
            == 0
        )


def test_synthetic_explicit_review_can_complete_gate_but_rejection_does_not_fail_validators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _sources(tmp_path)
    annotations = _sessions(tmp_path, manifest)
    _record_test_only_review(annotations, monkeypatch)
    out = tmp_path / "evidence"
    args = [
        "schema-evidence",
        "--manifest",
        str(manifest),
        "--annotations",
        str(annotations),
        "--out",
        str(out),
    ]
    assert main(args) == 0
    assert (
        main(
            [
                "annotate",
                "review",
                str(annotations / "synthetic-0"),
                "1",
                "--owner",
                "TEST-FIXTURE-ONLY",
                "--notes",
                "TEST ONLY: rejection.",
                "--meaningful-trajectories",
            ]
        )
        == 0
    )
    assert main(args) == 1
    report = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert report["mechanical_status"] == "passed"
    assert report["gate_status"] == "failed"
