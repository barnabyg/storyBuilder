import json
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from nde.controllability import replay_report
from nde.narrative.workflow import load_plan
from nde.plan.revision import RevisionSpec, identity


def packet(tmp_path: Path) -> Path:
    plan: dict[str, Any] = {
        "schema_version": "1",
        "id": "story",
        "title": "A trajectory",
        "scenes": [{"id": f"s{i}", "title": f"Scene {i}"} for i in range(9)],
        "reader_affect": [
            {
                "id": "hope",
                "reading": {"value": "hope"},
                "trajectory": {
                    "shape": "oscillating",
                    "points": [
                        {"scene_id": f"s{i}", "value": value}
                        for i, value in ((2, 0), (3, 3), (4, 0), (5, 3), (6, 0))
                    ],
                },
            }
        ],
    }
    (tmp_path / "before.json").write_text(json.dumps(plan), encoding="utf-8")
    plan["scenes"][4]["summary"] = "A concrete changed intention"
    (tmp_path / "after.json").write_text(json.dumps(plan), encoding="utf-8")
    value = {
        "before": "before.json",
        "after": "after.json",
        "revision_provenance": "revision.json",
        "revision": {
            "mask": {"units": ["scenes/s4"]},
            "locks": ["scenes/s0"],
            "instruction": "Change the middle scene",
        },
        "targets": [
            {"id": "edit", "kind": "changed", "selector": "scenes/s4/summary"},
            {
                "id": "hope-twice",
                "kind": "renewed_hope_middle",
                "annotation_id": "hope",
                "min_episodes": 2,
            },
        ],
    }
    request = tmp_path / "input.json"
    request.write_text(json.dumps(value), encoding="utf-8")
    return request


def report(tmp_path: Path, request: Path) -> dict[str, Any]:
    value = json.loads(request.read_text())
    if value.get("revision_provenance"):
        # Synthetic operator record for known-answer metric fixtures only.
        proof = {
            "status": "accepted",
            "before_sha256": identity(load_plan(request.parent / value["before"])),
            "after_sha256": identity(load_plan(request.parent / value["after"])),
            "revision": RevisionSpec.model_validate(value["revision"]).model_dump(mode="json"),
        }
        (request.parent / value["revision_provenance"]).write_text(json.dumps(proof))
    out = tmp_path / "report"
    assert main(["control-report", str(request), "--out", str(out)]) == 0
    value = json.loads((out / "control-report.json").read_text(encoding="utf-8"))
    assert "plan-only" in (out / "control-report.md").read_text(encoding="utf-8")
    assert "<!doctype html>" in (out / "control-report.html").read_text(encoding="utf-8")
    return value


def test_report_binds_changes_and_known_ordinal_episodes(tmp_path: Path) -> None:
    request = packet(tmp_path)
    value = report(tmp_path, request)
    assert value["status"] == "success"
    assert value["lock_stable"]
    assert value["authorised_scope"]
    assert value["validator_passed"]
    assert [target["status"] for target in value["targets"]] == ["success", "success"]
    assert value["before_sha256"] != value["after_sha256"]
    assert len(value["targets"][1]["request_sha256"]) == 64
    first = (tmp_path / "report/control-report.json").read_bytes()
    report(tmp_path, request)
    assert first == (tmp_path / "report/control-report.json").read_bytes()


def test_diversity_ignores_renamed_ids_and_exposes_unknown_functions(tmp_path: Path) -> None:
    packet(tmp_path)
    original = json.loads((tmp_path / "after.json").read_text())
    original.pop("reader_affect")
    original["causal_links"] = [{"id": "cause", "cause_scene_id": "s0", "effect_scene_id": "s4"}]
    (tmp_path / "original.json").write_text(json.dumps(original), encoding="utf-8")
    renamed = json.loads(json.dumps(original))
    for scene in renamed["scenes"]:
        scene["id"] = "renamed-" + scene["id"]
    renamed["causal_links"][0]["cause_scene_id"] = "renamed-s0"
    renamed["causal_links"][0]["effect_scene_id"] = "renamed-s4"
    renamed["id"] = "other-story"
    renamed["causal_links"][0]["id"] = "new-cause"
    (tmp_path / "renamed.json").write_text(json.dumps(renamed), encoding="utf-8")
    manifest: dict[str, Any] = {
        "plans": [
            {
                "path": "original.json",
                "seed": 1,
                "functions": {f"s{i}": {"value": "setup"} for i in range(9)},
            },
            {
                "path": "renamed.json",
                "seed": 2,
                "functions": {f"renamed-s{i}": {"value": "setup"} for i in range(9)},
            },
        ]
    }
    request = tmp_path / "diversity.json"
    request.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "diversity"
    assert main(["diversity-report", str(request), "--out", str(out)]) == 0
    result = json.loads((out / "diversity-report.json").read_text())
    assert result["pairs"][0]["function_distance"] == 0
    assert result["pairs"][0]["edge_distance"] == 0
    manifest["plans"][1]["functions"].pop("renamed-s4")
    renamed["causal_links"][0]["effect_scene_id"] = "renamed-s5"
    (tmp_path / "renamed.json").write_text(json.dumps(renamed), encoding="utf-8")
    request.write_text(json.dumps(manifest), encoding="utf-8")
    assert main(["diversity-report", str(request), "--out", str(out)]) == 0
    result = json.loads((out / "diversity-report.json").read_text())
    assert result["pairs"][0]["function_distance"] is None
    assert result["pairs"][0]["edge_distance"] == 1
    assert result["samples"][1]["unknown_functions"] == ["renamed-s4"]


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("locked", "failure"),
        ("outside", "failure"),
        ("noop", "failure"),
        ("missing-point", "unknown"),
        ("one-episode", "failure"),
        ("literary", "unknown"),
    ],
)
def test_reports_failures_and_pending_evidence_honestly(
    tmp_path: Path, change: str, expected: str
) -> None:
    request = packet(tmp_path)
    plan = json.loads((tmp_path / "after.json").read_text())
    value = json.loads(request.read_text())
    if change == "locked":
        plan["scenes"][0]["title"] = "Attack on locked scene"
    elif change == "outside":
        plan["scenes"][8]["title"] = "Outside authorised scope"
    elif change == "noop":
        plan = json.loads((tmp_path / "before.json").read_text())
    elif change == "missing-point":
        plan["reader_affect"][0]["trajectory"]["points"][2]["value"] = None
        value["revision"]["mask"]["channels"] = ["reader_affect"]
    elif change == "one-episode":
        plan["reader_affect"][0]["trajectory"]["points"][3]["value"] = 0
        value["revision"]["mask"]["channels"] = ["reader_affect"]
    elif change == "literary":
        value["targets"].append({"id": "beauty", "kind": "literary"})
    (tmp_path / "after.json").write_text(json.dumps(plan), encoding="utf-8")
    request.write_text(json.dumps(value), encoding="utf-8")
    assert report(tmp_path, request)["status"] == expected


def test_replay_rejects_changed_sources_instead_of_trusting_status(tmp_path: Path) -> None:
    request = packet(tmp_path)
    report(tmp_path, request)
    saved = tmp_path / "report/control-report.json"
    assert replay_report(saved).status == "success"
    plan = json.loads((tmp_path / "after.json").read_text())
    plan["scenes"][4]["summary"] = "Edited since the evidence was saved"
    (tmp_path / "after.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        replay_report(saved)


def test_unbound_plan_difference_cannot_claim_an_accepted_revision(tmp_path: Path) -> None:
    request = packet(tmp_path)
    raw = json.loads(request.read_text())
    raw.pop("revision_provenance")
    request.write_text(json.dumps(raw))
    value = report(tmp_path, request)
    assert value["status"] == "unknown"
    assert not value["accepted_revision"]
