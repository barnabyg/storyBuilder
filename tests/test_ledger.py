import json
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from tests.conftest import REPO


def test_replay_writes_ordered_planned_states_without_changing_source(tmp_path: Path) -> None:
    source = REPO / "examples/harbour-state.plan.json"
    original = source.read_bytes()
    out = tmp_path / "ledger"
    assert main(["plan", "replay", str(source), "--out", str(out)]) == 0
    data: dict[str, Any] = json.loads((out / "ledger.json").read_text())
    assert data["status"] == "planned-only"
    assert data["plan_id"] == "harbour-state"
    assert len(data["plan_sha256"]) == 64
    assert [scene["scene_id"] for scene in data["scenes"]] == ["s1", "s2"]
    first, last = data["scenes"]
    assert first["before"]["states"][0]["reading"]["value"] == "protect the lamp"
    assert first["after"]["states"][0]["reading"]["value"] == "save the boat"
    assert last["before"]["states"][0]["reading"]["value"] == "save the boat"
    assert last["after"]["states"][0]["reading"]["value"] == "guide the boat home"
    assert "allies" not in first["summary"]
    assert "cooperating" in last["summary"]
    assert "friendship" in last["after_summary"]
    assert "Requirement hc1 (locked; c1 value): oath = protect lives" in first["summary"]
    assert "planned" in (out / "ledger.md").read_text().lower()
    assert source.read_bytes() == original
    saved = (out / "ledger.json").read_bytes()
    assert main(["plan", "replay", str(source), "--out", str(out)]) == 0
    assert (out / "ledger.json").read_bytes() == saved


def test_replay_times_knowledge_setups_and_motifs_by_position(tmp_path: Path) -> None:
    data: dict[str, Any] = json.loads((REPO / "examples/harbour-state.plan.json").read_text())
    data["knowledge_facts"] = [
        {"id": "fact", "description": "The lamp needs oil", "initial_holder_ids": ["c2"]},
        {"id": "secret", "description": "The boat carries a letter"},
    ]
    data["reveal_links"] = [
        {"id": "rev1", "fact_id": "fact", "scene_id": "s1", "holder_ids": ["c1"]},
        {"id": "rev2", "fact_id": "secret", "scene_id": "s2", "holder_ids": ["c1", "c2"]},
    ]
    data["knowledge_uses"] = [
        {"id": "use", "fact_id": "fact", "scene_id": "s2", "holder_id": "c1"},
    ]
    data["setups"] = [{"id": "setup", "scene_id": "s1", "description": "An empty oil can"}]
    data["payoff_links"] = [{"id": "paid", "setup_id": "setup", "payoff_scene_id": "s2"}]
    data["motifs"] = [
        {"id": "motif1", "scene_ids": ["s1"], "reading": {"value": "darkness"}},
        {
            "id": "motif2",
            "scene_ids": ["s2"],
            "reading": {"status": "ambiguous", "alternatives": ["hope", "danger"]},
        },
    ]
    # ID spelling is intentionally opposite to chronological order.
    for scene, name in zip(data["scenes"], ["z-first", "a-last"], strict=True):
        old = scene["id"]
        scene["id"] = name
        for channel in ("states", "relationships", "reveal_links", "knowledge_uses", "setups"):
            for item in data[channel]:
                if item["scene_id"] == old:
                    item["scene_id"] = name
        for item in data["payoff_links"]:
            if item["payoff_scene_id"] == old:
                item["payoff_scene_id"] = name
        for item in data["motifs"]:
            item["scene_ids"] = [name if value == old else value for value in item["scene_ids"]]
    data["states"].reverse()
    data["reveal_links"].reverse()
    file = tmp_path / "plan.json"
    file.write_text(json.dumps(data))
    out = tmp_path / "ledger"
    assert main(["plan", "replay", str(file), "--out", str(out)]) == 0
    ledger: dict[str, Any] = json.loads((out / "ledger.json").read_text())
    first, last = ledger["scenes"]
    assert first["before"]["knowledge"] == [
        {"fact_id": "fact", "description": "The lamp needs oil", "holder_ids": ["c2"]},
    ]
    assert first["after"]["knowledge"][0]["holder_ids"] == ["c1", "c2"]
    assert last["before"]["knowledge"] == first["after"]["knowledge"]
    assert last["after"]["knowledge"][1]["holder_ids"] == ["c1", "c2"]
    assert first["before"]["open_setups"] == []
    assert last["before"]["open_setups"][0]["id"] == "setup"
    assert last["after"]["open_setups"] == []
    assert first["before"]["motifs"] == []
    assert last["before"]["motifs"][0]["id"] == "motif1"
    assert last["after"]["motifs"][1]["reading"]["status"] == "ambiguous"
    assert first["knowledge_uses"] == []
    assert last["knowledge_uses"][0]["holder_id"] == "c1"
    assert "letter" not in first["summary"]
    assert "letter" not in last["summary"]
    assert "hope" in last["after_summary"]


@pytest.mark.parametrize("tier", ["S", "M", "L"])
def test_replay_respects_sparse_tiers_and_keeps_unknown_readings(tmp_path: Path, tier: str) -> None:
    out = tmp_path / "projection"
    assert (
        main(
            [
                "plan",
                "project",
                str(REPO / "examples/harbour-state.plan.json"),
                "--tier",
                tier,
                "--out",
                str(out),
            ]
        )
        == 0
    )
    ledger_out = tmp_path / "ledger"
    assert main(["plan", "replay", str(out / "plan.json"), "--out", str(ledger_out)]) == 0
    data: dict[str, Any] = json.loads((ledger_out / "ledger.json").read_text())
    text = (ledger_out / "ledger.md").read_text()
    assert data["tier"] == tier
    if tier == "S":
        assert data["scenes"][0]["before"]["states"] == []
        assert "protect the lamp" not in text
    if tier in {"S", "M"}:
        assert data["constraints"] == []
        assert "protect lives" not in text
        assert "cooperating" not in text
    if tier in {"M", "L"}:
        belief = data["scenes"][1]["before"]["states"][-1]["reading"]
        assert belief["status"] == "unknown"
        assert belief["value"] is None
        assert "friendship" in text


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("constraint", "locked_value"),
        ("reference", "state_scene_reference"),
    ],
)
def test_invalid_plan_writes_diagnostics_without_a_valid_ledger(
    tmp_path: Path,
    mutation: str,
    code: str,
) -> None:
    data: dict[str, Any] = json.loads((REPO / "examples/harbour-state.plan.json").read_text())
    if mutation == "constraint":
        data["states"][2]["after"] = {"value": "protect property"}
    else:
        data["states"][0]["scene_id"] = "missing"
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(data))
    out = tmp_path / "ledger"
    assert main(["plan", "replay", str(file), "--out", str(out)]) == 1
    result: dict[str, Any] = json.loads((out / "validation.json").read_text())
    assert any(item["code"] == code for item in result["diagnostics"])
    assert not (out / "ledger.json").exists()


def test_current_scene_uncertainty_is_not_replaced_with_prior_certainty(tmp_path: Path) -> None:
    data: dict[str, Any] = json.loads((REPO / "examples/harbour-state.plan.json").read_text())
    data["states"][1]["before"] = {"status": "ambiguous", "alternatives": ["save", "escape"]}
    file = tmp_path / "uncertain.json"
    file.write_text(json.dumps(data))
    out = tmp_path / "ledger"
    assert main(["plan", "replay", str(file), "--out", str(out)]) == 0
    ledger: dict[str, Any] = json.loads((out / "ledger.json").read_text())
    reading = ledger["scenes"][1]["before"]["states"][0]["reading"]
    assert reading["status"] == "ambiguous"
    assert reading["value"] is None
    assert reading["alternatives"] == ["save", "escape"]
