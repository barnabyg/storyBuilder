import json
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from tests.conftest import REPO


def test_state_transitions_and_uncertainty_survive_saved_cards(tmp_path: Path) -> None:
    out = tmp_path / "cards"
    assert (
        main(["plan", "inspect", str(REPO / "examples/harbour-state.plan.json"), "--out", str(out)])
        == 0
    )
    saved: dict[str, Any] = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert saved["states"][3]["after"]["alternatives"] == ["duty", "friendship"]
    assert saved["states"][3]["before"]["status"] == "unknown"
    for name in ("cards.md", "cards.html"):
        text = (out / name).read_text(encoding="utf-8")
        assert "guide the boat home" in text
        assert "suspicious" in text
        assert "allies" in text
        assert "ambiguous" in text
        assert "friendship" in text
        assert "constraint_human_assessment" in text
        assert "hc1: satisfied" in text
    assert main(["plan", "validate", str(out / "plan.json")]) == 0


# Each mutation changes a valid saved plan and requires its intended invariant,
# rather than accepting an unrelated parse failure as a successful catch.
STATE_MUTATIONS = [
    ("unknown_state_character", "states", 0, "character_id", "ghost", "state_character_reference"),
    ("unknown_state_scene", "states", 0, "scene_id", "missing", "state_scene_reference"),
    (
        "state_before_contradiction",
        "states",
        1,
        "before",
        {"value": "abandon the boat"},
        "state_transition",
    ),
    (
        "relationship_unknown_participant",
        "relationships",
        0,
        "participant_ids",
        ["c1", "ghost"],
        "relationship_participant",
    ),
    (
        "relationship_duplicate_participant",
        "relationships",
        0,
        "participant_ids",
        ["c1", "c1"],
        "relationship_participant_duplicate",
    ),
    (
        "relationship_unknown_scene",
        "relationships",
        0,
        "scene_id",
        "missing",
        "relationship_scene_reference",
    ),
    (
        "relationship_before_contradiction",
        "relationships",
        1,
        "before",
        {"value": "enemies"},
        "relationship_transition",
    ),
    ("locked_value_changed", "states", 2, "after", {"value": "protect property"}, "locked_value"),
    (
        "ending_fact_contradicted",
        "states",
        1,
        "after",
        {"value": "abandon the boat"},
        "required_ending",
    ),
    (
        "constraint_unknown_character",
        "constraints",
        0,
        "character_id",
        "ghost",
        "constraint_character_reference",
    ),
    ("constraint_expected_missing", "constraints", 0, "expected", None, "constraint_expected"),
    ("state_character_required", "states", 0, "character_id", None, "state_character_required"),
    ("duplicate_state_transition", "states", 1, "scene_id", "s1", "state_duplicate"),
    (
        "duplicate_relationship_transition",
        "relationships",
        1,
        "scene_id",
        "s1",
        "relationship_duplicate",
    ),
    ("absent_state_character", "scenes", 0, "character_ids", ["c2"], "state_presence"),
    (
        "absent_relationship_participant",
        "scenes",
        0,
        "character_ids",
        ["c1"],
        "relationship_presence",
    ),
    (
        "constraint_character_required",
        "constraints",
        0,
        "character_id",
        None,
        "constraint_character_required",
    ),
]


@pytest.mark.parametrize("mutation", STATE_MUTATIONS, ids=[case[0] for case in STATE_MUTATIONS])
def test_named_state_mutations_receive_intended_diagnostic(
    tmp_path: Path,
    mutation: tuple[str, str, int, str, Any, str],
) -> None:
    category, channel, index, field, value, code = mutation
    data: dict[str, Any] = json.loads((REPO / "examples/harbour-state.plan.json").read_text())
    data[channel][index][field] = value
    file = tmp_path / f"{category}.json"
    file.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "cards"
    assert main(["plan", "inspect", str(file), "--out", str(out)]) == 1
    diagnostics = json.loads((out / "validation.json").read_text())["diagnostics"]
    assert any(item["code"] == code and item["severity"] == "error" for item in diagnostics)
    assert code in (out / "cards.md").read_text()
    assert code in (out / "cards.html").read_text()


def test_absent_characters_receive_state_and_relationship_diagnostics(tmp_path: Path) -> None:
    data: dict[str, Any] = json.loads((REPO / "examples/harbour-state.plan.json").read_text())
    data["scenes"][0]["character_ids"] = ["c2"]
    file = tmp_path / "absent.json"
    file.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "cards"
    assert main(["plan", "inspect", str(file), "--out", str(out)]) == 1
    diagnostics = json.loads((out / "validation.json").read_text())["diagnostics"]
    assert {"state_presence", "relationship_presence"} <= {item["code"] for item in diagnostics}
    assert any(
        "c1" in item["location"] and "s1" in item["location"]
        for item in diagnostics
        if item["code"] == "state_presence"
    )


def test_unknown_constraints_stay_unresolved_and_literary_questions_stay_human(
    tmp_path: Path,
) -> None:
    data: dict[str, Any] = json.loads((REPO / "examples/harbour-state.plan.json").read_text())
    data["states"][2]["after"] = {"status": "unknown"}
    data["constraints"][1]["key"] = "unrepresented ending"
    file = tmp_path / "unknown.json"
    file.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "cards"
    assert main(["plan", "inspect", str(file), "--out", str(out)]) == 0
    diagnostics = json.loads((out / "validation.json").read_text())["diagnostics"]
    assert {"constraint_unresolved", "constraint_fact_missing", "constraint_human_assessment"} <= {
        item["code"] for item in diagnostics if item["severity"] == "human"
    }
    assert "hc1: satisfied" not in (out / "cards.md").read_text()


@pytest.mark.parametrize(
    "reading",
    [
        {"value": "safe", "confidence": 1.1},
        {"value": "safe", "status": "unknown"},
        {"value": None, "status": "known"},
    ],
)
def test_malformed_readings_are_rejected(tmp_path: Path, reading: dict[str, Any]) -> None:
    data: dict[str, Any] = json.loads((REPO / "examples/harbour-state.plan.json").read_text())
    data["states"][0]["before"] = reading
    file = tmp_path / "bad-reading.json"
    file.write_text(json.dumps(data), encoding="utf-8")
    assert main(["plan", "validate", str(file)]) == 2


def test_transition_storage_order_does_not_override_scene_order(tmp_path: Path) -> None:
    data: dict[str, Any] = json.loads((REPO / "examples/harbour-state.plan.json").read_text())
    data["states"].reverse()
    data["relationships"].reverse()
    file = tmp_path / "reordered-records.json"
    file.write_text(json.dumps(data), encoding="utf-8")
    assert main(["plan", "validate", str(file)]) == 0
