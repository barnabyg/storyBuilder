import json
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from tests.conftest import REPO


def complete_plan(tmp_path: Path) -> Path:
    data: dict[str, Any] = json.loads(
        (REPO / "tests/fixtures/narrative-structure.json").read_text()
    )
    state: dict[str, Any] = json.loads((REPO / "examples/harbour-state.plan.json").read_text())
    interpretive: dict[str, Any] = json.loads(
        (REPO / "examples/trajectory-rising.plan.json").read_text()
    )
    for channel in ("states", "relationships", "constraints"):
        data[channel] = state[channel]
    for channel in (
        "arcs",
        "reader_affect",
        "tension",
        "themes",
        "motifs",
        "motif_echoes",
        "discourse",
        "subtext",
        "sources",
    ):
        if channel in interpretive:
            data[channel] = interpretive[channel]
    data["scenes"][0]["beats"] = [{"id": "b1", "summary": "A boat approaches."}]
    file = tmp_path / "complete.json"
    file.write_text(json.dumps(data), encoding="utf-8")
    assert main(["plan", "validate", str(file)]) == 0
    return file


@pytest.mark.parametrize("tier", ["S", "M", "L"])
def test_file_projection_has_exact_membership_and_is_repeatable(tmp_path: Path, tier: str) -> None:
    file = complete_plan(tmp_path)
    original = file.read_bytes()
    out = tmp_path / tier
    assert main(["plan", "project", str(file), "--tier", tier, "--out", str(out)]) == 0
    projected = out / "plan.json"
    assert main(["plan", "validate", str(projected)]) == 0
    data = json.loads(projected.read_text())
    cards = (out / "cards.md").read_text()
    assert [scene["id"] for scene in data["scenes"]] == ["s1", "s2", "s3"]
    assert data["scenes"][0]["beats"][0]["summary"] == "A boat approaches."
    assert "A boat approaches." in cards
    if tier == "S":
        assert "states" not in data
        assert "causal_links" not in data
        assert "spare flare" not in cards
        assert "State st1" not in cards
    elif tier == "M":
        assert {s["kind"] for s in data["states"]} == {"goal", "belief"}
        assert data["causal_links"][0]["id"] == "cause1"
        assert "relationships" not in data
        assert "constraints" not in data
        assert "State st3" not in cards
    else:
        assert data["arcs"][0]["reading"]["confidence"] == 0.6
        assert data["relationships"][0]["id"] == "r1"
    if tier != "L":
        assert "arcs" not in data
        assert "tension" not in data
        assert "motif_echoes" not in data
        assert "trajectory" not in cards
        assert "Alternative readings: Mara acts from fear" not in cards
    first = projected.read_bytes()
    assert main(["plan", "project", str(projected), "--tier", tier, "--out", str(out)]) == 0
    assert projected.read_bytes() == first
    assert file.read_bytes() == original
    assert (out / "cards.html").is_file()


@pytest.mark.parametrize("tier", ["S", "M", "L"])
def test_empty_channels_project_and_wrong_membership_is_rejected(tmp_path: Path, tier: str) -> None:
    source = REPO / "examples/harbour.plan.json"
    out = tmp_path / tier
    assert main(["plan", "project", str(source), "--tier", tier, "--out", str(out)]) == 0
    data = json.loads((out / "plan.json").read_text())
    envelope = {"schema_version", "id", "tier", "title", "characters", "sources", "scenes"}
    middle = {
        "causal_links",
        "setups",
        "payoff_links",
        "knowledge_facts",
        "reveal_links",
        "knowledge_uses",
        "states",
    }
    large = {
        "relationships",
        "constraints",
        "arcs",
        "reader_affect",
        "tension",
        "themes",
        "motifs",
        "motif_echoes",
        "discourse",
        "subtext",
    }
    assert set(data) == envelope | (middle if tier != "S" else set()) | (
        large if tier == "L" else set()
    )
    data["tier"] = "S"
    data["setups"] = [{"id": "leak", "scene_id": "s1", "description": "Excluded"}]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    assert main(["plan", "validate", str(bad)]) == 1
