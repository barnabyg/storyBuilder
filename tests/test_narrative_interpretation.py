import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from tests.conftest import REPO


def example() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "id": "lamp",
        "title": "The lamp",
        "characters": [{"id": "c1", "name": "Mara"}],
        "scenes": [
            {"id": "s1", "title": "Darkness", "character_ids": ["c1"]},
            {"id": "s2", "title": "A choice", "character_ids": ["c1"]},
            {"id": "s3", "title": "The light", "character_ids": ["c1"]},
        ],
        "arcs": [
            {
                "id": "a1",
                "character_id": "c1",
                "reading": {
                    "value": "Mara takes responsibility",
                    "confidence": 0.6,
                    "alternatives": ["Mara acts from fear"],
                },
                "trajectory": {
                    "shape": "rising",
                    "points": [
                        {"scene_id": "s1", "value": 0},
                        {"scene_id": "s2", "value": 2},
                        {"scene_id": "s3", "value": 4},
                    ],
                },
            }
        ],
        "reader_affect": [
            {
                "id": "aff1",
                "reading": {
                    "status": "ambiguous",
                    "alternatives": ["hope", "dread"],
                },
                "trajectory": {
                    "shape": "unknown",
                    "points": [
                        {"scene_id": "s1", "value": 1},
                        {"scene_id": "s2", "value": None},
                        {"scene_id": "s3", "value": 3},
                    ],
                },
            }
        ],
        "tension": [
            {
                "id": "ten1",
                "reading": {"value": "The boat approaches"},
                "trajectory": {
                    "shape": "peak",
                    "points": [
                        {"scene_id": "s1", "value": 1},
                        {"scene_id": "s2", "value": 4},
                        {"scene_id": "s3", "value": 0},
                    ],
                },
            }
        ],
        "themes": [{"id": "th1", "reading": {"status": "unknown"}}],
        "motifs": [
            {"id": "m1", "scene_ids": ["s1"], "reading": {"value": "dark lamp"}},
            {"id": "m2", "scene_ids": ["s3"], "reading": {"value": "fire"}},
        ],
        "motif_echoes": [
            {"id": "echo1", "from_motif_id": "m1", "to_motif_id": "m2", "kind": "transformation"}
        ],
        "discourse": [
            {
                "id": "d1",
                "kind": "observation",
                "scene_ids": ["s1"],
                "reading": {"value": "The opening uses short sentences"},
            }
        ],
        "subtext": [
            {
                "id": "sub1",
                "reading": {
                    "value": "Mara hides her fear",
                    "confidence": 0.4,
                    "alternatives": ["Mara feels no fear"],
                },
            }
        ],
    }


def inspect(data: dict[str, Any], directory: Path) -> tuple[int, Path]:
    source = directory / "input.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    out = directory / "cards"
    return main(["plan", "inspect", str(source), "--out", str(out)]), out


def test_interpretations_preserve_unknowns_alternatives_and_observations(tmp_path: Path) -> None:
    code, out = inspect(example(), tmp_path)
    assert code == 0
    saved = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert saved["themes"][0]["reading"]["status"] == "unknown"
    assert saved["themes"][0]["reading"]["value"] is None
    assert saved["reader_affect"][0]["reading"]["alternatives"] == ["hope", "dread"]
    assert saved["discourse"][0]["kind"] == "observation"
    for filename in ("cards.md", "cards.html"):
        cards = (out / filename).read_text(encoding="utf-8")
        assert "Mara acts from fear" in cards
        assert "confidence=0.6" in cards
        assert "unknown" in cards
        assert "observation" in cards
        assert "transformation" in cards
        assert "0 → 2 → 4" in cards


def test_motif_echo_must_name_existing_endpoints(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = example()
    data["motif_echoes"][0]["to_motif_id"] = "missing-motif"
    code, out = inspect(data, tmp_path)
    assert code == 1
    assert "motif_endpoint" in capsys.readouterr().err
    assert "echo1" in (out / "cards.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "expected", "exit_code"),
    [
        ("scene", "interpretive_scene_reference", 1),
        ("arc_character", "arc_character_reference", 1),
        ("trajectory_scene", "trajectory_scene_reference", 1),
        ("trajectory_order", "trajectory_order", 1),
        ("ordinal_bounds", "less_than_equal", 2),
        ("point_count", "too_short", 2),
        ("shape", "type=enum", 2),
        ("motif_self_link", "motif_self_link", 1),
        ("duplicate_id", "duplicate_id", 1),
    ],
)
def test_interpretive_mutation_categories(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
    expected: str,
    exit_code: int,
) -> None:
    data = example()
    if mutation == "scene":
        data["themes"][0]["scene_ids"] = ["missing-scene"]
    elif mutation == "arc_character":
        data["arcs"][0]["character_id"] = "missing-character"
    elif mutation == "trajectory_scene":
        data["arcs"][0]["trajectory"]["points"][0]["scene_id"] = "missing-scene"
    elif mutation == "trajectory_order":
        data["arcs"][0]["trajectory"]["points"].reverse()
    elif mutation == "ordinal_bounds":
        data["tension"][0]["trajectory"]["points"][0]["value"] = 5
    elif mutation == "point_count":
        data["tension"][0]["trajectory"]["points"].pop()
    elif mutation == "shape":
        data["tension"][0]["trajectory"]["shape"] = "precise-continuous"
    elif mutation == "motif_self_link":
        data["motif_echoes"][0]["to_motif_id"] = "m1"
    elif mutation == "duplicate_id":
        data["motifs"][0]["id"] = "s1"
    code, _ = inspect(data, tmp_path)
    assert code == exit_code
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("valid", None),
        ("source", "span_source"),
        ("version", "span_version"),
        ("bounds", "span_bounds"),
    ],
)
def test_interpretive_evidence_is_bound_to_exact_source_offsets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
    expected: str | None,
) -> None:
    data = example()
    text = "Mara lifts the dark lamp.\nThe boat passes.\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    data["sources"] = [
        {
            "id": "source1",
            "title": "Fixture source",
            "text": text,
            "sha256": digest,
            "provenance": "Deterministic test text",
        }
    ]
    span: dict[str, Any] = {"source_id": "source1", "source_sha256": digest, "start": 0, "end": 4}
    if mutation == "source":
        span["source_id"] = "missing-source"
    elif mutation == "version":
        span["source_sha256"] = "wrong-version"
    elif mutation == "bounds":
        span["end"] = 1000
    data["discourse"][0]["evidence"] = [span]
    code, out = inspect(data, tmp_path)
    if expected is None:
        assert code == 0
        assert "source1" in (out / "cards.md").read_text(encoding="utf-8")
    else:
        assert code == 1
        assert expected in capsys.readouterr().err


def test_arc_character_is_visible_without_a_trajectory(tmp_path: Path) -> None:
    data = example()
    del data["arcs"][0]["trajectory"]
    code, out = inspect(data, tmp_path)
    assert code == 0
    assert "Arc character: c1" in (out / "cards.md").read_text(encoding="utf-8")


def test_contrasting_saved_trajectories_survive_rendering_and_round_trip(tmp_path: Path) -> None:
    for name, shape, values, description in (
        ("rising", "rising", [0, 2, 4], "Mara takes responsibility"),
        ("falling", "falling", [4, 2, 0], "Mara gives up control"),
    ):
        source = REPO / f"examples/trajectory-{name}.plan.json"
        out = tmp_path / name
        assert main(["plan", "inspect", str(source), "--out", str(out)]) == 0
        saved = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        assert saved["arcs"][0]["trajectory"]["shape"] == shape
        assert [point["value"] for point in saved["arcs"][0]["trajectory"]["points"]] == values
        sequence = "0 → 2 → 4" if name == "rising" else "4 → 2 → 0"
        for filename in ("cards.md", "cards.html"):
            cards = (out / filename).read_text(encoding="utf-8")
            assert description in cards
            assert sequence in cards
            assert f"shape={shape}" in cards
        first = (out / "plan.json").read_bytes()
        assert main(["plan", "inspect", str(out / "plan.json"), "--out", str(out)]) == 0
        assert (out / "plan.json").read_bytes() == first
