import json
from pathlib import Path

import pytest

from nde.cli import main
from tests.test_corpus_acquisition import acquire


def segmented(tmp_path: Path, body: str) -> tuple[Path, Path]:
    work = acquire(tmp_path, f"BEGIN LAMP\n{body}\nEND LAMP")
    out = tmp_path / "segments"
    assert main(["corpus", "segment", str(work), "--out", str(out)]) == 0
    return work, out


def test_explicit_chapters_and_breaks_cover_exact_unicode_source(tmp_path: Path) -> None:
    work, out = segmented(
        tmp_path, "CHAPTER I\n\nÉva 😀 waited.\n\n***\n\nShe left.\n\nCHAPTER II\n\nHe stayed."
    )
    text = (work / "text.txt").read_text("utf-8")
    value = json.loads((out / "segmentation.json").read_text("utf-8"))
    assert len(value["chapters"]) == 2
    assert len(value["scenes"]) == 3
    assert value["offset_basis"] == "unicode-codepoints-lf"
    assert value["scenes"][0]["start"] == 0
    assert value["scenes"][-1]["end"] == len(text)
    assert text[value["scenes"][1]["start"] : value["scenes"][1]["end"]].strip() == "She left."
    assert "Éva 😀" in (out / "report.html").read_text("utf-8")
    assert main(["corpus", "segment-check", str(work), str(out)]) == 0


def test_ambiguous_unmarked_story_retains_unknown_boundary_confidence(tmp_path: Path) -> None:
    _, out = segmented(tmp_path, "She waited.\n\nA week later, he arrived.")
    value = json.loads((out / "segmentation.json").read_text("utf-8"))
    assert len(value["scenes"]) == 1
    assert value["scenes"][0]["confidence"] is None
    assert "unmarked" in value["scenes"][0]["reason"]


@pytest.mark.parametrize("kind", ["out-of-bounds", "overlap", "parent", "gap"])
def test_invalid_intervals_are_rejected_at_public_command(tmp_path: Path, kind: str) -> None:
    work, out = segmented(tmp_path, "A lamp.\n\n***\n\nShe left.")
    path = out / "segmentation.json"
    value = json.loads(path.read_text("utf-8"))
    if kind == "out-of-bounds":
        value["scenes"][0]["end"] = 9999
    elif kind == "overlap":
        value["scenes"][1]["start"] = 0
    elif kind == "parent":
        value["scenes"][0]["parent_id"] = "missing"
    else:
        value["scenes"][0]["start"] = 4
    path.write_text(json.dumps(value), encoding="utf-8")
    assert main(["corpus", "segment-check", str(work), str(out)]) == 1


def test_gold_comparison_requires_current_approved_real_evidence(tmp_path: Path) -> None:
    work, out = segmented(tmp_path, "A lamp.\n\n***\n\nShe left.")
    gold = tmp_path / "gold.json"
    segments = json.loads((out / "segmentation.json").read_text("utf-8"))
    gold.write_text(
        json.dumps(
            {
                "version": "1",
                "work_id": "lamp",
                "text_sha256": segments["text_sha256"],
                "segmentation_sha256": segments["sha256"],
                "approved": True,
                "owner": "Owner",
                "recorded_at": "2026-10-26T21:00:00Z",
                "synthetic": True,
                "scene_starts": [s["start"] for s in segments["scenes"]],
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "comparison"
    assert (
        main(
            [
                "corpus",
                "segment-compare",
                str(work),
                str(out),
                "--gold",
                str(gold),
                "--out",
                str(report),
            ]
        )
        == 0
    )
    result = json.loads((report / "comparison.json").read_text("utf-8"))
    assert result["status"] == "pending"
    assert result["matched"] == 1
    assert result["predicted"] == 1
    assert result["reference"] == 1
