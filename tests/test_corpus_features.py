import json
from pathlib import Path

import pytest

from nde.cli import main
from tests.test_corpus_segmentation import segmented


def test_features_define_dialogue_sentence_entity_and_viewpoint_denominators(
    tmp_path: Path,
) -> None:
    work, segments = segmented(tmp_path, "Dr. Éva said, “I can't stay.” Bea nodded.")
    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps({"eva": ["Éva"], "bea": ["Bea"]}), encoding="utf-8")
    out = tmp_path / "features"
    assert (
        main(
            [
                "corpus",
                "features",
                str(work),
                str(segments),
                "--aliases",
                str(aliases),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    value = json.loads((out / "features.json").read_text("utf-8"))
    assert value["work"]["tokens"] == 8
    assert value["work"]["sentence_lengths"] == [6, 2]
    assert value["work"]["mean_sentence_tokens"] == 4
    assert value["work"]["dialogue_tokens"] == 3
    assert value["work"]["dialogue_proportion"] == 0.375
    assert value["work"]["cooccurrence"] == {"bea|eva": 1}
    assert value["work"]["first_person_tokens"] == 1
    assert value["scenes"][0]["measurements"]["viewpoint"] == "first-person-cue"
    assert value["work"]["mentions"][0]["start"] == 4
    assert "heuristic" in (out / "report.html").read_text("utf-8")
    original = (out / "features.json").read_bytes()
    assert (
        main(
            [
                "corpus",
                "features",
                str(work),
                str(segments),
                "--aliases",
                str(aliases),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert (out / "features.json").read_bytes() == original


def test_apostrophes_are_not_dialogue_and_ambiguous_aliases_stay_unresolved(tmp_path: Path) -> None:
    work, segments = segmented(tmp_path, "Ann can't see O'Neil. Ann Smith waits.")
    aliases = tmp_path / "aliases.json"
    aliases.write_text(
        json.dumps({"ann-one": ["Ann", "Ann Smith"], "ann-two": ["Ann"], "neil": ["O'Neil"]}),
        encoding="utf-8",
    )
    out = tmp_path / "features"
    assert (
        main(
            [
                "corpus",
                "features",
                str(work),
                str(segments),
                "--aliases",
                str(aliases),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    value = json.loads((out / "features.json").read_text("utf-8"))["work"]
    assert value["dialogue_proportion"] == 0
    assert value["mentions"][0]["entity_id"] is None
    assert value["mentions"][0]["candidates"] == ["ann-one", "ann-two"]
    assert value["mentions"][-1]["text"] == "Ann Smith"
    assert value["mentions"][-1]["entity_id"] == "ann-one"


def test_missing_entities_and_unbalanced_quotes_are_not_measured_zero(tmp_path: Path) -> None:
    work, segments = segmented(tmp_path, 'She said "wait.')
    out = tmp_path / "features"
    assert main(["corpus", "features", str(work), str(segments), "--out", str(out)]) == 0
    value = json.loads((out / "features.json").read_text("utf-8"))["work"]
    assert value["dialogue_proportion"] is None
    assert value["quotation_status"] == "unbalanced"
    assert value["entity_status"] == "unsupported-no-aliases"
    assert value["cooccurrence"] is None


def test_features_inspection_rejects_changed_aliases_or_segmentation(tmp_path: Path) -> None:
    work, segments = segmented(tmp_path, "She waits.")
    aliases = tmp_path / "aliases.json"
    aliases.write_text('{"she": ["She"]}', encoding="utf-8")
    out = tmp_path / "features"
    assert (
        main(
            [
                "corpus",
                "features",
                str(work),
                str(segments),
                "--aliases",
                str(aliases),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    aliases.write_text('{"her": ["She"]}', encoding="utf-8")
    assert main(["corpus", "features-check", str(work), str(segments), str(out)]) == 1


@pytest.mark.parametrize("body", ["She said 'wait.", 'She said "wait.', "She said \u2018wait."])
def test_unmatched_quote_styles_have_unknown_dialogue_fraction(tmp_path: Path, body: str) -> None:
    work, segments = segmented(tmp_path, body)
    out = tmp_path / "features"
    assert main(["corpus", "features", str(work), str(segments), "--out", str(out)]) == 0
    value = json.loads((out / "features.json").read_text("utf-8"))["work"]
    assert value["quotation_status"] == "unbalanced"
    assert value["dialogue_proportion"] is None


def test_empty_break_intervals_do_not_invent_empty_scenes(tmp_path: Path) -> None:
    work, segments = segmented(tmp_path, "She waited.\n\n***\n\n***\n\nHe left.")
    out = tmp_path / "features"
    assert main(["corpus", "features", str(work), str(segments), "--out", str(out)]) == 0
    value = json.loads((out / "features.json").read_text("utf-8"))
    assert len(value["scenes"]) == 2
    assert [scene["measurements"]["tokens"] for scene in value["scenes"]] == [2, 2]
