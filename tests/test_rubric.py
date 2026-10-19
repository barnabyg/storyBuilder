from pathlib import Path
from typing import Any

import pytest
import yaml

from nde.cli import main
from nde.eval.rubric import RubricError, load_rubric
from tests.conftest import REPO

RUBRIC_DIR = REPO / "benchmark" / "rubric"


def test_the_committed_rubric_loads_dimensions_anchors_and_failure_tags() -> None:
    rubric = load_rubric(RUBRIC_DIR, 1)

    assert rubric.version == 1
    assert [dimension.id for dimension in rubric.dimensions] == [
        "engagement",
        "coherence",
        "emotional_impact",
        "character_believability",
        "pacing",
        "originality",
    ]
    assert (rubric.scale.min, rubric.scale.max) == (1, 5)
    for dimension in rubric.dimensions:
        assert dimension.definition
        assert all(dimension.anchors.model_dump().values())
    tags = {tag.id for tag in rubric.failure_tags}
    assert {"unearned-change", "dropped-thread", "continuity-error", "melodrama", "generic"} <= tags
    assert all(tag.definition and tag.example for tag in rubric.failure_tags)


def test_the_committed_rubric_matches_its_checksums(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["rubric", "lock", "--dir", str(RUBRIC_DIR), "--check"]) == 0

    assert "CHANGED" not in capsys.readouterr().err


def _copy_rubric(tmp_path: Path) -> Path:
    directory = tmp_path / "rubric"
    directory.mkdir()
    text = (RUBRIC_DIR / "v1.yaml").read_text(encoding="utf-8")
    (directory / "v1.yaml").write_text(text, encoding="utf-8")
    assert main(["rubric", "lock", "--dir", str(directory)]) == 0
    return directory


def test_changing_a_released_rubric_version_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _copy_rubric(tmp_path)
    path = directory / "v1.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("story", "tale", 1), encoding="utf-8")

    with pytest.raises(RubricError, match="rubric v1 changed after release"):
        load_rubric(directory, 1)
    assert main(["rubric", "lock", "--dir", str(directory)]) == 2
    assert "CHANGED rubric@1" in capsys.readouterr().err


def test_a_new_rubric_version_is_recorded_by_lock(tmp_path: Path) -> None:
    directory = _copy_rubric(tmp_path)
    raw = yaml.safe_load((directory / "v1.yaml").read_text(encoding="utf-8"))
    raw["version"] = 2
    raw["dimensions"][0]["definition"] = "How much the reader wants to keep reading."
    (directory / "v2.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(RubricError, match="not registered"):
        load_rubric(directory, 2)
    assert main(["rubric", "lock", "--dir", str(directory)]) == 0
    assert load_rubric(directory, 2).dimensions[0].definition.startswith("How much")


def _break(raw: dict[str, Any], breakage: str) -> None:
    if breakage == "duplicate":
        raw["dimensions"].append(raw["dimensions"][0])
    elif breakage == "missing-anchor":
        del raw["dimensions"][0]["anchors"]["middle"]
    else:
        raw["version"] = 3


@pytest.mark.parametrize(
    ("breakage", "message"),
    [
        ("duplicate", "dimension ids must be unique"),
        ("missing-anchor", "anchors.middle"),
        ("wrong-version", "declares version 3"),
    ],
)
def test_an_invalid_rubric_file_is_rejected(tmp_path: Path, breakage: str, message: str) -> None:
    directory = tmp_path / "rubric"
    directory.mkdir()
    raw = yaml.safe_load((RUBRIC_DIR / "v1.yaml").read_text(encoding="utf-8"))
    _break(raw, breakage)
    (directory / "v1.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert main(["rubric", "lock", "--dir", str(directory)]) == 0

    with pytest.raises(RubricError, match=message):
        load_rubric(directory, 1)
