import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from nde.cli import main
from nde.schema.brief import Brief, BriefContent
from tests.conftest import REPO, base_config, write_config
from tests.fake_openai import FakeOpenAI, completion

BRIEF: dict[str, Any] = {
    "id": "brief-01",
    "split": "dev",
    "genre": "literary fiction",
    "premise": "A lighthouse keeper must keep the lamp lit through a storm.",
    "protagonist": {"name": "Ada Venn", "description": "a widowed keeper in her sixties"},
    "central_conflict": "A smuggler wants the light out so a ship will founder.",
    "themes": ["duty", "grief"],
    "ending_constraint": {"kind": "bittersweet", "detail": "The ship is saved; the lamp is not."},
}


def test_a_brief_gets_the_default_target_length() -> None:
    brief = Brief.model_validate(BRIEF)

    assert (brief.target_words, brief.min_scenes, brief.max_scenes) == (2000, 5, 8)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"premise": None}, "premise"),
        ({"mood": "dark"}, "mood: Extra inputs are not permitted"),
        ({"target_words": 100}, "target_words: Input should be greater than or equal to 1000"),
        ({"target_words": 50_000}, "target_words: Input should be less than or equal to 8000"),
        ({"min_scenes": 9}, "min_scenes must not exceed max_scenes"),
        ({"split": "train"}, "split: Input should be 'dev' or 'test'"),
        ({"id": "Brief 1"}, "id: String should match pattern"),
    ],
    ids=["missing", "unknown", "too-short", "too-long", "scene-range", "split", "id"],
)
def test_the_brief_model_rejects_invalid_briefs_clearly(
    change: dict[str, object], message: str
) -> None:
    raw = copy.deepcopy(BRIEF)
    for key, value in change.items():
        if value is None:
            del raw[key]
        else:
            raw[key] = value

    with pytest.raises(ValidationError) as error:
        Brief.model_validate(raw)

    assert message in str(error.value).replace("\n  ", ": ")


def _write_brief(root: Path, brief: dict[str, Any]) -> Path:
    path = root / str(brief["split"]) / f"{brief['id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(brief, sort_keys=False), encoding="utf-8")
    return path


def _briefs(root: Path, split: str, count: int) -> list[Path]:
    return [
        _write_brief(root, BRIEF | {"id": f"brief-{index:02d}", "split": split})
        for index in range(1, count + 1)
    ]


def test_the_check_accepts_valid_briefs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _briefs(tmp_path, "dev", 3)

    assert main(["briefs", "check", "--dir", str(tmp_path)]) == 0

    assert "3 briefs valid (dev: 3, test: 0)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("breakage", "message"),
    [
        ("invalid", "brief-02.yaml: themes: Field required"),
        ("misnamed", "brief-02.yaml: file name must match id 'brief-07'"),
        ("misfiled", "brief-02.yaml: split 'test' does not match its folder 'dev'"),
        ("duplicate", "brief-02 appears in both dev and test"),
    ],
)
def test_the_check_names_each_broken_brief(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], breakage: str, message: str
) -> None:
    _, second = _briefs(tmp_path, "dev", 2)
    raw = yaml.safe_load(second.read_text(encoding="utf-8"))
    if breakage == "invalid":
        del raw["themes"]
    elif breakage == "misnamed":
        raw["id"] = "brief-07"
    elif breakage == "misfiled":
        raw["split"] = "test"
    else:
        _write_brief(tmp_path, raw | {"split": "test"})
    second.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert main(["briefs", "check", "--dir", str(tmp_path)]) == 1

    assert message in capsys.readouterr().err


def test_freezing_a_split_records_its_checksums(tmp_path: Path) -> None:
    _briefs(tmp_path, "test", 2)

    assert main(["briefs", "freeze", "test", "--dir", str(tmp_path)]) == 0

    frozen = json.loads((tmp_path / "frozen.json").read_text(encoding="utf-8"))
    assert sorted(frozen["test"]) == ["brief-01", "brief-02"]
    assert all(len(digest) == 64 for digest in frozen["test"].values())
    assert main(["briefs", "check", "--dir", str(tmp_path)]) == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("edit", "test/brief-02 changed after the test split was frozen"),
        ("add", "test/brief-03 was added after the test split was frozen"),
        ("remove", "test/brief-02 was removed after the test split was frozen"),
    ],
)
def test_a_frozen_split_rejects_edits_additions_and_removals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change: str, message: str
) -> None:
    _, second = _briefs(tmp_path, "test", 2)
    assert main(["briefs", "freeze", "test", "--dir", str(tmp_path)]) == 0
    if change == "edit":
        text = second.read_text(encoding="utf-8")
        second.write_text(text.replace("storm", "storms"), encoding="utf-8")
    elif change == "add":
        _write_brief(tmp_path, BRIEF | {"id": "brief-03", "split": "test"})
    else:
        second.unlink()

    assert main(["briefs", "check", "--dir", str(tmp_path)]) == 1

    assert message in capsys.readouterr().err


def test_a_frozen_split_cannot_be_refrozen_over_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    [first] = _briefs(tmp_path, "test", 1)
    assert main(["briefs", "freeze", "test", "--dir", str(tmp_path)]) == 0
    first.write_text(first.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")

    assert main(["briefs", "freeze", "test", "--dir", str(tmp_path)]) == 1

    assert "already frozen" in capsys.readouterr().err


def test_the_dev_split_is_not_frozen_so_dev_briefs_can_change(tmp_path: Path) -> None:
    [first] = _briefs(tmp_path, "dev", 1)
    first.write_text(first.read_text(encoding="utf-8").replace("storm", "gale"), encoding="utf-8")

    assert main(["briefs", "check", "--dir", str(tmp_path)]) == 0


def test_the_committed_briefs_pass_the_check(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["briefs", "check", "--dir", str(REPO / "benchmark")]) == 0

    assert "(dev: 10," in capsys.readouterr().out


def _draft(genre: str) -> dict[str, Any]:
    content = {key: value for key, value in BRIEF.items() if key not in {"id", "split"}}
    return content | {"genre": genre}


def test_the_drafting_config_writes_candidate_briefs_for_review(
    fake_llm: FakeOpenAI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NDE_OPENAI_BASE_URL", fake_llm.base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_llm.default = lambda _body: completion(
        json.dumps({"briefs": [_draft("horror"), _draft("western")]})
    )
    runs = tmp_path / "runs"
    assert main(["run", str(REPO / "configs" / "draft-briefs.yaml"), "--runs-dir", str(runs)]) == 0
    [run_dir] = list(runs.iterdir())
    out = tmp_path / "drafts"

    assert main(["briefs", "drafts", str(run_dir), "--out", str(out)]) == 0

    drafts = sorted(out.glob("*.yaml"))
    assert len(drafts) == 2 * fake_llm.call_count
    for path in drafts:
        BriefContent.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert yaml.safe_load(drafts[1].read_text(encoding="utf-8"))["genre"] == "western"
    # Drafts never land in a split by themselves.
    assert not (out / "dev").exists()
    assert not (out / "test").exists()


def test_drafts_from_a_run_without_brief_drafts_are_refused(
    fake_llm: FakeOpenAI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    runs = tmp_path / "runs"
    assert (
        main(
            [
                "run",
                str(write_config(tmp_path, base_config(fake_llm.base_url))),
                "--runs-dir",
                str(runs),
            ]
        )
        == 0
    )
    [run_dir] = list(runs.iterdir())

    assert main(["briefs", "drafts", str(run_dir), "--out", str(tmp_path / "drafts")]) == 1

    assert "no brief drafts" in capsys.readouterr().err
