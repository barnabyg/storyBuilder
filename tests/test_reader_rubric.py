import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from tests.eval_helpers import (
    BRIEFS,
    DIMENSIONS,
    INTACT,
    RUBRIC_DIR,
    Marks,
    build_batch,
    complete_form,
    reader,
    seeded_story,
    two_runs,
)
from tests.fake_openai import FakeOpenAI
from tests.stories import run, story_config


@pytest.fixture
def runs(
    fake_llm: FakeOpenAI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    return two_runs(fake_llm, tmp_path)


def _key(batch: Path) -> dict[str, Any]:
    return json.loads((batch / "sealed" / "key.json").read_text(encoding="utf-8"))


def _view(batch: Path) -> dict[str, Any]:
    return json.loads((batch / "view.json").read_text(encoding="utf-8"))


def _report(tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str = "demo") -> str:
    capsys.readouterr()
    assert main(["eval", "report", name, "--batches-dir", str(tmp_path / "batches")]) == 0
    return capsys.readouterr().out


def test_the_batch_records_the_rubric_version(runs: tuple[Path, Path], tmp_path: Path) -> None:
    batch = build_batch(tmp_path, runs)

    assert _view(batch)["rubric"]["version"] == 1
    assert _key(batch)["rubric_version"] == 1
    assert _view(batch)["guess_options"] == ["control", "intact"]


def test_the_pair_page_asks_for_scores_tags_and_a_guess(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    page = reader(build_batch(tmp_path, runs)).get("/pairs/1").text

    for side in ("a", "b"):
        for dimension in DIMENSIONS:
            assert f'name="score-{side}-{dimension}"' in page
        assert f'name="tags-{side}" value="dropped-thread"' in page
        assert f'name="free-tags-{side}"' in page
    assert "Hard to put down." in page  # an anchor, shown as guidance
    assert 'name="guess" value="intact"' in page


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"guess": None}, "guess"),
        ({"score-a-pacing": None}, "score-a-pacing"),
        ({"score-b-coherence": "6"}, "score-b-coherence"),
        ({"guess": "nope"}, "guess"),
        ({"tags-a": ["not-a-tag"]}, "tags-a"),
    ],
    ids=["no-guess", "missing-score", "score-range", "unknown-guess", "unknown-tag"],
)
def test_a_pair_cannot_be_submitted_incomplete(
    runs: tuple[Path, Path], tmp_path: Path, change: dict[str, Any], problem: str
) -> None:
    client = reader(build_batch(tmp_path, runs))
    form = complete_form()
    for field, value in change.items():
        if value is None:
            del form[field]
        else:
            form[field] = value

    response = client.post("/pairs/1", data=form)

    assert response.status_code == 422
    assert problem in response.text
    assert client.get("/api/progress").json()["judged"] == 0


def test_a_complete_judgement_records_scores_tags_and_the_guess(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    client = reader(build_batch(tmp_path, runs))
    form = complete_form(
        "B", 3, "control", Marks(2, ["generic", "melodrama"], "too long; flat"), Marks(4)
    )

    client.post("/pairs/1", data=form)

    judgement = client.get("/api/pairs/1").json()["judgement"]
    assert judgement["guess"] == "control"
    assert judgement["a"]["scores"] == dict.fromkeys(DIMENSIONS, 2)
    assert judgement["b"]["scores"] == dict.fromkeys(DIMENSIONS, 4)
    assert judgement["a"]["tags"] == ["generic", "melodrama"]
    assert judgement["a"]["free_tags"] == ["too long", "flat"]
    assert judgement["b"]["tags"] == []


def _judge(batch: Path, *, guess_right: list[bool]) -> None:
    """Prefer intact, score intact 4 and control 2, tag control generic, and guess as told."""
    client = reader(batch)
    for number, (pair, right) in enumerate(zip(_key(batch)["pairs"], guess_right, strict=True), 1):
        a_is_intact = pair["a"]["condition"] == "intact"
        actual = pair["a"]["condition"]
        other = "control" if actual == "intact" else "intact"
        client.post(
            f"/pairs/{number}",
            data=complete_form(
                "A" if a_is_intact else "B",
                2,
                actual if right else other,
                Marks(4) if a_is_intact else Marks(2, ["generic"], "odd ending"),
                Marks(2, ["generic"]) if a_is_intact else Marks(4),
            ),
        )
    assert client.post("/close").status_code in {200, 303}


def test_the_report_adds_rubric_means_with_intervals_tags_and_guess_accuracy(
    runs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = build_batch(tmp_path, runs)
    _judge(batch, guess_right=[True, True, True, True])

    report = _report(tmp_path, capsys)

    assert "## Rubric scores (rubric v1)" in report
    assert "| engagement | 4.00 [4.00, 4.00] | 2.00 [2.00, 2.00] |" in report
    assert "## Failure tags" in report
    assert "| generic | 0 | 4 |" in report
    assert "## Condition guesses" in report
    assert "4 of 4 guesses right (100%)" in report
    assert "**Possibly unblinded:**" in report


def test_a_batch_with_chance_guessing_is_not_flagged(
    runs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = build_batch(tmp_path, runs)
    _judge(batch, guess_right=[True, False, True, False])

    report = _report(tmp_path, capsys)

    assert "2 of 4 guesses right (50%)" in report
    assert "Possibly unblinded" not in report


def test_a_batch_built_before_rubric_scoring_still_reports_without_the_new_sections(
    runs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = build_batch(tmp_path, runs)
    # Rewrite the batch as the first reader wrote it: no rubric, no guess, no version.
    view = _view(batch)
    (batch / "view.json").write_text(json.dumps({"pairs": view["pairs"]}), encoding="utf-8")
    key = _key(batch)
    del key["rubric_version"]
    (batch / "sealed" / "key.json").write_text(json.dumps(key), encoding="utf-8")
    client = reader(batch)
    assert "score-a-pacing" not in client.get("/pairs/1").text
    for number in range(1, 5):
        response = client.post(f"/pairs/{number}", data={"preferred": "A", "strength": "1"})
        assert response.status_code in {200, 303}
    client.post("/close")

    report = _report(tmp_path, capsys)

    assert "## Preferences" in report
    assert "## Rubric scores" not in report
    assert "## Condition guesses" not in report


def _git(repo: Path, *args: str, date: datetime | None = None) -> None:
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    if date is not None:
        env |= {"GIT_AUTHOR_DATE": date.isoformat(), "GIT_COMMITTER_DATE": date.isoformat()}
    # S603/S607: fixed git arguments in a test repository; git is a developer prerequisite.
    subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
        env=os.environ | env,
    )


def _repo_with_generating_code_changed(repo: Path, hours_ago: float) -> None:
    _git(repo, "init", "-q")
    code = repo / "src" / "nde" / "stories" / "pipeline.py"
    code.parent.mkdir(parents=True)
    code.write_text("# generating code\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "change", date=datetime.now(UTC) - timedelta(hours=hours_ago))


def _no_server(*_args: object, **_kwargs: object) -> None:
    return None


def _serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr("uvicorn.run", _no_server)
    return main(["eval", "serve", "--batch", "demo", "--batches-dir", str(tmp_path / "batches")])


def test_the_reader_warns_when_generating_code_changed_within_48_hours(
    runs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch = build_batch(tmp_path, runs)
    _repo_with_generating_code_changed(tmp_path, hours_ago=3)

    assert _serve(tmp_path, monkeypatch) == 0

    assert "WARNING: the generating code last changed" in capsys.readouterr().err
    assert "less than 48 hours" in reader(batch).get("/pairs/1").text
    _judge(batch, guess_right=[True, False, True, False])
    report = _report(tmp_path, capsys)
    assert "## Warnings" in report
    assert "less than 48 hours" in report


def test_the_reader_does_not_warn_after_48_hours(
    runs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch = build_batch(tmp_path, runs)
    _repo_with_generating_code_changed(tmp_path, hours_ago=72)

    assert _serve(tmp_path, monkeypatch) == 0

    assert "WARNING" not in capsys.readouterr().err
    _judge(batch, guess_right=[True, False, True, False])
    assert "## Warnings" not in _report(tmp_path, capsys)


def test_same_named_conditions_are_offered_without_run_directory_names(
    fake_llm: FakeOpenAI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = seeded_story
    runs = []
    for seed in (100, 200):
        config = story_config(fake_llm.base_url, [INTACT], briefs={"ids": BRIEFS[:2]})
        config["seed"] = seed
        runs.append(run(tmp_path, config))
    code = main(
        [
            "eval",
            "batch",
            "same",
            "--from",
            str(runs[0]),
            "intact",
            "--from",
            str(runs[1]),
            "intact",
            "--batches-dir",
            str(tmp_path / "batches"),
            "--rubric-dir",
            str(RUBRIC_DIR),
        ]
    )
    assert code == 0

    options = _view(tmp_path / "batches" / "same")["guess_options"]

    assert options == ["intact (source 1)", "intact (source 2)"]
    page = reader(tmp_path / "batches" / "same").get("/pairs/1").text
    assert runs[0].name not in page
    assert runs[1].name not in page
