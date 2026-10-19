import json
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from tests.eval_helpers import BRIEFS, build_batch, reader, two_runs
from tests.fake_openai import FakeOpenAI


@pytest.fixture
def runs(
    fake_llm: FakeOpenAI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    return two_runs(fake_llm, tmp_path)


def _view(batch: Path) -> dict[str, Any]:
    return json.loads((batch / "view.json").read_text(encoding="utf-8"))


def _key(batch: Path) -> dict[str, Any]:
    return json.loads((batch / "sealed" / "key.json").read_text(encoding="utf-8"))


def _judge_all(batch: Path, preferred: str = "A") -> None:
    client = reader(batch)
    for number in range(1, len(_view(batch)["pairs"]) + 1):
        response = client.post(f"/pairs/{number}", data={"preferred": preferred, "strength": "2"})
        assert response.status_code in {200, 303}


def test_a_batch_pairs_stories_by_brief_with_a_sealed_key(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    batch = build_batch(tmp_path, runs)

    view, key = _view(batch), _key(batch)
    assert len(view["pairs"]) == len(BRIEFS)
    assert sorted((p["brief"], p["repetition"]) for p in key["pairs"]) == [
        (brief, 0) for brief in BRIEFS
    ]
    for shown, sealed in zip(view["pairs"], key["pairs"], strict=True):
        # Each side is the normalised story of the condition the key names.
        for side in ("a", "b"):
            story = Path(sealed[side]["run"]) / sealed[side]["story"]
            assert shown[side] == story.read_text(encoding="utf-8")
        assert {sealed["a"]["condition"], sealed["b"]["condition"]} == {"intact", "control"}
    assert set(view["pairs"][0]) == {"a", "b"}


def test_pair_order_and_sides_are_randomised(runs: tuple[Path, Path], tmp_path: Path) -> None:
    orders: set[tuple[str, ...]] = set()
    sides: set[tuple[str, ...]] = set()
    for seed in range(8):
        key = _key(build_batch(tmp_path, runs, f"b{seed}", "--seed", str(seed)))
        orders.add(tuple(pair["brief"] for pair in key["pairs"]))
        sides.add(tuple(pair["a"]["condition"] for pair in key["pairs"]))

    assert len(orders) > 1
    assert len(sides) > 1


def test_a_batch_name_cannot_be_reused(
    runs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_batch(tmp_path, runs)
    first, second = runs

    code = main(
        [
            "eval",
            "batch",
            "demo",
            "--from",
            str(first),
            "intact",
            "--from",
            str(second),
            "control",
            "--batches-dir",
            str(tmp_path / "batches"),
        ]
    )

    assert code == 2
    assert "already exists" in capsys.readouterr().err


def test_a_condition_missing_from_a_run_is_rejected(
    runs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, second = runs

    code = main(
        [
            "eval",
            "batch",
            "demo",
            "--from",
            str(first),
            "nope",
            "--from",
            str(second),
            "control",
            "--batches-dir",
            str(tmp_path / "batches"),
        ]
    )

    assert code == 2
    assert "has no stories for condition 'nope'" in capsys.readouterr().err


def test_no_page_or_api_response_reveals_the_key(runs: tuple[Path, Path], tmp_path: Path) -> None:
    batch = build_batch(tmp_path, runs)
    client = reader(batch)
    forbidden = [
        "intact",
        "control",
        "secret-model-id",
        "baseline_a",
        "brief-0",
        runs[0].name,
        runs[1].name,
        "sealed",
        "key.json",
    ]

    responses = [client.get("/", follow_redirects=True), client.get("/api/progress")]
    for number in range(1, len(BRIEFS) + 1):
        responses.append(client.get(f"/pairs/{number}"))
        responses.append(client.get(f"/api/pairs/{number}"))
        responses.append(
            client.post(
                f"/pairs/{number}", data={"preferred": "B", "strength": "3"}, follow_redirects=True
            )
        )
    responses.append(client.post("/close", follow_redirects=True))
    responses += [client.get("/sealed/key.json"), client.get("/key.json")]

    for response in responses:
        body = response.text.lower()
        for word in forbidden:
            assert word.lower() not in body, f"{response.request.url} leaks {word!r}"
    assert responses[-1].status_code == 404


def test_a_pair_page_shows_both_stories_and_the_choice_form(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    batch = build_batch(tmp_path, runs)

    page = reader(batch).get("/pairs/1").text

    first = _view(batch)["pairs"][0]
    assert first["a"].split("\n\n")[0] in page
    assert first["b"].split("\n\n")[0] in page
    assert 'name="preferred"' in page
    assert 'name="strength"' in page
    assert "Pair 1 of 4" in page


def test_judgements_are_saved_at_once_and_survive_a_restart(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    batch = build_batch(tmp_path, runs)
    reader(batch).post("/pairs/1", data={"preferred": "B", "strength": "3"})

    restarted = reader(batch)

    assert restarted.get("/api/progress").json() == {"pairs": 4, "judged": 1, "closed": False}
    assert restarted.get("/api/pairs/1").json()["judgement"] == {"preferred": "B", "strength": 3}
    # The reader resumes at the first unjudged pair.
    resumed = restarted.get("/", follow_redirects=False)
    assert resumed.headers["location"] == "/pairs/2"


def test_an_incomplete_judgement_is_rejected(runs: tuple[Path, Path], tmp_path: Path) -> None:
    client = reader(build_batch(tmp_path, runs))

    for data in (
        {"preferred": "B"},
        {"strength": "2"},
        {"preferred": "C", "strength": "2"},
        {"preferred": "A", "strength": "4"},
    ):
        assert client.post("/pairs/1", data=data).status_code == 422
    assert client.get("/api/progress").json()["judged"] == 0


def test_the_batch_closes_only_when_every_pair_is_judged(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    batch = build_batch(tmp_path, runs)
    client = reader(batch)
    client.post("/pairs/1", data={"preferred": "A", "strength": "1"})

    assert client.post("/close").status_code == 409
    _judge_all(batch)
    assert client.post("/close", follow_redirects=False).status_code == 303
    assert client.get("/api/progress").json()["closed"] is True


def test_a_closed_batch_rejects_new_judgements(runs: tuple[Path, Path], tmp_path: Path) -> None:
    batch = build_batch(tmp_path, runs)
    _judge_all(batch)
    reader(batch).post("/close")

    response = reader(batch).post("/pairs/1", data={"preferred": "B", "strength": "1"})

    assert response.status_code == 409
    assert reader(batch).get("/api/pairs/1").json()["judgement"]["preferred"] == "A"


def test_the_report_refuses_an_open_batch(
    runs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_batch(tmp_path, runs)

    code = main(["eval", "report", "demo", "--batches-dir", str(tmp_path / "batches")])

    assert code == 2
    assert "still open" in capsys.readouterr().err


def test_the_report_reveals_the_key_with_win_counts_and_a_binomial_p_value(
    runs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = build_batch(tmp_path, runs)
    client = reader(batch)
    # Prefer the intact story in every pair.
    for number, pair in enumerate(_key(batch)["pairs"], start=1):
        side = "A" if pair["a"]["condition"] == "intact" else "B"
        client.post(f"/pairs/{number}", data={"preferred": side, "strength": "2"})
    client.post("/close")
    capsys.readouterr()

    code = main(["eval", "report", "demo", "--batches-dir", str(tmp_path / "batches")])

    assert code == 0
    report = capsys.readouterr().out
    assert "| intact | 4 |" in report
    assert "| control | 0 |" in report
    # 4 of 4: two-sided exact binomial p = 2 / 16.
    assert "p = 0.125" in report
    assert (batch / "report.md").read_text(encoding="utf-8") == report.rstrip("\n") + "\n"
    for brief in BRIEFS:
        assert brief in report


def test_the_reader_binds_to_localhost_only(
    runs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_batch(tmp_path, runs)
    served: dict[str, Any] = {}

    def fake_run(app: object, **options: Any) -> None:
        served.update(options, app=app)

    monkeypatch.setattr("uvicorn.run", fake_run)

    code = main(["eval", "serve", "--batch", "demo", "--batches-dir", str(tmp_path / "batches")])

    assert code == 0
    assert served["host"] == "127.0.0.1"
    assert served["app"] is not None
