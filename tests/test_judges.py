import json
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from nde.eval.judge_report import PairLabels, judges_section
from nde.eval.judges import JudgeRecord
from tests.conftest import base_config, write_config
from tests.eval_helpers import Marks, build_batch, complete_form, reader, seeded_story, two_runs
from tests.fake_openai import FakeOpenAI, Json, Reply, completion
from tests.stories import last_user_message, run

# two_runs writes the intact stories with seed 100, so their words read "w100".
INTACT_MARKER = "w100"


def _judge_reply(body: Json) -> Reply:
    """A biased judge always prefers story 1; a fair judge always prefers the intact story."""
    prompt = last_user_message(body)
    if "Score this short story" in prompt:
        return completion(json.dumps({**dict.fromkeys(_DIMENSIONS, 3), "failure_tags": []}))
    if "Story 1:" in prompt:
        first = prompt.split("Story 1:", 1)[1].split("Story 2:", 1)[0]
        preferred = 1 if body["model"] == "biased-judge" or INTACT_MARKER in first else 2
        return completion(json.dumps({"preferred": preferred, "strength": 2, "reason": "r"}))
    return seeded_story(body)


_DIMENSIONS = [
    "engagement",
    "coherence",
    "emotional_impact",
    "character_believability",
    "pacing",
    "originality",
]


@pytest.fixture
def batch(fake_llm: FakeOpenAI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    built = build_batch(tmp_path, two_runs(fake_llm, tmp_path))
    fake_llm.default = _judge_reply
    return built


def _panel(*judges: tuple[str, str | None]) -> list[dict[str, Any]]:
    panel: list[dict[str, Any]] = []
    for model, family in judges:
        binding: dict[str, Any] = {"provider": "fake", "model": model, "params": {}}
        if family is not None:
            binding["family"] = family
        panel.append(binding)
    return panel


def judge_config(
    base_url: str, tmp_path: Path, panel: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    config = base_config(base_url)
    del config["calls"]
    config["roles"] = {}
    config["judges"] = {
        "batch": "demo",
        "batches_dir": str(tmp_path / "batches"),
        "panel": panel or _panel(("biased-judge", "family-one"), ("fair-judge", "family-two")),
    }
    config["prices"] = {
        f"fake/{model}": {"input_per_mtok": 0.1, "output_per_mtok": 0.2}
        for model in ("biased-judge", "fair-judge", "other-judge")
    }
    return config


def _judge_records(batch: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for path in sorted((batch / "judges").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_judges_run_pairwise_in_both_orders_and_in_rubric_mode(
    fake_llm: FakeOpenAI, batch: Path, tmp_path: Path
) -> None:
    before = fake_llm.call_count

    run_dir = run(tmp_path, judge_config(fake_llm.base_url, tmp_path))

    # 4 pairs x 2 judges x (2 orders + 2 stories).
    assert fake_llm.call_count - before == 32
    records = _judge_records(batch)
    pairwise = [r for r in records if r["mode"] == "pairwise"]
    assert {(r["family"], r["pair"], r["order"]) for r in pairwise} == {
        (family, pair, order)
        for family in ("family-one", "family-two")
        for pair in range(1, 5)
        for order in ("AB", "BA")
    }
    rubric = [r for r in records if r["mode"] == "rubric"]
    assert len(rubric) == 16
    assert rubric[0]["scores"] == dict.fromkeys(_DIMENSIONS, 3)
    assert (run_dir / "judgements.jsonl").is_file()
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text("utf-8").splitlines()]
    assert {c["step"] for c in calls} == {"pairwise", "rubric"}


def test_judges_see_only_the_normalised_text_in_both_orders(
    fake_llm: FakeOpenAI, batch: Path, tmp_path: Path
) -> None:
    view = json.loads((batch / "view.json").read_text(encoding="utf-8"))
    before = fake_llm.call_count

    run(
        tmp_path,
        judge_config(
            fake_llm.base_url, tmp_path, _panel(("fair-judge", "f2"), ("other-judge", "f3"))
        ),
    )

    prompts = [last_user_message(r.body) for r in fake_llm.requests[before:]]
    first_pair = view["pairs"][0]
    ab, ba = [p for p in prompts if "Story 1:" in p and first_pair["a"] in p][:2]
    assert ab.index(first_pair["a"]) < ab.index(first_pair["b"])
    assert ba.index(first_pair["b"]) < ba.index(first_pair["a"])
    assert all("brief-0" not in prompt and "intact" not in prompt for prompt in prompts)


@pytest.mark.usefixtures("batch")
def test_a_rerun_of_the_judges_is_free(fake_llm: FakeOpenAI, tmp_path: Path) -> None:
    config = judge_config(fake_llm.base_url, tmp_path)
    run(tmp_path, config)
    calls = fake_llm.call_count

    run(tmp_path, config)

    assert fake_llm.call_count == calls


@pytest.mark.parametrize(
    ("panel", "message"),
    [
        (_panel(("fair-judge", "same"), ("other-judge", "same")), "at least two model families"),
        (_panel(("fair-judge", "f2")), "at least two model families"),
        (
            _panel(("fair-judge", "generator-family"), ("other-judge", "f3")),
            "judge fake/fair-judge is in the family 'generator-family'",
        ),
        (
            _panel(("fair-judge", None), ("other-judge", "f3")),
            "judge fake/fair-judge has no family",
        ),
    ],
    ids=["one-family", "one-judge", "same-family-as-generator", "no-family"],
)
@pytest.mark.usefixtures("batch")
def test_an_invalid_panel_is_rejected_before_any_call(
    fake_llm: FakeOpenAI,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    panel: list[dict[str, Any]],
    message: str,
) -> None:
    before = fake_llm.call_count
    config = judge_config(fake_llm.base_url, tmp_path, panel)

    code = main(["run", str(write_config(tmp_path, config)), "--runs-dir", str(tmp_path / "r")])

    assert code == 2
    assert message in capsys.readouterr().err
    assert fake_llm.call_count == before


def test_generators_without_a_recorded_family_are_rejected(
    fake_llm: FakeOpenAI,
    batch: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    key_path = batch / "sealed" / "key.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    for generator in key["generators"]:
        generator["family"] = None
    key_path.write_text(json.dumps(key), encoding="utf-8")

    code = main(
        [
            "run",
            str(write_config(tmp_path, judge_config(fake_llm.base_url, tmp_path))),
            "--runs-dir",
            str(tmp_path / "r"),
        ]
    )

    assert code == 2
    assert "family of generator fake/secret-model-id is not recorded" in capsys.readouterr().err


def test_judges_run_on_an_open_batch_without_revealing_the_key(
    fake_llm: FakeOpenAI, batch: Path, tmp_path: Path
) -> None:
    run(tmp_path, judge_config(fake_llm.base_url, tmp_path))

    assert not (batch / "closed.json").exists()
    judged = "\n".join(p.read_text("utf-8") for p in (batch / "judges").glob("*.jsonl"))
    assert "intact" not in judged
    assert "control" not in judged
    assert "intact" not in reader(batch).get("/overview").text


def _owner_reads(batch: Path, prefer_intact: list[bool]) -> None:
    key = json.loads((batch / "sealed" / "key.json").read_text(encoding="utf-8"))
    client = reader(batch)
    for number, (pair, intact) in enumerate(zip(key["pairs"], prefer_intact, strict=True), 1):
        a_intact = pair["a"]["condition"] == "intact"
        side = "A" if a_intact == intact else "B"
        client.post(f"/pairs/{number}", data=complete_form(side, 2, "intact", Marks(), Marks()))
    client.post("/close")


def test_the_report_gives_each_family_wins_agreement_kappa_and_consistency(
    fake_llm: FakeOpenAI, batch: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run(tmp_path, judge_config(fake_llm.base_url, tmp_path))
    # The owner prefers intact in pairs 1-3 and control in pair 4.
    _owner_reads(batch, [True, True, True, False])
    capsys.readouterr()

    assert main(["eval", "report", "demo", "--batches-dir", str(tmp_path / "batches")]) == 0

    report = capsys.readouterr().out
    assert "## LLM judges" in report
    assert "Judges never pass a gate" in report
    # family-one always prefers story 1, so it flips with the order: never consistent.
    assert "| family-one | fake/biased-judge | 4 | 0 | 0 | 4 | 0 | 0% | 0.00 | 0% |" in report
    # family-two always prefers intact: it agrees with the owner in 3 of 4 pairs. The owner
    # rates [i, i, i, c] and the judge [i, i, i, i]: observed 0.75, chance 0.75, kappa 0.
    assert "| family-two | fake/fair-judge | 4 | 4 | 0 | 0 | 0 | 75% | 0.00 | 100% |" in report
    assert "### Judge rubric scores" in report


def test_a_missing_verdict_is_incomplete_not_position_inconsistent() -> None:
    pairs = [PairLabels(a="intact", b="control", owner="intact") for _ in range(2)]
    records = [
        JudgeRecord(judge="f/j", family="f", mode="pairwise", pair=1, order="AB", preferred="A"),
        JudgeRecord(judge="f/j", family="f", mode="pairwise", pair=1, order="BA", preferred="A"),
        JudgeRecord(judge="f/j", family="f", mode="pairwise", pair=2, order="AB", preferred="A"),
        JudgeRecord(judge="f/j", family="f", mode="pairwise", pair=2, order="BA", error="bad"),
    ]

    lines = judges_section(records, pairs, ["intact", "control"], None)

    # One consistent pair; pair 2 is incomplete, so 0 inconsistent and 100% consistency.
    assert "| f | f/j | 2 | 1 | 0 | 0 | 1 | 50% |" in "\n".join(lines)
    assert "| 100% |" in "\n".join(lines)
