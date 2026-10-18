import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nde.cli import main
from nde.runs.benchmark import benchmark_rows
from nde.runs.records import CallRecord, StructuredMetrics
from nde.schema.benchmark import StoryOutline
from tests.conftest import REPO, base_config, write_config
from tests.fake_openai import FakeOpenAI, Json, Reply, completion

OUTLINE: dict[str, Any] = {
    "title": "The Last Lamp",
    "logline": "A keeper must relight a dead lighthouse before a ship founders.",
    "characters": [
        {"id": "c1", "name": "Ada", "role": "protagonist", "want": "save the ship"},
        {"id": "c2", "name": "Bram", "role": "antagonist", "want": "wreck it", "flaw": "greed"},
    ],
    "locations": [{"id": "l1", "name": "Lighthouse"}, {"id": "l2", "name": "Cove"}],
    "scenes": [
        {
            "id": f"s{i}",
            "title": f"Scene {i}",
            "location_id": "l1" if i % 2 else "l2",
            "pov_character_id": "c1",
            "character_ids": ["c1", "c2"],
            "beats": [{"kind": "complication", "summary": "things get worse"}],
            "tension": min(i, 5),
            "pays_off_scene_ids": ["s1"] if i == 5 else [],
        }
        for i in range(1, 6)
    ],
}


def test_the_test_schema_accepts_a_consistent_outline() -> None:
    assert StoryOutline.model_validate(OUTLINE).scenes[4].pays_off_scene_ids == ["s1"]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("scenes", 0, "location_id"), "l9", "unknown location_id 'l9'"),
        (("scenes", 1, "character_ids"), ["c1", "c7"], "unknown character id 'c7'"),
        (("scenes", 0, "pays_off_scene_ids"), ["s3"], "must name earlier scenes"),
        (("characters", 1, "id"), "c1", "character ids must be unique"),
        (("characters", 0, "role"), "hero", "Input should be 'protagonist'"),
    ],
    ids=["location", "character", "payoff-order", "duplicate-id", "enum"],
)
def test_the_test_schema_rejects_broken_references_and_enums(
    path: tuple[str | int, ...], value: object, message: str
) -> None:
    outline = copy.deepcopy(OUTLINE)
    node: Any = outline
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value

    with pytest.raises(ValidationError, match=message):
        StoryOutline.model_validate(outline)


def _call(model: str, repetition: int, attempt: int, latency: float) -> CallRecord:
    provider, name = model.split("/")
    return CallRecord(
        call="outline",
        repetition=repetition,
        attempt=attempt,
        provider=provider,
        model=name,
        reported_model=name,
        template="g0_outline",
        template_version=1,
        seed=repetition,
        input_tokens=50,
        output_tokens=100,
        latency_s=latency,
        cached=attempt == 1,
        cost_usd=0.0 if attempt == 1 else 0.1,
        token_cost_usd=0.1,
    )


def _metrics(calls: int, valid_first: int, valid: int, repaired: int) -> StructuredMetrics:
    metrics = StructuredMetrics()
    for index in range(calls):
        attempts = 1 if index < valid_first else 2
        metrics.record(attempts=attempts, valid=index < valid)
    assert metrics.repaired_calls == repaired
    return metrics


def test_benchmark_rows_summarise_validity_latency_and_cost_per_model() -> None:
    calls = [
        _call("a/x", 0, 0, 1.0),
        _call("a/x", 0, 1, 2.0),
        _call("a/x", 1, 0, 1.0),
        _call("b/y", 0, 0, 4.0),
        _call("b/y", 1, 0, 6.0),
    ]
    structured = {"a/x": _metrics(2, 1, 2, 1), "b/y": _metrics(2, 1, 1, 1)}

    good, bad = benchmark_rows(calls, structured, threshold=0.95)

    assert good.model == "a/x"
    assert (good.calls, good.validity_after_repair, good.meets_threshold) == (2, 1.0, True)
    assert (good.first_attempt_validity, good.repair_rate) == (0.5, 0.5)
    # Latency per logical call, all attempts included: [1.0 + 2.0, 1.0].
    assert (good.latency_median_s, good.latency_p95_s) == (2.0, 3.0)
    # Three requests of 100 output tokens cost $0.30 at list price.
    assert good.cost_per_1k_output_usd == pytest.approx(1.0)
    assert good.token_cost_usd == pytest.approx(0.3)
    assert good.new_spend_usd == pytest.approx(0.2)
    assert (bad.model, bad.validity_after_repair, bad.meets_threshold) == ("b/y", 0.5, False)
    assert (bad.latency_median_s, bad.latency_p95_s) == (5.0, 6.0)


def _reply_by_model(body: Json) -> Reply:
    """`good` answers with a valid outline; `bad` always breaks a reference."""
    if body["model"] == "good":
        return completion(json.dumps(OUTLINE), completion_tokens=400)
    broken = copy.deepcopy(OUTLINE)
    broken["scenes"][0]["location_id"] = "l9"
    return completion(json.dumps(broken), completion_tokens=400)


@pytest.fixture
def benchmark_config(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = _reply_by_model
    config = base_config(fake_llm.base_url)
    config["roles"] = {}
    config["calls"] = [
        {
            "id": "outline",
            "role": "planner",
            "prompt": {"template": "g0_outline", "version": 1},
            "variables": {"brief": "A lighthouse keeper in a storm."},
            "response_model": "story_outline",
            "repeat": 3,
        }
    ]
    config["benchmark"] = {
        "role": "planner",
        "candidates": [
            {"provider": "fake", "model": "good", "params": {"max_tokens": 1000}},
            {"provider": "fake", "model": "bad", "params": {"max_tokens": 1000}},
        ],
    }
    config["structured"] = {"max_repairs": 1}
    config["prices"] = {
        "fake/good": {"input_per_mtok": 1.0, "output_per_mtok": 2.0},
        "fake/bad": {"input_per_mtok": 1.0, "output_per_mtok": 2.0},
    }
    config["budget"] = {"per_run_usd": 1.0}
    return config


def _run(tmp_path: Path, config: dict[str, Any]) -> Path:
    runs = tmp_path / "runs"
    before: set[Path] = set(runs.iterdir()) if runs.exists() else set()
    assert main(["run", str(write_config(tmp_path, config)), "--runs-dir", str(runs)]) == 0
    [run_dir] = set(runs.iterdir()) - before
    return run_dir


def _benchmark_table(report: str) -> list[str]:
    section = report.split("## Benchmark (gate G0)", 1)[1].split("\n## ", 1)[0]
    return [line for line in section.splitlines() if line.startswith("| fake/")]


def test_a_benchmark_run_compares_candidates_and_applies_the_g0_threshold(
    fake_llm: FakeOpenAI, benchmark_config: dict[str, Any], tmp_path: Path
) -> None:
    run_dir = _run(tmp_path, benchmark_config)

    # good: 3 valid calls. bad: 3 calls, each with one failed repair.
    assert fake_llm.call_count == 3 + 3 * 2
    assert {request.body["model"] for request in fake_llm.requests} == {"good", "bad"}
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = {row["model"]: row for row in manifest["benchmark"]["rows"]}
    assert rows["fake/good"]["meets_threshold"] is True
    assert rows["fake/bad"]["meets_threshold"] is False
    assert [m["model"] for m in manifest["models"]] == ["good", "bad"]
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    good, bad = _benchmark_table(report)
    assert good.startswith("| fake/good | 3 | 100.0% | 100.0% | 0.0% |")
    assert good.endswith("| yes |")
    assert bad.startswith("| fake/bad | 3 | 0.0% | 0.0% | 100.0% |")
    assert bad.endswith("| no |")
    assert "threshold: 95%" in report


def test_a_second_benchmark_run_is_free_and_reports_the_same_table(
    fake_llm: FakeOpenAI, benchmark_config: dict[str, Any], tmp_path: Path
) -> None:
    first = _run(tmp_path, benchmark_config)
    calls = fake_llm.call_count

    second = _run(tmp_path, benchmark_config)

    assert fake_llm.call_count == calls
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert second_manifest["cost"]["usd"] == 0.0
    first_report = (first / "report.md").read_text(encoding="utf-8")
    second_report = (second / "report.md").read_text(encoding="utf-8")
    assert _benchmark_table(second_report) == _benchmark_table(first_report)


def test_the_per_run_cap_stops_a_benchmark_before_it_overspends(
    fake_llm: FakeOpenAI, benchmark_config: dict[str, Any], tmp_path: Path
) -> None:
    # Each request reserves at least $0.002 (1,000 output tokens at $2 per million).
    benchmark_config["budget"] = {"per_run_usd": 0.005}
    runs = tmp_path / "runs"

    code = main(["run", str(write_config(tmp_path, benchmark_config)), "--runs-dir", str(runs)])

    assert code == 1
    assert 0 < fake_llm.call_count < 9
    [run_dir] = list(runs.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["abort_reason"] == "budget"
    assert manifest["cost"]["usd"] <= 0.005
    assert len(manifest["benchmark"]["rows"]) == 2


def test_the_committed_benchmark_config_defines_candidates_and_repetitions() -> None:
    config = (REPO / "configs" / "benchmark-g0.yaml").read_text(encoding="utf-8")

    assert "candidates:" in config
    assert "repeat:" in config


def test_the_report_regenerates_from_the_recorded_run_without_network(
    fake_llm: FakeOpenAI, benchmark_config: dict[str, Any], tmp_path: Path
) -> None:
    run_dir = _run(tmp_path, benchmark_config)
    original = (run_dir / "report.md").read_text(encoding="utf-8")
    (run_dir / "report.md").unlink()
    fake_llm.stop()

    assert main(["report", str(run_dir)]) == 0

    assert (run_dir / "report.md").read_text(encoding="utf-8") == original
