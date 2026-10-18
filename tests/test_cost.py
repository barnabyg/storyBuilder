import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from nde.llm.config import ModelBinding, ProviderConfig
from nde.llm.gateway import Gateway, Prompt
from nde.llm.ledger import BudgetExceededError, CostLedger, Price, Pricing
from tests.conftest import base_config, write_config
from tests.fake_openai import FakeOpenAI, completion

# Output tokens cost $1,000 per million so that one 100-token reply costs $0.10.
PRICES = {"fake/fake-model": {"input_per_mtok": 0.0, "output_per_mtok": 1000.0}}


@pytest.fixture
def priced(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = lambda _body: completion("reply", prompt_tokens=10, completion_tokens=100)
    config = base_config(fake_llm.base_url)
    config["roles"]["prose"]["params"] = {"max_tokens": 100}
    config["prices"] = PRICES
    return config


def _run(tmp_path: Path, config: dict[str, Any], *flags: str) -> tuple[int, Path]:
    runs = tmp_path / "runs"
    before: set[Path] = set(runs.iterdir()) if runs.exists() else set()
    code = main(["run", str(write_config(tmp_path, config)), "--runs-dir", str(runs), *flags])
    [run_dir] = set(runs.iterdir()) - before
    return code, run_dir


def _manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _calls(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_each_call_records_tokens_and_cost_with_totals_in_manifest_and_report(
    priced: dict[str, Any], tmp_path: Path
) -> None:
    priced["calls"][0]["repeat"] = 2

    code, run_dir = _run(tmp_path, priced)

    assert code == 0
    assert [(c["input_tokens"], c["output_tokens"], c["cost_usd"]) for c in _calls(run_dir)] == [
        (10, 100, pytest.approx(0.1)),
        (10, 100, pytest.approx(0.1)),
    ]
    cost = _manifest(run_dir)["cost"]
    assert (cost["input_tokens"], cost["output_tokens"]) == (20, 200)
    assert cost["usd"] == pytest.approx(0.2)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Cost" in report
    assert "$0.2000" in report


def test_cache_hits_cost_nothing(
    fake_llm: FakeOpenAI, priced: dict[str, Any], tmp_path: Path
) -> None:
    _run(tmp_path, priced)

    code, rerun = _run(tmp_path, priced)

    assert code == 0
    assert fake_llm.call_count == 1
    [call] = _calls(rerun)
    assert (call["cached"], call["cost_usd"]) == (True, 0.0)
    assert _manifest(rerun)["cost"]["usd"] == 0.0
    assert "New spend this run: $0.0000" in (rerun / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("limit", ["max_tokens", "max_completion_tokens"])
def test_a_run_cap_below_one_call_stops_before_calling(
    fake_llm: FakeOpenAI, priced: dict[str, Any], tmp_path: Path, limit: str
) -> None:
    # Newer OpenAI models take max_completion_tokens; both bound the estimate.
    priced["roles"]["prose"]["params"] = {limit: 100}
    priced["budget"] = {"per_run_usd": 0.05}

    code, run_dir = _run(tmp_path, priced, "--no-cache")

    assert code == 1
    assert fake_llm.call_count == 0
    manifest = _manifest(run_dir)
    assert (manifest["status"], manifest["abort_reason"]) == ("aborted", "budget")
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "aborted: budget" in report
    assert "per-run cap of $0.0500" in report


def test_a_run_stops_cleanly_when_its_next_call_would_exceed_the_run_cap(
    fake_llm: FakeOpenAI, priced: dict[str, Any], tmp_path: Path
) -> None:
    priced["calls"][0]["repeat"] = 3
    priced["budget"] = {"per_run_usd": 0.25}

    code, run_dir = _run(tmp_path, priced)

    assert code == 1
    assert fake_llm.call_count == 2
    assert _manifest(run_dir)["status"] == "aborted"
    assert len(_calls(run_dir)) == 2
    outputs = (run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(outputs) == 2


def test_the_daily_total_persists_across_runs(
    fake_llm: FakeOpenAI, priced: dict[str, Any], tmp_path: Path
) -> None:
    priced["budget"] = {"per_day_usd": 0.25}

    first, _ = _run(tmp_path, priced, "--no-cache")
    second, _ = _run(tmp_path, priced, "--no-cache")
    third, run_dir = _run(tmp_path, priced, "--no-cache")

    assert (first, second, third) == (0, 0, 1)
    assert fake_llm.call_count == 2
    assert "per-day cap of $0.2500" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_an_unpriced_model_fails_fast(
    fake_llm: FakeOpenAI,
    priced: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    priced["prices"] = {}

    code = main(["run", str(write_config(tmp_path, priced)), "--runs-dir", str(tmp_path / "r")])

    assert code == 2
    assert "fake/fake-model has no price" in capsys.readouterr().err
    assert fake_llm.call_count == 0


def test_an_unpriced_model_can_be_allowed_with_a_recorded_warning(
    priced: dict[str, Any], tmp_path: Path
) -> None:
    priced["prices"] = {}
    priced["budget"] = {"allow_unpriced_models": True}

    code, run_dir = _run(tmp_path, priced)

    assert code == 0
    [warning] = _manifest(run_dir)["warnings"]
    assert "fake/fake-model has no price" in warning
    assert warning in (run_dir / "report.md").read_text(encoding="utf-8")


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 18, 23, 59, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def gateway_factory(fake_llm: FakeOpenAI, tmp_path: Path) -> Iterator[Any]:
    made: list[Gateway] = []

    def make(clock: Clock) -> Gateway:
        fake_llm.default = lambda _body: completion("x", completion_tokens=100)
        gateway = Gateway(
            {"fake": ProviderConfig(base_url=fake_llm.base_url)},
            environ={},
            pricing=Pricing({"fake/m": Price(input_per_mtok=0.0, output_per_mtok=1000.0)}),
            ledger=CostLedger(tmp_path / "ledger.sqlite", per_day_usd=0.15, clock=clock),
        )
        made.append(gateway)
        return gateway

    yield make
    for gateway in made:
        gateway.close()


def test_the_daily_cap_resets_at_the_next_utc_day(gateway_factory: Any) -> None:
    clock = Clock()
    gateway: Gateway = gateway_factory(clock)
    binding = ModelBinding(provider="fake", model="m", params={"max_tokens": 100})
    prompt = Prompt("t", 1, ({"role": "user", "content": "hi"},))

    gateway.complete(binding, prompt, seed=1)
    with pytest.raises(BudgetExceededError) as exceeded:
        gateway.complete(binding, prompt, seed=2)
    assert exceeded.value.scope == "day"

    clock.now = datetime(2026, 10, 19, 0, 1, tzinfo=UTC)
    assert gateway.complete(binding, prompt, seed=3).cost_usd == pytest.approx(0.1)
