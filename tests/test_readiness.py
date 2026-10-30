import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from nde import briefs
from nde.cli import main
from nde.eval.batch import BatchError, Scoring, Source
from nde.eval.readiness import GateBatchSettings, GateCohort, build_gate_batch
from nde.files import checksum_lf
from tests.conftest import REPO
from tests.fake_openai import FakeOpenAI
from tests.stories import reply_with_story, run, story_config


@pytest.fixture
def cohort_run(
    fake_llm: FakeOpenAI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = lambda _body: reply_with_story(2000)
    benchmark = tmp_path / "benchmark"
    (benchmark / "dev").mkdir(parents=True)
    shutil.copy(REPO / "benchmark/dev/brief-01.yaml", benchmark / "dev")
    briefs.freeze(benchmark, "dev")
    config = story_config(
        fake_llm.base_url,
        [{"name": "A", "pipeline": "baseline_a"}, {"name": "B", "pipeline": "baseline_a"}],
        briefs_dir=str(benchmark),
        briefs={"ids": ["brief-01"]},
        equal_budget={"conditions": ["A", "B"]},
    )
    run_dir = run(tmp_path, config)
    cohort = {
        "id": "synthetic-test-only",
        "brief_checksums": {
            "brief-01": checksum_lf((benchmark / "dev/brief-01.yaml").read_bytes())
        },
        "split_checksums": {"dev": briefs.split_checksum(benchmark, "dev")},
        "conditions": ["A", "B"],
        "repeat": 1,
        "length_tolerance": 0.1,
        "budget_measure": "generation_tokens",
        "budget_tolerance": 0.1,
    }
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(cohort), encoding="utf-8")
    return path, benchmark, run_dir


def readiness(cohort_run: tuple[Path, Path, Path], out: Path) -> tuple[int, dict[str, Any]]:
    cohort, benchmark, run_dir = cohort_run
    code = main(
        [
            "eval",
            "readiness",
            str(cohort),
            "--benchmark",
            str(benchmark),
            "--from",
            str(run_dir),
            "A",
            "--from",
            str(run_dir),
            "B",
            "--out",
            str(out),
        ]
    )
    return code, json.loads((out / "readiness.json").read_text(encoding="utf-8"))


def test_exact_saved_cohort_is_ready_and_has_inspectable_reports(
    cohort_run: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    code, result = readiness(cohort_run, tmp_path / "admission")
    assert code == 0
    assert result["status"] == "ready"
    assert len(result["pairs"]) == 1
    assert result["pairs"][0]["problems"] == []
    assert "ready" in (tmp_path / "admission/readiness.md").read_text(encoding="utf-8")


def batch_command(cohort_run: tuple[Path, Path, Path], tmp_path: Path, *extra: str) -> list[str]:
    _, _, run_dir = cohort_run
    return [
        "eval",
        "batch",
        "gate",
        "--from",
        str(run_dir),
        "A",
        "--from",
        str(run_dir),
        "B",
        "--batches-dir",
        str(tmp_path / "batches"),
        "--rubric-dir",
        str(REPO / "benchmark/rubric"),
        "--optional-scores",
        *extra,
    ]


def test_batch_with_a_cohort_stores_readiness_only_for_an_admitted_cohort(
    cohort_run: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    cohort, benchmark, _ = cohort_run
    command = batch_command(
        cohort_run, tmp_path, "--cohort", str(cohort), "--benchmark", str(benchmark)
    )
    assert main(command) == 0
    assert (tmp_path / "batches/gate/sealed/readiness/readiness.json").is_file()


def test_batch_with_a_cohort_refuses_an_unready_cohort_without_building(
    cohort_run: tuple[Path, Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cohort, benchmark, _ = cohort_run
    raw = json.loads(cohort.read_text(encoding="utf-8"))
    raw["brief_checksums"]["brief-01"] = "0" * 64
    cohort.write_text(json.dumps(raw), encoding="utf-8")
    command = batch_command(
        cohort_run, tmp_path, "--cohort", str(cohort), "--benchmark", str(benchmark)
    )
    assert main(command) == 2
    assert "not ready" in capsys.readouterr().err
    assert not (tmp_path / "batches/gate").exists()


@pytest.mark.parametrize("flag", ["--cohort", "--benchmark"])
def test_batch_needs_a_cohort_and_benchmark_together(
    cohort_run: tuple[Path, Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    flag: str,
) -> None:
    assert main(batch_command(cohort_run, tmp_path, flag, str(tmp_path))) == 2
    assert "together" in capsys.readouterr().err


def test_batch_reports_an_unreadable_cohort(
    cohort_run: tuple[Path, Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, benchmark, _ = cohort_run
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    command = batch_command(
        cohort_run, tmp_path, "--cohort", str(broken), "--benchmark", str(benchmark)
    )
    assert main(command) == 2
    assert "cannot read cohort" in capsys.readouterr().err


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "failed",
        "duplicate",
        "extra",
        "short",
        "long",
        "changed",
        "budget",
        "calls",
        "unmatched",
        "frozen",
    ],
)
def test_every_invalid_pair_is_retained_and_blocks_admission(
    cohort_run: tuple[Path, Path, Path], tmp_path: Path, mutation: str
) -> None:
    _, benchmark, run_dir = cohort_run
    story_index = run_dir / "stories.jsonl"
    records = [json.loads(line) for line in story_index.read_text(encoding="utf-8").splitlines()]
    first = run_dir / records[0]["path"]
    if mutation == "missing":
        (first / "story.md").unlink()
    elif mutation == "failed":
        records[0]["status"] = "failed"
    elif mutation == "duplicate":
        records.append(records[0])
    elif mutation == "extra":
        records.append(records[0] | {"brief": "brief-99"})
    elif mutation in {"short", "long", "changed"}:
        words = {"short": 1500, "long": 2500, "changed": 2000}[mutation]
        (first / "story.md").write_text("altered " * words, encoding="utf-8")
    elif mutation == "budget":
        records[0]["budget"]["input_tokens"] += 1000
    elif mutation == "calls":
        (run_dir / "calls.jsonl").write_text("", encoding="utf-8")
    elif mutation == "unmatched":
        manifest_file = run_dir / "manifest.json"
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifest["stories"]["equal_budget"]["matches"][0]["matched"] = False
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "frozen":
        with (benchmark / "dev/brief-01.yaml").open("a", encoding="utf-8") as handle:
            handle.write("\n# changed frozen input\n")
    story_index.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    code, result = readiness(cohort_run, tmp_path / "admission")

    assert code == 1
    assert result["status"] == "incomplete"
    assert [(pair["brief"], pair["repetition"]) for pair in result["pairs"]] == [("brief-01", 0)]
    assert result["problems"] or result["pairs"][0]["problems"]


def test_malformed_registration_writes_report_and_uses_usage_exit(
    cohort_run: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    cohort_run[0].write_text('{"budget_measure": "unregistered"}', encoding="utf-8")
    code, report = readiness(cohort_run, tmp_path / "admission")
    assert code == 2
    assert report["status"] == "malformed"


def test_gate_batch_cannot_drop_missing_pairs_by_intersection(
    cohort_run: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    cohort_path, benchmark, run_dir = cohort_run
    index = run_dir / "stories.jsonl"
    first = index.read_text(encoding="utf-8").splitlines()[0]
    index.write_text(first + "\n", encoding="utf-8")
    cohort = GateCohort.model_validate_json(cohort_path.read_text(encoding="utf-8"))
    with pytest.raises(BatchError, match="not ready"):
        build_gate_batch(
            cohort,
            benchmark,
            (Source(run_dir, "A"), Source(run_dir, "B")),
            GateBatchSettings(tmp_path / "batches", "synthetic", Scoring(None), seed=7),
        )
    assert not (tmp_path / "batches/synthetic").exists()


@pytest.mark.parametrize(("tokens", "expected"), [(1809, 0), (1808, 1)])
def test_budget_admission_recomputes_inclusive_registered_boundary(
    cohort_run: tuple[Path, Path, Path], tmp_path: Path, tokens: int, expected: int
) -> None:
    run_dir = cohort_run[2]
    calls_path = run_dir / "calls.jsonl"
    calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
    # The original budget is 2,010 tokens; 1,809 is exactly within 10%.
    calls[0]["output_tokens"] = tokens - 10
    calls_path.write_text("".join(json.dumps(call) + "\n" for call in calls), encoding="utf-8")
    index = run_dir / "stories.jsonl"
    records = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    records[0]["budget"]["output_tokens"] = tokens - 10
    index.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stories"]["equal_budget"]["matches"][0]["after"]["A"] = tokens
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    code, _ = readiness(cohort_run, tmp_path / "admission")
    assert code == expected
