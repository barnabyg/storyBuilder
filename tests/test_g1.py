"""Offline protocol checks; all evidence here is explicitly test-only."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from nde import briefs
from nde.cli import main
from nde.eval.g1 import G1Binding, G1Protocol, OriginalitySafeguard, binding_problems
from nde.eval.readiness import PairReadiness, ReadinessReport, cohort_checksum, write_readiness
from nde.files import checksum_lf
from nde.runs.config import load_config
from tests.conftest import REPO, write_config
from tests.fake_openai import FakeOpenAI
from tests.g1_fixtures import synthetic_schema_packet
from tests.stories import story_config
from tests.test_tiers import tier_provider


def git_fixture(repo: Path, *args: str) -> None:
    # S603/S607: fixed git arguments in a temporary synthetic test repository.
    # Review these exceptions if arguments accept non-test data or the executable changes.
    subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
        env=os.environ
        | {
            "GIT_AUTHOR_NAME": "Synthetic test only",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Synthetic test only",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        },
    )


def test_g1_prerequisite_check_reports_absent_protocol_without_generation(tmp_path: Path) -> None:
    out = tmp_path / "check"
    code = main(
        [
            "eval",
            "g1-check",
            str(tmp_path / "protocol.json"),
            str(tmp_path / "config.yaml"),
            "--selection",
            str(tmp_path / "selection.json"),
            "--schema-evidence",
            str(tmp_path / "schema"),
            "--out",
            str(out),
        ]
    )
    assert code == 2
    report = json.loads((out / "g1-check.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["problems"]
    assert not (tmp_path / "runs").exists()


def test_g1_replays_schema_evidence_before_rejecting_changed_selection(
    registered_inputs: dict[str, Path],
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = registered_inputs
    schema = synthetic_schema_packet(tmp_path / "passed-schema", fake_llm, monkeypatch)
    raw = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    raw["schema_evidence_sha256"] = checksum_lf((schema / "evidence.json").read_bytes())
    paths["protocol"].write_text(json.dumps(raw), encoding="utf-8")
    paths["selection"].write_text("{}", encoding="utf-8")
    git_fixture(tmp_path, "add", "protocol.json")
    git_fixture(tmp_path, "commit", "-q", "-m", "synthetic passed schema protocol")
    out = tmp_path / "checked"
    assert (
        main(
            [
                "eval",
                "g1-check",
                str(paths["protocol"]),
                str(paths["config"]),
                "--selection",
                str(paths["selection"]),
                "--schema-evidence",
                str(schema),
                "--out",
                str(out),
            ]
        )
        == 2
    )
    result = json.loads((out / "g1-check.json").read_text(encoding="utf-8"))
    assert "tier selection checksum differs" in " ".join(result["problems"])


def test_synthetic_owner_evidence_cannot_enable_real_g1_generation(
    registered_inputs: dict[str, Path],
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = registered_inputs
    schema = synthetic_schema_packet(tmp_path / "passed-schema", fake_llm, monkeypatch)
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    config["stories"]["briefs_dir"] = str(REPO / "benchmark")
    config["stories"]["briefs"] = {"split": "test"}
    paths["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    raw = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    raw["test_only"] = False
    raw["win_threshold"] = 21
    raw["cohort"]["brief_checksums"] = briefs.read_frozen(REPO / "benchmark")["test"]
    raw["cohort"]["split_checksums"] = {"test": briefs.split_checksum(REPO / "benchmark", "test")}
    raw["config_sha256"] = load_config(paths["config"]).hash
    raw["schema_evidence_sha256"] = checksum_lf((schema / "evidence.json").read_bytes())
    paths["protocol"].write_text(json.dumps(raw), encoding="utf-8")
    git_fixture(tmp_path, "add", "protocol.json")
    git_fixture(tmp_path, "commit", "-q", "-m", "synthetic invalid real-mode preregistration")
    before = fake_llm.call_count
    assert (
        main(
            [
                "eval",
                "g1-run",
                str(paths["protocol"]),
                str(paths["config"]),
                "--selection",
                str(paths["selection"]),
                "--schema-evidence",
                str(schema),
                "--runs-dir",
                str(tmp_path / "held-out-runs"),
            ]
        )
        == 2
    )
    assert fake_llm.call_count == before
    assert not (tmp_path / "held-out-runs").exists()


@pytest.mark.parametrize("mutation", ["protocol", "readiness", "cohort_checksum"])
def test_bound_result_cannot_reinterpret_saved_protocol_or_admission(
    registered_inputs: dict[str, Path], tmp_path: Path, mutation: str
) -> None:
    paths = registered_inputs
    paths["selection"].write_text("synthetic selection identity", encoding="utf-8")
    raw = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    raw["selection_sha256"] = checksum_lf(paths["selection"].read_bytes())
    paths["protocol"].write_text(json.dumps(raw), encoding="utf-8")
    protocol = G1Protocol.model_validate(raw)
    batch = tmp_path / "synthetic-bound-result"
    (batch / "sealed").mkdir(parents=True)
    (batch / "view.json").write_text("{}", encoding="utf-8")
    (batch / "sealed/key.json").write_text("{}", encoding="utf-8")
    readiness = ReadinessReport(
        status="ready",
        cohort=protocol.cohort,
        cohort_sha256=cohort_checksum(protocol.cohort),
        benchmark=str(paths["benchmark"]),
        sources=[],
    )
    write_readiness(readiness, batch / "sealed/readiness")
    binding = G1Binding(
        protocol=protocol,
        protocol_sha256=checksum_lf(paths["protocol"].read_bytes()),
        protocol_path=str(paths["protocol"]),
        protocol_commit="synthetic-test-only",
        selection_path=str(paths["selection"]),
        schema_evidence_path=str(paths["schema"]),
        readiness=readiness,
        view_sha256=checksum_lf((batch / "view.json").read_bytes()),
        key_sha256=checksum_lf((batch / "sealed/key.json").read_bytes()),
        artifact_checksums={},
    )
    assert binding_problems(batch, binding) == []
    if mutation == "protocol":
        changed_protocol = protocol.model_copy(
            update={
                "originality": OriginalitySafeguard(analysis="mean_paired_difference", margin=0.75)
            }
        )
        binding = binding.model_copy(update={"protocol": changed_protocol})
    elif mutation == "readiness":
        changed = readiness.model_copy(
            update={"pairs": [PairReadiness(brief="brief-01", repetition=0, problems=["changed"])]}
        )
        binding = binding.model_copy(update={"readiness": changed})
    else:
        changed = readiness.model_copy(update={"cohort_sha256": "b" * 64})
        binding = binding.model_copy(update={"readiness": changed})
    assert binding_problems(batch, binding)


def prepare_registered_workflow(
    registered_inputs: dict[str, Path],
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = registered_inputs
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = tier_provider
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    config["providers"]["fake"]["base_url"] = fake_llm.base_url
    paths["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    comparison = tmp_path / "dev-comparison"
    assert main(["tiers", "compare", str(paths["config"]), "--out", str(comparison)]) == 0
    packet = json.loads((comparison / "comparison.json").read_text(encoding="utf-8"))
    review = tmp_path / "synthetic-review.json"
    review.write_text(
        json.dumps(
            {
                "comparison_sha256": packet["id"],
                "reviewer": "synthetic",
                "test_only": True,
                "judgements": [
                    {
                        "brief": "brief-01",
                        "tier": "L",
                        "coherence": 4,
                        "originality": 4,
                        "note": "TEST ONLY",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main(["tiers", "review", str(comparison), str(review)]) == 0
    assert (
        main(
            [
                "tiers",
                "select",
                str(comparison),
                "--tier",
                "L",
                "--owner",
                "synthetic",
                "--rationale",
                "Synthetic test-only choice",
                "--test-only",
                "--out",
                str(paths["selection"]),
            ]
        )
        == 0
    )
    schema = synthetic_schema_packet(tmp_path / "passed-schema", fake_llm, monkeypatch)
    raw = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    raw["selection_sha256"] = checksum_lf(paths["selection"].read_bytes())
    raw["schema_evidence_sha256"] = checksum_lf((schema / "evidence.json").read_bytes())
    raw["config_sha256"] = load_config(paths["config"]).hash
    paths["protocol"].write_text(json.dumps(raw), encoding="utf-8")
    git_fixture(tmp_path, "add", "protocol.json")
    git_fixture(tmp_path, "commit", "-q", "-m", "synthetic registered G1 dependencies")
    common = [
        str(paths["protocol"]),
        str(paths["config"]),
        "--selection",
        str(paths["selection"]),
        "--schema-evidence",
        str(schema),
    ]
    return common, packet, config, raw


def test_registered_synthetic_workflow_generates_then_admits_a_sealed_exact_batch(
    registered_inputs: dict[str, Path],
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = registered_inputs
    common, packet, config, raw = prepare_registered_workflow(
        paths, tmp_path, fake_llm, monkeypatch
    )
    assert main(["eval", "g1-check", *common, "--out", str(tmp_path / "checked")]) == 0
    batches = tmp_path / "batches"
    ordinary_run = packet["runs"]["L"]
    assert (
        main(
            [
                "eval",
                "g1-batch",
                *common,
                "--name",
                "ordinary-refused",
                "--batches-dir",
                str(batches),
                "--from",
                ordinary_run,
                "C",
                "--from",
                ordinary_run,
                "B",
                "--rubric-dir",
                str(REPO / "benchmark/rubric"),
            ]
        )
        == 2
    )
    assert not (batches / "ordinary-refused").exists()
    runs = tmp_path / "registered-runs"
    assert main(["eval", "g1-run", *common, "--runs-dir", str(runs)]) == 0
    [run_dir] = list(runs.iterdir())
    assert (run_dir / "g1-generation.json").is_file()
    assert (
        main(
            [
                "eval",
                "g1-batch",
                *common,
                "--name",
                "synthetic",
                "--batches-dir",
                str(batches),
                "--from",
                str(run_dir),
                "C",
                "--from",
                str(run_dir),
                "B",
                "--seed",
                "7",
                "--rubric-dir",
                str(REPO / "benchmark/rubric"),
            ]
        )
        == 0
    )
    batch = batches / "synthetic"
    binding = G1Binding.model_validate_json((batch / "sealed/g1.json").read_text(encoding="utf-8"))
    assert binding.test_only
    assert binding.readiness.status == "ready"
    assert len(binding.readiness.pairs) == 1
    assert binding_problems(batch, binding) == []
    original = (run_dir / "stories/C/brief-01/r1/story.md").read_text(encoding="utf-8")
    (run_dir / "stories/C/brief-01/r1/story.md").write_text(original + " altered", encoding="utf-8")
    assert binding_problems(batch, binding)
    (run_dir / "stories/C/brief-01/r1/story.md").write_text(original, encoding="utf-8")
    index = run_dir / "stories.jsonl"
    records = index.read_text(encoding="utf-8").splitlines()
    index.write_text(records[0] + "\n", encoding="utf-8")
    assert (
        main(
            [
                "eval",
                "g1-batch",
                *common,
                "--name",
                "incomplete-refused",
                "--batches-dir",
                str(batches),
                "--from",
                str(run_dir),
                "C",
                "--from",
                str(run_dir),
                "B",
                "--rubric-dir",
                str(REPO / "benchmark/rubric"),
            ]
        )
        == 2
    )
    assert not (batches / "incomplete-refused").exists()
    # Register the changed config exactly, so stale dev selection is the blocking evidence.
    config["roles"]["prose"]["params"]["temperature"] = 0.5
    paths["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    raw["config_sha256"] = load_config(paths["config"]).hash
    paths["protocol"].write_text(json.dumps(raw), encoding="utf-8")
    git_fixture(tmp_path, "add", "protocol.json")
    git_fixture(tmp_path, "commit", "-q", "-m", "synthetic stale tier settings")
    out = tmp_path / "stale-check"
    assert main(["eval", "g1-check", *common, "--out", str(out)]) == 2
    result = json.loads((out / "g1-check.json").read_text(encoding="utf-8"))
    assert "tier selection is stale" in " ".join(result["problems"])


@pytest.fixture
def registered_inputs(tmp_path: Path) -> dict[str, Path]:
    benchmark = tmp_path / "benchmark"
    (benchmark / "dev").mkdir(parents=True)
    shutil.copy(REPO / "benchmark/dev/brief-01.yaml", benchmark / "dev")
    briefs.freeze(benchmark, "dev")
    config = story_config(
        "http://127.0.0.1:1",
        [{"name": "C", "pipeline": "baseline_c"}, {"name": "B", "pipeline": "baseline_b"}],
        briefs_dir=str(benchmark),
        briefs={"ids": ["brief-01"]},
        equal_budget={"conditions": ["C", "B"]},
    )
    config["budget"] = {"per_run_usd": 0.1, "per_day_usd": 1}
    path = write_config(tmp_path, config)
    loaded = load_config(path)
    schema = tmp_path / "schema"
    assert (
        main(
            [
                "schema-evidence",
                "--manifest",
                str(REPO / "data/manifest.calibration.yaml"),
                "--out",
                str(schema),
            ]
        )
        == 3
    )
    raw = {
        "id": "synthetic-test-only",
        "test_only": True,
        "cohort": {
            "id": "synthetic-test-only",
            "brief_checksums": briefs.read_frozen(benchmark)["dev"],
            "split_checksums": {"dev": briefs.split_checksum(benchmark, "dev")},
            "conditions": ["C", "B"],
            "repeat": 1,
            "length_tolerance": 0.1,
            "budget_measure": "generation_tokens",
            "budget_tolerance": 0.1,
        },
        "tier": "L",
        "win_threshold": 1,
        "originality": {"analysis": "mean_paired_difference", "margin": 0.25},
        "reading_mode": "required_only",
        "config_sha256": loaded.hash,
        "selection_sha256": "a" * 64,
        "schema_evidence_sha256": checksum_lf((schema / "evidence.json").read_bytes()),
        "generation_limits": {
            "max_repairs": loaded.config.structured.max_repairs,
            "max_length_attempts": 3,
            "max_revision_passes": 4,
            "per_run_usd": 0.1,
            "per_day_usd": 1,
        },
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(raw), encoding="utf-8")
    git_fixture(tmp_path, "init", "-q")
    git_fixture(tmp_path, "add", "protocol.json")
    git_fixture(tmp_path, "commit", "-q", "-m", "synthetic test-only protocol")
    return {
        "protocol": protocol,
        "config": path,
        "selection": tmp_path / "selection.json",
        "schema": schema,
        "benchmark": benchmark,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("pending", "G-schema owner gate has not passed"),
        ("uncommitted", "pre-registration is uncommitted"),
        ("dirty", "differs from its committed contents"),
        ("configuration", "configuration differs from registration"),
        ("frozen", "changed after"),
        ("unsupported", "mean_paired_difference"),
        ("absent_schema", "evidence.json"),
    ],
)
def test_g1_blocks_generation_until_registered_prerequisites_are_valid(
    registered_inputs: dict[str, Path], tmp_path: Path, mutation: str, message: str
) -> None:
    paths = registered_inputs
    if mutation == "uncommitted":
        git_fixture(tmp_path, "rm", "--cached", "protocol.json")
    elif mutation == "dirty":
        with paths["protocol"].open("a", encoding="utf-8") as handle:
            handle.write("\n")
    elif mutation == "configuration":
        with paths["config"].open("a", encoding="utf-8") as handle:
            handle.write("\nseed: 99\n")
    elif mutation == "frozen":
        with (paths["benchmark"] / "dev/brief-01.yaml").open("a", encoding="utf-8") as handle:
            handle.write("\n# changed frozen fixture\n")
    elif mutation == "unsupported":
        raw = json.loads(paths["protocol"].read_text(encoding="utf-8"))
        raw["originality"]["analysis"] = "decide_after_reading"
        paths["protocol"].write_text(json.dumps(raw), encoding="utf-8")
    elif mutation == "absent_schema":
        (paths["schema"] / "evidence.json").unlink()
    out = tmp_path / "checked"
    code = main(
        [
            "eval",
            "g1-check",
            str(paths["protocol"]),
            str(paths["config"]),
            "--selection",
            str(paths["selection"]),
            "--schema-evidence",
            str(paths["schema"]),
            "--out",
            str(out),
        ]
    )
    result = json.loads((out / "g1-check.json").read_text(encoding="utf-8"))
    assert code == 2
    assert message in " ".join(result["problems"])
    assert not (tmp_path / "runs").exists()
