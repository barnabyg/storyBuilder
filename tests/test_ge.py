"""Registered G-E preparation through public commands: a tiny CPU checkpoint and fake providers.

Every fixture here is synthetic and explicitly test-only. It exercises readiness,
registration, the bound dev run, replay and the owner packet; it never establishes
G3, G-E or any literary result.
"""

import copy
import importlib.util
import json
import os
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from nde import briefs
from nde.cli import main
from nde.conditional import ge, ge_review
from nde.corpus.records_io import create_record, read_sealed
from nde.runs.config import ConditionSpec, load_config
from nde.runs.runner import RunOptions
from nde.stories.execute import story_conditions, story_templates
from nde.tiers import load_selection
from tests.conditional_support import trained_checkpoint
from tests.fake_openai import FakeOpenAI, Json, Reply, completion
from tests.stories import last_user_message, story_config
from tests.test_baseline_d import BASELINE_D
from tests.test_candidate_e import Provider, e_condition
from tests.test_conditional_report import FIELDS, LOCK, Planner, tier_selection
from tests.test_g1 import git_fixture
from tests.test_reconstruction_g3 import complete, g3_inputs

HELD_OUT = "{not: [valid yaml held-out brief"
QUESTION = "Which plan has the more coherent long-range structure for this brief?"


class GEProvider(Planner):
    """E completion and prose, C's four-scene plans and its channel pass."""

    def __call__(self, body: Json) -> Reply:
        if "narrative scene-channel planner" in str(body["messages"][0]["content"]):
            request = last_user_message(body)
            return completion(request.split("Accepted plan:\n", 1)[1].split("\nCURRENT_SCENE")[0])
        return super().__call__(body)


@dataclass(frozen=True)
class Fixture:
    root: Path
    checkpoint: Path
    selection: Path
    config: Path
    registration: Path
    protocol: Path
    run: Path


def _benchmark(root: Path) -> Path:
    """Frozen dev briefs E can (brief-01) and cannot (brief-02) satisfy.

    The held-out split holds an unreadable file throughout: nothing G-E runs,
    verifies or packets may open it.
    """
    benchmark = root / "benchmark"
    (benchmark / "dev").mkdir(parents=True)
    repository = Path(__file__).resolve().parents[1] / "benchmark/dev"
    for brief, low in (("brief-01", 2), ("brief-02", 3)):
        value = yaml.safe_load((repository / f"{brief}.yaml").read_text())
        value.update(min_scenes=low, max_scenes=4)
        (benchmark / f"dev/{brief}.yaml").write_text(yaml.safe_dump(value))
    briefs.freeze(benchmark, "dev")
    (benchmark / "test").mkdir()
    (benchmark / "test/brief-99.yaml").write_text(HELD_OUT)
    return benchmark


def _config(root: Path, fake: FakeOpenAI, checkpoint: Path, selection: Path) -> Path:
    # The owner's selected C contract: the reviewed C condition at the selected tier.
    chosen = load_selection(selection)
    [c] = [item for item in chosen.config["stories"]["conditions"] if item["name"] == "C"]
    c = c | {"params": c["params"] | {"tier": chosen.tier}}
    value = story_config(
        fake.base_url,
        [e_condition(checkpoint, condition_fields=FIELDS, locks=LOCK), c],
        briefs={"split": "dev", "ids": ["brief-01", "brief-02"]},
        briefs_dir=str(_benchmark(root)),
        repeat=2,
        equal_budget={"conditions": ["E", "C"], "resource_policy": {"version": "1"}},
    )
    value["budget"] = {"per_run_usd": 1, "per_day_usd": 5}
    path = root / "ge.yaml"
    path.write_text(yaml.safe_dump(value))
    return path


RESOURCES = {
    "max_attempts": 3,
    "max_tokens": 1000000,
    "max_attention_cells": 100000000,
    "max_seconds": 60,
    "completion_max_tokens": 1024,
    "max_repairs": 2,
    "max_length_attempts": 3,
    "max_revision_passes": 4,
    "budget_tolerance": 0.1,
    "per_run_usd": 1,
    "per_day_usd": 5,
}


def registration(directory: Path, fixture: Fixture | None = None, **changes: Any) -> Path:
    """A registration file in `directory`; paths are relative to it."""
    root = directory if fixture is None else fixture.root
    value: dict[str, Any] = {
        "id": "synthetic-test-only-ge",
        "test_only": True,
        "config": os.path.relpath(root / "ge.yaml", directory),
        "reference": {
            "condition": "C",
            "c_selection": os.path.relpath(root / "tiers/selection.json", directory),
        },
        "cohort": {"briefs": ["brief-01", "brief-02"], "repeat": 2},
        "resources": RESOURCES,
        "validator_comparison": {
            "analysis": "pass_rate_difference",
            "margin": 0.1,
            "minimum_candidate_rate": 0.5,
            "aggregation": "pooled",
        },
        "structural_criterion": {
            "analysis": "owner_paired_competitive_share",
            "question": QUESTION,
            "minimum_share": 0.5,
        },
        "owner_review": {
            "reviewer": "Synthetic owner",
            "sample_size": 3,
            "sample_seed": 11,
            "minutes_per_item": 10,
            "max_minutes": 40,
        },
        "iteration": {"attempt": 1},
        "prior_exposure": "Synthetic test-only fixtures; no owner reading.",
    }
    value.update(changes)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "registration.json"
    path.write_text(json.dumps(value, indent=2))
    return path


def ge_command(*arguments: str | Path) -> int:
    return main(["ge", *(str(argument) for argument in arguments)])


def commit(root: Path, *names: str) -> None:
    git_fixture(root, "add", *names)
    git_fixture(root, "commit", "-q", "-m", "synthetic test-only G-E registration")


@pytest.fixture(scope="module")
def registered(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    if importlib.util.find_spec("torch") is None:
        if os.environ.get("NDE_REQUIRE_ML") == "1":
            pytest.fail("CPU ML lane requires the pinned torch runtime")
        pytest.skip("optional CPU ML runtime")
    root = tmp_path_factory.mktemp("ge")
    git_fixture(root, "init", "-q")
    fake = FakeOpenAI()
    fake.start()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("FAKE_API_KEY", "test-key")
        try:
            checkpoint = trained_checkpoint(root / "training", fake)
            selection, _ = tier_selection(root / "tiers", fake)
            config = _config(root, fake, checkpoint, selection)
            path = registration(root)
            assert ge_command("readiness", path, "--out", root / "readiness") == 0
            protocol = root / "protocol.json"
            assert ge_command("register", path, "--out", protocol) == 0
            # Registration is frozen before running: an uncommitted protocol cannot run.
            assert ge_command("check", protocol, "--out", root / "uncommitted") == 1
            assert ge_command("run", protocol, "--runs-dir", root / "runs") == 2
            assert not (root / "runs").exists()
            commit(root, "protocol.json")
            fake.default = GEProvider()
            runs = root / "runs"
            assert (
                ge_command(
                    "run",
                    protocol,
                    "--runs-dir",
                    runs,
                    "--cache",
                    root / "cache.sqlite",
                    "--ledger",
                    root / "ledger.sqlite",
                )
                == 0
            )
        finally:
            fake.stop()
    [run] = list(runs.iterdir())
    return Fixture(root, checkpoint, selection, config, path, protocol, run)


@contextmanager
def restored(*paths: Path) -> Generator[None]:
    """Change shared fixture evidence inside the block; its bytes come back afterwards."""
    saved = {path: path.read_bytes() for path in paths}
    try:
        yield
    finally:
        for path, data in saved.items():
            path.write_bytes(data)


@pytest.mark.ml
def test_protocol_freezes_e_reference_cohort_resources_and_criteria(
    registered: Fixture,
) -> None:
    protocol = read_sealed(registered.protocol)
    frozen = protocol["frozen"]
    # Frozen before running: E, data, conditioning, guidance, completion and the reference.
    candidate = frozen["candidate"]
    assert candidate["checkpoint"]["id"] == read_sealed(registered.checkpoint)["id"]
    assert candidate["conditioning"] == {"condition_fields": FIELDS, "locks": LOCK}
    assert candidate["guidance"]["scale"] == 0
    assert candidate["completion"]["role"] == "planner"
    assert frozen["lineage"]["fine_tuning"]["status"] == "verified"
    assert frozen["g3"]["status"] == "test-only"
    assert frozen["reference"]["identification"] == "owner C selection"
    assert frozen["cohort"]["slots"] == ["brief-01/r1", "brief-01/r2", "brief-02/r1", "brief-02/r2"]
    assert set(frozen["cohort"]["briefs"]) == {"brief-01", "brief-02"}
    assert frozen["resources"] == RESOURCES | {"resource_policy": "1"}
    assert protocol["owner_review"]["sample"] == frozen["owner_sample"]
    assert len(frozen["owner_sample"]) == 3
    assert protocol["criteria"]["judges"] == "diagnostic_only"
    assert protocol["criteria"]["validator_comparison"]["denominator"] == "all_expected_units"
    assert protocol["registration"]["config"] == "ge.yaml"


@pytest.mark.ml
def test_registered_run_records_denominators_failures_costs_and_evidence(
    registered: Fixture,
) -> None:
    protocol = read_sealed(registered.protocol)
    record = ge.read_run(registered.run)
    assert record["protocol"]["id"] == protocol["id"]
    assert record["protocol"]["commit"]
    assert record["test_only"] is True
    # Every registered slot stays in each denominator, failures included.
    validator = record["validator"]
    assert validator["candidate"]["counts"] == {
        "expected": 4,
        "success": 2,
        "failure": 0,
        "no-output": 2,
        "unknown": 0,
    }
    assert validator["reference"]["counts"]["expected"] == 4
    assert validator["reference"]["counts"]["success"] == 4
    assert validator["candidate"]["rate"] == 0.5
    assert validator["difference"] == -0.5
    assert validator["complete"] is True
    failed = {(f["condition"], f["slot"]) for f in record["failures"]}
    assert failed == {("E", "brief-02/r1"), ("E", "brief-02/r2")}
    assert all(f["error"] for f in record["failures"])
    # Costs keep provider calls and local model work apart, per condition.
    costs = record["costs"]["conditions"]
    assert costs["E"]["local"]["passes"] > 0
    assert costs["C"]["local"]["passes"] == 0
    assert costs["E"]["provider"]["completion"]["calls"] > 0
    assert costs["C"]["provider"]["planning"]["calls"] > 0
    assert record["costs"]["equal_budget"]["pairs"] == 4
    # Adherence, diversity and overlap come from the bound #83 evidence report.
    evidence = record["evidence"]
    report = read_sealed(registered.run / "ge-evidence/report/report.json")
    assert evidence["report"]["id"] == report["id"]
    assert evidence["adherence"]["deterministic"]["by_predicate"]["validity"]["expected"] == 4
    assert evidence["reference"]["identified"] is True
    assert evidence["diversity"]
    assert "protected_overlap" in evidence["overlap"]
    # Owner review is pending and no decision or later gate exists.
    assert record["owner_review"]["status"] == "pending"
    assert record["judges"] == "diagnostic_only"
    assert record["statement"] == ge.NOT_A_DECISION
    assert not list(registered.run.rglob("decision.json"))
    assert record["artifacts"]["manifest.json"]
    assert "stories/E/brief-01/r1/plan.json" in record["artifacts"]
    markdown = (registered.run / "ge-run.md").read_text()
    assert "## Validator pass rate (measured, not judged)" in markdown
    assert ge_command("verify", registered.protocol, registered.run) == 0
    # The fixed cohort is run once: no top-up or substituted rerun, in any runs folder.
    claim = read_sealed(ge.claim_path(registered.protocol))
    assert claim["protocol_id"] == protocol["id"]
    assert record["claim"]["id"] == claim["id"]
    runs = registered.run.parent
    assert ge_command("run", registered.protocol, "--runs-dir", runs) == 2
    other = registered.root / "other-runs"
    assert ge_command("run", registered.protocol, "--runs-dir", other) == 2
    assert list(runs.iterdir()) == [registered.run]
    assert not other.exists()


@pytest.mark.ml
def test_a_run_whose_evidence_fails_is_still_recorded_and_cannot_be_rerun(
    tmp_path: Path, registered: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = registered.root / "evidence-failure"
    path = registration(folder, registered, id="synthetic-test-only-ge-evidence-failure")
    protocol = folder / "protocol.json"
    assert ge_command("register", path, "--out", protocol) == 0
    commit(registered.root, "evidence-failure/protocol.json")
    generations: list[Path] = []

    def generate(_: object, options: RunOptions) -> Path:
        # The fixture's generated stories, without spending again.
        run = options.runs_dir / f"generated-{len(generations)}"
        ignored = shutil.ignore_patterns("ge-run.json", "ge-run.md", "ge-evidence")
        shutil.copytree(registered.run, run, ignore=ignored)
        generations.append(run)
        return run

    def fail(*_: object) -> dict[str, Any]:
        raise ValueError("synthetic evidence failure")

    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    monkeypatch.setattr(ge, "execute", generate)
    monkeypatch.setattr(ge.evidence_report, "write", fail)
    assert ge_command("run", protocol, "--runs-dir", tmp_path / "runs") == 1
    [run] = generations
    record = ge.read_run(run)
    assert record["evidence"] == {"status": "unavailable", "error": "synthetic evidence failure"}
    assert "evidence report unavailable: synthetic evidence failure" in record["problems"]
    # Costs, failures and complete denominators are kept; E's plans stay unverified.
    assert record["validator"]["candidate"]["counts"]["expected"] == 4
    assert record["validator"]["candidate"]["counts"]["unknown"] == 2
    assert record["validator"]["complete"] is False
    assert {(f["condition"], f["slot"]) for f in record["failures"]} == {
        ("E", "brief-02/r1"),
        ("E", "brief-02/r2"),
    }
    assert record["costs"]["conditions"]["E"]["local"]["passes"] > 0
    # It is never current evidence and cannot be replaced by regenerating elsewhere.
    assert ge.verify(protocol, run)[-1:] == [
        "evidence report unavailable: synthetic evidence failure"
    ]
    assert ge_command("packet", protocol, run, "--out", tmp_path / "packet") == 1
    for runs in (tmp_path / "runs", tmp_path / "elsewhere"):
        assert ge_command("run", protocol, "--runs-dir", runs) == 2
    assert len(generations) == 1


def relabelled(path: Path, checkpoint: Path, **evidence: Any) -> Path:
    """The fixture checkpoint relabelled as real, optionally claiming a G3 pass."""
    value = read_sealed(checkpoint)
    value["binding"]["config"]["test_only"] = False
    if evidence:
        value["binding"]["config"]["g3_protocol"] = str(evidence.pop("protocol"))
        value["binding"]["scientific_evidence"] = {"status": "passed", **evidence, "inputs": {}}
    del value["id"], value["version"]
    create_record(path, value)
    return path


def readiness(directory: Path, fixture: Fixture, checkpoint: Path | None = None) -> dict[str, Any]:
    """A real (not test-only) registration, run through the public readiness command."""
    config = yaml.safe_load(fixture.config.read_text())
    if checkpoint is not None:
        config["stories"]["conditions"][0]["params"]["checkpoint"] = str(checkpoint)
    directory.mkdir(parents=True)
    (directory / "ge.yaml").write_text(yaml.safe_dump(config))
    path = registration(directory, fixture, test_only=False, config="ge.yaml")
    assert ge_command("readiness", path, "--out", directory / "out") == 1
    assert ge_command("register", path, "--out", directory / "protocol.json") == 2
    assert not (directory / "protocol.json").exists()
    return json.loads((directory / "out/ge-readiness.json").read_text())


@pytest.mark.ml
def test_synthetic_prerequisites_cannot_authorise_real_e(
    tmp_path: Path, registered: Fixture
) -> None:
    # A test-only checkpoint and selection are software mechanics only.
    result = readiness(tmp_path / "fixture", registered)
    assert result["authority"]["status"] == "test-only"
    assert any("test-only checkpoint cannot authorise" in p for p in result["problems"])
    assert any("test-only C selection" in p for p in result["problems"])
    # Relabelling the fixture as real does not supply G3 evidence.
    unbound = relabelled(tmp_path / "unbound.json", registered.checkpoint)
    result = readiness(tmp_path / "unbound", registered, unbound)
    assert result["authority"]["status"] == "blocked"
    assert "genuine passed G3 evidence is missing" in result["authority"]["reason"]
    # A real writer's synthetic G3 criteria pass, claimed by the checkpoint, is rejected.
    g3_dir = tmp_path / "g3"
    g3_dir.mkdir()
    config, values = g3_inputs(g3_dir)
    g3_protocol = g3_dir / "protocol.json"
    assert main(["reconstruction-g3", "register", str(config), "--out", str(g3_protocol)]) == 0
    complete(g3_dir, values)
    decision = ge.g3.assess(g3_protocol)
    assert decision["criteria_outcome"] == "passed"
    claimed = relabelled(
        tmp_path / "claimed.json",
        registered.checkpoint,
        protocol=g3_protocol,
        decision_id=decision["id"],
    )
    result = readiness(tmp_path / "claimed", registered, claimed)
    assert result["authority"]["status"] == "blocked"
    assert "synthetic (test-only) G3 evidence" in result["authority"]["reason"]
    # No registration field accepts an issue state or a declared pass in place of evidence.
    path = registration(tmp_path / "issue", registered, g3_issue="closed #75")
    assert ge_command("readiness", path, "--out", tmp_path / "issue/out") == 2

    # Fixtures cannot hold a genuine pass, so the rule is exercised on the real writer's
    # synthetic decision promoted in memory only.
    checkpoint = read_sealed(claimed)
    promoted = decision | {"test_only": False, "outcome": "passed"}
    assert ge.g3_authority(checkpoint, promoted)["status"] == "genuine"
    assert ge.g3_authority(checkpoint, promoted | {"id": "f" * 64})["status"] == "blocked"
    pending = promoted | {"outcome": "pending"}
    assert ge.g3_authority(checkpoint, pending)["status"] == "blocked"
    assert ge.g3_authority(checkpoint, None, "missing")["status"] == "blocked"


def rejected(directory: Path, capsys: pytest.CaptureFixture[str]) -> str:
    assert ge_command("register", directory / "registration.json", "--out", directory / "p") == 2
    assert not (directory / "p").exists()
    return capsys.readouterr().err


@pytest.mark.ml
def test_unsupported_rules_unregistered_thresholds_and_mismatched_identities_are_rejected(
    tmp_path: Path, registered: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    rule = {
        "analysis": "pass_rate_difference",
        "margin": 0.1,
        "minimum_candidate_rate": 0.5,
        "aggregation": "pooled",
    }
    structural = {"analysis": "owner_paired_competitive_share", "question": QUESTION}
    cases: list[tuple[dict[str, Any], str]] = [
        ({"validator_comparison": rule | {"analysis": "t_test"}}, "analysis"),
        ({"validator_comparison": {k: v for k, v in rule.items() if k != "margin"}}, "margin"),
        ({"validator_comparison": rule | {"validator_target": 0.95}}, "validator_target"),
        ({"validator_comparison": rule | {"aggregation": "median"}}, "aggregation"),
        ({"structural_criterion": structural}, "minimum_share"),
        ({"structural_criterion": structural | {"minimum_share": 0.5, "ties": "half"}}, "ties"),
        ({"judges": "validated"}, "judges"),
        ({"iteration": {"attempt": 3}}, "attempt"),
        ({"iteration": {"attempt": 2}}, "development iteration"),
        ({"resources": RESOURCES | {"max_tokens": 5}}, "resource limits differ"),
        ({"resources": RESOURCES | {"per_day_usd": 9}}, "per_day_usd"),
        ({"cohort": {"briefs": ["brief-01"], "repeat": 2}}, "registered dev cohort"),
        ({"cohort": {"briefs": ["brief-01", "brief-02"], "repeat": 3}}, "registered dev cohort"),
        ({"reference": {"condition": "D"}}, "replayed actual G-D pass"),
        (
            {
                "owner_review": {
                    "reviewer": "Owner",
                    "sample_size": 5,
                    "sample_seed": 1,
                    "minutes_per_item": 1,
                    "max_minutes": 10,
                }
            },
            "larger than the registered dev cohort",
        ),
        (
            {
                "owner_review": {
                    "reviewer": "Owner",
                    "sample_size": 3,
                    "sample_seed": 1,
                    "minutes_per_item": 20,
                    "max_minutes": 40,
                }
            },
            "review time budget",
        ),
    ]
    for index, (change, message) in enumerate(cases):
        directory = tmp_path / f"case-{index}"
        registration(directory, registered, **change)
        assert message in rejected(directory, capsys), change

    # A G-D decision that does not replay cannot identify D (or anything else).
    forged = tmp_path / "gd"
    forged.mkdir()
    (forged / "decision.json").write_text(json.dumps({"outcome": "passed", "test_only": False}))
    gd = {"condition": "C", "c_selection": "x", "gd_decision": "../gd", "gd_batch": "../none"}
    registration(tmp_path / "forged", registered, reference=gd)
    assert "does not replay" in rejected(tmp_path / "forged", capsys)

    # The reference must be the owner's selected C contract, not another C.
    config = yaml.safe_load(registered.config.read_text())
    config["stories"]["conditions"][1]["params"]["tier"] = "L"
    other = tmp_path / "other-c"
    other.mkdir()
    (other / "ge.yaml").write_text(yaml.safe_dump(config))
    registration(other, registered, config="ge.yaml")
    assert "reference identity mismatch" in rejected(other, capsys)

    # Held-out briefs are refused before anything is read or generated.
    config = yaml.safe_load(registered.config.read_text())
    config["stories"]["briefs"] = {"split": "test"}
    held_out = tmp_path / "held-out"
    held_out.mkdir()
    (held_out / "ge.yaml").write_text(yaml.safe_dump(config))
    registration(held_out, registered, config="ge.yaml")
    assert "held-out cohort is never generated" in rejected(held_out, capsys)
    # IDs without `split: dev` would validate every split, the held-out one included.
    config["stories"]["briefs"] = {"ids": ["brief-01", "brief-02"]}
    (held_out / "ge.yaml").write_text(yaml.safe_dump(config))
    assert "split: dev" in rejected(held_out, capsys)


@pytest.mark.ml
def test_g_e_reads_only_dev_briefs(tmp_path: Path, registered: Fixture) -> None:
    # The fixture's run was generated beside an unreadable held-out brief file.
    held_out = registered.root / "benchmark/test/brief-99.yaml"
    assert held_out.read_text() == HELD_OUT
    path = registration(tmp_path / "dev-only", registered)
    assert ge_command("readiness", path, "--out", tmp_path / "out") == 0
    assert ge_command("verify", registered.protocol, registered.run) == 0
    assert ge_command("packet", registered.protocol, registered.run, "--out", tmp_path / "p") == 0
    # Only the selected split is checked: the shared whole-benchmark check still fails.
    assert briefs.check(held_out.parents[1]).problems


@pytest.mark.ml
def test_d_reference_identity_follows_the_g_d_baseline(registered: Fixture) -> None:
    """Fixtures hold no actual G-D pass; the D identity rule is exercised in memory."""
    loaded = load_config(registered.config)
    roles = {str(name): binding for name, binding in loaded.config.roles.items()}
    spec = ConditionSpec(name="D", pipeline="baseline_d", params={"critic_iterations": 1})
    baseline = {
        "pipeline": "baseline_d",
        "params": {"critic_iterations": 1},
        "schema_sha256": ge.digest(ge.schema_json()),
        "models": {name: binding.model_dump(mode="json") for name, binding in roles.items()},
        "prompts": {"skeleton@1": "a" * 64},
    }
    assert ge.d_reference_problems(spec, roles, baseline, {"other@1": "b" * 64}) == []
    changed = spec.model_copy(update={"params": {"critic_iterations": 2}})
    assert ge.d_reference_problems(changed, roles, baseline, {}) == [
        "D params differs from the G-D reference baseline"
    ]
    assert ge.d_reference_problems(spec, roles, baseline, {"skeleton@1": "c" * 64}) == [
        "D prompt differs from the G-D reference baseline: skeleton@1"
    ]
    assert ge.d_reference_problems(spec, {}, baseline, {}) == [
        f"D {role} model binding differs from the G-D reference baseline" for role in sorted(roles)
    ]


def _d_workflow(folder: Path, registered: Fixture, fake: FakeOpenAI) -> dict[str, Any]:
    """An E-versus-D dev config, and the reference baseline an actual G-D pass would name."""
    config = yaml.safe_load(registered.config.read_text())
    config["providers"]["fake"]["base_url"] = fake.base_url
    config["stories"]["conditions"][1] = BASELINE_D
    config["stories"]["briefs"] = {"split": "dev", "ids": ["brief-01"]}
    config["stories"]["repeat"] = 1
    config["stories"]["equal_budget"]["conditions"] = ["E", "D"]
    folder.mkdir()
    (folder / "ge.yaml").write_text(yaml.safe_dump(config))
    loaded = load_config(folder / "ge.yaml")
    templates: dict[str, Any] = {}
    story_templates(loaded, story_conditions(loaded), templates)
    return {
        "condition": "D",
        "pipeline": "baseline_d",
        "params": {},
        "schema_sha256": ge.digest(ge.schema_json()),
        # The baseline's own roles; G-D's test configuration had no E and other briefs.
        "models": {
            role: binding.model_dump(mode="json")
            for role, binding in loaded.config.roles.items()
            if role in {"planner", "prose"}
        },
        "prompts": {label: template.sha256 for label, template in templates.items()},
        "config_sha256": "0" * 64,
        "planning_artifacts": {},
    }


@pytest.mark.ml
def test_an_actual_g_d_pass_identifies_d_in_the_protocol_and_the_report(
    registered: Fixture,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    folder = registered.root / "d-reference"
    baseline = _d_workflow(folder, registered, fake_llm)
    # Fixtures hold no actual G-D pass: the real decision folder layout is promoted in
    # memory only, by replaying the saved decision as `gd-report` would rebuild it.
    decision: dict[str, Any] = {"id": "promoted-in-test", "outcome": "passed", "test_only": False}
    decision["reference_baseline"] = baseline
    (folder / "gd").mkdir()
    (folder / "gd/decision.json").write_text(json.dumps(decision))
    (folder / "gd/reference-baseline.json").write_text(json.dumps(baseline))
    (folder / "batch").mkdir()

    def replay(*_: object) -> dict[str, Any]:
        return json.loads((folder / "gd/decision.json").read_text())

    monkeypatch.setattr(ge.gd_report, "decision", replay)
    monkeypatch.setattr(ge.evidence_report, "gd_decision", replay)
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = Provider()
    change = {
        "config": "ge.yaml",
        "reference": {"condition": "D", "gd_decision": "gd", "gd_batch": "batch"},
        "cohort": {"briefs": ["brief-01"], "repeat": 1},
        "owner_review": {
            "reviewer": "Synthetic owner",
            "sample_size": 1,
            "sample_seed": 3,
            "minutes_per_item": 10,
            "max_minutes": 10,
        },
    }
    # A D planner that differs from the named baseline is not that reference.
    wrong = folder / "wrong"
    wrong.mkdir()
    shutil.copy2(folder / "ge.yaml", wrong / "ge.yaml")
    changed = yaml.safe_load((wrong / "ge.yaml").read_text())
    changed["stories"]["conditions"][1]["params"] = {"critic_iterations": 0}
    (wrong / "ge.yaml").write_text(yaml.safe_dump(changed))
    elsewhere = {"condition": "D", "gd_decision": "../gd", "gd_batch": "../batch"}
    registration(wrong, registered, **change | {"reference": elsewhere})
    assert "D params differs" in rejected(wrong, capsys)
    # The baseline's own contract identifies D; G-D's test-configuration checksum is not part.
    path = registration(folder, registered, id="synthetic-test-only-ge-d", **change)
    protocol = folder / "protocol.json"
    assert ge_command("register", path, "--out", protocol) == 0
    reference = read_sealed(protocol)["frozen"]["reference"]
    assert (reference["condition"], reference["identification"]) == (
        "D",
        "replayed actual G-D pass",
    )
    commit(registered.root, "d-reference/protocol.json")
    assert ge_command("run", protocol, "--runs-dir", folder / "runs") == 0
    [run] = list((folder / "runs").iterdir())
    evidence = ge.read_run(run)["evidence"]["reference"]
    assert evidence["permitted"] == "D"
    assert evidence["identity_problems"] == []
    assert evidence["identified"] is True
    report = read_sealed(run / "ge-evidence/report/report.json")
    assert report["reference"]["identified"] is True
    assert ge_command("verify", protocol, run) == 0


@pytest.mark.ml
def test_changed_checkpoints_runs_and_missing_results_are_stale(
    tmp_path: Path, registered: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    protocol, run = registered.protocol, registered.run
    # A changed checkpoint blocks the protocol and staleness reaches the run and packet.
    with restored(registered.checkpoint):
        value = json.loads(registered.checkpoint.read_text())
        value["step"] += 1
        registered.checkpoint.write_text(json.dumps(value))
        assert ge_command("check", protocol, "--out", tmp_path / "check") == 1
        checked = json.loads((tmp_path / "check/ge-check.json").read_text())
        assert any("checkpoint" in problem for problem in checked["problems"])
        assert ge_command("verify", protocol, run) == 1
        assert ge_command("packet", protocol, run, "--out", tmp_path / "packet") == 1
        assert not (tmp_path / "packet").exists()
    # A changed E plan after the run.
    plan = run / "stories/E/brief-01/r1/plan.json"
    with restored(plan):
        plan.write_text(plan.read_text().replace('"Ann"', '"Bea"'))
        capsys.readouterr()
        assert ge_command("verify", protocol, run) == 1
        assert "stories/E/brief-01/r1/plan.json" in capsys.readouterr().err
    # A missing result.
    story = run / "stories/C/brief-02/r2/story.md"
    with restored(story):
        story.unlink()
        assert ge_command("verify", protocol, run) == 1
    # A changed configuration or protocol after commit.
    with restored(registered.config):
        registered.config.write_text(registered.config.read_text().replace("seed: 7", "seed: 8"))
        assert ge_command("check", protocol, "--out", tmp_path / "config") == 1
    with restored(protocol):
        value = json.loads(protocol.read_text())
        protocol.write_text(json.dumps(value, indent=1))
        assert ge_command("check", protocol, "--out", tmp_path / "edited") == 1
        problems = json.loads((tmp_path / "edited/ge-check.json").read_text())["problems"]
        assert any("differs from its committed contents" in p for p in problems)
    assert ge_command("verify", protocol, run) == 0


def review_file(path: Path, packet: dict[str, Any], *, rating: str = "A", **fields: Any) -> Path:
    ratings = [
        {
            "item": item["item"],
            "plans": {label: plan["sha256"] for label, plan in item["plans"].items()},
            "rating": rating,
            "note": "Synthetic fixture note.",
        }
        for item in packet["items"]
        if item["kind"] == "owner"
    ]
    value = {"packet_id": packet["id"], "reviewer": "Synthetic owner", "ratings": ratings}
    path.write_text(json.dumps(value | fields))
    return path


@pytest.mark.ml
def test_owner_packet_is_fixed_budgeted_and_missing_review_stays_pending(
    tmp_path: Path, registered: Fixture
) -> None:
    folder = tmp_path / "packet"
    assert ge_command("packet", registered.protocol, registered.run, "--out", folder) == 0
    packet = ge_review.read_packet(folder)
    protocol = read_sealed(registered.protocol)
    assert [item["slot"] for item in packet["items"]] == protocol["owner_review"]["sample"]
    kinds = {item["slot"]: item["kind"] for item in packet["items"]}
    assert all(
        kind == ("candidate-no-output" if slot.startswith("brief-02") else "owner")
        for slot, kind in kinds.items()
    )
    owner_items = sum(kind == "owner" for kind in kinds.values())
    assert packet["budget"]["estimated_minutes"] == owner_items * 10
    assert packet["budget"]["estimated_minutes"] <= packet["budget"]["max_minutes"]
    assert packet["question"] == QUESTION
    text = (folder / "packet.md").read_text()
    assert QUESTION in text
    assert "Plan A" in text
    assert "candidate_e" not in text
    # Nothing the owner reads names a condition; the sealed key holds conditions and paths.
    assert "stories/E" not in (folder / "packet.json").read_text()
    assert "stories/E" not in (folder / "review-template.json").read_text()
    # Without any review the owner evidence is pending.
    assert ge_command("review", folder) == 1
    assert ge_review.status(folder)["status"] == "pending"
    # A model judge agreeing on every item stays diagnostic.
    judge = review_file(
        tmp_path / "judge.json",
        packet,
        diagnostic_provenance={"model": "judge-model", "prompt": "structure@1"},
    )
    assert ge_command("review", folder, judge) == 1
    result = ge_review.status(folder)
    assert (result["status"], result["evidence"]) == ("pending", None)
    assert result["rated"]["diagnostic"] == owner_items
    assert {o["outcome"] for o in result["outcomes"]} == {"pending"}
    # A review of other plans, or of another packet, is rejected outright.
    stale = review_file(tmp_path / "stale.json", packet)
    value = json.loads(stale.read_text())
    value["ratings"][0]["plans"]["A"] = "0" * 64
    stale.write_text(json.dumps(value))
    assert ge_command("review", folder, stale) == 2
    other = review_file(tmp_path / "other.json", packet, packet_id="1" * 64)
    assert ge_command("review", folder, other) == 2
    # The synthetic owner finishes the mechanics; the packet is test-only, so it stays pending.
    owner = review_file(tmp_path / "owner.json", packet, rating="equivalent")
    assert ge_command("review", folder, owner) == 1
    result = ge_review.status(folder)
    assert (result["status"], result["evidence"]) == ("pending", "synthetic")
    outcomes = {o["slot"]: o["outcome"] for o in result["outcomes"]}
    assert all(
        outcome == ("not_competitive" if slot.startswith("brief-02") else "competitive")
        for slot, outcome in outcomes.items()
    )
    # A changed output makes the packet and its reviews stale.
    [first] = [item for item in packet["items"] if item["kind"] == "owner"][:1]
    plan = registered.run / read_sealed(folder / "key.json")["paths"][first["item"]]["A"]
    with restored(plan):
        plan.write_text(plan.read_text() + " ")
        assert ge_command("review", folder) == 1
        assert ge_review.status(folder)["status"] == "stale"


@pytest.mark.ml
def test_only_the_registered_owners_actual_review_completes_a_real_packet(
    tmp_path: Path, registered: Fixture
) -> None:
    folder = tmp_path / "packet"
    assert ge_command("packet", registered.protocol, registered.run, "--out", folder) == 0
    # Fixtures cannot produce a real packet; promote this one in a copy, within the test.
    real = tmp_path / "real"
    real.mkdir()
    shutil.copy2(folder / "key.json", real / "key.json")
    value = copy.deepcopy(read_sealed(folder / "packet.json"))
    value["test_only"] = False
    value["run"] = ge.relative(registered.run.resolve(), real.resolve())
    del value["id"], value["version"]
    packet = create_record(real / "packet.json", value)
    someone = review_file(tmp_path / "someone.json", packet, reviewer="Someone else")
    assert ge_command("review", real, someone) == 1
    assert ge_review.status(real)["rated"]["other-reviewer"] > 0
    synthetic = review_file(tmp_path / "synthetic.json", packet, test_only=True)
    assert ge_command("review", real, synthetic) == 1
    assert ge_review.status(real)["evidence"] == "synthetic"
    actual = review_file(tmp_path / "actual.json", packet, rating="B")
    assert ge_command("review", real, actual) == 0
    result = ge_review.status(real)
    assert (result["status"], result["evidence"]) == ("complete", "actual")
    key = read_sealed(real / "key.json")["items"]
    for outcome in result["outcomes"]:
        if outcome["kind"] == "owner":
            expected = "competitive" if key[outcome["item"]]["B"] == "E" else "not_competitive"
            assert outcome["outcome"] == expected
    assert len(list((real / "reviews").glob("review-*.json"))) == 3


@pytest.mark.ml
def test_the_single_development_iteration_is_bound_to_the_first_attempt(
    tmp_path: Path, registered: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    previous = os.path.relpath(registered.protocol, tmp_path / "same")
    iteration = {"attempt": 2, "previous_protocol": previous, "change": "Raise guidance."}
    # An identical E is a rerun, not an iteration.
    registration(tmp_path / "same", registered, iteration=iteration)
    assert "must change E" in rejected(tmp_path / "same", capsys)
    config = yaml.safe_load(registered.config.read_text())
    config["stories"]["conditions"][0]["params"]["guidance_scale"] = 1.5
    changed = tmp_path / "changed"
    changed.mkdir()
    (changed / "ge.yaml").write_text(yaml.safe_dump(config))
    iteration["previous_protocol"] = os.path.relpath(registered.protocol, changed)
    # Criteria cannot be loosened for the iteration.
    loosened = registration(
        changed,
        registered,
        config="ge.yaml",
        iteration=iteration,
        structural_criterion={
            "analysis": "owner_paired_competitive_share",
            "question": QUESTION,
            "minimum_share": 0.25,
        },
    )
    assert "keeps the first attempt's" in rejected(changed, capsys)
    loosened.unlink()
    # Nor can the cohort's seeds: another run seed is another cohort.
    reseeded = changed / "reseeded"
    reseeded.mkdir()
    (reseeded / "ge.yaml").write_text(yaml.safe_dump(config | {"seed": 8}))
    later = iteration | {"previous_protocol": os.path.relpath(registered.protocol, reseeded)}
    registration(reseeded, registered, config="ge.yaml", iteration=later)
    assert "keeps the first attempt's" in rejected(reseeded, capsys)
    # An attempt 1 that was never run cannot be iterated on.
    unrun = tmp_path / "unrun"
    path = registration(unrun, registered, id="synthetic-test-only-ge-unrun")
    assert ge_command("register", path, "--out", unrun / "protocol.json") == 0
    (unrun / "next").mkdir()
    shutil.copy2(changed / "ge.yaml", unrun / "next/ge.yaml")
    first = {"attempt": 2, "previous_protocol": "../protocol.json", "change": "Raise guidance."}
    registration(unrun / "next", registered, config="ge.yaml", iteration=first)
    assert "was never run" in rejected(unrun / "next", capsys)
    path = registration(changed, registered, config="ge.yaml", iteration=iteration)
    assert ge_command("register", path, "--out", changed / "protocol.json") == 0
    second = read_sealed(changed / "protocol.json")
    assert second["frozen"]["iteration"]["previous"]["id"] == read_sealed(registered.protocol)["id"]
    assert (
        second["frozen"]["owner_sample"]
        == read_sealed(registered.protocol)["owner_review"]["sample"]
    )
    # A second iteration chained to the iteration is refused.
    third = changed / "third"
    third.mkdir()
    shutil.copy2(changed / "ge.yaml", third / "ge.yaml")
    registration(
        third,
        registered,
        config="ge.yaml",
        iteration={"attempt": 2, "previous_protocol": "../protocol.json", "change": "Again."},
    )
    assert "follow the first registered attempt" in rejected(third, capsys)
