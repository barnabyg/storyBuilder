"""Synthetic protocols and provider data only; never scientific owner evidence."""

import json
from pathlib import Path

import pytest
import yaml

from nde.annotations.records import digest
from nde.cli import main
from nde.controllability import ControlTarget
from nde.files import checksum_lf
from nde.narrative.workflow import schema_json
from nde.plan.revision import identity
from nde.runs.config import load_config
from nde.runs.prompts import PromptTemplate
from nde.stories.execute import prepare_stories
from tests.conftest import REPO
from tests.eval_helpers import reader
from tests.fake_openai import FakeOpenAI
from tests.test_baseline_d import critic_provider
from tests.test_g1 import git_fixture, prepare_registered_workflow


def prepare_gd(
    paths: dict[str, Path], tmp_path: Path, fake: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], Path]:
    # Record family metadata before selecting C and establishing synthetic G1.
    source = yaml.safe_load(paths["config"].read_text())
    for role in ("planner", "prose"):
        source["roles"][role]["family"] = "generator"
    paths["config"].write_text(yaml.safe_dump(source))
    common, _, config, _ = prepare_registered_workflow(paths, tmp_path, fake, monkeypatch)
    runs, batches = tmp_path / "g1-runs", tmp_path / "g1-batches"
    assert main(["eval", "g1-run", *common, "--runs-dir", str(runs)]) == 0
    [run] = list(runs.iterdir())
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
                str(run),
                "C",
                "--from",
                str(run),
                "B",
                "--rubric-dir",
                str(REPO / "benchmark/rubric"),
            ]
        )
        == 0
    )
    batch = batches / "synthetic"
    client = reader(batch)
    fields = {"preferred": "A", "strength": "1"}
    fields.update(
        {
            f"score-{side}-{name}": "4"
            for side in ("a", "b")
            for name in ("coherence", "originality")
        }
    )
    key = json.loads((batch / "sealed/key.json").read_text())
    side = "a" if key["pairs"][0]["a"]["condition"] == "C" else "b"
    fields[f"score-{side}-coherence"] = "5"
    assert client.post("/pairs/1", data=fields).status_code == 200
    assert client.post("/close").status_code == 200
    result = tmp_path / "g1-decision"
    assert main(["g1-report", str(batch), "--out", str(result)]) == 0
    config["stories"]["conditions"] = [
        {"name": "C", "pipeline": "baseline_c"},
        {"name": "D", "pipeline": "baseline_d", "params": {"critic_iterations": 1}},
    ]
    config["roles"]["critic"] = {"provider": "fake", "model": "fake-model", "family": "critic"}
    config["stories"]["equal_budget"]["conditions"] = ["C", "D"]
    target = ControlTarget(id="preserve", kind="unchanged", selector="scenes/scene-0")
    config_path = tmp_path / "cd.yaml"
    config_path.write_text(yaml.safe_dump(config))
    loaded = load_config(config_path)
    templates: dict[str, PromptTemplate] = {}
    prepare_stories(loaded, templates)
    gd_protocol = {
        "id": "synthetic-test-only-G-D",
        "test_only": True,
        "cohort": {
            "id": "synthetic-test-only",
            "brief_checksums": json.loads(paths["protocol"].read_text())["cohort"][
                "brief_checksums"
            ],
            "split_checksums": json.loads(paths["protocol"].read_text())["cohort"][
                "split_checksums"
            ],
            "conditions": ["D", "C"],
            "length_tolerance": 0.1,
            "budget_tolerance": 0.1,
        },
        "n": 1,
        "tier": "L",
        "coherence": {"analysis": "mean_paired_difference", "margin": 0.5},
        "originality": {"analysis": "mean_paired_difference", "margin": 0.25},
        "reading_mode": "required_only",
        "g1_decision_sha256": checksum_lf((result / "decision.json").read_bytes()),
        "selection_sha256": checksum_lf(paths["selection"].read_bytes()),
        "config_sha256": loaded.hash,
        "schema_sha256": digest(schema_json()),
        "models_sha256": digest(
            json.dumps(
                {
                    name: binding.model_dump(mode="json")
                    for name, binding in loaded.config.roles.items()
                },
                sort_keys=True,
            )
        ),
        "prompt_sha256": {label: template.sha256 for label, template in templates.items()},
        "control_targets": {target.id: identity(target)},
        "critic_iterations": 1,
        "generation_limits": {
            "max_repairs": 2,
            "max_length_attempts": 3,
            "max_revision_passes": 4,
            "per_run_usd": 0.1,
            "per_day_usd": 1,
        },
        "prior_benchmark_exposure": "Synthetic test-only fixtures, not owner reading",
    }
    protocol = tmp_path / "gd.json"
    protocol.write_text(json.dumps(gd_protocol))
    git_fixture(tmp_path, "add", "gd.json")
    git_fixture(tmp_path, "commit", "-q", "-m", "synthetic test-only G-D registration")
    fake.default = critic_provider
    return [
        str(protocol),
        str(config_path),
        "--selection",
        str(paths["selection"]),
        "--g1-batch",
        str(batch),
        "--g1-decision",
        str(result / "decision.json"),
    ], config_path


def gd_batch(
    common: list[str], tmp_path: Path, target_kind: str = "unchanged"
) -> tuple[Path, Path, Path]:
    runs, batches = tmp_path / "gd-runs", tmp_path / "gd-batches"
    assert (
        main(
            [
                "gd",
                "run",
                *common,
                "--runs-dir",
                str(runs),
                "--cache",
                str(tmp_path / "gd-cache.sqlite"),
                "--ledger",
                str(tmp_path / "gd-ledger.sqlite"),
            ]
        )
        == 0
    )
    [run] = list(runs.iterdir())
    plan = run / "stories/D/brief-01/r1/plan.json"
    revision = plan.parent / "revisions/critic-1"
    provenance = revision / "provenance.json"
    recorded = json.loads(provenance.read_text())
    request = tmp_path / "controls.json"
    request.write_text(
        json.dumps(
            {
                "before": str(revision / "source-plan.json"),
                "after": str(plan),
                "revision": recorded["revision"],
                "revision_provenance": str(provenance),
                "targets": [{"id": "preserve", "kind": target_kind, "selector": "scenes/scene-0"}],
            }
        )
    )
    out = tmp_path / "controls"
    assert main(["control-report", str(request), "--out", str(out)]) == 0
    control = out / "control-report.json"
    assert (
        main(
            [
                "gd",
                "batch",
                *common,
                "--name",
                "synthetic",
                "--batches-dir",
                str(batches),
                "--from",
                str(run),
                "D",
                "--from",
                str(run),
                "C",
                "--control",
                "brief-01",
                str(control),
                "--rubric-dir",
                str(REPO / "benchmark/rubric"),
            ]
        )
        == 0
    )
    return batches / "synthetic", run, control
