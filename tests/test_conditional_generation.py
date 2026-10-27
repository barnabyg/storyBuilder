"""Full-generation authority is checked at the public command/file boundary."""

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from nde.conditional.generation import completion_fields
from nde.corpus.records_io import create_record, read_json, read_sealed
from nde.narrative.models import NarrativePlan
from tests.conditional_support import trained_checkpoint
from tests.fake_openai import FakeOpenAI, completion


def proposal_file(directory: Path, **updates: Any) -> Path:
    plan = NarrativePlan.model_validate(
        {
            "id": "generated-plan",
            "title": "Unfilled generated text",
            "scenes": [{"id": "scenes.0", "title": "Unfilled generated text"}],
        }
    )
    plan_data = plan.model_dump(mode="json")
    plan_data.update(updates)
    create_record(
        directory / "proposal.json",
        {
            "kind": "conditional-structural-proposal",
            "status": "structural-proposal",
            "mode": "full-generation",
            "plan": plan_data,
            "locks": {},
            "completion_fields": completion_fields(plan, {}),
            "conditions": {},
            "test_only": True,
            "inputs": {},
        },
    )
    return directory / "proposal.json"


def test_inspection_rejects_missing_facts_for_an_explicit_hard_constraint(tmp_path: Path) -> None:
    proposal = proposal_file(
        tmp_path,
        constraints=[
            {
                "id": "lock-ending",
                "kind": "required_ending",
                "state_kind": "canon",
                "key": "door",
                "expected": "closed",
            }
        ],
    )
    assert main(["conditional-generate", "inspect", str(proposal)]) == 2


def test_inference_request_rejects_hidden_target_scaffolding_before_model_access(
    tmp_path: Path,
) -> None:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"checkpoint": "does-not-exist.json", "target_plan": {"scenes": ["secret"]}})
    )
    assert (
        main(["conditional-generate", "generate", str(request), "--out", str(tmp_path / "out")])
        == 2
    )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("locks", [[], "scenes/scenes.0/title", {"title": []}])
def test_inspect_rejects_malformed_saved_lock_authority(tmp_path: Path, locks: Any) -> None:
    proposal = proposal_file(tmp_path / "source")
    record = read_sealed(proposal)
    record["locks"] = locks
    malicious = tmp_path / "malicious.json"
    create_record(malicious, record)
    assert main(["conditional-generate", "inspect", str(malicious)]) == 2


@pytest.fixture(scope="module")
def learned_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if importlib.util.find_spec("torch") is None:
        if os.environ.get("NDE_REQUIRE_ML") == "1":
            pytest.fail("CPU ML lane requires the pinned torch runtime")
        pytest.skip("optional CPU ML runtime")
    directory = tmp_path_factory.mktemp("conditional-generation-cpu")
    fake = FakeOpenAI()
    fake.start()
    try:
        checkpoint = trained_checkpoint(directory / "training", fake)
    finally:
        fake.stop()
    detached = directory / "model.json"
    detached.write_bytes(checkpoint.read_bytes())
    # All training paths and base files are now unavailable at their recorded locations.
    (directory / "training").rename(directory / "unavailable-training")
    return detached


def generation_request(directory: Path, checkpoint: Path, **options: Any) -> Path:
    request = directory / "request.json"
    request.write_text(json.dumps({"checkpoint": str(checkpoint), **options}))
    return request


def generate_command(request: Path, out: Path) -> int:
    return main(["conditional-generate", "generate", str(request), "--out", str(out)])


@pytest.mark.ml
def test_fully_masked_generation_needs_no_targets_and_preserves_locks_with_conditions_dropped(
    tmp_path: Path, learned_checkpoint: Path
) -> None:
    request = generation_request(
        tmp_path,
        learned_checkpoint,
        guidance_scale=0,
        locks={"characters/characters.0/name": "Caller name"},
        seed=41,
    )
    out = tmp_path / "first"
    assert generate_command(request, out) == 0
    proposal = read_sealed(out / "proposal.json")
    assert proposal["status"] == "structural-proposal"
    assert proposal["test_only"]
    assert len(proposal["plan"]["scenes"]) == 2
    assert proposal["plan"]["characters"][0]["name"] == "Caller name"
    assert proposal["plan"]["sources"] == []
    assert "characters/characters.0/name" not in proposal["completion_fields"]
    assert not (out / "plan.json").exists()
    assert main(["conditional-generate", "inspect", str(out)]) == 0
    first = read_sealed(out / "attempt-001.json")
    assert first["candidate"]["work"]["passes"] == 1
    assert first["candidate"]["conditions"]["genre"] is None
    assert generate_command(request, tmp_path / "second") == 0
    second = read_sealed(tmp_path / "second/attempt-001.json")
    assert second["candidate"]["generated_tokens"] == first["candidate"]["generated_tokens"]
    assert (out / "report.md").is_file()
    assert generate_command(request, out) == 2


@pytest.mark.ml
def test_invalid_reference_lock_rejects_every_candidate_and_stops_at_attempt_limit(
    tmp_path: Path, learned_checkpoint: Path
) -> None:
    request = generation_request(
        tmp_path,
        learned_checkpoint,
        locks={"causal_links/causal_links.0/effect_scene_id": "missing-scene"},
        max_attempts=2,
    )
    out = tmp_path / "out"
    assert generate_command(request, out) == 1
    report = read_json(out / "report.json")
    assert report["status"] == "attempt-limit"
    assert len(report["attempts"]) == 2
    assert all(row["status"] == "rejected" for row in report["attempts"])
    assert "missing-scene" in json.dumps(report["attempts"])
    assert report["work"]["tokens"] > 0
    assert report["work"]["tokens"] == sum(
        row["candidate"]["work"]["tokens"] for row in report["attempts"]
    )
    assert not (out / "proposal.json").exists()


@pytest.mark.ml
@pytest.mark.parametrize("allowance", [{"max_tokens": 1}, {"max_seconds": 0.00000001}])
def test_resource_stop_performs_no_unbudgeted_model_work(
    tmp_path: Path, learned_checkpoint: Path, allowance: dict[str, Any]
) -> None:
    request = generation_request(tmp_path, learned_checkpoint, **allowance)
    out = tmp_path / "out"
    assert generate_command(request, out) == 1
    report = read_json(out / "report.json")
    assert report["status"] == "resource-stop"
    assert report["work"]["tokens"] == 0
    assert report["attempts"] == []


@pytest.mark.ml
def test_unsupported_scene_range_fails_before_attempts(
    tmp_path: Path, learned_checkpoint: Path
) -> None:
    request = generation_request(tmp_path, learned_checkpoint, conditions={"scene_range": [1, 5]})
    assert generate_command(request, tmp_path / "out") == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.ml
def test_checkpoint_inference_cap_stops_retries_without_hiding_the_reason(
    tmp_path: Path, learned_checkpoint: Path
) -> None:
    checkpoint = read_sealed(learned_checkpoint)
    checkpoint["binding"]["config"]["inference_max_tokens"] = 1
    limited = tmp_path / "limited-model.json"
    create_record(limited, checkpoint)
    request = generation_request(tmp_path, limited, max_attempts=3)
    out = tmp_path / "out"
    assert generate_command(request, out) == 1
    report = read_json(out / "report.json")
    assert report["status"] == "resource-stop"
    assert len(report["attempts"]) == 1
    assert report["work"]["tokens"] == 0


@pytest.mark.ml
def test_relabelled_fixture_cannot_start_a_real_e_experiment_without_g3(
    tmp_path: Path, learned_checkpoint: Path
) -> None:
    checkpoint = read_sealed(learned_checkpoint)
    checkpoint["binding"]["config"]["test_only"] = False
    relabelled = tmp_path / "relabelled.json"
    create_record(relabelled, checkpoint)
    request = generation_request(tmp_path, relabelled)
    assert generate_command(request, tmp_path / "out") == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.ml
def test_bounded_rejection_recovers_without_discarding_invalid_learned_candidates(
    tmp_path: Path, learned_checkpoint: Path
) -> None:
    checkpoint = read_sealed(learned_checkpoint)
    checkpoint["binding"]["config"]["sampling_temperature"] = 1
    stochastic = tmp_path / "stochastic.json"
    create_record(stochastic, checkpoint)
    request = generation_request(tmp_path, stochastic, seed=8, max_attempts=8)
    out = tmp_path / "out"
    assert generate_command(request, out) == 0
    report = read_json(out / "report.json")
    assert report["attempts"][0]["status"] == "rejected"
    assert report["attempts"][-1]["status"] == "selected"
    assert report["attempts"][0]["candidate"]["generated_tokens"]
    assert report["attempts"][0]["diagnostics"]
    assert report["work"]["tokens"] == sum(
        row["candidate"]["work"]["tokens"] for row in report["attempts"]
    )


@pytest.mark.ml
def test_generated_structure_completes_through_gateway_into_validated_plan_and_cards(
    tmp_path: Path, learned_checkpoint: Path, fake_llm: FakeOpenAI
) -> None:
    request = generation_request(tmp_path, learned_checkpoint, conditions={"genre": "parable"})
    proposals = tmp_path / "proposals"
    assert generate_command(request, proposals) == 0
    source = proposals / "proposal.json"
    original_bytes = source.read_bytes()
    proposal = read_sealed(source)
    fields = dict.fromkeys(proposal["completion_fields"], "A promise is kept.")
    fake_llm.queue(completion(json.dumps({"fields": fields})))
    config = tmp_path / "completion.json"
    config.write_text(
        json.dumps(
            {
                "proposal": str(source),
                "fields": list(fields),
                "provider": {"base_url": fake_llm.base_url, "retry": {"max_retries": 0}},
                "model": {"provider": "fake", "model": "completion-model"},
                "price": {"input_per_mtok": 1, "output_per_mtok": 2},
            }
        )
    )
    out = tmp_path / "completed"
    assert main(["conditional-complete", "run", str(config), "--out", str(out)]) == 0
    assert main(["conditional-complete", "inspect", str(out)]) == 0
    completed = json.loads((out / "plan.json").read_text())
    assert completed["title"] == "A promise is kept."
    assert completed["causal_links"] == proposal["plan"]["causal_links"]
    assert completed["states"] == proposal["plan"]["states"]
    assert [scene["id"] for scene in completed["scenes"]] == ["scenes.0", "scenes.1"]
    assert completed["sources"] == []
    assert json.loads((out / "validation.json").read_text())["diagnostics"] == []
    assert "A promise is kept." in (out / "cards.html").read_text()
    assert source.read_bytes() == original_bytes
    assert (out / "source-proposal.json").read_bytes() == original_bytes
