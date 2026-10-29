"""Shared D/E revision-contract cases at the public `nde plan revise` boundary."""

import copy
import hashlib
import importlib.util
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from nde.conditional import inpainting
from nde.conditional.data import Conditions
from nde.conditional.layout import textless
from nde.conditional.training import inspect_checkpoint, sample_candidate
from nde.corpus.records_io import create_record, read_sealed
from nde.narrative.models import NarrativePlan
from nde.narrative.workflow import PlanError
from nde.plan.backends import ERevisionBackend
from nde.plan.revision import RevisionPlanningParams, RevisionSpec, identity
from nde.reconstruction.codec import (
    BYTE_OFFSET,
    MASK,
    PAD,
    canonical_json,
    encode_plan,
    tokenize,
)
from tests.conditional_support import trained_checkpoint
from tests.conftest import REPO, write_config
from tests.fake_openai import FakeOpenAI, Json, Reply, completion, error
from tests.stories import jsonl, manifest
from tests.test_plan_generation import config as plan_config


def lamp_plan() -> dict[str, Any]:
    """The tiny fixture checkpoint's own approved training annotation, as saved by the pilot."""
    text = "Ann saw the lamp.\n\n***\n\nAnn lit the lamp.\n"
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()

    def scene(number: int, start: int, end: int) -> dict[str, Any]:
        span = {"source_id": "edition-1", "source_sha256": checksum, "start": start, "end": end}
        return {
            "id": f"lamp.scene-{number + 1}",
            "title": f"Moment {number}",
            "summary": "",
            "character_ids": ["ann"],
            "beats": [],
            "evidence": [span],
        }

    def state(kind: str, value: str) -> dict[str, Any]:
        unknown: dict[str, Any] = {
            "value": None,
            "status": "unknown",
            "confidence": None,
            "alternatives": [],
        }
        known: dict[str, Any] = {
            "value": value,
            "status": "known",
            "confidence": None,
            "alternatives": [],
        }
        return {
            "id": kind,
            "scene_id": "lamp.scene-1",
            "character_id": "ann",
            "kind": kind,
            "key": "lamp",
            "before": unknown,
            "after": known,
        }

    channels = ["setups", "payoff_links", "reveal_links", "knowledge_uses", "relationships"]
    channels += ["constraints", "arcs", "reader_affect", "tension", "themes", "motifs"]
    channels += ["motif_echoes", "discourse", "subtext"]
    return {
        "id": "fixture-plan",
        "schema_version": "1",
        "tier": "L",
        "title": "The Lamp",
        "characters": [{"id": "ann", "name": "Ann"}],
        "sources": [
            {
                "id": "edition-1",
                "title": "The Lamp",
                "text": text,
                "sha256": checksum,
                "provenance": "https://example.org/lamp",
                "span_basis": "unicode-codepoints-lf",
            }
        ],
        "scenes": [scene(0, 0, 19), scene(1, 22, 42)],
        "causal_links": [
            {
                "id": "cause",
                "kind": "causal",
                "cause_scene_id": "lamp.scene-1",
                "effect_scene_id": "lamp.scene-2",
            }
        ],
        "knowledge_facts": [
            {"id": "lamp-fact", "description": "The lamp is present", "initial_holder_ids": ["ann"]}
        ],
        "states": [state("goal", "light the lamp"), state("belief", "the lamp works")],
        **{key: [] for key in channels},
    }


SOURCE = lamp_plan()
SIGNAL = json.loads((REPO / "tests/fixtures/narrative-structure.json").read_text())
# The approved channel policy of the tiny fixture checkpoint's training targets.
POLICY = {
    "belief": "hard",
    "causality": "hard",
    "goal": "hard",
    "information": "hard",
    "interpretation": "excluded",
    "plot": "hard",
    "relationships": "excluded",
}
DESCRIPTION = "knowledge_facts/lamp-fact/description"
TEXT = "A cracked lamp that still holds oil."
UNITS: dict[str, Any] = {
    "mask": {"units": ["states/belief/key", DESCRIPTION]},
    "locks": ["states/goal"],
}
SPAN: dict[str, Any] = {"mask": {"scene_ids": ["lamp.scene-2"], "channels": ["causal_links"]}}
BACKENDS = ["d", pytest.param("e", marks=pytest.mark.ml)]
type Mutation = Callable[[dict[str, Any]], None]


@pytest.fixture(scope="module")
def learned(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if importlib.util.find_spec("torch") is None:
        if os.environ.get("NDE_REQUIRE_ML") == "1":
            pytest.fail("CPU ML lane requires the pinned torch runtime")
        pytest.skip("optional CPU ML runtime")
    directory = tmp_path_factory.mktemp("revision-backends-cpu")
    fake = FakeOpenAI()
    fake.start()
    try:
        checkpoint = trained_checkpoint(directory / "training", fake)
    finally:
        fake.stop()
    annotation = directory / "training/data/pilot/stories/lamp/family-0/annotation/plan.json"
    # The shared source is the tiny model's own approved training annotation.
    assert json.loads(annotation.read_text()) == SOURCE
    detached = directory / "model.json"
    detached.write_bytes(checkpoint.read_bytes())
    (directory / "training").rename(directory / "unavailable-training")
    return detached


@pytest.fixture
def llm(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    return fake_llm


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> dict[str, Any]:
    """The same public request selecting D, or E with the tiny trained CPU checkpoint."""
    if request.param == "d":
        return {"kind": "d"}
    return {
        "kind": "e",
        "checkpoint": str(request.getfixturevalue("learned")),
        "target_policy": POLICY,
        "conditions": {"genre": "parable"},
    }


def run_config(
    tmp_path: Path,
    base_url: str,
    revision: dict[str, Any],
    backend: dict[str, Any] | None,
    plan: dict[str, Any] = SOURCE,
) -> dict[str, Any]:
    source = tmp_path / "source.json"
    if not source.exists():
        source.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    value = plan_config(base_url)
    value["plans"]["params"] = {
        "saved_plan": str(source),
        "revision": {"instruction": "Revise only the authorised units", **revision},
        **({} if backend is None else {"backend": backend}),
    }
    value["structured"] = {"max_repairs": 0}
    return value


def revise(tmp_path: Path, value: dict[str, Any], expected: int) -> Path | None:
    runs = tmp_path / "runs"
    before = set(runs.iterdir()) if runs.exists() else set[Path]()
    arguments = ["plan", "revise", str(write_config(tmp_path, value)), "--runs-dir", str(runs)]
    arguments += ["--cache", str(tmp_path / "cache.sqlite")]
    assert main([*arguments, "--ledger", str(tmp_path / "ledger.sqlite")]) == expected
    created = set(runs.iterdir()) - before if runs.exists() else set[Path]()
    return created.pop() if created else None


def mutated(plan_change: Mutation, source: dict[str, Any] = SOURCE) -> dict[str, Any]:
    plan = copy.deepcopy(source)
    plan_change(plan)
    return plan


def accepted_plan(plan: dict[str, Any]) -> None:
    plan["states"][1]["key"] = "lakp"
    plan["knowledge_facts"][0]["description"] = TEXT


def responder(plan: dict[str, Any]) -> Callable[[Json], Reply]:
    """D receives a complete candidate; E's hybrid step receives only text selectors."""

    def reply(body: Json) -> Reply:
        if "authorised text of a masked structural revision" in str(body["messages"][0]):
            return completion(json.dumps({"fields": {DESCRIPTION: TEXT}}))
        return completion(json.dumps(plan))

    return reply


def folder_of(run: Path) -> Path:
    [record] = jsonl(run / "plans.jsonl")
    return run / record["path"]


def provenance(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "revisions/revision/provenance.json").read_text())


def test_both_backends_accept_authorised_units_with_hybrid_text_and_shared_artifacts(
    tmp_path: Path,
    llm: FakeOpenAI,
    backend: dict[str, Any],
) -> None:
    llm.default = responder(mutated(accepted_plan))
    run = revise(tmp_path, run_config(tmp_path, llm.base_url, UNITS, backend), 0)
    assert run is not None
    folder = folder_of(run)
    record = provenance(folder)
    plan = json.loads((folder / "plan.json").read_text())
    assert record["status"] == "accepted"
    assert record["before_sha256"] == identity(NarrativePlan.model_validate(SOURCE))
    assert record["after_sha256"] == identity(NarrativePlan.model_validate(plan))
    assert record["backend"]["kind"] == backend["kind"]
    assert record["revision"]["mask"]["units"] == UNITS["mask"]["units"]
    # Learned or LLM structure may differ, but only authorised units change.
    assert set(record["changed_units"]) <= {"states/belief/key", DESCRIPTION}
    assert plan["knowledge_facts"][0]["description"] == TEXT
    assert plan["states"][0] == SOURCE["states"][0]
    assert plan["sources"] == SOURCE["sources"]
    assert len(record["calls"]) == llm.call_count == 1
    assert record["calls"][0]["cost_usd"] > 0
    assert json.loads((folder / "revisions/revision/boundary.json").read_text())["valid"]
    assert TEXT in (folder / "cards.html").read_text()
    assert (tmp_path / "source.json").read_text() == json.dumps(SOURCE, indent=2)
    if backend["kind"] == "e":
        e = record["backend"]
        assert e["test_only"]
        assert e["masked_leaves"] == ["states/belief/key"]
        assert e["text_fields"] == [DESCRIPTION]
        assert e["work"]["tokens"] > 0
        assert e["work"]["passes"] == 1
        assert e["attempts"][0]["status"] == "selected"
        # Actual tiny-model inference changed the masked key; the LLM supplied only text.
        assert record["changed_units"] == [DESCRIPTION, "states/belief/key"]
        assert plan["states"][1]["key"] != "lamp"
        assert record["calls"][0]["step"] == "revision-revision-text"
    report = json.loads((run / "alternatives.json").read_text())
    assert report["consistent"]
    assert report["counts"]["distinct_changed"] == 1
    assert main(["plan", "alternatives", str(run)]) == 0
    # Existing plan-only control evidence consumes either backend's provenance unchanged.
    control = folder / "control.json"
    control.write_text(
        json.dumps(
            {
                "before": "revisions/revision/source-plan.json",
                "after": "plan.json",
                "revision": record["revision"],
                "revision_provenance": "revisions/revision/provenance.json",
                "targets": [{"id": "fact", "kind": "changed", "selector": DESCRIPTION}],
            }
        )
    )
    assert main(["control-report", str(control), "--out", str(tmp_path / "control")]) == 0
    evidence = json.loads((tmp_path / "control/control-report.json").read_text())
    assert evidence["status"] == "success"
    assert evidence["accepted_revision"] or evidence["authorised_scope"]


def scope_attack(plan: dict[str, Any]) -> None:
    plan["states"][1]["after"]["value"] = "the lamp fails"


def lock_attack(plan: dict[str, Any]) -> None:
    plan["states"][0]["after"]["value"] = "light the lump"


def crossing_attack(plan: dict[str, Any]) -> None:
    link = plan["causal_links"][0]
    link["cause_scene_id"], link["effect_scene_id"] = (
        link["effect_scene_id"],
        link["cause_scene_id"],
    )


def identity_attack(plan: dict[str, Any]) -> None:
    plan["scenes"][1]["id"] = "lamp.scene-9"


def view_identity_attack(view: dict[str, Any]) -> None:
    view["scenes"][1]["id"] = "scenes.9"


def knowledge_attack(plan: dict[str, Any]) -> None:
    # The use moves out of the s3 span into s1, before the s2 reveal: the fragment looks local.
    plan["knowledge_uses"][1]["scene_id"] = plan["scenes"][0]["id"]


def payoff_attack(plan: dict[str, Any]) -> None:
    plan["payoff_links"][0]["payoff_scene_id"] = plan["scenes"][0]["id"]


def other(revision: dict[str, Any]) -> dict[str, Any]:
    return {"mask": {"scene_ids": ["s3"], **revision}}


ATTACKS: dict[str, tuple[dict[str, Any], dict[str, Any], Mutation, Mutation, str]] = {
    # name: (source, revision authority, plan mutation, E-view mutation, required diagnostic)
    "out_of_scope": (SOURCE, UNITS, scope_attack, scope_attack, "revision_scope"),
    "lock": (
        SOURCE,
        {"mask": {"channels": ["states"]}, "locks": ["states/goal"]},
        lock_attack,
        lock_attack,
        "revision_lock",
    ),
    "crossing_link": (SOURCE, SPAN, crossing_attack, crossing_attack, "causal_order"),
    "crossing_knowledge": (
        SIGNAL,
        other({"channels": ["knowledge_uses"]}),
        knowledge_attack,
        knowledge_attack,
        "knowledge_before_acquisition",
    ),
    "crossing_payoff": (
        SIGNAL,
        other({"channels": ["payoff_links"]}),
        payoff_attack,
        payoff_attack,
        "payoff_order",
    ),
    "immutable_identity": (
        SOURCE,
        SPAN,
        identity_attack,
        view_identity_attack,
        "revision_identity",
    ),
}


def adversarial_sampler(
    mutation: Mutation, source: dict[str, Any]
) -> Callable[..., dict[str, Any]]:
    """A learned-candidate fixture that ignores model quality entirely."""
    view = textless(encode_plan(NarrativePlan.model_validate(source), POLICY, {})["view"])
    mutation(view)

    def sample(
        checkpoint: Path, *_args: Any, observed: list[int], **_options: Any
    ) -> dict[str, Any]:
        tokens = tokenize(view)
        return {
            "status": "complete",
            "checkpoint_id": read_sealed(checkpoint)["id"],
            "generated_tokens": tokens + [PAD] * (len(observed) - len(tokens)),
            "diagnostics": [],
            "work": {"tokens": 1, "attention_cells": 1, "passes": 1, "elapsed_seconds": 0.0},
        }

    return sample


@pytest.mark.parametrize("attack", list(ATTACKS))
def test_adversarial_candidates_are_rejected_identically_and_preserve_originals(
    tmp_path: Path,
    llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    backend: dict[str, Any],
    attack: str,
) -> None:
    source, revision, plan_change, view_change, code = ATTACKS[attack]
    llm.default = responder(mutated(plan_change, source))
    monkeypatch.setattr(inpainting, "sample_candidate", adversarial_sampler(view_change, source))
    run = revise(tmp_path, run_config(tmp_path, llm.base_url, revision, backend, source), 1)
    assert run is not None
    folder = folder_of(run)
    record = provenance(folder)
    evidence = (
        (folder / "planning/revision-revision/1.diagnostics.json").read_text()
        if backend["kind"] == "d"
        else (folder / "revisions/revision/inpainting/attempt-001.json").read_text()
    )
    assert code in evidence
    assert record["status"] == "failed"
    assert record["before_sha256"] == record["after_sha256"]
    assert record["changed_units"] == []
    assert not (folder / "plan.json").exists()
    last = json.loads((folder / "revisions/revision/last-accepted/plan.json").read_text())
    assert last == NarrativePlan.model_validate(source).model_dump(mode="json")
    assert (tmp_path / "source.json").read_text() == json.dumps(source, indent=2)
    assert manifest(run)["status"] == "failed"
    # E stops before hybrid text completion; no rejected structure is ever completed.
    assert llm.call_count == (1 if backend["kind"] == "d" else 0)
    report = json.loads((run / "alternatives.json").read_text())
    assert report["counts"]["failed"] == 1
    assert report["counts"]["distinct_changed"] == 0
    assert report["consistent"]


@pytest.mark.parametrize(
    "revision",
    [
        {"mask": {"units": ["states/belief/key"]}, "locks": ["states/belief"]},
        {"mask": {"scene_ids": ["lamp.scene-2", "lamp.scene-1"]}},
        {"mask": {"units": ["scenes/lamp.scene-1/id"]}},
        {"mask": {"channels": ["constraints"]}},
    ],
)
def test_invalid_authority_fails_before_calls_or_inference_for_both_backends(
    tmp_path: Path,
    llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    backend: dict[str, Any],
    revision: dict[str, Any],
) -> None:

    def forbidden(*_args: Any, **_options: Any) -> dict[str, Any]:
        raise AssertionError("inference must not start")

    monkeypatch.setattr(inpainting, "sample_candidate", forbidden)
    value = run_config(tmp_path, llm.base_url, revision, backend)
    assert revise(tmp_path, value, 2) is None
    assert llm.call_count == 0


def test_duplicate_alternatives_are_reported_not_counted_as_diversity(
    tmp_path: Path,
    llm: FakeOpenAI,
    backend: dict[str, Any],
) -> None:
    llm.default = responder(mutated(accepted_plan))
    value = run_config(tmp_path, llm.base_url, UNITS, backend)
    value["plans"]["repeat"] = 3
    run = revise(tmp_path, value, 0)
    assert run is not None
    report = json.loads((run / "alternatives.json").read_text())
    rows = report["alternatives"]
    assert [row["seed"] for row in rows] == [7, 8, 9]
    assert {row["before_sha256"] for row in rows} == {report["source"]["plan_sha256"]}
    assert report["counts"] | {"expected": 3} == {
        "expected": 3,
        "accepted": 3,
        "noop": 0,
        "failed": 0,
        "missing": 0,
        "inconsistent": 0,
        "distinct_changed": 1,
        "exact_duplicates": 2,
        "structurally_distinct": 1,
        "structural_duplicates": 2,
    }
    assert [row["duplicate_of"] for row in rows] == [None, rows[0]["path"], rows[0]["path"]]
    assert all(row["calls"] == 1 for row in rows)
    assert "never counted as diversity" in (run / "alternatives.md").read_text()
    assert main(["plan", "alternatives", str(run)]) == 0
    changed = run / rows[1]["path"] / "plan.json"
    original = changed.read_text()
    changed.write_text(original.replace(TEXT, "Edited after the run"))
    assert main(["plan", "alternatives", str(run)]) == 1
    changed.write_text(original)
    assert main(["plan", "alternatives", str(run)]) == 0
    (tmp_path / "source.json").write_text(json.dumps(SOURCE))
    assert main(["plan", "alternatives", str(run)]) == 1


def test_omitted_backend_keeps_released_d_provenance_and_explicit_d_adds_metadata(
    tmp_path: Path, llm: FakeOpenAI
) -> None:
    llm.default = responder(mutated(accepted_plan))
    released = revise(tmp_path, run_config(tmp_path, llm.base_url, UNITS, None), 0)
    explicit = revise(tmp_path, run_config(tmp_path, llm.base_url, UNITS, {"kind": "d"}), 0)
    assert released is not None
    assert explicit is not None
    old, new = provenance(folder_of(released)), provenance(folder_of(explicit))
    assert list(old) == [
        "version",
        "status",
        "error",
        "revision",
        "before_sha256",
        "after_sha256",
        "changed_units",
        "lock_identities",
        "schema_sha256",
        "git_sha",
        "model",
        "prompt_sha256",
        "seed",
        "max_repairs",
        "brief",
        "calls",
    ]
    assert new.pop("backend") == {"version": "1", "kind": "d"}
    assert {**new, "calls": None} == {**old, "calls": None}
    assert (folder_of(released) / "plan.json").read_bytes() == (
        folder_of(explicit) / "plan.json"
    ).read_bytes()
    # The second run replays the cached call: identical work, no new provider spend.
    assert llm.call_count == 1
    assert json.loads((explicit / "alternatives.json").read_text())["backends"] == ["d"]


def test_existing_revision_requests_parse_without_a_backend() -> None:
    params = RevisionPlanningParams.model_validate(
        {
            "saved_plan": "plan.json",
            "revision": {"version": "1", "mask": {"channels": ["arcs"]}, "instruction": "x"},
        }
    )
    assert params.backend is None
    assert "backend" not in params.revision.model_dump()


@pytest.mark.ml
def test_actual_e_inference_rejects_invalid_learned_references_and_keeps_work(
    tmp_path: Path, llm: FakeOpenAI, learned: Path
) -> None:
    backend = {"kind": "e", "checkpoint": str(learned), "target_policy": POLICY}
    run = revise(
        tmp_path,
        run_config(tmp_path, llm.base_url, {"mask": {"scene_ids": ["lamp.scene-2"]}}, backend),
        1,
    )
    assert run is not None
    folder = folder_of(run)
    record = provenance(folder)
    attempt = json.loads((folder / "revisions/revision/inpainting/attempt-001.json").read_text())
    assert record["status"] == "failed"
    assert record["backend"]["status"] == "deterministic-rejection"
    assert record["backend"]["masked_leaves"] == ["scenes/lamp.scene-2/character_ids"]
    assert attempt["status"] == "rejected"
    assert attempt["diagnostics"]
    assert record["backend"]["work"]["tokens"] == attempt["work"]["tokens"] > 0
    assert llm.call_count == 0
    assert not (folder / "plan.json").exists()
    assert (folder / "revisions/revision/last-accepted/plan.json").exists()


@pytest.mark.ml
def test_stochastic_e_alternatives_use_their_own_seeds_and_honest_counts(
    tmp_path: Path, llm: FakeOpenAI, learned: Path
) -> None:
    checkpoint = read_sealed(learned)
    checkpoint["binding"]["config"]["sampling_temperature"] = 1
    stochastic = tmp_path / "stochastic.json"
    create_record(stochastic, checkpoint)
    backend = {
        "kind": "e",
        "checkpoint": str(stochastic),
        "target_policy": POLICY,
        "guidance_scale": 2,
        "max_attempts": 4,
    }
    value = run_config(tmp_path, llm.base_url, {"mask": {"units": ["states/goal/key"]}}, backend)
    value["plans"]["repeat"] = 4
    run = revise(tmp_path, value, 0)
    assert run is not None
    report = json.loads((run / "alternatives.json").read_text())
    rows = report["alternatives"]
    seeds = [
        attempt["seed"]
        for row in rows
        for attempt in provenance(run / row["path"])["backend"]["attempts"]
    ]
    assert len(seeds) == len(set(seeds))
    assert all(
        row["checkpoint_id"] == read_sealed(stochastic)["id"] and row["guidance_scale"] == 2
        for row in rows
    )
    assert all(row["local_work"]["passes"] >= 2 for row in rows)
    accepted = {row["after_sha256"] for row in rows if row["status"] == "accepted"}
    assert report["counts"]["distinct_changed"] == len(accepted)
    assert report["counts"]["exact_duplicates"] == report["counts"]["accepted"] - len(accepted)
    assert report["counts"]["noop"] == sum(row["status"] == "noop" for row in rows)
    assert llm.call_count == 0
    assert report["test_only"]


@pytest.mark.ml
@pytest.mark.parametrize(
    "case",
    [
        ({"test_only": False}, "G3"),
        ({"target_policy": {**POLICY, "goal": "excluded", "belief": "excluded"}}, "represent"),
    ],
)
def test_e_preflight_rejects_unauthorised_checkpoints_and_unrepresentable_masks(
    tmp_path: Path,
    llm: FakeOpenAI,
    capsys: pytest.CaptureFixture[str],
    learned: Path,
    case: tuple[dict[str, Any], str],
) -> None:
    change, message = case
    checkpoint = learned
    if "test_only" in change:
        record = read_sealed(learned)
        record["binding"]["config"]["test_only"] = False
        checkpoint = tmp_path / "relabelled.json"
        create_record(checkpoint, record)
    backend = {
        "kind": "e",
        "checkpoint": str(checkpoint),
        "target_policy": change.get("target_policy", POLICY),
    }
    value = run_config(tmp_path, llm.base_url, {"mask": {"units": ["states/goal/key"]}}, backend)
    assert revise(tmp_path, value, 2) is None
    assert message in capsys.readouterr().err
    assert llm.call_count == 0


@pytest.mark.ml
def test_e_resource_stop_performs_no_unbudgeted_inference(
    tmp_path: Path, llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, learned: Path
) -> None:
    def forbidden(*_args: Any, **_options: Any) -> dict[str, Any]:
        raise AssertionError("inference exceeds the registered allowance")

    monkeypatch.setattr(inpainting, "sample_candidate", forbidden)
    backend = {"kind": "e", "checkpoint": str(learned), "target_policy": POLICY, "max_tokens": 1}
    run = revise(tmp_path, run_config(tmp_path, llm.base_url, UNITS, backend), 1)
    assert run is not None
    record = provenance(folder_of(run))
    assert record["status"] == "failed"
    assert record["backend"]["status"] == "resource-stop"
    assert record["backend"]["attempts"] == []
    assert record["backend"]["work"]["tokens"] == 0
    assert llm.call_count == 0


@pytest.mark.ml
@pytest.mark.parametrize("failure", ["unauthorised_text", "provider"])
def test_e_hybrid_text_failures_keep_learned_work_and_the_source(
    tmp_path: Path, llm: FakeOpenAI, learned: Path, failure: str
) -> None:
    if failure == "provider":
        llm.queue(error(401, "synthetic provider failure"))
    else:
        fields = {DESCRIPTION: TEXT, "scenes/lamp.scene-1/title": "Unrequested"}
        llm.default = lambda _body: completion(json.dumps({"fields": fields}))
    backend = {"kind": "e", "checkpoint": str(learned), "target_policy": POLICY}
    run = revise(tmp_path, run_config(tmp_path, llm.base_url, UNITS, backend), 1)
    assert run is not None
    folder = folder_of(run)
    record = provenance(folder)
    assert record["status"] == "failed"
    assert record["backend"]["status"] == "selected"
    assert record["backend"]["work"]["tokens"] > 0
    assert not (folder / "plan.json").exists()
    assert (folder / "revisions/revision/last-accepted/plan.json").exists()
    assert (tmp_path / "source.json").read_text() == json.dumps(SOURCE, indent=2)
    if failure == "provider":
        assert "synthetic provider failure" in record["error"]
        assert manifest(run)["error"]["type"] == "authentication"
    else:
        diagnostics = folder / "planning/revision-revision-text/1.diagnostics.json"
        assert "revision_text" in diagnostics.read_text()
        assert len(record["calls"]) == 1


def test_plan_command_reports_bad_evidence_but_not_internal_faults(
    tmp_path: Path,
    llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    llm.default = responder(mutated(accepted_plan))
    run = revise(tmp_path, run_config(tmp_path, llm.base_url, UNITS, {"kind": "d"}), 0)
    assert run is not None
    # Malformed provenance is an inconsistent alternative, not a crash or a config error.
    path = folder_of(run) / "revisions/revision/provenance.json"
    record = json.loads(path.read_text())
    del record["seed"]
    path.write_text(json.dumps(record))
    capsys.readouterr()
    assert main(["plan", "alternatives", str(run)]) == 1
    [row] = json.loads(capsys.readouterr().out)["alternatives"]
    assert "provenance is malformed" in row["diagnostics"][0]
    # An unreadable saved report is a usage error with a message.
    (run / "alternatives.json").write_text("{")
    assert main(["plan", "alternatives", str(run)]) == 2
    assert "nde plan: saved alternatives evidence" in capsys.readouterr().err
    # An internal fault keeps its traceback instead of exiting like a config error.

    def fault(_plan: object) -> object:
        raise KeyError("internal")

    monkeypatch.setattr("nde.narrative.cli.validate_plan", fault)
    with pytest.raises(KeyError, match="internal"):
        main(["plan", "validate", str(tmp_path / "source.json")])


@pytest.mark.ml
def test_observation_masks_exactly_the_editable_spans_and_sampling_fills_only_them(
    learned: Path,
) -> None:
    checkpoint = inspect_checkpoint(learned)
    spec = RevisionSpec.model_validate({**UNITS, "instruction": "x"})
    backend = ERevisionBackend.model_validate(
        {"kind": "e", "checkpoint": learned, "target_policy": POLICY}
    )
    observation = inpainting.observe(
        NarrativePlan.model_validate(SOURCE), spec, backend, checkpoint
    )
    clear = tokenize(observation.view)
    clear += [PAD] * (len(observation.tokens) - len(clear))
    spans = [observation.leaves[path] for path in observation.editable]
    assert [leaf.selector for leaf in spans] == ["states/belief/key"]
    expected = {index for leaf in spans for index in range(leaf.start, leaf.end)}
    assert {i for i, token in enumerate(observation.tokens) if token == MASK} == expected
    for leaf, path in zip(spans, observation.editable, strict=True):
        # Each masked span is exactly the leaf's serialized value, at its full width.
        value = bytes(token - BYTE_OFFSET for token in clear[leaf.start : leaf.end])
        source: Any = observation.view
        for part in path:
            source = source[part]
        assert value.decode() == canonical_json(source)
    assert all(token == clear[i] for i, token in enumerate(observation.tokens) if i not in expected)
    result = sample_candidate(
        learned, Conditions(), seed=11, guidance_scale=0, observed=observation.tokens
    )
    generated = result["generated_tokens"]
    assert result["work"]["passes"] == 1
    assert len(generated) == len(observation.tokens)
    assert all(
        token == known
        for token, known in zip(generated, observation.tokens, strict=True)
        if known != MASK
    )
    assert all(generated[index] != MASK for index in expected)


@pytest.fixture
def bound_dataset(learned: Path) -> Iterator[Path]:
    """The fixture checkpoint with its training inputs back where it was trained."""
    hidden, training = learned.parent / "unavailable-training", learned.parent / "training"
    hidden.rename(training)
    try:
        yield learned
    finally:
        training.rename(hidden)


@pytest.mark.ml
def test_e_target_policy_is_verified_against_the_checkpoints_training_dataset(
    tmp_path: Path,
    llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bound_dataset: Path,
) -> None:
    llm.default = lambda _body: completion(json.dumps({"fields": {DESCRIPTION: TEXT}}))
    backend = {"kind": "e", "checkpoint": str(bound_dataset), "target_policy": POLICY}
    run = revise(tmp_path, run_config(tmp_path, llm.base_url, UNITS, backend), 0)
    assert run is not None
    check = provenance(folder_of(run))["backend"]["target_policy_check"]
    assert check["status"] == "verified"
    assert check["dataset_id"] == read_sealed(bound_dataset)["binding"]["dataset_id"]

    def forbidden(*_args: Any, **_options: Any) -> dict[str, Any]:
        raise AssertionError("a mismatched target policy must not reach inference")

    monkeypatch.setattr(inpainting, "sample_candidate", forbidden)
    calls = llm.call_count
    capsys.readouterr()
    for channel in ("relationships", "interpretation"):
        enabled = {**backend, "target_policy": {**POLICY, channel: "hard"}}
        value = run_config(tmp_path, llm.base_url, UNITS, enabled)
        assert revise(tmp_path, value, 2) is None
        assert "differs from the checkpoint's training policy" in capsys.readouterr().err
    assert llm.call_count == calls


@pytest.mark.ml
def test_e_target_policy_without_its_training_dataset_is_declared_or_refused(
    tmp_path: Path, llm: FakeOpenAI, learned: Path
) -> None:
    llm.default = lambda _body: completion(json.dumps({"fields": {DESCRIPTION: TEXT}}))
    backend = {"kind": "e", "checkpoint": str(learned), "target_policy": POLICY}
    run = revise(tmp_path, run_config(tmp_path, llm.base_url, UNITS, backend), 0)
    assert run is not None
    check = provenance(folder_of(run))["backend"]["target_policy_check"]
    assert check["status"] == "declared-unverified"
    # A real checkpoint whose training policy cannot be recovered is refused.
    real = read_sealed(learned)
    real["binding"]["config"]["test_only"] = False
    with pytest.raises(PlanError, match="cannot be verified"):
        inpainting.check_target_policy(
            real,
            ERevisionBackend.model_validate(
                {"kind": "e", "checkpoint": learned, "target_policy": POLICY}
            ),
        )


@pytest.mark.ml
@pytest.mark.parametrize("drift", ["replaced", "unreadable"])
def test_e_checkpoint_drift_during_inference_is_a_recorded_rejection(
    tmp_path: Path,
    llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    learned: Path,
    drift: str,
) -> None:
    model = tmp_path / "model.json"
    model.write_bytes(learned.read_bytes())
    preflight_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    real = inpainting.sample_candidate

    def drifting(*args: Any, **options: Any) -> dict[str, Any]:
        if drift == "replaced":
            changed = read_sealed(model)
            changed["binding"]["config"]["inference_max_seconds"] = 61
            model.unlink()
            create_record(model, {k: v for k, v in changed.items() if k not in {"id", "version"}})
        else:
            model.unlink()
        return real(*args, **options)

    monkeypatch.setattr(inpainting, "sample_candidate", drifting)
    backend = {"kind": "e", "checkpoint": str(model), "target_policy": POLICY}
    value = run_config(tmp_path, llm.base_url, UNITS, backend)
    value["plans"]["repeat"] = 2
    run = revise(tmp_path, value, 1)
    assert run is not None
    records = jsonl(run / "plans.jsonl")
    assert len(records) == 2
    assert records[0]["status"] == "failed"
    first = run / records[0]["path"]
    record = provenance(first)
    assert record["status"] == "failed"
    assert "checkpoint changed or became unreadable" in record["error"]
    assert record["backend"]["status"] == "checkpoint-changed"
    assert record["backend"]["checkpoint_file_sha256"] == preflight_sha256
    [attempt] = record["backend"]["attempts"]
    assert attempt["status"] == "rejected"
    assert "checkpoint" in attempt["diagnostics"][0]
    if drift == "replaced":
        # The drifted inference really ran, so its work stays counted.
        assert record["backend"]["work"]["tokens"] > 0
    else:
        assert record["backend"]["work"]["tokens"] == 0
    assert (first / "revisions/revision/boundary.json").is_file()
    assert (first / "revisions/revision/last-accepted/plan.json").is_file()
    assert not (first / "plan.json").exists()
    assert record["calls"] == []
    # The next alternative is still run and recorded rather than aborting the run.
    second = provenance(run / records[1]["path"])
    if drift == "unreadable":
        assert second["backend"]["status"] == "preflight-failed"


@pytest.mark.ml
def test_unexpected_e_sampler_failures_are_recorded_and_later_alternatives_still_run(
    tmp_path: Path, llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, learned: Path
) -> None:
    llm.default = responder(mutated(accepted_plan))
    real = inpainting.sample_candidate
    calls: list[int] = []

    def exhausted(*args: Any, **options: Any) -> dict[str, Any]:
        calls.append(options["seed"])
        if len(calls) == 1:
            # What torch raises when a local device runs out of memory.
            raise RuntimeError("out of memory: tried to allocate 2.00 GiB")
        return real(*args, **options)

    monkeypatch.setattr(inpainting, "sample_candidate", exhausted)
    backend = {"kind": "e", "checkpoint": str(learned), "target_policy": POLICY}
    value = run_config(tmp_path, llm.base_url, UNITS, backend)
    value["plans"]["repeat"] = 2
    run = revise(tmp_path, value, 1)
    assert run is not None
    records = jsonl(run / "plans.jsonl")
    assert [r["status"] for r in records] == ["failed", "succeeded"]
    assert "unexpected RuntimeError: out of memory" in records[0]["error"]
    first = run / records[0]["path"]
    record = provenance(first)
    assert record["status"] == "failed"
    assert "unexpected RuntimeError" in record["error"]
    assert record["backend"]["status"] == "sampler-failed"
    assert record["backend"]["checkpoint_id"] == inspect_checkpoint(learned)["id"]
    [attempt] = record["backend"]["attempts"]
    assert attempt["status"] == "failed"
    assert attempt["seed"] == calls[0]
    assert attempt["reserved_work"]["tokens"] > 0
    assert attempt["diagnostics"] == [
        "unexpected RuntimeError: out of memory: tried to allocate 2.00 GiB"
    ]
    saved = json.loads((first / "revisions/revision/inpainting/attempt-001.json").read_text())
    assert saved == attempt
    # No inference result came back, so no completed work is claimed.
    assert record["backend"]["work"]["tokens"] == 0
    trace = (first / "revisions/revision/inpainting/sampler-error.txt").read_text()
    assert "RuntimeError: out of memory" in trace
    assert (first / "revisions/revision/boundary.json").is_file()
    assert (first / "revisions/revision/source-plan.json").is_file()
    assert (first / "revisions/revision/last-accepted/plan.json").is_file()
    assert not (first / "plan.json").exists()
    assert record["calls"] == []
    # The run continues: the next alternative is recorded and reported.
    assert provenance(run / records[1]["path"])["backend"]["status"] == "selected"
    report = json.loads((run / "alternatives.json").read_text())
    assert [a["status"] for a in report["alternatives"]] == ["failed", "accepted"]


@pytest.mark.ml
def test_e_text_only_masks_run_no_learned_inference(
    tmp_path: Path, llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, learned: Path
) -> None:
    def forbidden(*_args: Any, **_options: Any) -> dict[str, Any]:
        raise AssertionError("no structural value is editable, so the checkpoint must not run")

    monkeypatch.setattr(inpainting, "sample_candidate", forbidden)
    llm.default = lambda _body: completion(json.dumps({"fields": {DESCRIPTION: TEXT}}))
    backend = {"kind": "e", "checkpoint": str(learned), "target_policy": POLICY}
    value = run_config(tmp_path, llm.base_url, {"mask": {"units": [DESCRIPTION]}}, backend)
    run = revise(tmp_path, value, 0)
    assert run is not None
    folder = folder_of(run)
    record = provenance(folder)
    assert record["status"] == "accepted"
    assert record["changed_units"] == [DESCRIPTION]
    metadata = record["backend"]
    assert metadata["status"] == "text-only"
    assert metadata["attempts"] == []
    assert metadata["masked_tokens"] == 0
    assert metadata["work"] == {"tokens": 0, "attention_cells": 0, "passes": 0} | {
        "elapsed_seconds": metadata["work"]["elapsed_seconds"]
    }
    assert metadata["unrepresented"] == []
    assert len(record["calls"]) == 1
    [row] = json.loads((run / "alternatives.json").read_text())["alternatives"]
    assert row["local_work"]["tokens"] == 0


@pytest.mark.ml
def test_e_reports_selected_structure_it_cannot_represent_per_unit(learned: Path) -> None:
    checkpoint = inspect_checkpoint(learned)
    spec = RevisionSpec.model_validate(
        {"mask": {"units": ["states/goal/key", "states/belief/key"]}, "instruction": "x"}
    )
    # Declared for an unverifiable fixture only: goal is outside this policy.
    backend = ERevisionBackend.model_validate(
        {"kind": "e", "checkpoint": learned, "target_policy": {**POLICY, "goal": "excluded"}}
    )
    plan = NarrativePlan.model_validate(SOURCE)
    observation = inpainting.observe(plan, spec, backend, checkpoint)
    assert [observation.leaves[path].selector for path in observation.editable] == [
        "states/belief/key"
    ]
    assert observation.summary()["unrepresented"] == [
        {"unit": "states/goal/key", "values": ["states/goal/key"]}
    ]
    # Every structural value of a fully represented selection is editable.
    full = RevisionSpec.model_validate(
        {"mask": {"channels": ["causal_links", "states"]}, "instruction": "x"}
    )
    policy = ERevisionBackend.model_validate(
        {"kind": "e", "checkpoint": learned, "target_policy": POLICY}
    )
    assert inpainting.observe(plan, full, policy, checkpoint).summary()["unrepresented"] == []
