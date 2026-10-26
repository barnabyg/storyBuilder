import errno
import json
import os
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from tests.conftest import REPO, write_config
from tests.fake_openai import FakeOpenAI, completion, error
from tests.stories import jsonl, manifest
from tests.test_plan_generation import config, skeleton


def revision_config(tmp_path: Path, base_url: str, revision: dict[str, Any]) -> dict[str, Any]:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(skeleton() | {"schema_version": "1"}), encoding="utf-8")
    value = config(base_url)
    value["plans"]["params"] = {"saved_plan": str(source), "revision": revision}
    return value


def revise(tmp_path: Path, value: dict[str, Any], expected: int = 0) -> Path:
    runs = tmp_path / "runs"
    before: set[Path] = set(runs.iterdir()) if runs.exists() else set()
    assert (
        main(
            [
                "plan",
                "revise",
                str(write_config(tmp_path, value)),
                "--runs-dir",
                str(runs),
                "--cache",
                str(tmp_path / "cache.sqlite"),
                "--ledger",
                str(tmp_path / "ledger.sqlite"),
            ]
        )
        == expected
    )
    [result] = set(runs.iterdir()) - before
    return result


@pytest.mark.skipif(os.name != "nt", reason="Windows directory sharing semantics")
@pytest.mark.parametrize("persistent", [False, True])
def test_windows_publication_locks_preserve_artifacts_without_repeating_provider_calls(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    persistent: bool,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = lambda _body: completion(json.dumps(skeleton()))
    value = revision_config(
        tmp_path,
        fake_llm.base_url,
        {"mask": {"channels": ["arcs"]}, "instruction": "Keep the accepted arc"},
    )
    source = (tmp_path / "source.json").read_bytes()
    native_rename = os.rename
    failures = 0

    def locked_rename(origin: Path, destination: Path) -> None:
        nonlocal failures
        # Fault injection at the OS filesystem boundary, as a Windows scanner lock.
        if origin.name.startswith(".plan-") and (persistent or failures < 2):
            failures += 1
            raise PermissionError(errno.EACCES, "Synthetic sharing lock", str(origin), 5)
        native_rename(origin, destination)

    monkeypatch.setattr(os, "rename", locked_rename)
    result = revise(tmp_path, value, expected=2 if persistent else 0)
    assert (tmp_path / "source.json").read_bytes() == source
    assert fake_llm.call_count == 1
    folder = result / "plans/brief-01"
    if persistent:
        assert not (folder / "r1").exists()
        [staging] = folder.glob(".plan-*")
        assert (staging / "revisions/revision/provenance.json").exists()
        assert (staging / "planning/revision-revision/1.txt").exists()
    else:
        assert (folder / "r1/plan.json").exists()
        assert (folder / "r1/cards.html").exists()


def test_channel_revision_saves_cards_changes_and_lock_identities(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    changed = skeleton()
    changed["arcs"][0]["reading"]["value"] = "Accepts shared responsibility"
    fake_llm.default = lambda _body: completion(json.dumps(changed))
    value = revision_config(
        tmp_path,
        fake_llm.base_url,
        {
            "mask": {"channels": ["arcs"]},
            "locks": ["scenes/scene-0", "constraints"],
            "instruction": "Revise the arc to accept shared responsibility",
        },
    )
    source = (tmp_path / "source.json").read_bytes()
    run = revise(tmp_path, value)
    [record] = jsonl(run / "plans.jsonl")
    folder = run / record["path"]
    assert (tmp_path / "source.json").read_bytes() == source
    assert json.loads((folder / "plan.json").read_text())["arcs"][0]["reading"]["value"] == (
        "Accepts shared responsibility"
    )
    assert "Accepts shared responsibility" in (folder / "cards.html").read_text()
    provenance = json.loads((folder / "revisions/revision/provenance.json").read_text())
    assert provenance["status"] == "accepted"
    assert provenance["before_sha256"] != provenance["after_sha256"]
    assert provenance["changed_units"] == ["arcs/arc-keeper/reading/value"]
    assert (
        provenance["lock_identities"]["scenes/scene-0"]["before"]
        == (provenance["lock_identities"]["scenes/scene-0"]["after"])
    )
    assert len(provenance["calls"]) == 1
    assert manifest(run)["status"] == "succeeded"


@pytest.mark.parametrize("attack", ["scope", "lock", "constraint", "identity", "schema"])
def test_rejected_channel_replacements_keep_source_and_bounded_attempts(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    attack: str,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    original = skeleton()
    changed = skeleton()
    if attack == "scope":
        changed["scenes"][1]["summary"] = "Unrequested change"
    elif attack == "lock":
        changed["arcs"][0]["reading"]["value"] = "Unlawful locked change"
    elif attack == "constraint":
        changed["constraints"] = [{"id": "unlock", "kind": "literary", "key": "ending"}]
    elif attack == "identity":
        changed["id"] = "unrelated-plan"
    fake_llm.default = lambda _body: completion(
        "not JSON" if attack == "schema" else json.dumps(changed)
    )
    value = revision_config(
        tmp_path,
        fake_llm.base_url,
        {
            "mask": {"channels": ["arcs"]},
            "locks": ["arcs/arc-keeper/reading"],
            "instruction": "Change only the unlocked arc fields",
        },
    )
    value["structured"] = {"max_repairs": 1}
    run = revise(tmp_path, value, expected=1)
    [record] = jsonl(run / "plans.jsonl")
    folder = run / record["path"]
    assert not (folder / "plan.json").exists()
    assert (
        json.loads((folder / "revisions/revision/last-accepted/plan.json").read_text())["id"]
        == (original["id"])
    )
    provenance = json.loads((folder / "revisions/revision/provenance.json").read_text())
    assert provenance["status"] == "failed"
    assert provenance["before_sha256"] == provenance["after_sha256"]
    assert len(provenance["calls"]) == 2
    assert fake_llm.call_count == 2
    assert (folder / "planning/revision-revision/2.diagnostics.json").exists()


@pytest.mark.parametrize(
    ("mask", "locks"),
    [
        ({"channels": []}, []),
        ({"channels": ["not-a-channel"]}, []),
        ({"units": ["arcs/missing"]}, []),
        ({"units": ["arcs/arc-keeper/reading"]}, ["arcs/arc-keeper"]),
        ({"channels": ["arcs"]}, ["scenes/missing"]),
        ({"channels": ["constraints"]}, []),
        ({"units": ["scenes/scene-0/id"]}, []),
    ],
)
def test_invalid_authority_fails_before_provider_calls(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mask: dict[str, Any],
    locks: list[str],
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    value = revision_config(
        tmp_path,
        fake_llm.base_url,
        {
            "mask": mask,
            "locks": locks,
            "instruction": "Revise",
        },
    )
    assert main(["plan", "revise", str(write_config(tmp_path, value))]) == 2
    assert fake_llm.call_count == 0


def test_channel_repairs_receive_scope_diagnostics_and_cached_replay_is_free(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    bad = skeleton()
    bad["scenes"][0]["summary"] = "Outside mask"
    good = skeleton()
    good["arcs"][0]["reading"]["value"] = "New arc"
    fake_llm.queue(completion(json.dumps(bad)), completion(json.dumps(good)))
    value = revision_config(
        tmp_path,
        fake_llm.base_url,
        {
            "mask": {"channels": ["arcs"]},
            "instruction": "Change the arc",
        },
    )
    first = revise(tmp_path, value)
    assert "revision_scope" in str(fake_llm.requests[1].body["messages"][-1]["content"])
    second = revise(tmp_path, value)
    [a] = jsonl(first / "plans.jsonl")
    [b] = jsonl(second / "plans.jsonl")
    assert (first / a["path"] / "plan.json").read_bytes() == (
        second / b["path"] / "plan.json"
    ).read_bytes()
    assert manifest(second)["cost"]["usd"] == 0
    assert manifest(second)["cost"]["input_tokens"] == manifest(first)["cost"]["input_tokens"]
    assert len(jsonl(second / "calls.jsonl")) == 2
    assert fake_llm.call_count == 2


def test_noop_is_explicit_and_budget_abort_retains_last_accepted(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = lambda _body: completion(json.dumps(skeleton()))
    value = revision_config(
        tmp_path,
        fake_llm.base_url,
        {
            "mask": {"channels": ["arcs"]},
            "instruction": "Keep the arc if no valid change",
        },
    )
    run = revise(tmp_path, value)
    [record] = jsonl(run / "plans.jsonl")
    folder = run / record["path"]
    assert (
        json.loads((folder / "revisions/revision/provenance.json").read_text())["status"] == "noop"
    )
    value["seed"] += 1
    value["budget"] = {"per_run_usd": 0}
    aborted = revise(tmp_path, value, expected=1)
    assert manifest(aborted)["status"] == "aborted"
    [record] = jsonl(aborted / "plans.jsonl")
    assert (aborted / record["path"] / "revisions/revision/last-accepted/plan.json").exists()
    assert fake_llm.call_count == 1


def test_provider_failure_retains_revision_provenance_and_source(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.queue(error(401, "synthetic provider failure"))
    value = revision_config(
        tmp_path,
        fake_llm.base_url,
        {
            "mask": {"channels": ["arcs"]},
            "instruction": "Revise the arc",
        },
    )
    run = revise(tmp_path, value, expected=1)
    [record] = jsonl(run / "plans.jsonl")
    folder = run / record["path"]
    provenance = json.loads((folder / "revisions/revision/provenance.json").read_text())
    assert provenance["status"] == "failed"
    assert "synthetic provider failure" in provenance["error"]
    assert (folder / "revisions/revision/last-accepted/plan.json").exists()
    assert not (folder / "plan.json").exists()
    assert manifest(run)["error"]["type"] == "authentication"


@pytest.mark.parametrize(
    "span",
    [
        ["scene-0"],
        ["scene-1", "scene-2"],
        ["scene-4"],
        [f"scene-{i}" for i in range(5)],
    ],
)
def test_scene_spans_save_changed_content_with_unchanged_surroundings(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    span: list[str],
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    changed = skeleton()
    for scene in changed["scenes"]:
        if scene["id"] in span:
            scene["summary"] = "A revised scene function"
    fake_llm.default = lambda _body: completion(json.dumps(changed))
    value = revision_config(
        tmp_path,
        fake_llm.base_url,
        {
            "mask": {"scene_ids": span},
            "instruction": "Revise these scene functions",
        },
    )
    run = revise(tmp_path, value)
    [record] = jsonl(run / "plans.jsonl")
    folder = run / record["path"]
    plan = json.loads((folder / "plan.json").read_text())
    for scene in plan["scenes"]:
        assert scene["summary"] == (
            "A revised scene function"
            if scene["id"] in span
            else f"Function {scene['id'].split('-')[-1]}"
        )
    boundary = json.loads((folder / "revisions/revision/boundary.json").read_text())
    assert boundary["scene_ids"] == span
    assert boundary["valid"]


@pytest.mark.parametrize(
    ("span", "locks"),
    [
        ([], []),
        (["scene-0", "scene-2"], []),
        (["scene-2", "scene-1"], []),
        (["missing"], []),
        (["scene-1", "scene-1"], []),
        (["scene-1"], ["scenes/scene-1"]),
        (["scene-1"], ["scenes"]),
    ],
)
def test_invalid_spans_and_lock_conflicts_fail_before_spend(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    span: list[str],
    locks: list[str],
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    value = revision_config(
        tmp_path,
        fake_llm.base_url,
        {
            "mask": {"scene_ids": span},
            "locks": locks,
            "instruction": "Revise span",
        },
    )
    assert main(["plan", "revise", str(write_config(tmp_path, value))]) == 2
    assert fake_llm.call_count == 0


@pytest.mark.parametrize(
    ("attack", "diagnostic"),
    [
        ("dangling", "causal_endpoint"),
        ("timing", "causal_order"),
        ("knowledge", "knowledge_before_acquisition"),
        ("presence", "knowledge_presence"),
        ("locked_edge", "revision_lock"),
        ("scope_edge", "revision_scope"),
        ("nested_lock", "revision_lock"),
        ("scene_identity", "revision_identity"),
        ("scene_count", "revision_identity"),
    ],
)
def test_boundary_mutations_and_lock_attacks_are_rejected(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    attack: str,
    diagnostic: str,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    value = revision_config(
        tmp_path,
        fake_llm.base_url,
        {
            "mask": {"scene_ids": ["s2"], "channels": ["causal_links", "knowledge_uses"]},
            "locks": ["causal_links/cause1", "scenes/s2/title"],
            "instruction": "Revise span",
        },
    )
    fixture = json.loads((REPO / "tests/fixtures/narrative-structure.json").read_text())
    (tmp_path / "source.json").write_text(json.dumps(fixture))
    changed = json.loads(json.dumps(fixture))
    if attack == "dangling":
        changed["causal_links"][0]["cause_scene_id"] = "missing"
    elif attack == "timing":
        changed["causal_links"][0]["cause_scene_id"] = "s3"
        changed["causal_links"][0]["effect_scene_id"] = "s2"
    elif attack == "knowledge":
        changed["knowledge_uses"][1]["scene_id"] = "s1"
    elif attack == "presence":
        changed["scenes"][1]["character_ids"] = []
    elif attack in {"locked_edge", "scope_edge"}:
        changed["causal_links"][0]["effect_scene_id"] = "s2"
        if attack == "scope_edge":
            value["plans"]["params"]["revision"]["mask"]["channels"] = []
    elif attack == "nested_lock":
        changed["scenes"][1]["title"] = "Locked title attack"
    elif attack == "scene_identity":
        changed["scenes"][1]["id"] = "replacement"
    elif attack == "scene_count":
        changed["scenes"].pop()
    fake_llm.default = lambda _body: completion(json.dumps(changed))
    value["structured"] = {"max_repairs": 0}
    run = revise(tmp_path, value, expected=1)
    [record] = jsonl(run / "plans.jsonl")
    folder = run / record["path"]
    diagnostics = (folder / "planning/revision-revision/1.diagnostics.json").read_text()
    assert diagnostic in diagnostics
    assert (folder / "revisions/revision/last-accepted/plan.json").exists()
    assert not (folder / "plan.json").exists()


def test_valid_boundary_preserving_span_retains_causal_payoff_and_knowledge_context(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    value = revision_config(
        tmp_path,
        fake_llm.base_url,
        {
            "mask": {"scene_ids": ["s2", "s3"]},
            "locks": ["causal_links", "constraints"],
            "instruction": "Add concrete scene summaries without changing structural context",
        },
    )
    fixture = json.loads((REPO / "tests/fixtures/narrative-structure.json").read_text())
    (tmp_path / "source.json").write_text(json.dumps(fixture))
    changed = json.loads(json.dumps(fixture))
    changed["scenes"][1]["summary"] = "The warning transmits knowledge of the spare flare"
    changed["scenes"][2]["summary"] = "Ivo uses the acquired knowledge to save the boat"
    fake_llm.default = lambda _body: completion(json.dumps(changed))
    run = revise(tmp_path, value)
    [record] = jsonl(run / "plans.jsonl")
    boundary = json.loads((run / record["path"] / "revisions/revision/boundary.json").read_text())
    assert boundary["incoming_causal_links"] == ["cause1"]
    assert boundary["crossing_payoff_links"] == ["payoff1"]
    assert [event["id"] for event in boundary["knowledge_events"]] == ["reveal1", "use1", "use2"]
    assert boundary["valid"]
