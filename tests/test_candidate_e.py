"""Candidate E through the public story run: tiny CPU inference and fake-provider calls."""

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest

from nde.annotations.records import digest
from nde.cli import main
from nde.corpus.records_io import create_record, identity, read_sealed
from nde.narrative.ledger import plan_identity
from nde.narrative.workflow import load_plan
from nde.plan.revision import identity as canonical_identity
from nde.runs.config import load_config
from nde.runs.records import read_stories
from nde.runs.report import render_report
from nde.stories.resources import call_category
from nde.stories.selected import selected_text
from tests.conditional_support import trained_checkpoint
from tests.conftest import REPO, write_config
from tests.fake_openai import FakeOpenAI, Json, Reply, completion, error
from tests.stories import (
    jsonl,
    last_user_message,
    manifest,
    reply_with_story,
    run,
    stories,
    story_config,
    story_text,
)
from tests.test_baseline_d import BASELINE_D, fake_d

FILLED = "A promise is kept."
POLICY = {"version": "1"}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if importlib.util.find_spec("torch") is None:
        if os.environ.get("NDE_REQUIRE_ML") == "1":
            pytest.fail("CPU ML lane requires the pinned torch runtime")
        pytest.skip("optional CPU ML runtime")
    directory = tmp_path_factory.mktemp("candidate-e-cpu")
    fake = FakeOpenAI()
    fake.start()
    try:
        trained = trained_checkpoint(directory / "training", fake)
    finally:
        fake.stop()
    detached = directory / "model.json"
    detached.write_bytes(trained.read_bytes())
    return detached


def e_condition(model: Path, **params: Any) -> dict[str, Any]:
    # The tiny fixture model has two-scene capacity, so the brief's scene range stays unknown.
    return {
        "name": "E",
        "pipeline": "candidate_e",
        "params": {
            "checkpoint": str(model),
            "condition_fields": ["genre", "themes", "ending_intent", "target_words"],
            "guidance_scale": 0,
            **params,
        },
    }


def fill(body: Json) -> Reply:
    request, _ = json.JSONDecoder().raw_decode(str(body["messages"][1]["content"]))
    return completion(json.dumps({"fields": dict.fromkeys(request["fields"], FILLED)}))


def is_completion(body: Json) -> bool:
    return str(body["messages"][0]["content"]).startswith("Complete only the requested text")


class Provider:
    """E completion, E's two scenes, a story revision, and otherwise Candidate D."""

    def __init__(self, *, e_scene_tokens: int = 1000, short_scenes: int = 0) -> None:
        self.e_scene_tokens = e_scene_tokens
        self.short_scenes = short_scenes
        self.e_scenes = 0

    def __call__(self, body: Json) -> Reply:
        if is_completion(body):
            return fill(body)
        if "revising your own short story" in str(body["messages"][0]["content"]):
            return completion(story_text(2000), completion_tokens=1000)
        message = last_user_message(body)
        if "Now write scene" in message and " of 2," in message:
            self.e_scenes += 1
            words = 100 if self.e_scenes <= self.short_scenes else 1000
            return completion(story_text(words), completion_tokens=self.e_scene_tokens)
        if "Now write scene" in message:
            return reply_with_story(400)
        return fake_d(body)


def e_config(fake: FakeOpenAI, conditions: list[dict[str, Any]], **stories: Any) -> dict[str, Any]:
    return story_config(fake.base_url, conditions, briefs={"ids": ["brief-01"]}, **stories)


def story_folder(run_dir: Path, condition: str = "E") -> tuple[dict[str, Any], Path]:
    [story] = [row for row in stories(run_dir) if row["condition"] == condition]
    return story, run_dir / story["path"]


@pytest.mark.ml
def test_e_plan_uses_shared_prose_ledger_and_records_every_resource(
    fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint: Path
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = Provider()
    config = e_config(fake_llm, [e_condition(checkpoint)])
    first = run(tmp_path, config)
    story, folder = story_folder(first)
    assert story["status"] == "ok"
    assert story["words"] == 2000
    proposal = read_sealed(folder / "conditional/generation/proposal.json")
    completed = read_sealed(folder / "conditional/completion/completion.json")
    plan = load_plan(folder / "plan.json")
    assert (folder / "plan.json").read_bytes() == (
        folder / "conditional/completion/plan.json"
    ).read_bytes()
    assert plan.scenes[0].summary == FILLED
    assert [scene["id"] for scene in proposal["plan"]["scenes"]] == [s.id for s in plan.scenes]
    selection = json.loads((folder / "conditional-selection.json").read_text())
    assert selection == {
        "pipeline": "candidate_e",
        "source": "generated",
        "checkpoint_id": read_sealed(checkpoint)["id"],
        "proposal_id": proposal["id"],
        "completion_id": completed["id"],
        "plan_sha256": plan_identity(plan),
    }
    # The shared ledger and scene notes, replayed from the completed learned plan.
    ledger = json.loads((folder / "ledger.json").read_text())
    assert ledger["plan_sha256"] == plan_identity(plan)
    inputs = [
        json.loads((folder / f"attempts/1/scene-0{n}.inputs.json").read_text()) for n in (1, 2)
    ]
    assert [row["planned_state_before"] for row in inputs] == [
        scene["summary"] for scene in ledger["scenes"]
    ]
    calls = jsonl(first / "calls.jsonl")
    assert [(c["step"], c["template"]) for c in calls] == [
        ("conditional-completion", "conditional_text_completion"),
        ("scene-01", "scene_prose"),
        ("scene-02", "scene_prose"),
    ]
    prose_requests = [
        r.body for r in fake_llm.requests if "Now write scene" in last_user_message(r.body)
    ]
    assert all(
        "You are writing a short story from this brief" in last_user_message(b)
        for b in prose_requests
    )
    assert FILLED in last_user_message(prose_requests[0])
    # Local model work is recorded in its own units, never as a provider call.
    [work] = jsonl(first / "local-work.jsonl")
    assert work["status"] == "structural-proposal"
    assert work["proposal_id"] == proposal["id"]
    assert work["passes"] == 1
    assert work["attempts"][0]["pass_modes"] == ["unconditional"]
    assert work["tokens"] > 0
    assert work["attention_cells"] > 0
    assert work["test_only"] is True
    assert manifest(first)["files"]["local_work"] == "local-work.jsonl"
    provenance = json.loads((folder / "planning-provenance.json").read_text())
    assert provenance["status"] == "succeeded"
    assert provenance["scientific_status"].startswith("test-only")
    assert provenance["local_work"] == [work]
    assert "conditional/generation/proposal.json" in story["files"]
    report = (first / "report.md").read_text()
    for heading in (
        "## Resource accounting",
        "### Local model work",
        "### Selected conditional plans",
    ):
        assert heading in report
    assert f"`{story['sha256'][:12]}`" in report
    assert story["sha256"] == digest(selected_text(first, read_stories(first)[0]))
    # Saved conditional records still pass their own public inspections.
    assert main(["conditional-generate", "inspect", str(folder / "conditional/generation")]) == 0
    assert main(["conditional-complete", "inspect", str(folder / "conditional/completion")]) == 0
    # A replay is free on the provider side, but local model work is run, and counted, again.
    second = run(tmp_path, config)
    _, replayed = story_folder(second)
    assert fake_llm.call_count == 3
    assert manifest(second)["cost"]["usd"] == 0
    assert (replayed / "plan.json").read_bytes() == (folder / "plan.json").read_bytes()
    assert jsonl(second / "local-work.jsonl")[0]["tokens"] == work["tokens"]


@pytest.mark.ml
def test_e_length_attempts_reset_scene_and_ledger_state_and_keep_every_attempt(
    fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint: Path
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = Provider(short_scenes=2)
    result = run(tmp_path, e_config(fake_llm, [e_condition(checkpoint)]))
    story, folder = story_folder(result)
    assert [attempt["in_range"] for attempt in story["attempts"]] == [False, True]
    assert story["chosen_attempt"] == 1
    for attempt in (1, 2):
        assert (folder / f"attempts/{attempt}/scene-01.inputs.json").is_file()
        assert (folder / f"attempts/{attempt}/scene-02.md").is_file()
    first_scenes = [
        last_user_message(r.body)
        for r in fake_llm.requests
        if "Now write scene 1 of 2" in last_user_message(r.body)
    ]
    assert len(first_scenes) == 2
    assert all("This is the first scene." in message for message in first_scenes)
    assert (folder / "story.md").read_text() == selected_text(result, read_stories(result)[0])
    # The plan, and so the local model and the completion, is never regenerated.
    steps = [call["step"] for call in jsonl(result / "calls.jsonl")]
    assert steps.count("conditional-completion") == 1
    assert len(jsonl(result / "local-work.jsonl")) == 1


@pytest.mark.ml
def test_equal_budget_with_e_requires_a_registered_resource_policy(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    config = e_config(
        fake_llm, [BASELINE_D, e_condition(checkpoint)], equal_budget={"conditions": ["D", "E"]}
    )
    assert main(["run", str(write_config(tmp_path, config))]) == 2
    assert "resource_policy" in capsys.readouterr().err
    assert fake_llm.call_count == 0


@pytest.mark.ml
def test_e_budget_revision_uses_final_selected_prose_and_discloses_local_compute(
    fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint: Path
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = Provider(e_scene_tokens=500)
    config = e_config(
        fake_llm,
        [BASELINE_D, e_condition(checkpoint)],
        equal_budget={"conditions": ["D", "E"], "resource_policy": POLICY},
    )
    result = run(tmp_path, config)
    story, folder = story_folder(result)
    assert story["revision_passes"] == 1
    assert (folder / "story.md").read_bytes() == (folder / "revisions/1/story.md").read_bytes()
    assert (folder / "story.original.md").is_file()
    [match] = manifest(result)["stories"]["equal_budget"]["matches"]
    assert match["matched"] is True
    assert match["revised"] == "E"
    assert match["local_work"]["D"]["tokens"] == 0
    assert match["local_work"]["E"]["tokens"] > 0
    comparison = json.loads((result / "comparison.json").read_text())
    [pair] = comparison["pairs"]
    assert comparison["resource_policy"]["local_compute"] == "report-separately"
    assert pair["local_compute_equal"] is False
    assert pair["budget_basis"].startswith("provider generation tokens only")
    final = selected_text(result, read_stories(result)[1])
    assert pair["conditions"]["E"]["final_sha256"] == digest(final)
    assert pair["conditions"]["E"]["resources"]["local"]["passes"] == 1
    assert pair["conditions"]["E"]["resources"]["provider"]["revision"]["calls"] == 1
    assert pair["conditions"]["D"]["resources"]["provider"]["planning"]["calls"] == 7
    assert pair["conditions"]["E"]["resources"]["provider"]["completion"]["calls"] == 1
    report = (result / "report.md").read_text()
    assert "a match is not an equal total budget" in report
    assert "local compute equal: no" in report
    revision_calls = [c for c in jsonl(result / "calls.jsonl") if c["step"] == "revision-01"]
    assert [c["condition"] for c in revision_calls] == ["E"]
    packet = tmp_path / "packet"
    assert (
        main(
            [
                "realisation",
                "create",
                str(result),
                "--condition",
                "E",
                "--brief",
                "brief-01",
                "--out",
                str(packet),
            ]
        )
        == 0
    )
    saved = json.loads((packet / "packet.json").read_text())
    assert saved["story_sha256"] == digest(final)
    assert saved["alignment"] == "unresolved"


@pytest.mark.ml
@pytest.mark.parametrize("stage", ["generation", "completion"])
def test_e_failures_keep_failed_records_and_local_work_in_the_comparison(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint: Path,
    stage: str,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    provider = Provider()

    def failing(body: Json) -> Reply:
        if stage == "completion" and is_completion(body):
            return completion('{"fields": {"title": "only one field"}}')
        return provider(body)

    fake_llm.default = failing
    params: dict[str, Any] = {"max_attempts": 2}
    if stage == "generation":
        params["locks"] = {"causal_links/causal_links.0/effect_scene_id": "missing-scene"}
    config = e_config(
        fake_llm,
        [BASELINE_D, e_condition(checkpoint, **params)],
        equal_budget={"conditions": ["D", "E"], "resource_policy": POLICY},
    )
    config["structured"] = {"max_repairs": 1}
    result = run(tmp_path, config)
    story, folder = story_folder(result)
    assert story["status"] == "failed"
    assert f"conditional-{stage}" in story["error"]
    assert not (folder / "story.md").exists()
    assert story_folder(result, "D")[0]["status"] == "ok"
    [work] = jsonl(result / "local-work.jsonl")
    assert work["tokens"] == sum(attempt["tokens"] for attempt in work["attempts"]) > 0
    provenance = json.loads((folder / "planning-provenance.json").read_text())
    assert provenance["status"] == "failed"
    assert provenance["local_work"] == [work]
    completion_calls = [c for c in jsonl(result / "calls.jsonl") if c["condition"] == "E"]
    if stage == "generation":
        assert work["status"] == "attempt-limit"
        assert len(work["attempts"]) == 2
        assert completion_calls == []
    else:
        assert work["status"] == "structural-proposal"
        assert [c["attempt"] for c in completion_calls] == [0, 1]
        assert story["budget"]["calls"] == 2
        assert read_sealed(folder / "conditional/completion/completion.json")["status"] == (
            "repair-exhausted"
        )
    [match] = manifest(result)["stories"]["equal_budget"]["matches"]
    assert match["reason"] == "no story to match: E failed"
    [pair] = json.loads((result / "comparison.json").read_text())["pairs"]
    assert pair["conditions"]["E"]["status"] == "failed"
    assert "E: final prose unavailable or failed" in pair["problems"]
    assert "### Failed stories" in (result / "report.md").read_text()


@pytest.mark.ml
def test_budget_stop_during_completion_keeps_local_work_and_partial_records(
    fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint: Path
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = Provider()
    config = e_config(fake_llm, [e_condition(checkpoint)])
    config["budget"] = {"per_run_usd": 0}
    result = run(tmp_path, config, expect=1)
    story, folder = story_folder(result)
    assert manifest(result)["status"] == "aborted"
    assert story["status"] == "failed"
    assert fake_llm.call_count == 0
    assert jsonl(result / "local-work.jsonl")[0]["status"] == "structural-proposal"
    assert read_sealed(folder / "conditional/completion/completion.json")["status"] == "budget-stop"
    assert json.loads((folder / "planning-provenance.json").read_text())["status"] == "failed"


def import_params(folder: Path) -> dict[str, str]:
    return {
        "saved_plan": str(folder / "plan.json"),
        "provenance": str(folder / "planning-provenance.json"),
    }


@pytest.mark.ml
def test_e_import_attributes_original_local_work_and_calls_once(
    fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint: Path
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = Provider()
    source = run(tmp_path, e_config(fake_llm, [e_condition(checkpoint)]))
    original, folder = story_folder(source)
    [original_work] = jsonl(source / "local-work.jsonl")
    imported = run(tmp_path, e_config(fake_llm, [e_condition(checkpoint, **import_params(folder))]))
    story, target = story_folder(imported)
    assert story["status"] == "ok"
    assert story["budget"] == original["budget"]
    assert fake_llm.call_count == 3
    calls = jsonl(imported / "calls.jsonl")
    assert calls[0]["step"] == "imported-conditional-completion"
    assert calls[0]["imported_from"]
    assert calls[0]["cost_usd"] == 0
    [work] = jsonl(imported / "local-work.jsonl")
    assert work["step"] == "imported-conditional-generation"
    assert work["imported_from"] == calls[0]["imported_from"]
    for key in ("tokens", "attention_cells", "passes", "attempts", "checkpoint_id"):
        assert work[key] == original_work[key]
    assert json.loads((target / "conditional-selection.json").read_text())["source"] == "imported"
    assert (target / "plan.json").read_bytes() == (folder / "plan.json").read_bytes()
    assert "imported" in (imported / "report.md").read_text()


def tamper_provenance(raw: dict[str, Any], tamper: str) -> None:
    if tamper == "completion":
        record = raw["completion"]
        record["selected_fields"]["title"] = "A different title"
        record["id"] = identity(record)
    elif tamper == "local-work":
        raw["local_work"] = []
        raw["local_work_sha256"] = canonical_identity([])
    elif tamper == "calls":
        raw["calls"] = []
        raw["calls_sha256"] = canonical_identity([])
    elif tamper == "attempts":
        raw["generation"]["work"]["tokens"] -= 1


REJECTIONS = {
    "checkpoint": "different conditional checkpoint",
    "guidance": "settings differ from this run",
    "conditions": "settings differ from this run",
    "completion": "not the completed learned proposal",
    "local-work": "local model work is missing, partial or changed",
    "calls": "completion calls differ from its attributed calls",
    "attempts": "generation work differs from its attempts",
    "plan": "checksum differs from its provenance",
}


@pytest.mark.ml
@pytest.mark.parametrize(
    "tamper",
    [
        "checkpoint",
        "guidance",
        "conditions",
        "completion",
        "local-work",
        "calls",
        "attempts",
        "plan",
    ],
)
def test_e_import_rejects_incompatible_or_incomplete_provenance_before_prose(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint: Path,
    tamper: str,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = Provider()
    source = run(tmp_path, e_config(fake_llm, [e_condition(checkpoint)]))
    _, folder = story_folder(source)
    params: dict[str, Any] = import_params(folder)
    if tamper == "checkpoint":
        changed = read_sealed(checkpoint)
        changed["binding"]["config"]["inference_max_seconds"] = 61
        other = tmp_path / "other-model.json"
        create_record(other, {k: v for k, v in changed.items() if k not in {"id", "version"}})
        params["checkpoint"] = str(other)
    elif tamper == "guidance":
        params["guidance_scale"] = 1
    elif tamper == "conditions":
        params["condition_fields"] = ["genre"]
    elif tamper == "plan":
        (folder / "plan.json").write_text((folder / "plan.json").read_text() + " ")
    else:
        provenance = folder / "planning-provenance.json"
        raw = json.loads(provenance.read_text())
        tamper_provenance(raw, tamper)
        provenance.write_text(json.dumps(raw))
    condition = e_condition(checkpoint)
    condition["params"].update(params)
    result = run(tmp_path, e_config(fake_llm, [condition]))
    story, _ = story_folder(result)
    assert story["status"] == "failed"
    assert "saved-E-plan" in story["error"]
    assert REJECTIONS[tamper] in story["error"]
    assert fake_llm.call_count == 3
    assert jsonl(result / "local-work.jsonl") == []


@pytest.mark.ml
def test_relabelled_fixture_checkpoint_cannot_run_e_without_g3(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    relabelled = read_sealed(checkpoint)
    relabelled["binding"]["config"]["test_only"] = False
    path = tmp_path / "relabelled.json"
    create_record(path, {k: v for k, v in relabelled.items() if k not in {"id", "version"}})
    config = e_config(fake_llm, [e_condition(path)])
    assert main(["run", str(write_config(tmp_path, config))]) == 2
    assert "genuine passed G3" in capsys.readouterr().err
    assert fake_llm.call_count == 0


def test_e_rejects_missing_checkpoints_and_lock_authority_before_any_call(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    missing = e_config(fake_llm, [e_condition(tmp_path / "missing.json")])
    assert main(["run", str(write_config(tmp_path, missing))]) == 2
    assert "checkpoint cannot serve E" in capsys.readouterr().err
    forged = e_config(fake_llm, [e_condition(tmp_path / "missing.json", locks={"title": "x"})])
    assert main(["run", str(write_config(tmp_path, forged, "forged.yaml"))]) == 2
    assert "locks must select" in capsys.readouterr().err
    assert fake_llm.call_count == 0


@pytest.mark.parametrize(
    ("step", "category"),
    [
        ("conditional-completion", "completion"),
        ("imported-conditional-completion", "completion"),
        ("scene-03", "prose"),
        ("story", "prose"),
        ("revision-01", "revision"),
        ("revision-critic-1", "planning"),
        ("imported-channels-scene-0", "planning"),
        ("outline", "planning"),
    ],
)
def test_calls_are_accounted_by_what_they_served(step: str, category: str) -> None:
    assert call_category(step) == category


def test_c_and_d_configs_and_run_files_keep_their_released_form(
    fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Hashes of released configs on main before E existed: registrations bind them.
    assert {
        name: load_config(REPO / "configs" / name).hash
        for name in ("stories-bc-equal-budget.yaml", "stories-ab-equal-budget.yaml")
    } == {
        "stories-bc-equal-budget.yaml": (
            # pragma: allowlist nextline secret - a public config content hash
            "eba016f04b345e4850ae97ff0b3df27f4fa34f9a560e1fc7ea0eb73f554f32d8"
        ),
        "stories-ab-equal-budget.yaml": (
            # pragma: allowlist nextline secret - a public config content hash
            "9992188da56dd19fe9b74fe10a3696caef4b50eb8c7eef4e30a29a86993126e5"
        ),
    }
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = Provider()
    result = run(
        tmp_path,
        story_config(
            fake_llm.base_url,
            [BASELINE_D, {"name": "D2", "pipeline": "baseline_d"}],
            briefs={"ids": ["brief-01"]},
            equal_budget={"conditions": ["D", "D2"]},
        ),
    )
    saved = (result / "manifest.json").read_text()
    assert "resource_policy" not in saved
    assert "local_work" not in saved
    assert not (result / "local-work.jsonl").exists()
    assert "budget_basis" not in (result / "comparison.json").read_text()
    assert "Resource accounting" not in (result / "report.md").read_text()


@pytest.mark.ml
def test_e_rejects_repair_limits_its_completion_cannot_honour_before_any_work(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    config = e_config(fake_llm, [e_condition(checkpoint)])
    config["structured"] = {"max_repairs": 9}
    assert main(["run", str(write_config(tmp_path, config))]) == 2
    assert "max_repairs" in capsys.readouterr().err
    assert fake_llm.call_count == 0
    assert not list(tmp_path.rglob("local-work.jsonl"))


@pytest.mark.ml
def test_unexpected_local_model_failure_is_a_recorded_failed_e_story(
    fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint: Path
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = Provider()

    def out_of_memory(*_: object, **__: object) -> dict[str, Any]:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("nde.conditional.generation.sample_candidate", out_of_memory)
    config = e_config(
        fake_llm,
        [BASELINE_D, e_condition(checkpoint)],
        equal_budget={"conditions": ["D", "E"], "resource_policy": POLICY},
    )
    result = run(tmp_path, config)
    story, folder = story_folder(result)
    assert story["status"] == "failed"
    assert "conditional-generation" in story["error"]
    assert "RuntimeError: CUDA out of memory" in story["error"]
    assert story_folder(result, "D")[0]["status"] == "ok"
    [work] = jsonl(result / "local-work.jsonl")
    assert work["status"] == "interrupted"
    assert work["tokens"] == work["attempts"][0]["tokens"] > 0
    provenance = json.loads((folder / "planning-provenance.json").read_text())
    assert provenance["status"] == "failed"
    assert provenance["local_work"] == [work]
    # The unexpected failure keeps its traceback beside the story, never silently dropped.
    assert "RuntimeError: CUDA out of memory" in (folder / "planning-error.txt").read_text()
    assert "planning-error.txt" in story["files"]
    [match] = manifest(result)["stories"]["equal_budget"]["matches"]
    assert match["reason"] == "no story to match: E failed"


@pytest.mark.ml
def test_interrupted_e_revision_still_discloses_local_work_in_the_match(
    fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint: Path
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    provider = Provider(e_scene_tokens=500)

    def refusing(body: Json) -> Reply:
        if "revising your own short story" in str(body["messages"][0]["content"]):
            return error(400, "revision refused")
        return provider(body)

    fake_llm.default = refusing
    config = e_config(
        fake_llm,
        [BASELINE_D, e_condition(checkpoint)],
        equal_budget={"conditions": ["D", "E"], "resource_policy": POLICY},
    )
    result = run(tmp_path, config, expect=1)
    [match] = manifest(result)["stories"]["equal_budget"]["matches"]
    assert match["reason"].startswith("revision interrupted")
    assert match["local_work"]["D"]["tokens"] == 0
    assert match["local_work"]["E"]["tokens"] == jsonl(result / "local-work.jsonl")[0]["tokens"]
    report = (result / "report.md").read_text()
    [line] = [row for row in report.splitlines() if "local work:" in row]
    assert "E 1 passes" in line
    # A match saved before local work was kept on this path says so, not an empty line.
    saved = manifest(result)
    del saved["stories"]["equal_budget"]["matches"][0]["local_work"]
    (result / "manifest.json").write_text(json.dumps(saved, indent=2))
    assert "#1 local work: not recorded" in render_report(result)
