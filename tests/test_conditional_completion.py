import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from nde.conditional import completion as workflow
from nde.corpus.records_io import create_record, read_sealed
from nde.narrative.models import NarrativePlan
from nde.narrative.workflow import load_plan
from tests.fake_openai import FakeOpenAI, completion, error


def proposal(directory: Path, **changes: Any) -> Path:
    plan = NarrativePlan.model_validate(
        {
            "id": "generated-plan",
            "title": "Unfilled generated text",
            "characters": [{"id": "character-1", "name": "Unfilled generated text"}],
            "scenes": [
                {
                    "id": "scene-1",
                    "title": "Unfilled generated text",
                    "character_ids": ["character-1"],
                },
                {"id": "scene-2", "title": "The promise", "summary": "A promise is kept."},
            ],
            "causal_links": [
                {"id": "cause-1", "cause_scene_id": "scene-1", "effect_scene_id": "scene-2"}
            ],
            "subtext": [
                {
                    "id": "subtext-1",
                    "scene_ids": ["scene-1"],
                    "reading": {"value": "Unfilled generated text", "status": "known"},
                }
            ],
        }
    )
    fields = [
        "title",
        "characters/character-1/name",
        "scenes/scene-1/title",
        "scenes/scene-1/summary",
        "subtext/subtext-1/reading/value",
    ]
    create_record(
        directory / "proposal.json",
        {
            "kind": "conditional-structural-proposal",
            "status": "structural-proposal",
            "plan": plan.model_dump(mode="json"),
            "conditions": {"genre": "mystery"},
            "locks": {"scenes/scene-2/title": "The promise"},
            "completion_fields": fields,
            "test_only": True,
            "checkpoint_id": "synthetic-checkpoint",
            "inputs": {},
            **changes,
        },
    )
    return directory / "proposal.json"


def fields() -> dict[str, str]:
    return {
        "title": "The hidden letter",
        "characters/character-1/name": "Mara",
        "scenes/scene-1/title": "A discovery",
        "scenes/scene-1/summary": "Mara finds a letter beneath the door.",
        "subtext/subtext-1/reading/value": "The letter conceals an old debt.",
    }


def configuration(directory: Path, fake: FakeOpenAI, source: Path, **changes: Any) -> Path:
    config = directory / "complete.json"
    config.write_text(
        json.dumps(
            {
                "proposal": str(source),
                "fields": list(fields()),
                "provider": {"base_url": fake.base_url, "retry": {"max_retries": 0}},
                "model": {"provider": "fake", "model": "completion-model"},
                "price": {"input_per_mtok": 1, "output_per_mtok": 2},
                "max_repairs": 1,
                **changes,
            }
        ),
        encoding="utf-8",
    )
    return config


def command(config: Path, out: Path) -> int:
    parser = argparse.ArgumentParser()
    workflow.configure(parser)
    return workflow.run(parser.parse_args(["run", str(config), "--out", str(out)]))


def test_ambiguous_subtext_and_alternative_text_complete_without_changing_uncertainty(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    source = proposal(tmp_path / "initial")
    saved = read_sealed(source)
    saved["plan"]["subtext"][0]["reading"] = {
        "status": "ambiguous",
        "value": "Unfilled generated text",
        "confidence": 0.4,
        "alternatives": ["Unfilled generated text", "The letter is an ordinary invitation."],
    }
    alternative = "subtext/subtext-1/reading/alternatives/0"
    saved["completion_fields"].append(alternative)
    saved["locks"]["subtext/subtext-1/reading/status"] = "ambiguous"
    source = tmp_path / "ambiguous.json"
    create_record(source, saved)
    selected = {**fields(), alternative: "The letter offers reconciliation."}
    fake_llm.queue(completion(json.dumps({"fields": selected})))
    config = configuration(tmp_path, fake_llm, source, fields=list(selected))
    out = tmp_path / "completed"
    assert command(config, out) == 0
    reading = load_plan(out / "plan.json").subtext[0].reading
    assert reading.status == "ambiguous"
    assert reading.confidence == 0.4
    assert reading.value == "The letter conceals an old debt."
    assert reading.alternatives == [
        "The letter offers reconciliation.",
        "The letter is an ordinary invitation.",
    ]


def test_public_completion_publishes_cards_and_preserves_learned_structure(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    source = proposal(tmp_path)
    original = source.read_bytes()
    fake_llm.queue(completion(json.dumps({"fields": fields()})))
    config = configuration(tmp_path, fake_llm, source)
    out = tmp_path / "result"
    assert command(config, out) == 0
    result = read_sealed(out / "completion.json")
    plan = load_plan(out / "plan.json")
    assert result["status"] == "complete"
    assert plan.characters[0].name == "Mara"
    assert plan.scenes[1].title == "The promise"
    assert plan.causal_links[0].effect_scene_id == "scene-2"
    assert plan.subtext[0].reading.status == "known"
    assert plan.sources == []
    assert source.read_bytes() == original == (out / "source-proposal.json").read_bytes()
    assert "Mara finds a letter" in (out / "cards.md").read_text()
    assert "Mara finds a letter" in (out / "cards.html").read_text()
    assert result["tokens"] == 15
    assert result["cost_usd"] == pytest.approx(0.00002)
    assert result["calls"][0]["cached"] is False
    assert result["selected_fields"] == fields()
    assert (out / "response-01.txt").is_file()


@pytest.mark.parametrize("attack", ["graph", "entity", "evidence", "lock", "approval", "empty"])
def test_unauthorized_or_invalid_text_responses_are_rejected_without_publishing(
    tmp_path: Path, fake_llm: FakeOpenAI, attack: str
) -> None:
    source = proposal(tmp_path)
    original = source.read_bytes()
    response: dict[str, Any] = {"fields": fields()}
    if attack == "approval":
        response["owner_approved"] = True
    elif attack == "empty":
        response["fields"]["title"] = " "
    else:
        selector = {
            "graph": "causal_links/cause-1/effect_scene_id",
            "entity": "characters/new-character/name",
            "evidence": "scenes/scene-1/evidence",
            "lock": "scenes/scene-2/title",
        }[attack]
        response["fields"][selector] = "invented"
    fake_llm.default = lambda _body: completion(json.dumps(response))
    config = configuration(tmp_path, fake_llm, source)
    out = tmp_path / "rejected"
    assert command(config, out) == 1
    result = read_sealed(out / "completion.json")
    assert result["status"] == "repair-exhausted"
    assert len(result["attempts"]) == 2
    assert all(row["status"] == "rejected" for row in result["attempts"])
    assert result["tokens"] == 30
    assert result["cost_usd"] == pytest.approx(0.00004)
    assert source.read_bytes() == original
    assert not (out / "plan.json").exists()
    assert load_plan(out / "last-valid/plan.json").title == "Unfilled generated text"
    assert (out / "response-01.txt").is_file()
    assert (out / "response-02.txt").is_file()


def test_repair_uses_exact_failures_and_replaces_only_requested_text(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    source = proposal(tmp_path)
    fake_llm.queue(completion('{"fields": {"title": "incomplete"}}'))
    fake_llm.queue(completion(json.dumps({"fields": fields()})))
    out = tmp_path / "repaired"
    assert command(configuration(tmp_path, fake_llm, source), out) == 0
    result = read_sealed(out / "completion.json")
    assert [row["status"] for row in result["attempts"]] == ["rejected", "accepted"]
    feedback = fake_llm.requests[1].body["messages"][-1]["content"]
    assert "response fields must exactly match authorized selectors" in feedback
    assert result["calls"][0]["prompt_id"] != result["calls"][1]["prompt_id"]


def test_partial_completion_remains_unpublished(tmp_path: Path, fake_llm: FakeOpenAI) -> None:
    source = proposal(tmp_path)
    fake_llm.queue(completion(json.dumps({"fields": {"title": "A new title"}})))
    out = tmp_path / "partial"
    config = configuration(tmp_path, fake_llm, source, fields=["title"])
    assert command(config, out) == 0
    assert read_sealed(out / "completion.json")["status"] == "partial"
    assert not (out / "plan.json").exists()
    assert load_plan(out / "last-valid/plan.json").title == "A new title"


@pytest.mark.parametrize("failure", ["provider", "budget", "actual-budget", "truncated"])
def test_failure_and_budget_stops_keep_last_valid_plan(
    tmp_path: Path, fake_llm: FakeOpenAI, failure: str
) -> None:
    source = proposal(tmp_path)
    changes: dict[str, Any] = {"max_repairs": 0}
    if failure == "provider":
        fake_llm.queue(error(401, "fixture authentication failure"))
    elif failure == "budget":
        changes["max_cost_usd"] = 0
    elif failure == "actual-budget":
        changes["max_cost_usd"] = 0.01
        fake_llm.queue(completion(json.dumps({"fields": fields()}), prompt_tokens=100000))
    else:
        fake_llm.queue(completion(json.dumps({"fields": fields()}), finish_reason="length"))
    out = tmp_path / "failed"
    assert command(configuration(tmp_path, fake_llm, source, **changes), out) == 1
    result = read_sealed(out / "completion.json")
    assert (
        result["status"]
        == {
            "provider": "provider-failed",
            "budget": "budget-stop",
            "actual-budget": "budget-stop",
            "truncated": "repair-exhausted",
        }[failure]
    )
    assert not (out / "plan.json").exists()
    assert load_plan(out / "last-valid/plan.json").title == "Unfilled generated text"
    if failure == "actual-budget":
        assert result["cost_usd"] == pytest.approx(0.10001)


def test_cached_completion_records_tokens_with_zero_additional_spend(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    source = proposal(tmp_path)
    fake_llm.queue(completion(json.dumps({"fields": fields()})))
    config = configuration(tmp_path, fake_llm, source, cache=str(tmp_path / "shared-cache.sqlite"))
    assert command(config, tmp_path / "first") == 0
    assert command(config, tmp_path / "second") == 0
    result = read_sealed(tmp_path / "second/completion.json")
    assert result["tokens"] == 15
    assert result["cost_usd"] == 0
    assert result["calls"][0]["cached"] is True
    assert fake_llm.call_count == 1


def test_malformed_saved_authority_fails_clearly_before_provider_call(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    source = proposal(tmp_path, locks=["scenes/scene-2/title"])
    assert command(configuration(tmp_path, fake_llm, source), tmp_path / "invalid") == 2
    assert fake_llm.call_count == 0


def test_interruption_keeps_source_raw_response_call_intent_and_last_valid_plan(
    tmp_path: Path, fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = proposal(tmp_path)
    original = source.read_bytes()
    fake_llm.queue(completion(json.dumps({"fields": fields()})))
    config = configuration(tmp_path, fake_llm, source)
    out = tmp_path / "interrupted"
    native_write = Path.write_text

    def interrupted_write(path: Path, text: str, *args: Any, **kwargs: Any) -> int:
        count = native_write(path, text, *args, **kwargs)
        if path.name == "response-01.txt":
            raise KeyboardInterrupt("fixture interruption at file publication boundary")
        return count

    monkeypatch.setattr(Path, "write_text", interrupted_write)
    with pytest.raises(KeyboardInterrupt):
        command(config, out)
    result = read_sealed(out / "completion.json")
    assert result["status"] == "interrupted"
    assert result["tokens"] == 15
    assert result["attempts"][0]["status"] == "pending"
    assert result["pending_call_usage"] is not None
    assert (out / "response-01.txt").is_file()
    assert (out / "attempt-01.json").is_file()
    assert load_plan(out / "last-valid/plan.json").title == "Unfilled generated text"
    assert source.read_bytes() == original
    assert not (out / "plan.json").exists()


def test_changed_source_during_completion_is_not_published(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    source = proposal(tmp_path)
    original = source.read_bytes()

    def concurrent_change(_body: dict[str, Any]) -> Any:
        source.write_text("{}")
        return completion(json.dumps({"fields": fields()}))

    fake_llm.default = concurrent_change
    out = tmp_path / "stale"
    assert command(configuration(tmp_path, fake_llm, source), out) == 2
    result = read_sealed(out / "completion.json")
    assert result["status"] == "failed"
    assert "changed" in result["diagnostics"][0]
    assert (out / "source-proposal.json").read_bytes() == original
    assert not (out / "plan.json").exists()


@pytest.mark.parametrize("invalid", ["unknown", "reference", "source-evidence", "manifest", "lock"])
def test_invalid_proposals_and_forged_completion_authority_never_reach_provider(
    tmp_path: Path, fake_llm: FakeOpenAI, invalid: str
) -> None:
    original = proposal(tmp_path / "original")
    record = read_sealed(original)
    plan = record["plan"]
    changes: dict[str, Any] = {"plan": plan}
    if invalid == "unknown":
        plan["subtext"][0]["reading"] = {"status": "unknown", "value": None}
    elif invalid == "reference":
        plan["causal_links"][0]["effect_scene_id"] = "missing-scene"
    elif invalid == "source-evidence":
        plan["scenes"][0]["evidence"] = [
            {"source_id": "invented", "source_sha256": "0" * 64, "start": 0, "end": 1}
        ]
    elif invalid == "manifest":
        changes["completion_fields"] = ["title"]
    else:
        changes["locks"] = {"scenes/scene-1/title": "Unfilled generated text"}
    source = proposal(tmp_path / "invalid", **changes)
    assert command(configuration(tmp_path, fake_llm, source), tmp_path / "rejected") == 2
    assert fake_llm.call_count == 0


def test_inspection_checks_generated_artifacts_and_reuse_cannot_overwrite_run(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    source = proposal(tmp_path)
    fake_llm.queue(completion(json.dumps({"fields": fields()})))
    config = configuration(tmp_path, fake_llm, source)
    out = tmp_path / "completed"
    assert command(config, out) == 0
    assert command(config, out) == 2
    assert fake_llm.call_count == 1
    parser = argparse.ArgumentParser()
    workflow.configure(parser)
    assert workflow.run(parser.parse_args(["inspect", str(out)])) == 0
    (out / "cards.md").write_text("changed report")
    assert workflow.run(parser.parse_args(["inspect", str(out)])) == 2
