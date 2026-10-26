import argparse
import json
from pathlib import Path

import pytest

from nde.narrative.models import NarrativePlan
from nde.reconstruction import codec
from tests.fake_openai import FakeOpenAI
from tests.g2_support import approved_export


def command(*arguments: str) -> int:
    parser = argparse.ArgumentParser()
    codec.configure(parser)
    return codec.run(parser.parse_args(arguments))


def test_approved_export_roundtrips_only_registered_retained_content(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    source = approved_export(tmp_path, fake_llm)
    out = tmp_path / "encoded"
    assert command("encode", str(source), "--out", str(out)) == 0
    assert command("inspect", str(out)) == 0
    saved = codec.load_encoding(out)
    record = saved["records"][0]
    assert record["view"]["scenes"][0]["id"] == "scenes.0"
    assert record["view"]["causal_links"][0]["cause_scene_id"] == "scenes.0"
    assert record["view"]["states"][0]["before"]["status"] == "unknown"
    assert "interpretation" not in {slot["channel"] for slot in record["slots"]}
    model_text = json.dumps(record["view"])
    assert "BEGIN LAMP" not in model_text
    assert "source_sha256" not in model_text
    decoded = tmp_path / "decoded.json"
    assert command("decode", str(out), "--record", "0", "--out", str(decoded)) == 0
    assert json.loads(decoded.read_text())["retained_view"] == record["view"]
    assert codec.decode_record(record) == record["view"]
    assert len(record["tokens"]) > 0
    assert saved["specification"]["vocabulary_size"] == 260
    Path(record["provenance"]["annotation_path"]).write_text("{}")
    assert command("inspect", str(out)) == 2


def test_canonical_views_preserve_unknowns_and_reject_capacity_without_truncation() -> None:
    plan = NarrativePlan.model_validate(
        {
            "id": "a",
            "title": "source title",
            "characters": [{"id": "person", "name": "Reader"}],
            "scenes": [
                {"id": "first", "title": "Opening", "character_ids": ["person"]},
                {"id": "last", "title": "Ending"},
            ],
            "states": [
                {
                    "id": "goal",
                    "kind": "goal",
                    "scene_id": "first",
                    "character_id": "person",
                    "key": "x",
                    "before": {"status": "unknown"},
                    "after": {"value": "wait", "confidence": 0.5},
                }
            ],
            "causal_links": [{"id": "edge", "cause_scene_id": "first", "effect_scene_id": "last"}],
        }
    )
    policy = {"plot": "hard", "goal": "soft", "causality": "hard"}
    encoded = codec.encode_plan(plan, policy, {}, ineligible_channels=("causality",))
    assert (
        codec.encode_plan(
            plan, dict(reversed(list(policy.items()))), {}, ineligible_channels=("causality",)
        )
        == encoded
    )
    assert codec.decode_record(encoded)["states"][0]["before"]["status"] == "unknown"
    assert all(
        not slot["eligible"]
        for slot in encoded["slots"]
        if slot["channel"] == "causality" or "before" in slot["path"]
    )
    assert any(slot["eligible"] and slot["policy"] == "soft" for slot in encoded["slots"])
    with pytest.raises(ValueError, match="scene capacity"):
        codec.encode_plan(plan, policy, {}, max_scenes=1)
    with pytest.raises(ValueError, match="token capacity"):
        codec.encode_plan(plan, policy, {}, max_tokens=len(encoded["tokens"]) - 1)
    assert codec.encode_plan(plan, policy, {}, max_tokens=len(encoded["tokens"]))["tokens"]
    bad = plan.model_copy(
        update={
            "causal_links": [plan.causal_links[0].model_copy(update={"effect_scene_id": "missing"})]
        }
    )
    with pytest.raises(ValueError, match="invalid source"):
        codec.encode_plan(bad, policy, {})
