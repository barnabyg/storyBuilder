import argparse
import json
from pathlib import Path

import pytest

from nde.conditional import layout
from nde.corpus.records_io import identity, read_sealed
from nde.narrative.models import NarrativePlan
from nde.reconstruction.codec import encode_plan, tokenize
from tests.conditional_support import tiny_conditional_dataset
from tests.fake_openai import FakeOpenAI


def command(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    layout.configure(parser)
    return layout.run(parser.parse_args(arguments))


def test_saved_layout_encoding_rejects_stale_approved_training_inputs(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    dataset = tiny_conditional_dataset(tmp_path / "data", fake_llm)
    capacity = tmp_path / "capacity.json"
    capacity.write_text(json.dumps({"max_tokens": 2048}))
    out = tmp_path / "layouts"
    assert command(["encode", str(dataset), str(capacity), "--out", str(out)]) == 0
    assert command(["inspect", str(out / "layouts.json")]) == 0
    saved = read_sealed(out / "layouts.json")
    saved["examples"][0]["layout"]["counts"]["scenes"] = 99
    saved["id"] = identity(saved)
    attack = tmp_path / "resealed-layout.json"
    attack.write_text(json.dumps(saved))
    assert command(["inspect", str(attack)]) == 2
    capacity.write_text(json.dumps({"max_tokens": 4096}))
    assert command(["inspect", str(out / "layouts.json")]) == 2


@pytest.mark.parametrize("scenes", [1, 2, 4])
def test_layout_roundtrip_has_learnable_counts_identities_and_no_original_ids(scenes: int) -> None:
    plan = NarrativePlan.model_validate(
        {
            "id": "original",
            "title": "Original",
            "characters": [{"id": "original-person", "name": "Private name"}],
            "scenes": [
                {
                    "id": f"original-{number}",
                    "title": "Private title",
                    "character_ids": ["original-person"],
                }
                for number in range(scenes)
            ],
        }
    )
    capacity = layout.Capacity(max_scenes=4, max_tokens=2048)
    target = layout.encode_target(encode_plan(plan, {"plot": "hard"}, {}), capacity)
    view = layout.decode_layout(target["tokens"], capacity)
    assert target["counts"]["scenes"] == scenes
    assert view["characters"] == [{"id": "characters.0"}]
    assert len(view["scenes"]) == scenes
    assert "Private" not in json.dumps(view)
    assert "original" not in json.dumps(view)
    assert layout.materialize(view).scenes[-1].id == f"scenes.{scenes - 1}"


def test_layout_rejects_invalid_references_unterminated_output_and_capacity() -> None:
    capacity = layout.Capacity(max_scenes=1, max_tokens=512)
    with pytest.raises(ValueError, match="character_reference"):
        layout.decode_layout(
            tokenize({"scenes": [{"id": "scenes.0", "character_ids": ["missing"]}]}), capacity
        )
    with pytest.raises(ValueError, match="BOS/EOS"):
        layout.decode_layout([2, 127], capacity)
    with pytest.raises(ValueError, match="scene capacity"):
        layout.decode_layout(tokenize({"scenes": [{"id": "a"}, {"id": "b"}]}), capacity)
    with pytest.raises(ValueError, match="duplicate_id"):
        layout.decode_layout(
            tokenize({"scenes": [{"id": "a"}], "characters": [{"id": "a"}]}), capacity
        )
    with pytest.raises(ValueError, match="source/evaluator"):
        layout.decode_layout(tokenize({"sources": [], "scenes": [{"id": "a"}]}), capacity)


def test_context_scaffolding_cannot_be_promoted_to_unapproved_layout_targets() -> None:
    plan = NarrativePlan.model_validate(
        {"id": "p", "title": "P", "scenes": [{"id": "a", "title": "A"}]}
    )
    target = encode_plan(plan, {"plot": "excluded"}, {})
    with pytest.raises(ValueError, match="approved plot/count"):
        layout.encode_target(target, layout.Capacity())


def test_conditional_layout_loss_preserves_soft_and_unknown_target_policies() -> None:
    plan = NarrativePlan.model_validate(
        {
            "id": "p",
            "title": "P",
            "characters": [{"id": "c", "name": "C"}],
            "scenes": [{"id": "a", "title": "A", "character_ids": ["c"]}],
            "states": [
                {
                    "id": "state",
                    "scene_id": "a",
                    "character_id": "c",
                    "kind": "goal",
                    "key": "lamp",
                    "before": {"status": "ambiguous", "value": "maybe", "confidence": 0.5},
                    "after": {"value": "light"},
                }
            ],
        }
    )
    target = layout.encode_target(
        encode_plan(plan, {"plot": "hard", "goal": "soft"}, {}), layout.Capacity(max_tokens=2048)
    )
    assert len(target["loss_weights"]) == len(target["tokens"])
    assert set(target["loss_weights"]) == {0.0, 0.25, 1.0}
