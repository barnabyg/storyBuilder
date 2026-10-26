import json
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from tests.conftest import REPO


def _fixture() -> dict[str, Any]:
    return json.loads(
        (REPO / "tests/fixtures/narrative-structure.json").read_text(encoding="utf-8")
    )


def test_structural_id_collision_is_reported_once(tmp_path: Path) -> None:
    data = _fixture()
    data["causal_links"][0]["id"] = "s1"
    code, _, _, result = _inspect(tmp_path, data)
    assert code == 1
    assert [d["code"] for d in result["diagnostics"]].count("duplicate_id") == 1


def _inspect(tmp_path: Path, data: dict[str, Any]) -> tuple[int, str, str, dict[str, Any]]:
    file = tmp_path / "input.json"
    file.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "cards"
    code = main(["plan", "inspect", str(file), "--out", str(out)])
    return (
        code,
        (out / "cards.md").read_text(encoding="utf-8"),
        (out / "cards.html").read_text(encoding="utf-8"),
        json.loads((out / "validation.json").read_text(encoding="utf-8")),
    )


def test_saved_structural_plan_traces_causes_payoffs_and_multiple_holders(tmp_path: Path) -> None:
    code, markdown, html, result = _inspect(tmp_path, _fixture())
    assert code == 0
    assert result["diagnostics"] == []
    for rendered in (markdown, html):
        assert "cause1" in rendered
        assert "s1" in rendered
        assert "s3" in rendered
        assert "setup1" in rendered
        assert "payoff1" in rendered
        assert "fact2" in rendered
        assert "reveal1" in rendered
        assert "c1, c2" in rendered
        assert "use2" in rendered
    saved = tmp_path / "cards/plan.json"
    assert main(["plan", "validate", str(saved)]) == 0
    round_trip = json.loads(saved.read_text(encoding="utf-8"))
    assert round_trip["reveal_links"][0]["holder_ids"] == ["c1", "c2"]
    assert round_trip["causal_links"][0]["kind"] == "causal"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("dangling-cause", "causal_endpoint"),
        ("self-cause", "causal_self_link"),
        ("reverse-cause", "causal_order"),
        ("cycle", "causal_cycle"),
        ("reordered-scenes", "causal_order"),
        ("duplicate-link-id", "duplicate_id"),
    ],
)
def test_causal_mutations_have_the_intended_diagnostic(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    data = _fixture()
    link = data["causal_links"][0]
    if mutation == "dangling-cause":
        link["cause_scene_id"] = "missing"
    elif mutation == "self-cause":
        link["cause_scene_id"] = "s3"
    elif mutation == "reverse-cause":
        link["cause_scene_id"], link["effect_scene_id"] = "s3", "s1"
    elif mutation == "cycle":
        data["causal_links"].append(
            {"id": "cause2", "cause_scene_id": "s3", "effect_scene_id": "s1"}
        )
    elif mutation == "duplicate-link-id":
        link["id"] = "s1"
    else:
        data["scenes"].reverse()
    code, markdown, html, result = _inspect(tmp_path, data)
    assert code == 1
    assert expected in {item["code"] for item in result["diagnostics"]}
    assert expected in markdown
    assert expected in html


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("dangling-setup-scene", "setup_scene_reference"),
        ("dangling-setup", "payoff_setup_reference"),
        ("dangling-payoff-scene", "payoff_scene_reference"),
        ("same-scene-payoff", "payoff_order"),
        ("reverse-payoff", "payoff_order"),
    ],
)
def test_setup_mutations_have_the_intended_diagnostic(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    data = _fixture()
    if mutation == "dangling-setup-scene":
        data["setups"][0]["scene_id"] = "missing"
    elif mutation == "dangling-setup":
        data["payoff_links"][0]["setup_id"] = "missing"
    elif mutation == "dangling-payoff-scene":
        data["payoff_links"][0]["payoff_scene_id"] = "missing"
    elif mutation == "same-scene-payoff":
        data["payoff_links"][0]["payoff_scene_id"] = "s1"
    else:
        data["setups"][0]["scene_id"] = "s3"
        data["payoff_links"][0]["payoff_scene_id"] = "s2"
    code, markdown, html, result = _inspect(tmp_path, data)
    assert code == 1
    assert expected in {item["code"] for item in result["diagnostics"]}
    assert expected in markdown
    assert expected in html


def test_unpaid_setup_is_visible_as_advisory_without_failing_valid_control(tmp_path: Path) -> None:
    data = _fixture()
    data["payoff_links"] = []
    code, markdown, html, result = _inspect(tmp_path, data)
    assert code == 0
    assert result["diagnostics"] == [
        {
            "code": "unpaid_setup",
            "location": "setup1",
            "message": "setup setup1 at s1 has no payoff",
            "severity": "advisory",
        }
    ]
    assert "advisory unpaid_setup" in markdown
    assert "advisory unpaid_setup" in html


@pytest.mark.parametrize(
    ("channel", "field", "value", "expected"),
    [
        ("knowledge_facts", "initial_holder_ids", ["ghost"], "knowledge_holder"),
        ("knowledge_facts", "initial_holder_ids", ["c1", "c1"], "duplicate_fact_reference"),
        ("reveal_links", "scene_id", "missing", "knowledge_scene"),
        ("reveal_links", "fact_id", "missing", "knowledge_fact"),
        ("reveal_links", "holder_ids", ["ghost"], "knowledge_holder"),
        ("reveal_links", "holder_ids", ["c2", "c2"], "duplicate_fact_reference"),
        ("knowledge_uses", "fact_id", "missing", "knowledge_fact"),
        ("knowledge_uses", "scene_id", "missing", "knowledge_scene"),
        ("knowledge_uses", "holder_id", "ghost", "knowledge_holder"),
        ("knowledge_uses", "fact_id", "fact2", "knowledge_before_acquisition"),
    ],
)
def test_knowledge_mutations_have_the_intended_diagnostic(
    tmp_path: Path,
    channel: str,
    field: str,
    value: Any,
    expected: str,
) -> None:
    data = _fixture()
    data[channel][0][field] = value
    code, markdown, html, result = _inspect(tmp_path, data)
    assert code == 1
    assert expected in {item["code"] for item in result["diagnostics"]}
    assert expected in markdown
    assert expected in html


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("duplicate-fact-id", "duplicate_fact_id"),
        ("duplicate-acquisition", "duplicate_acquisition"),
        ("duplicate-use", "duplicate_knowledge_use"),
        ("absent-holder", "knowledge_presence"),
    ],
)
def test_knowledge_identity_and_presence_mutations(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    data = _fixture()
    if mutation == "duplicate-fact-id":
        data["knowledge_facts"][1]["id"] = "fact1"
    elif mutation == "duplicate-acquisition":
        data["reveal_links"].append(
            {"id": "repeat-reveal", "fact_id": "fact2", "scene_id": "s3", "holder_ids": ["c2"]}
        )
    elif mutation == "duplicate-use":
        data["knowledge_uses"].append(
            {"id": "repeat-use", "fact_id": "fact1", "scene_id": "s1", "holder_id": "c1"}
        )
    else:
        data["scenes"][1]["character_ids"] = ["c1"]
    code, _, _, result = _inspect(tmp_path, data)
    assert code == 1
    assert expected in {item["code"] for item in result["diagnostics"]}


def test_knowledge_can_be_acquired_at_first_scene_and_used_at_last_scene(tmp_path: Path) -> None:
    data = _fixture()
    data["reveal_links"][0]["scene_id"] = "s1"
    data["knowledge_uses"][0]["fact_id"] = "fact2"
    code, _, _, result = _inspect(tmp_path, data)
    assert code == 0
    assert result["diagnostics"] == []


def test_reveal_at_last_scene_can_be_used_in_that_scene(tmp_path: Path) -> None:
    data = _fixture()
    data["reveal_links"][0]["scene_id"] = "s3"
    code, _, _, result = _inspect(tmp_path, data)
    assert code == 0
    assert result["diagnostics"] == []
