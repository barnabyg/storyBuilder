"""Synthetic closed-batch fixtures; none of these judgements is owner evidence."""

import json
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from nde.eval.batch import Key, ShownPair, View
from nde.eval.readiness import GateCohort, cohort_checksum
from nde.eval.rubric import load_rubric
from nde.files import checksum_lf, write_text_lf
from tests.conftest import REPO
from tests.eval_helpers import reader
from tests.test_realisation import create_packet, packet_source


def synthetic_batch(
    tmp_path: Path,
    wins: int,
    *,
    ties: int = 0,
    originality: int = 3,
    coherence_only: bool = False,
) -> Path:
    batch = tmp_path / "batch"
    (batch / "sealed").mkdir(parents=True)
    rubric = load_rubric(REPO / "benchmark/rubric", 1)
    view = View(
        pairs=[
            ShownPair(a=f"Candidate story {i}\n", b=f"Comparator story {i}\n") for i in range(30)
        ],
        rubric=rubric,
        optional_scores=True,
        primary_dimension="coherence",
    )
    write_text_lf(batch / "view.json", view.model_dump_json(indent=2) + "\n")
    key_data: dict[str, Any] = {
        "created_at": "2026-10-26T18:00:00+00:00",
        "seed": 7,
        "sources": [
            {"run": str(tmp_path / "run"), "condition": name, "story": ""} for name in ("C", "B")
        ],
        "pairs": [
            {
                "brief": f"brief-{i:02d}",
                "repetition": 0,
                "a": {"run": str(tmp_path / "run"), "condition": "C", "story": f"C/{i}.md"},
                "b": {"run": str(tmp_path / "run"), "condition": "B", "story": f"B/{i}.md"},
            }
            for i in range(30)
        ],
        "rubric_version": 1,
    }
    key = Key.model_validate(key_data)
    write_text_lf(batch / "sealed/key.json", key.model_dump_json(indent=2) + "\n")
    client = reader(batch)
    for i in range(30):
        c_score = 4 if i < wins else 3 if i < wins + ties else 2
        response = client.post(
            f"/pairs/{i + 1}",
            data={
                "preferred": "A",
                "strength": "1",
                "score-a-coherence": str(c_score),
                "score-b-coherence": "3",
                **(
                    {}
                    if coherence_only
                    else {"score-a-originality": str(originality), "score-b-originality": "3"}
                ),
            },
        )
        assert response.status_code == 200
    assert client.post("/close").status_code == 200
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    evidence = tmp_path / "schema"
    evidence.mkdir()
    (evidence / "evidence.json").write_text("{}")
    cohort = {
        "id": "synthetic-test-only",
        "brief_checksums": {f"brief-{i:02d}": "a" * 64 for i in range(30)},
        "split_checksums": {"test": "b" * 64},
        "conditions": ["C", "B"],
        "repeat": 1,
        "length_tolerance": 0.1,
        "budget_measure": "generation_tokens",
        "budget_tolerance": 0.1,
    }
    protocol: dict[str, Any] = {
        "id": "synthetic-test-only",
        "test_only": True,
        "cohort": cohort,
        "tier": "L",
        "win_threshold": 21,
        "originality": {"analysis": "mean_paired_difference", "margin": 0.25},
        "reading_mode": "required_only",
        "config_sha256": "c" * 64,
        "selection_sha256": checksum_lf(selection.read_bytes()),
        "schema_evidence_sha256": checksum_lf((evidence / "evidence.json").read_bytes()),
        "generation_limits": {
            "max_repairs": 2,
            "max_length_attempts": 3,
            "max_revision_passes": 4,
            "per_run_usd": 1,
            "per_day_usd": 5,
        },
    }
    if coherence_only:
        del protocol["originality"]
        protocol["required_dimensions"] = ["coherence"]
    protocol_file = tmp_path / "protocol.json"
    protocol_file.write_text(json.dumps(protocol))
    admitted_pairs: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}
    for i in range(30):
        admitted: list[dict[str, Any]] = []
        for condition, text in (("C", view.pairs[i].a), ("B", view.pairs[i].b)):
            path = tmp_path / "run" / condition / f"{i}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            checksum = checksum_lf(path.read_bytes())
            artifacts[str(path)] = checksum
            admitted.append(
                {
                    "condition": condition,
                    "run": str(tmp_path / "run"),
                    "path": f"{condition}/{i}.md",
                    "sha256": checksum,
                    "words": 3,
                    "budget": {"calls": 1, "input_tokens": 10, "output_tokens": 10},
                    "new_spend_usd": 0,
                }
            )
        admitted_pairs.append(
            {"brief": f"brief-{i:02d}", "repetition": 0, "stories": admitted, "problems": []}
        )
    readiness = {
        "status": "ready",
        "cohort": cohort,
        "cohort_sha256": cohort_checksum(GateCohort.model_validate(cohort)),
        "benchmark": str(tmp_path),
        "sources": [],
        "pairs": admitted_pairs,
    }
    binding = {
        "protocol": protocol,
        "protocol_sha256": checksum_lf(protocol_file.read_bytes()),
        "protocol_path": str(protocol_file),
        "protocol_commit": "fixture-test-only",
        "selection_path": str(selection),
        "schema_evidence_path": str(evidence),
        "readiness": readiness,
        "view_sha256": checksum_lf((batch / "view.json").read_bytes()),
        "key_sha256": checksum_lf((batch / "sealed/key.json").read_bytes()),
        "artifact_checksums": artifacts,
    }
    (batch / "sealed/readiness").mkdir()
    (batch / "sealed/readiness/readiness.json").write_text(json.dumps(readiness))
    (batch / "sealed/g1.json").write_text(json.dumps(binding))
    return batch


@pytest.mark.parametrize(
    ("wins", "expected"), [(20, "failed_inconclusive"), (21, "passed"), (0, "failed_inconclusive")]
)
def test_registered_coherence_direction_and_threshold(
    tmp_path: Path, wins: int, expected: str
) -> None:
    batch = synthetic_batch(tmp_path, wins)
    out = tmp_path / "decision"
    assert main(["g1-report", str(batch), "--out", str(out)]) == 0
    result = json.loads((out / "decision.json").read_text())
    assert result["criteria_outcome"] == expected
    assert result["outcome"] == "pending"
    assert result["test_only"]
    assert result["coherence"]["candidate_wins"] == wins
    assert result["coherence"]["denominator"] == 30


def test_ties_are_non_wins_and_originality_is_an_independent_safeguard(tmp_path: Path) -> None:
    batch = synthetic_batch(tmp_path, 21, ties=9, originality=2)
    out = tmp_path / "decision"
    assert main(["g1-report", str(batch), "--out", str(out)]) == 0
    result = json.loads((out / "decision.json").read_text())
    assert result["coherence"]["ties"] == 9
    assert result["originality"]["mean_candidate_minus_comparator"] == -1
    assert result["criteria_outcome"] == "failed_inconclusive"


@pytest.mark.parametrize(("wins", "expected"), [(21, "passed"), (20, "failed_inconclusive")])
def test_a_coherence_only_protocol_is_decided_by_coherence_alone(
    tmp_path: Path, wins: int, expected: str
) -> None:
    batch = synthetic_batch(tmp_path, wins, coherence_only=True)
    out = tmp_path / "decision"
    assert main(["g1-report", str(batch), "--out", str(out)]) == 0
    result = json.loads((out / "decision.json").read_text())
    assert result["pending"] == [
        "Synthetic test-only judgements cannot establish a scientific gate."
    ]
    assert result["criteria_outcome"] == expected
    assert result["originality"] == {"analysis": "not_registered"}
    assert "Coherence decides this gate." in (out / "decision.md").read_text()


@pytest.mark.parametrize("change", ["artifact", "missing_score", "invalid_score", "cohort", "open"])
def test_incomplete_or_changed_evidence_cannot_produce_a_decision(
    tmp_path: Path, change: str
) -> None:
    batch = synthetic_batch(tmp_path, 30)
    if change == "artifact":
        (tmp_path / "run/C/0.md").write_text("Replaced prose")
    elif change == "open":
        (batch / "closed.json").unlink()
    elif change == "cohort":
        path = batch / "sealed/g1.json"
        binding = json.loads(path.read_text())
        binding["readiness"]["pairs"].pop()
        path.write_text(json.dumps(binding))
    else:
        path = batch / "judgements.jsonl"
        judgements = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "missing_score":
            judgements[0]["a"]["scores"].pop("originality")
        else:
            judgements[0]["a"]["scores"]["coherence"] = 9
        path.write_text("\n".join(json.dumps(j) for j in judgements) + "\n")
    out = tmp_path / "decision"
    assert main(["g1-report", str(batch), "--out", str(out)]) == (2 if change == "open" else 0)
    if change != "open":
        result = json.loads((out / "decision.json").read_text())
        assert result["criteria_outcome"] == "pending"
        assert result["outcome"] == "pending"
        assert result["pending"]


def test_results_are_reproducible_and_completed_reports_are_preserved(tmp_path: Path) -> None:
    batch = synthetic_batch(tmp_path, 21)
    first, second = tmp_path / "first", tmp_path / "second"
    assert main(["g1-report", str(batch), "--out", str(first)]) == 0
    assert main(["g1-report", str(batch), "--out", str(second)]) == 0
    assert (first / "decision.json").read_bytes() == (second / "decision.json").read_bytes()
    assert main(["g1-report", str(batch), "--out", str(first)]) == 2
    result = json.loads((first / "decision.json").read_text())
    assert any("timing absent" in caveat for caveat in result["reading_caveats"])
    assert any("guesses absent" in caveat for caveat in result["reading_caveats"])


def test_identical_prose_from_another_run_cannot_supply_current_realisation(tmp_path: Path) -> None:
    batch = synthetic_batch(tmp_path, 21)
    source_root = tmp_path / "foreign"
    source_root.mkdir()
    run = packet_source(source_root)
    folder = run / "stories/C/brief-01/r1"
    for name in ("story.md", "story.raw.md", "attempts/1/scene-01.md"):
        write_text_lf(folder / name, "Candidate story 0\n")
    packet = tmp_path / "foreign-packet"
    assert create_packet(run, packet) == 0
    out = tmp_path / "decision"
    assert main(["g1-report", str(batch), "--out", str(out), "--realisation", str(packet)]) == 0
    result = json.loads((out / "decision.json").read_text())
    [evidence] = result["realisation"]
    assert evidence["stale"]
    assert evidence["owner_status"] == "pending"
