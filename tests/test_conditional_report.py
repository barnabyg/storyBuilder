"""Known-answer E evidence reports from public runs, a tiny CPU checkpoint and fake providers."""

import copy
import importlib.util
import json
import os
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from nde.annotations.records import digest
from nde.cli import main
from nde.conditional import report as evidence
from nde.conditional.report import retained_structure
from nde.corpus.records_io import create_record, read_sealed
from nde.diversity import multiset_distance, typed_edges
from nde.files import checksum_lf, write_text_lf
from nde.narrative.models import NarrativePlan
from nde.reconstruction.codec import CODEC_ID
from nde.reconstruction.overlap import view_distance
from tests.conditional_support import trained_checkpoint
from tests.conftest import REPO
from tests.fake_openai import FakeOpenAI, Json, Reply, completion
from tests.stories import jsonl, run, story_config
from tests.test_baseline_c import BASELINE_C
from tests.test_candidate_e import Provider, e_condition
from tests.test_plan_generation import skeleton
from tests.test_revision_backends import (
    POLICY,
    UNITS,
    accepted_plan,
    lamp_plan,
    mutated,
    responder,
    revise,
    run_config,
)

# The E story lock and the conditions its fixture briefs request.
LOCK = {"characters/characters.0/name": "Ann"}
FIELDS = ["genre", "themes", "ending_intent", "target_words", "scene_range"]


@dataclass(frozen=True)
class Evidence:
    root: Path
    checkpoint: Path
    stories: Path
    revisions: Path
    dataset: Path


def _benchmark(root: Path) -> Path:
    """Dev briefs the two-scene fixture model can (brief-01) and cannot (brief-02) satisfy.

    Four scenes is the tiny checkpoint's registered capacity and C's fixture plan length.
    """
    benchmark = root / "benchmark"
    (benchmark / "dev").mkdir(parents=True)
    for brief, low in (("brief-01", 2), ("brief-02", 3)):
        value = yaml.safe_load((REPO / f"benchmark/dev/{brief}.yaml").read_text())
        value.update(min_scenes=low, max_scenes=4)
        (benchmark / f"dev/{brief}.yaml").write_text(yaml.safe_dump(value))
    return benchmark


class Planner(Provider):
    """E's completion and prose as in the E tests; C plans four scenes."""

    def __call__(self, body: Json) -> Reply:
        if "whole-story skeleton planner" in str(body["messages"][0]["content"]):
            plan = skeleton()
            plan["scenes"] = plan["scenes"][:4]
            plan["arcs"][0]["scene_ids"] = ["scene-0", "scene-3"]
            return completion(json.dumps(plan))
        return super().__call__(body)


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Evidence:
    if importlib.util.find_spec("torch") is None:
        if os.environ.get("NDE_REQUIRE_ML") == "1":
            pytest.fail("CPU ML lane requires the pinned torch runtime")
        pytest.skip("optional CPU ML runtime")
    root = tmp_path_factory.mktemp("conditional-report")
    fake = FakeOpenAI()
    fake.start()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("FAKE_API_KEY", "test-key")
        try:
            checkpoint = trained_checkpoint(root / "training", fake)
            fake.default = Planner()
            conditions = [BASELINE_C, e_condition(checkpoint, condition_fields=FIELDS, locks=LOCK)]
            config = story_config(
                fake.base_url,
                conditions,
                briefs={"ids": ["brief-01", "brief-02"]},
                briefs_dir=str(_benchmark(root)),
                repeat=2,
            )
            stories = run(root, config)
            fake.default = responder(mutated(accepted_plan))
            backend = {
                "kind": "e",
                "checkpoint": str(checkpoint),
                "target_policy": POLICY,
                "conditions": {"genre": "parable"},
            }
            value = run_config(root, fake.base_url, UNITS, backend)
            value["plans"]["repeat"] = 3
            revisions = revise(root, value, 0)
            assert revisions is not None
        finally:
            fake.stop()
    dataset = root / "training/data/conditional/dataset.json"
    return Evidence(root, checkpoint, stories, revisions, dataset)


def request(directory: Path, runs: Evidence, **changes: Any) -> Path:
    value: dict[str, Any] = {
        "story_runs": [{"run": str(runs.stories), "label": "e-stories"}],
        "revision_runs": [{"run": str(runs.revisions), "label": "e-revisions"}],
        "reference": {"runs": [{"run": str(runs.stories), "condition": "C"}]},
        **changes,
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "request.json"
    path.write_text(json.dumps(value, indent=2))
    return path


def build(path: Path, out: Path, expected: int = 0) -> dict[str, Any]:
    assert main(["conditional-report", "build", str(path), "--out", str(out)]) == expected
    return read_sealed(out / "report.json") if expected == 0 else {}


def check(out: Path) -> int:
    return main(["conditional-report", "check", str(out)])


def rows(report: dict[str, Any], section: str, **match: Any) -> list[dict[str, Any]]:
    return [
        row
        for row in report["adherence"][section]["rows"]
        if all(row[key] == value for key, value in match.items())
    ]


@contextmanager
def restored(*paths: Path) -> Generator[None]:
    """Mutate shared fixture evidence inside the block; its bytes come back afterwards."""
    saved = {path: path.read_bytes() for path in paths}
    try:
        yield
    finally:
        for path, data in saved.items():
            path.write_bytes(data)


def renamed(plan: dict[str, Any]) -> dict[str, Any]:
    """The same plan with every identity and reference renamed."""
    text = json.dumps(plan)
    for old, new in (
        ("lamp.scene-1", "opening"),
        ("lamp.scene-2", "closing"),
        ('"ann"', '"heroine"'),
        ('"cause"', '"because"'),
        ("lamp-fact", "oil-fact"),
        ('"goal"', '"aim"'),
        ('"belief"', '"conviction"'),
    ):
        text = text.replace(old, new)
    result = json.loads(text)
    # Kinds are labels, not identities: only the state records' own IDs change.
    for state, kind in zip(result["states"], ("goal", "belief"), strict=True):
        state["kind"] = kind
    return result


def test_retained_structure_is_id_independent_and_measures_changed_structure() -> None:
    source = NarrativePlan.model_validate(lamp_plan())
    same = NarrativePlan.model_validate(renamed(lamp_plan()))
    assert [s.id for s in same.scenes] != [s.id for s in source.scenes]
    retitled = copy.deepcopy(lamp_plan())
    retitled["scenes"][0]["title"] = "A different title"
    changed = copy.deepcopy(lamp_plan())
    changed["states"][1]["after"]["value"] = "the lamp is broken"
    policy = {key: value for key, value in POLICY.items() if value != "excluded"}
    views = {
        name: retained_structure(NarrativePlan.model_validate(plan), policy)
        for name, plan in (
            ("source", lamp_plan()),
            ("renamed", renamed(lamp_plan())),
            ("retitled", retitled),
            ("changed", changed),
        )
    }
    assert all(view is not None for view in views.values())
    source_view, renamed_view = views["source"], views["renamed"]
    assert source_view is not None
    assert renamed_view is not None
    # Renamed identities and descriptive text alone are structurally identical.
    assert renamed_view == source_view == views["retitled"]
    assert view_distance(source_view, renamed_view)["distance"] == 0
    assert multiset_distance(typed_edges(source), typed_edges(same)) == 0
    # One of eight retained records differs: 1 - 6 shared / 8 combined.
    changed_view = views["changed"]
    assert changed_view is not None
    result = view_distance(source_view, changed_view)
    assert (result["intersection"], result["union"], result["distance"]) == (6, 8, 0.25)
    # The unknown "before" readings are kept as features, so the comparison is uncertain.
    assert result["uncertain"] is True
    invalid = lamp_plan()
    invalid["causal_links"][0]["effect_scene_id"] = "nowhere"
    assert retained_structure(NarrativePlan.model_validate(invalid), policy) is None


@pytest.mark.ml
def test_report_binds_inputs_and_evaluates_predicates_with_complete_denominators(
    tmp_path: Path, runs: Evidence
) -> None:
    out = tmp_path / "report"
    report = build(request(tmp_path, runs), out)
    assert report["kind"] == "conditional-evidence-report"
    assert report["version"] == "1"
    assert report["test_only"] is True
    assert report["consistent"] is True, report["problems"]
    checkpoint = read_sealed(runs.checkpoint)
    assert report["e_identity"]["checkpoint"]["id"] == checkpoint["id"]
    assert report["e_identity"]["target_policy_status"] == "verified"
    # Briefs, constraints, checkpoint, histories, final plans/stories are all bound.
    bound = set(report["inputs"])
    story = runs.stories / "stories/E/brief-01/r1"
    alternative = runs.revisions / "plans/brief-01/r1"
    for path in (
        runs.checkpoint,
        runs.root / "benchmark/dev/brief-01.yaml",
        runs.stories / "config.json",
        runs.stories / "stories.jsonl",
        runs.stories / "local-work.jsonl",
        story / "plan.json",
        story / "story.md",
        story / "planning-provenance.json",
        story / "conditional-selection.json",
        runs.revisions / "alternatives.json",
        alternative / "revisions/revision/provenance.json",
        alternative / "revisions/revision/source-plan.json",
        alternative / "plan.json",
        runs.dataset,
    ):
        assert str(path.resolve()) in bound, path
    units = {unit["unit"]: unit for unit in report["units"]}
    assert len(units) == 4 + 3
    first = units["story/e-stories/brief-01/r1"]
    assert first["allowed_constraints"]["locks"] == LOCK
    assert first["conditions"]["scene_range"] == [2, 4]
    assert units["story/e-stories/brief-02/r1"]["status"] == "failed"
    assert units["revision/e-revisions/brief-01/r1"]["allowed_constraints"]["revision"][
        "locks"
    ] == ["states/goal"]
    deterministic = report["adherence"]["deterministic"]
    expected = {
        # brief-01's two scenes are in range; brief-02's stories failed operationally and
        # left no plan, which is no output rather than non-adherence.
        "scene_range": (4, 2, 0, 0, 2),
        "target_words": (4, 2, 0, 0, 2),
        "lock": (4, 2, 0, 0, 2),
        "revision_lock": (3, 3, 0, 0, 0),
        "revision_scope": (3, 3, 0, 0, 0),
        "validity": (7, 5, 0, 0, 2),
    }
    assert report["report_version"] == "2"
    assert {
        name: (c["expected"], c["success"], c["failure"], c["unknown"], c["no-output"])
        for name, c in deterministic["by_predicate"].items()
    } == expected
    assert deterministic["total"]["expected"] == sum(row[0] for row in expected.values())
    [words] = rows(
        report, "deterministic", unit="story/e-stories/brief-01/r1", predicate="target_words"
    )
    assert words["status"] == "success"
    assert words["reason"].startswith("2000 words in the final prose")
    [failed] = rows(report, "deterministic", unit="story/e-stories/brief-02/r1", predicate="lock")
    assert failed == {
        "unit": "story/e-stories/brief-02/r1",
        "predicate": "lock",
        "request": {"selector": "characters/characters.0/name", "value": "Ann"},
        "status": "no-output",
        "reason": "no saved plan: an operational failure, not non-adherence",
    }
    diagnostics = report["model_diagnostics"]["units"]["story/e-stories/brief-02/r1"]
    assert diagnostics["generation_status"] == "attempt-limit"
    assert len(diagnostics["attempts"]) == 3
    markdown = (out / "report.md").read_text()
    assert "| scene_range | 4 | 2 | 0 | 0 | 2 |" in markdown
    assert "not a G-E decision" in markdown
    assert check(out) == 0
    # A report folder is never overwritten.
    build(request(tmp_path, runs), out, expected=2)


@pytest.mark.ml
def test_missing_outputs_stay_unknown_and_held_out_runs_are_refused(
    tmp_path: Path, runs: Evidence
) -> None:
    index = runs.stories / "stories.jsonl"
    with restored(index):
        kept = [
            row
            for row in jsonl(index)
            if (row["condition"], row["brief"], row["repetition"]) != ("E", "brief-01", 1)
        ]
        write_text_lf(index, "".join(json.dumps(row) + "\n" for row in kept))
        report = build(request(tmp_path, runs), tmp_path / "missing")
    missing = "story/e-stories/brief-01/r2"
    assert {unit["unit"]: unit["status"] for unit in report["units"]}[missing] == "missing"
    statuses = {row["status"] for row in rows(report, "deterministic", unit=missing)}
    assert statuses == {"unknown"}
    assert report["adherence"]["deterministic"]["by_predicate"]["scene_range"] == {
        "expected": 4,
        "success": 1,
        "failure": 0,
        "unknown": 1,
        "no-output": 2,
    }
    assert {row["status"] for row in rows(report, "judgement", unit=missing)} == {"missing"}
    config = runs.stories / "config.json"
    with restored(config):
        value = json.loads(config.read_text())
        value["stories"]["briefs"] = {"split": "test", "ids": ["brief-11"]}
        heldout = runs.root / "benchmark/test"
        heldout.mkdir(exist_ok=True)
        shutil.copy(REPO / "benchmark/test/brief-11.yaml", heldout / "brief-11.yaml")
        config.write_text(json.dumps(value))
        try:
            with pytest.raises(ValueError, match="held-out briefs"):
                evidence.build(request(tmp_path / "held-out", runs))
            build(request(tmp_path / "held-out", runs), tmp_path / "held-out/report", expected=2)
        finally:
            shutil.rmtree(heldout)
    assert not (tmp_path / "held-out/report").exists()


@pytest.mark.ml
def test_unreadable_evidence_is_unknown_not_failure(tmp_path: Path, runs: Evidence) -> None:
    revision = runs.revisions / "plans/brief-01/r2/revisions/revision/provenance.json"
    planning = runs.stories / "stories/E/brief-01/r2/planning-provenance.json"
    with restored(revision, planning):
        revision.unlink()
        planning.unlink()
        report = build(request(tmp_path, runs), tmp_path / "report")
    statuses = {unit["unit"]: unit["status"] for unit in report["units"]}
    for unit in ("revision/e-revisions/brief-01/r2", "story/e-stories/brief-01/r2"):
        assert statuses[unit] == "unreadable"
        assert {row["status"] for row in rows(report, "deterministic", unit=unit)} == {"unknown"}
        assert {row["status"] for row in rows(report, "judgement", unit=unit)} == {"missing"}
    assert report["consistent"] is False
    assert any("provenance unavailable" in problem for problem in report["problems"])


@pytest.mark.ml
def test_revision_predicates_need_the_recorded_source_plan(tmp_path: Path, runs: Evidence) -> None:
    source = runs.revisions / "plans/brief-01/r3/revisions/revision/source-plan.json"
    with restored(source):
        plan = json.loads(source.read_text())
        plan["states"][0]["after"]["value"] = "carry the lamp"
        source.write_text(json.dumps(plan))
        report = build(request(tmp_path, runs), tmp_path / "report")
    unit = "revision/e-revisions/brief-01/r3"
    assert {u["unit"]: u["status"] for u in report["units"]}[unit] == "unverified"
    # A substituted baseline would make the lock and scope predicates meaningless.
    assert {row["status"] for row in rows(report, "deterministic", unit=unit)} == {"unknown"}
    assert any(
        f"{unit}: source plan differs from the recorded original" in problem
        for problem in report["problems"]
    )
    assert report["consistent"] is False


@pytest.mark.ml
def test_operational_failure_with_a_saved_plan_keeps_plan_adherence(
    tmp_path: Path, runs: Evidence
) -> None:
    index = runs.stories / "stories.jsonl"
    with restored(index):
        records = jsonl(index)
        for row in records:
            if row["path"] == "stories/E/brief-01/r2":
                # Planning succeeded and saved its plan; prose then failed.
                row.update(status="failed", error="step 'scene-01': provider error")
        write_text_lf(index, "".join(json.dumps(row) + "\n" for row in records))
        report = build(request(tmp_path, runs), tmp_path / "report")
    unit = "story/e-stories/brief-01/r2"
    statuses = {row["predicate"]: row["status"] for row in rows(report, "deterministic", unit=unit)}
    assert statuses == {
        "scene_range": "success",
        "lock": "success",
        "validity": "success",
        "target_words": "no-output",
    }
    assert {row["status"] for row in rows(report, "judgement", unit=unit)} == {"no-output"}
    [summary] = [u for u in report["units"] if u["unit"] == unit]
    assert summary["plan_sha256"] is not None
    assert summary["output_sha256"] is None


def judgement_file(path: Path, **review: Any) -> Path:
    path.write_text(json.dumps({"reviewer": "Owner", **review}))
    return path


@pytest.mark.ml
def test_literary_adherence_stays_pending_until_actual_owner_assessment(
    tmp_path: Path, runs: Evidence
) -> None:
    unit = "story/e-stories/brief-01/r1"
    story = runs.stories / "stories/E/brief-01/r1/story.md"
    current = digest(story.read_text(encoding="utf-8"))

    def item(condition: str, sha256: str = current, rating: str = "adheres") -> dict[str, Any]:
        return {
            "unit": unit,
            "output_sha256": sha256,
            "condition": condition,
            "rating": rating,
            "note": "Read against the brief.",
        }

    reviews = [
        # A model estimate of every judgement is diagnostic only.
        judgement_file(
            tmp_path / "estimate.json",
            reviewer="judge-model",
            diagnostic_provenance={"model": "fake-model", "prompt": "adherence@1"},
            assessments=[item(name) for name in ("genre", "themes", "ending_intent")],
        ),
        judgement_file(tmp_path / "synthetic.json", test_only=True, assessments=[item("themes")]),
        judgement_file(tmp_path / "stale.json", assessments=[item("ending_intent", "0" * 64)]),
        judgement_file(
            tmp_path / "owner.json", assessments=[item("genre", rating="does_not_adhere")]
        ),
    ]
    path = request(tmp_path, runs, judgements=[str(review) for review in reviews])
    report = build(path, tmp_path / "report")
    by_condition = {row["condition"]: row for row in rows(report, "judgement", unit=unit)}
    assert by_condition["genre"]["status"] == "does_not_adhere"
    assert by_condition["genre"]["evidence"] == "owner"
    assert by_condition["genre"]["diagnostic_estimates"] == 1
    assert by_condition["themes"]["status"] == "pending"
    assert by_condition["themes"]["synthetic_test_only_assessments"] == 1
    assert by_condition["themes"]["diagnostic_estimates"] == 1
    assert by_condition["ending_intent"]["status"] == "pending"
    assert by_condition["ending_intent"]["stale_assessments"] == 1
    judgement = report["adherence"]["judgement"]
    # Two brief-01 stories carry all three; brief-02's failed stories and the
    # revision alternatives' genre-only request complete the denominator.
    assert judgement["by_condition"]["genre"] == {
        "expected": 7,
        "pending": 4,
        "adheres": 0,
        "does_not_adhere": 1,
        "unknown": 0,
        "no-output": 2,
        "missing": 0,
    }
    assert judgement["total"]["pending"] == 4 + 3 + 3 - 2
    # Literary conditions never appear among the deterministic predicates.
    assert not {"genre", "themes", "ending_intent"} & set(
        report["adherence"]["deterministic"]["by_predicate"]
    )
    unknown = judgement_file(tmp_path / "unknown.json", assessments=[item("genre") | {"unit": "x"}])
    report = build(request(tmp_path / "bad", runs, judgements=[str(unknown)]), tmp_path / "bad/r")
    assert report["consistent"] is False
    assert any("x has no requested genre" in problem for problem in report["problems"])


@pytest.mark.ml
def test_duplicate_alternatives_and_unknown_labels_are_preserved_in_diversity(
    tmp_path: Path, runs: Evidence
) -> None:
    report = build(request(tmp_path, runs), tmp_path / "report")
    groups = {group["group"]: group for group in report["diversity"]["groups"]}
    revisions = groups["revision/e-revisions/brief-01"]
    # Argmax inference makes all three seeds' alternatives identical: kept, not dropped.
    assert [s["seed"] for s in revisions["samples"]] == [7, 8, 9]
    assert [s["exact_duplicate_of"] for s in revisions["samples"]] == [
        None,
        "revision/e-revisions/brief-01/r1",
        "revision/e-revisions/brief-01/r1",
    ]
    assert revisions["summary"] | {"mean_structure_distance": 0, "mean_edge_distance": 0} == {
        "expected": 3,
        "with_output": 3,
        "without_output": 0,
        "exact_distinct": 1,
        "exact_duplicates": 2,
        "structurally_distinct": 1,
        "structural_duplicates": 2,
        "pairs": 3,
        "unavailable_pairs": 0,
        "uncertain_pairs": 3,
        "mean_structure_distance": 0,
        "mean_edge_distance": 0,
    }
    assert {p["structure_distance"] for p in revisions["pairs"]} == {0}
    assert all(s["unknown_labels"] == 2 for s in revisions["samples"])
    failed = groups["story/e-stories/brief-02"]
    assert failed["summary"]["expected"] == 2
    assert failed["summary"]["without_output"] == 2
    assert failed["pairs"] == []
    stories = groups["story/e-stories/brief-01"]
    assert stories["summary"]["exact_duplicates"] == 1


@pytest.mark.ml
def test_training_similarity_reports_complete_lineage_and_renamed_reproduction(
    tmp_path: Path, runs: Evidence
) -> None:
    report = build(request(tmp_path, runs), tmp_path / "report")
    training = report["training_similarity"]
    [population] = training["populations"]
    dataset = read_sealed(runs.dataset)
    assert population | {"nearest": None, "summary": None} == {
        "name": "conditional-fine-tuning",
        "origin": "checkpoint binding",
        "role": "other",
        "path": str(runs.dataset.resolve()),
        "status": "verified",
        "id": dataset["id"],
        "kind": "conditional-dataset",
        "split": "train",
        "test_only": True,
        "records": len(dataset["examples"]),
        "works": 1,
        "authors": 1,
        "nearest": None,
        "summary": None,
    }
    nearest = population["nearest"]
    # E's story plans reproduce the training annotation under renamed IDs: distance 0.
    story_plan = json.loads((runs.stories / "stories/E/brief-01/r1/plan.json").read_text())
    assert story_plan["scenes"][0]["id"] == "scenes.0"
    assert nearest["story/e-stories/brief-01/r1"]["matches"][0]["distance"] == 0
    assert nearest["story/e-stories/brief-01/r1"]["matches"][0]["work_id"] == "lamp"
    # The revision changed one of eight retained records.
    assert nearest["revision/e-revisions/brief-01/r1"]["matches"][0]["distance"] == 0.25
    assert population["summary"] == {
        "queries": 5,
        "exact_structural_matches": 2,
        "min_distance": 0,
        "mean_nearest_distance": 0.15,
    }
    assert training["pretraining"]["status"] == "none"
    assert training["g3_population"]["status"] == "not-applicable"
    assert training["lineage"] == {"complete": True, "missing": []}
    assert any("not proof of memorisation" in text for text in training["limitations"])
    # Without the bound fine-tuning dataset the lineage is reported incomplete, not omitted.
    data = runs.root / "training/data"
    hidden = runs.root / "training/hidden-data"
    data.rename(hidden)
    try:
        build(request(tmp_path / "unavailable", runs), tmp_path / "unavailable/r", expected=2)
        policy = request(tmp_path / "declared", runs, target_policy=POLICY)
        declared = build(policy, tmp_path / "declared/r")
    finally:
        hidden.rename(data)
    assert declared["e_identity"]["target_policy_status"] == "declared-unverified"
    [missing] = declared["training_similarity"]["populations"]
    assert missing["status"] == "unavailable"
    assert declared["training_similarity"]["lineage"] == {
        "complete": False,
        "missing": ["conditional-fine-tuning"],
    }
    assert any("training lineage incomplete" in item for item in declared["pending"])


@pytest.mark.ml
def test_g3_lineage_is_verified_against_the_checkpoints_g3_evidence(
    tmp_path: Path, runs: Evidence
) -> None:
    supplied = [{"name": "g3", "id": "a" * 64}]
    checkpoint = read_sealed(runs.checkpoint)
    assert evidence.g3_lineage(supplied, checkpoint)["status"] == "unverified"
    assert evidence.g3_lineage([], checkpoint)["status"] == "not-applicable"
    # A checkpoint claiming passed G3 evidence that does not replay cannot be completed
    # by declaring any population to be its G3 population.
    claimed = copy.deepcopy(checkpoint)
    claimed["binding"]["config"]["g3_protocol"] = str(tmp_path / "missing-protocol.json")
    claimed["binding"]["scientific_evidence"] = {
        "status": "passed",
        "decision_id": "b" * 64,
        "inputs": {},
    }
    lineage = evidence.g3_lineage(supplied, claimed)
    assert lineage["status"] == "missing"
    assert lineage["required"] is True
    assert "G3 evidence unavailable" in lineage["reason"]
    # Matching is by the G3 protocol's registered training population identity.
    assert evidence.g3_population_status(supplied, "a" * 64) == {
        "status": "verified",
        "training_id": "a" * 64,
        "populations": ["g3"],
        "mismatched": [],
    }
    other = [*supplied, {"name": "relabelled", "id": "c" * 64}]
    mismatch = evidence.g3_population_status(other, "a" * 64)
    assert (mismatch["status"], mismatch["mismatched"]) == ("mismatch", ["relabelled"])
    assert evidence.g3_population_status([], "a" * 64)["status"] == "missing"


def population(path: Path, records: list[dict[str, Any]], **fields: Any) -> Path:
    create_record(
        path,
        {
            "codec_id": CODEC_ID,
            "test_only": True,
            "examples": [{"id": f"r{n}", "record": record} for n, record in enumerate(records)],
            "inputs": {},
            **fields,
        },
    )
    return path


@pytest.mark.ml
def test_protected_material_is_audited_against_every_training_population(
    tmp_path: Path, runs: Evidence
) -> None:
    examples = read_sealed(runs.dataset)["examples"]
    target = examples[0]["target"]
    leaked = {"view": target["view"], "provenance": target["provenance"]}
    other = {
        "view": target["view"],
        "provenance": {"work_id": "other", "author_id": "other", "source_sha256": "f" * 64},
    }
    g3 = population(tmp_path / "g3.json", [other], role="optimizer", split="train")
    protected = population(tmp_path / "test.json", [leaked], role="evaluator-only", split="test")
    clean = population(tmp_path / "clean.json", [other], role="evaluator-only", split="test")
    path = request(
        tmp_path,
        runs,
        training_populations=[{"name": "g3", "path": str(g3), "role": "g3"}],
        # One structural neighbour per query; the overlap list must still be exhaustive.
        neighbours=1,
        protected_populations=[
            {"name": "held-out", "path": str(protected)},
            {"name": "clean", "path": str(clean)},
        ],
    )
    training = build(path, tmp_path / "report")["training_similarity"]
    assert [p["name"] for p in training["populations"]] == ["conditional-fine-tuning", "g3"]
    # A test-only checkpoint has no G3 decision, so a population tagged g3 stays unverified.
    assert training["g3_population"]["status"] == "unverified"
    assert training["g3_population"]["populations"] == ["g3"]
    assert training["lineage"] == {"complete": True, "missing": []}
    audits = {(a["training"], a["protected"]): a for a in training["protected_audits"]}
    leak = audits["conditional-fine-tuning", "held-out"]
    assert leak["protected_overlap"] is True
    # The leaked work's every training family shares its source, work and author.
    assert {o["record_id"] for o in leak["overlaps"]} == {e["target"]["id"] for e in examples}
    assert all(o["exact_source"] and o["exact_work"] for o in leak["overlaps"])
    assert all(o["author_overlap"] for o in leak["overlaps"])
    distances = {o["record_id"]: o["distance"] for o in leak["overlaps"]}
    assert distances[target["id"]] == 0
    assert not audits["g3", "held-out"]["protected_overlap"]
    assert not audits["conditional-fine-tuning", "clean"]["protected_overlap"]
    assert training["protected_overlap"] is True
    # Protected material can only be queried; it never becomes an index or tuning input.
    misuse = request(
        tmp_path / "misuse", runs, training_populations=[{"name": "t", "path": str(protected)}]
    )
    build(misuse, tmp_path / "misuse/report", expected=2)
    unprotected = request(
        tmp_path / "unprotected", runs, protected_populations=[{"name": "p", "path": str(g3)}]
    )
    build(unprotected, tmp_path / "unprotected/report", expected=2)
    # The report itself can never be loaded as training data.
    with pytest.raises(ValueError, match="optimizer"):
        evidence.check_index(read_sealed(tmp_path / "report/report.json"))


def relabelled_d(source: Path, out: Path) -> Path:
    """A copy of the C outputs recorded as a D condition, standing in for D reference runs."""
    shutil.copytree(source, out)
    config = json.loads((out / "config.json").read_text())
    config["stories"]["conditions"] = [
        {**condition, "name": "D", "pipeline": "baseline_d"}
        for condition in config["stories"]["conditions"]
        if condition["name"] == "C"
    ]
    (out / "config.json").write_text(json.dumps(config))
    records = [
        row | {"condition": "D"} for row in jsonl(out / "stories.jsonl") if row["condition"] == "C"
    ]
    write_text_lf(out / "stories.jsonl", "".join(json.dumps(row) + "\n" for row in records))
    return out


def gd_folder(directory: Path, *, outcome: str, test_only: bool) -> Path:
    directory.mkdir(parents=True)
    condition = "D" if outcome == "passed" else "C"
    decision: dict[str, Any] = {
        "version": "1",
        "outcome": "pending" if test_only else outcome,
        "criteria_outcome": outcome,
        "test_only": test_only,
        "batch": str(directory / "batch"),
        "reference_baseline": None if test_only else {"condition": condition},
        "realisation": [],
    }
    decision["id"] = digest(json.dumps(decision, sort_keys=True))
    (directory / "decision.json").write_text(json.dumps(decision))
    return directory


@pytest.mark.ml
def test_reference_is_c_unless_actual_gd_evidence_permits_d(
    tmp_path: Path, runs: Evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = build(request(tmp_path, runs), tmp_path / "c")
    reference = report["reference"]
    assert reference["permitted"] == "C"
    assert reference["reason"].startswith("no G-D decision supplied")
    [c] = reference["identities"]
    assert (c["condition"], c["pipeline"]) == ("C", "baseline_c")
    assert reference["summary"] == {
        "expected": 4,
        "compared": 2,
        "uncompared": 2,
        "mean_nearest_structure_distance": reference["summary"]["mean_nearest_structure_distance"],
    }
    assert 0 < reference["summary"]["mean_nearest_structure_distance"] <= 1
    d_run = relabelled_d(runs.stories, tmp_path / "d-run")
    both = {
        "runs": [
            {"run": str(runs.stories), "condition": "C"},
            {"run": str(d_run), "condition": "D"},
        ]
    }
    # A D outcome that does not replay from its bound evidence cannot promote D.
    unbound = gd_folder(tmp_path / "unbound", outcome="passed", test_only=False)
    path = request(tmp_path / "u", runs, reference={**both, "gd_decision": str(unbound)})
    reference = build(path, tmp_path / "u/r")["reference"]
    assert reference["permitted"] == "C"
    assert "no longer replays" in reference["reason"]
    assert any("D outputs" in problem for problem in reference["problems"])

    # Simulates a decision that replays exactly; fixtures cannot hold actual owner evidence.
    def replay(batch: Path, _packets: list[Path]) -> dict[str, Any]:
        return json.loads((batch.parent / "decision.json").read_text())

    monkeypatch.setattr(evidence, "gd_decision", replay)
    synthetic = gd_folder(tmp_path / "synthetic", outcome="passed", test_only=True)
    path = request(tmp_path / "s", runs, reference={**both, "gd_decision": str(synthetic)})
    reference = build(path, tmp_path / "s/r")["reference"]
    assert reference["permitted"] == "C"
    assert "synthetic test-only" in reference["reason"]
    failed = gd_folder(tmp_path / "failed", outcome="failed", test_only=False)
    path = request(tmp_path / "f", runs, reference={**both, "gd_decision": str(failed)})
    assert build(path, tmp_path / "f/r")["reference"]["permitted"] == "C"
    actual = gd_folder(tmp_path / "actual", outcome="passed", test_only=False)
    path = request(tmp_path / "a", runs, reference={**both, "gd_decision": str(actual)})
    reference = build(path, tmp_path / "a/r")["reference"]
    assert reference["permitted"] == "D"
    assert [i["pipeline"] for i in reference["identities"]] == ["baseline_d"]
    assert any("C outputs" in problem for problem in reference["problems"])
    assert reference["summary"]["compared"] == 2


@pytest.mark.ml
def test_realisation_binds_final_prose_and_changed_outputs_invalidate_reviews(
    tmp_path: Path, runs: Evidence
) -> None:
    packet = tmp_path / "packet"
    arguments = ["realisation", "create", str(runs.stories), "--condition", "E"]
    assert main([*arguments, "--brief", "brief-01", "--out", str(packet)]) == 0
    saved = json.loads((packet / "packet.json").read_text())
    [item] = saved["items"]
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "packet_sha256": saved["packet_sha256"],
                "reviewer": "Owner",
                "assessments": [{"item_id": item["id"], "rating": "realised", "note": "Clear."}],
            }
        )
    )
    assert main(["realisation", "assess", str(packet), str(review)]) == 0
    folder = runs.stories / "stories/E/brief-01/r1"
    owner = judgement_file(
        tmp_path / "owner.json",
        assessments=[
            {
                "unit": "story/e-stories/brief-01/r1",
                "output_sha256": saved["story_sha256"],
                "condition": "genre",
                "rating": "adheres",
                "note": "Literary fiction throughout.",
            }
        ],
    )
    path = request(tmp_path, runs, realisation=[str(packet)], judgements=[str(owner)])
    out = tmp_path / "report"
    report = build(path, out)
    realised = {row["unit"]: row for row in report["realisation"]["rows"]}
    first = realised["story/e-stories/brief-01/r1"]
    assert first["status"] == "complete"
    assert first["story_sha256"] == saved["story_sha256"]
    assert realised["story/e-stories/brief-01/r2"]["status"] == "missing"
    assert report["realisation"]["summary"] == {
        "expected": 2,
        "complete": 1,
        "pending": 0,
        "stale": 0,
        "missing": 1,
    }
    [genre] = rows(report, "judgement", unit="story/e-stories/brief-01/r1", condition="genre")
    assert genre["status"] == "adheres"
    index = runs.stories / "stories.jsonl"
    with restored(folder / "story.md", folder / "story.raw.md", index):
        changed = "A different final story.\n"
        write_text_lf(folder / "story.md", changed)
        write_text_lf(folder / "story.raw.md", changed)
        records = jsonl(index)
        for row in records:
            if row["path"] == "stories/E/brief-01/r1":
                row["sha256"] = checksum_lf(changed.encode())
        write_text_lf(index, "".join(json.dumps(row) + "\n" for row in records))
        # The saved report no longer holds, and a new one invalidates the old reviews.
        assert check(out) == 1
        stale = build(path, tmp_path / "after")
    assert check(out) == 0
    first = {row["unit"]: row for row in stale["realisation"]["rows"]}[
        "story/e-stories/brief-01/r1"
    ]
    assert first["status"] == "stale"
    assert first["story_sha256"] == digest(changed)
    [genre] = rows(stale, "judgement", unit="story/e-stories/brief-01/r1", condition="genre")
    assert (genre["status"], genre["stale_assessments"]) == ("pending", 1)
    [words] = rows(
        stale, "deterministic", unit="story/e-stories/brief-01/r1", predicate="target_words"
    )
    assert words["status"] == "failure"
