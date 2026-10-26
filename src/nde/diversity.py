"""ID-independent structural distances over explicit labels and typed edges."""

import argparse
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from nde.controllability import write_evidence
from nde.files import checksum_lf
from nde.narrative.base import Strict
from nde.narrative.ledger import plan_identity
from nde.narrative.models import NarrativePlan
from nde.narrative.uncertainty import Reading
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import load_plan


class PlanSample(Strict):
    path: str
    seed: int
    functions: dict[str, Reading] = Field(default_factory=dict[str, Reading])


class DiversityInputs(Strict):
    plans: list[PlanSample] = Field(min_length=2)

    @model_validator(mode="after")
    def _distinct_seeds(self) -> Self:
        if len({sample.seed for sample in self.plans}) != len(self.plans):
            raise ValueError("diversity comparisons require distinct seeds")
        return self


class SampleEvidence(Strict):
    seed: int
    plan_sha256: str
    validator_passed: bool
    unknown_functions: list[str]
    functions: dict[str, Reading]


class Distance(Strict):
    left: int
    right: int
    function_distance: float | None
    edge_distance: float | None


class DiversityReport(Strict):
    version: Literal["1"] = "1"
    scope: Literal["plan-only"] = "plan-only"
    metric: Literal["ordinal-position-multiset-jaccard-v1"] = "ordinal-position-multiset-jaccard-v1"
    inputs: dict[str, str]
    samples: list[SampleEvidence]
    pairs: list[Distance]


type Features = Counter[tuple[str, ...]]


def _edges(plan: NarrativePlan) -> Features:
    scenes = {scene.id: str(index) for index, scene in enumerate(plan.scenes)}
    facts = {fact.id: str(index) for index, fact in enumerate(plan.knowledge_facts)}
    people = {person.id: str(index) for index, person in enumerate(plan.characters)}
    setups = {setup.id: scenes[setup.scene_id] for setup in plan.setups}
    result: Features = Counter()
    result.update(
        ("causal", scenes[edge.cause_scene_id], scenes[edge.effect_scene_id])
        for edge in plan.causal_links
    )
    result.update(
        ("setup_payoff", setups[edge.setup_id], scenes[edge.payoff_scene_id])
        for edge in plan.payoff_links
    )
    result.update(
        ("reveal", facts[edge.fact_id], scenes[edge.scene_id], people[holder])
        for edge in plan.reveal_links
        for holder in edge.holder_ids
    )
    result.update(
        ("knowledge_use", facts[edge.fact_id], scenes[edge.scene_id], people[edge.holder_id])
        for edge in plan.knowledge_uses
    )
    return result


def _distance(left: Features, right: Features) -> float:
    union = sum((left | right).values())
    return 0.0 if union == 0 else 1.0 - sum((left & right).values()) / union


def diversity_report(path: Path) -> DiversityReport:
    request = DiversityInputs.model_validate_json(path.read_text(encoding="utf-8"))
    inputs = {str(path.resolve()): checksum_lf(path.read_bytes())}
    samples: list[SampleEvidence] = []
    functions: list[Features] = []
    edges: list[Features] = []
    for sample in request.plans:
        source = path.parent / sample.path
        plan = load_plan(source)
        inputs[str(source.resolve())] = checksum_lf(source.read_bytes())
        ids = {scene.id for scene in plan.scenes}
        if set(sample.functions) - ids:
            raise ValueError("function annotations include unknown scene identities")
        unknown = [
            scene.id
            for scene in plan.scenes
            if scene.id not in sample.functions or sample.functions[scene.id].status != "known"
        ]
        valid = validate_plan(plan).valid
        samples.append(
            SampleEvidence(
                seed=sample.seed,
                plan_sha256=plan_identity(plan),
                validator_passed=valid,
                unknown_functions=unknown,
                functions=sample.functions,
            )
        )
        functions.append(
            Counter(
                ("function", str(index), sample.functions[scene.id].value or "")
                for index, scene in enumerate(plan.scenes)
                if scene.id in sample.functions and sample.functions[scene.id].status == "known"
            )
        )
        edges.append(_edges(plan) if valid else Counter())
    pairs: list[Distance] = []
    for left, right in combinations(range(len(samples)), 2):
        a, b = samples[left], samples[right]
        pairs.append(
            Distance(
                left=a.seed,
                right=b.seed,
                function_distance=None
                if a.unknown_functions or b.unknown_functions
                else _distance(functions[left], functions[right]),
                edge_distance=_distance(edges[left], edges[right])
                if a.validator_passed and b.validator_passed
                else None,
            )
        )
    return DiversityReport(inputs=inputs, samples=samples, pairs=pairs)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)


def run(args: argparse.Namespace) -> int:
    try:
        write_evidence(diversity_report(args.input), args.out, "diversity-report")
    except (OSError, ValueError) as error:
        print(f"nde diversity-report: {error}", file=sys.stderr)
        return 2
    print(args.out)
    return 0
