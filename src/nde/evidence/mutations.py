"""Replay named semantic mutations and require their intended diagnostic."""

import copy
import json
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from nde.annotations.records import digest
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan

from .records import Mutation, MutationResult, Suite, SuiteResult

DEFAULT_SUITE = Path(__file__).with_name("mutations.json")
MIN_CATEGORIES = 20
MIN_CATCH_RATE = 0.95
REQUIRED_FAMILIES = {
    "structure",
    "causality",
    "setups",
    "knowledge",
    "state",
    "constraints",
    "interpretation",
    "evidence",
}


def load_suite(path: Path) -> Suite:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    raw["controls"] = {
        name: json.loads((path.parent / filename).read_text(encoding="utf-8"))
        for name, filename in cast(dict[str, str], raw["controls"]).items()
    }
    return Suite.model_validate(raw)


def _mutate(control: dict[str, Any], mutation: Mutation) -> dict[str, Any]:
    data = copy.deepcopy(control)
    target: Any = data
    for part in mutation.path[:-1]:
        target = target[part]
    target[mutation.path[-1]] = copy.deepcopy(mutation.value)
    return data


def _run_case(control: dict[str, Any], mutation: Mutation) -> MutationResult:
    result = MutationResult(
        category=mutation.category,
        family=mutation.family,
        description=mutation.description,
        expected_code=mutation.expected_code,
        expected_severity=mutation.expected_severity,
        caught=False,
    )
    try:
        plan = NarrativePlan.model_validate(_mutate(control, mutation))
    except (ValidationError, KeyError, IndexError, TypeError) as error:
        result.parse_error = str(error)
        return result
    result.diagnostics = validate_plan(plan).diagnostics
    result.caught = any(
        item.code == mutation.expected_code and item.severity == mutation.expected_severity
        for item in result.diagnostics
    )
    return result


def run_suite(suite: Suite) -> SuiteResult:
    names = [mutation.category for mutation in suite.mutations]
    if len(set(names)) != len(names):
        raise ValueError("mutation category names must be unique")
    controls: dict[str, bool] = {}
    control_diagnostics: dict[str, list[Any]] = {}
    for name, data in suite.controls.items():
        plan = NarrativePlan.model_validate(data)
        result = validate_plan(plan)
        controls[name] = result.valid
        control_diagnostics[name] = result.diagnostics
    results = [
        _run_case(suite.controls[mutation.control], mutation) for mutation in suite.mutations
    ]
    caught = sum(result.caught for result in results)
    rate = caught / len(results) if results else 0
    problems: list[str] = []
    if len(results) < MIN_CATEGORIES:
        problems.append(f"at least {MIN_CATEGORIES} distinct mutation categories are required")
    missing = REQUIRED_FAMILIES - {case.family for case in suite.mutations}
    if missing:
        problems.append(f"missing mutation families: {', '.join(sorted(missing))}")
    if rate < MIN_CATCH_RATE:
        problems.append("intended-validator mutation catch rate is below 95%")
    if not controls or not all(controls.values()):
        problems.append("valid mutation controls did not all pass")
    return SuiteResult(
        suite_sha256=digest(suite.model_dump_json()),
        total=len(results),
        caught=caught,
        catch_rate=rate,
        control_results=controls,
        control_diagnostics=control_diagnostics,
        results=results,
        problems=problems,
    )
