"""Pipelines, registered by the name conditions use for them."""

from collections.abc import Mapping
from typing import Any

from pydantic import JsonValue

from nde.stories.baseline_a import BaselineA
from nde.stories.baseline_b import BaselineB
from nde.stories.baseline_c import BaselineC
from nde.stories.baseline_d import BaselineD
from nde.stories.pipeline import Pipeline, PipelineFactory
from nde.stories.scrambled import BaselineBScrambled


def _candidate_e(raw: Mapping[str, JsonValue]) -> Pipeline[Any]:
    # Imported on use: the conditional workflows import the corpus and gate modules,
    # which in turn import the run machinery that reads this registry.
    from nde.stories.candidate_e import CandidateE  # noqa: PLC0415

    return CandidateE.from_params(raw)


PIPELINES: dict[str, PipelineFactory] = {
    "baseline_a": BaselineA.from_params,
    "baseline_b": BaselineB.from_params,
    "baseline_c": BaselineC.from_params,
    "baseline_d": BaselineD.from_params,
    "candidate_e": _candidate_e,
    "baseline_b_scrambled": BaselineBScrambled.from_params,
}
