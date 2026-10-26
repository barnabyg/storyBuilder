"""Pipelines, registered by the name conditions use for them."""

from nde.stories.baseline_a import BaselineA
from nde.stories.baseline_b import BaselineB
from nde.stories.baseline_c import BaselineC
from nde.stories.baseline_d import BaselineD
from nde.stories.pipeline import PipelineFactory
from nde.stories.scrambled import BaselineBScrambled

PIPELINES: dict[str, PipelineFactory] = {
    "baseline_a": BaselineA.from_params,
    "baseline_b": BaselineB.from_params,
    "baseline_c": BaselineC.from_params,
    "baseline_d": BaselineD.from_params,
    "baseline_b_scrambled": BaselineBScrambled.from_params,
}
