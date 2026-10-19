"""Story generation: pipelines registered by the name conditions use for them."""

from nde.stories.baseline_a import BaselineA
from nde.stories.pipeline import PipelineFactory

PIPELINES: dict[str, PipelineFactory] = {
    "baseline_a": BaselineA.from_params,
}
