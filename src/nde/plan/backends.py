"""Selectable structural-revision backends behind the shared mask/lock operator contract.

An omitted backend is the released D operator with unchanged provenance. Naming a
backend adds versioned `backend` metadata to each revision's provenance.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from nde.narrative.base import Strict
from nde.runs.config import PromptRef

INPAINTING_TEXT = PromptRef(template="plan_inpainting_text", version=1)
type Policy = Literal["hard", "soft", "excluded"]


class DRevisionBackend(Strict):
    """The LLM planner returns a complete candidate under the explicit mask."""

    version: Literal["1"] = "1"
    kind: Literal["d"]


class ERevisionBackend(Strict):
    """Conditional denoiser inpainting of masked structure, then authorised text completion."""

    version: Literal["1"] = "1"
    kind: Literal["e"]
    checkpoint: Path
    # The approved channel policy of the checkpoint's training targets. Channels it
    # excludes are not represented by E and stay as the source has them.
    target_policy: dict[str, Policy] = Field(min_length=1)
    # Validated as the registered `Conditions` object before any call; unknown by default.
    conditions: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    guidance_scale: float = Field(default=1, ge=0, le=8, allow_inf_nan=False)
    max_attempts: int = Field(default=3, ge=1, le=100)
    max_tokens: int = Field(default=1000000, ge=1)
    max_attention_cells: int = Field(default=100000000, ge=1)
    max_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    text_prompt: PromptRef = Field(default_factory=lambda: INPAINTING_TEXT)


type RevisionBackend = Annotated[DRevisionBackend | ERevisionBackend, Field(discriminator="kind")]
