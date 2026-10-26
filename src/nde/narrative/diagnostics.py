"""Shared validation results independent of any channel validator."""

from typing import Literal

from pydantic import Field

from nde.llm.config import Strict


class Diagnostic(Strict):
    code: str
    location: str
    message: str
    severity: Literal["error", "advisory", "human"] = "error"


class Validation(Strict):
    diagnostics: list[Diagnostic] = Field(default_factory=list[Diagnostic])

    @property
    def valid(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)
