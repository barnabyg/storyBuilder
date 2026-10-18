"""Structured output: responses validated against a Pydantic model, with bounded repair.

Where the provider supports JSON-schema response mode, the request asks for it;
otherwise the schema is appended to the prompt. A response that fails
validation is sent back to the model with the specific validation errors, up to
a configurable number of repairs. Every attempt is an ordinary gateway call, so
it is cached and costed like any other.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from nde.llm.client import Message

if TYPE_CHECKING:
    from nde.llm.gateway import Completion

DEFAULT_MAX_REPAIRS = 2

_FENCED = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    value: T
    attempts: "Sequence[Completion]"

    @property
    def repaired(self) -> bool:
        return len(self.attempts) > 1


class StructuredOutputError(Exception):
    """No attempt produced a valid response before the repair limit."""

    def __init__(self, model: str, errors: list[str], attempts: "Sequence[Completion]") -> None:
        self.errors = errors
        self.attempts = attempts
        super().__init__(
            f"{model}: no valid response after {len(attempts)} attempts: {'; '.join(errors)}"
        )


def response_format(model: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": model.__name__, "schema": model.model_json_schema()},
    }


def with_schema_instruction(
    messages: tuple[Message, ...], model: type[BaseModel]
) -> tuple[Message, ...]:
    """Append the JSON schema to the final user message, for providers without schema mode."""
    schema = json.dumps(model.model_json_schema(), indent=2)
    instruction = (
        "\n\nRespond with only a JSON object, with no other text, that conforms to this "
        f"JSON Schema:\n```json\n{schema}\n```"
    )
    *head, last = messages
    return (*head, {"role": last["role"], "content": last["content"] + instruction})


def parse[T: BaseModel](text: str, model: type[T]) -> T:
    """Validate a response; raises pydantic.ValidationError (including for malformed JSON)."""
    stripped = text.strip()
    fenced = _FENCED.match(stripped)
    return model.model_validate_json(fenced.group(1) if fenced else stripped)


def describe_errors(error: ValidationError) -> list[str]:
    described: list[str] = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        described.append(f"{location}: {item['msg']}" if location else item["msg"])
    return described


def repair_messages(previous: str, errors: list[str]) -> tuple[Message, Message]:
    feedback = "\n".join(f"- {error}" for error in errors)
    return (
        {"role": "assistant", "content": previous},
        {
            "role": "user",
            "content": (
                "Your previous response did not validate against the schema:\n"
                f"{feedback}\n"
                "Reply with only the corrected JSON object."
            ),
        },
    )
