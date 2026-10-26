"""Optional CPU attention backbone; importing this module does not import torch."""

import importlib
import math
from dataclasses import dataclass
from typing import Any, cast

from nde.reconstruction.codec import VOCABULARY_SIZE


def runtime() -> Any:
    try:
        return cast(Any, importlib.import_module("torch"))
    except ImportError as error:
        raise ValueError(
            "CPU ML runtime unavailable; install the pinned ml dependency group"
        ) from error


@dataclass
class Backbone:
    """One residual attention/MLP block with explicit causal or bidirectional attention."""

    weights: dict[str, Any]
    width: int
    max_positions: int

    def parameters(self) -> list[Any]:
        return list(self.weights.values())

    def forward(self, tokens: list[int], *, causal: bool, context_length: int = 0) -> Any:
        torch = runtime()
        if not tokens or len(tokens) > self.max_positions:
            raise ValueError("model position capacity exceeded; truncation is forbidden")
        ids = torch.tensor(tokens, dtype=torch.long)
        weight = self.weights
        hidden = weight["embedding"][ids] + weight["position"][: len(tokens)]
        query = hidden[context_length:] @ weight["query"]
        key, value = (hidden @ weight[name] for name in ("key", "value"))
        scores = query @ key.T / math.sqrt(self.width)
        if causal:
            future = (
                torch.arange(len(tokens))[None, :]
                > torch.arange(context_length, len(tokens))[:, None]
            )
            scores = scores.masked_fill(future, float("-inf"))
        hidden = (
            hidden[context_length:]
            + torch.softmax(scores, dim=-1) @ value @ weight["attention_output"]
        )
        hidden = torch.nn.functional.layer_norm(hidden, (self.width,))
        hidden = hidden + torch.nn.functional.gelu(hidden @ weight["ff_in"]) @ weight["ff_out"]
        return hidden @ weight["output"]

    def state(self) -> dict[str, Any]:
        return {name: tensor.detach().cpu().tolist() for name, tensor in self.weights.items()}


def build(
    width: int, max_positions: int, seed: int, state: dict[str, Any] | None = None
) -> Backbone:
    torch = runtime()
    torch.manual_seed(seed)
    shapes = {
        "embedding": (VOCABULARY_SIZE, width),
        "position": (max_positions, width),
        "query": (width, width),
        "key": (width, width),
        "value": (width, width),
        "attention_output": (width, width),
        "ff_in": (width, width * 2),
        "ff_out": (width * 2, width),
        "output": (width, VOCABULARY_SIZE),
    }
    weights = {
        name: torch.nn.Parameter(
            torch.tensor(state[name], dtype=torch.float32)
            if state is not None
            else torch.randn(shape, dtype=torch.float32) * 0.02
        )
        for name, shape in shapes.items()
    }
    if any(tuple(weights[name].shape) != shape for name, shape in shapes.items()):
        raise ValueError("local backbone weight shapes differ from the registered architecture")
    return Backbone(weights, width, max_positions)


def pack(value: Any) -> Any:
    torch = runtime()
    if isinstance(value, torch.Tensor):
        return {"tensor": value.detach().cpu().tolist(), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): pack(item) for key, item in cast(dict[Any, Any], value).items()}
    if isinstance(value, (list, tuple)):
        return [pack(item) for item in cast(list[Any], value)]
    return value


def unpack(value: Any) -> Any:
    torch = runtime()
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        if set(mapping) == {"tensor", "dtype"}:
            dtype = {"torch.float32": torch.float32, "torch.int64": torch.int64}.get(
                mapping["dtype"]
            )
            if dtype is None:
                raise ValueError("unsupported checkpoint tensor dtype")
            return torch.tensor(mapping["tensor"], dtype=dtype)
        return {key: unpack(item) for key, item in mapping.items()}
    if isinstance(value, list):
        return [unpack(item) for item in cast(list[Any], value)]
    return value
