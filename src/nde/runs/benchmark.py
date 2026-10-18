"""The G0 model benchmark: compare candidate models on structured-output reliability.

Rows are computed from a run's call log and structured-output metrics, so a
report can be regenerated from recorded responses without any network access.
"""

import math
import statistics
from collections.abc import Mapping, Sequence

from nde.runs.records import BenchmarkRow, CallRecord, StructuredMetrics

_PER_THOUSAND = 1000
_P95 = 0.95


def _p95(values: Sequence[float]) -> float:
    """Nearest-rank 95th percentile."""
    ordered = sorted(values)
    return ordered[math.ceil(_P95 * len(ordered)) - 1]


def _row(
    model: str, calls: Sequence[CallRecord], metrics: StructuredMetrics, threshold: float
) -> BenchmarkRow:
    latencies: dict[tuple[str, int], float] = {}
    for call in calls:
        key = (call.call, call.repetition)
        latencies[key] = latencies.get(key, 0.0) + call.latency_s
    output_tokens = sum(call.output_tokens for call in calls)
    token_cost = sum(call.token_cost_usd for call in calls)
    return BenchmarkRow(
        model=model,
        calls=metrics.calls,
        first_attempt_validity=metrics.first_attempt_validity,
        validity_after_repair=metrics.validity_after_repair,
        repair_rate=metrics.repair_rate,
        latency_median_s=statistics.median(latencies.values()) if latencies else None,
        latency_p95_s=_p95(list(latencies.values())) if latencies else None,
        cost_per_1k_output_usd=(
            token_cost / output_tokens * _PER_THOUSAND if output_tokens else None
        ),
        token_cost_usd=token_cost,
        new_spend_usd=sum(call.cost_usd for call in calls),
        meets_threshold=metrics.calls > 0 and metrics.validity_after_repair >= threshold,
    )


def benchmark_rows(
    calls: Sequence[CallRecord],
    structured: Mapping[str, StructuredMetrics],
    *,
    threshold: float,
    models: Sequence[str] | None = None,
) -> list[BenchmarkRow]:
    """One row per model, in `models` order (default: order of first call)."""
    by_model: dict[str, list[CallRecord]] = {}
    for call in calls:
        by_model.setdefault(f"{call.provider}/{call.model}", []).append(call)
    order = list(models) if models is not None else list(by_model)
    return [
        _row(model, by_model.get(model, []), structured.get(model, StructuredMetrics()), threshold)
        for model in order
    ]
