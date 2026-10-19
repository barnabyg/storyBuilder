"""The judge-panel section of a batch report: judges against the owner's blind verdicts.

Per judge (one per family on a typical panel):

- **Wins** per condition, counting only position-consistent verdicts (both
  orders preferred the same story).
- **Position-inconsistent** pairs: the two orders disagreed. They are reported,
  not averaged away, and count as disagreement with the owner.
- **Incomplete** pairs: a verdict is missing or invalid in one or both orders.
  They also count as disagreement, and are left out of position consistency.
- **Agreement with the owner**: the share of pairs where the judge's
  consistent verdict matches the owner's preference.
- **Cohen's kappa** between the owner's preferred condition and the judge's
  verdict (with "inconsistent" as its own category).
- **Position consistency**: consistent pairs as a share of the pairs judged in
  both orders.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from nde.eval.judges import ORDERS, JudgeRecord
from nde.eval.rubric import Rubric
from nde.eval.stats import cohens_kappa

INCONSISTENT = "inconsistent"
# A pair missing a valid verdict in one or both orders (an invalid or absent response).
INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class PairLabels:
    """A pair's revealed conditions and the owner's preferred condition."""

    a: str
    b: str
    owner: str

    def condition(self, side: str) -> str:
        return self.a if side == "A" else self.b


def _latest(records: Sequence[JudgeRecord]) -> list[JudgeRecord]:
    """The latest record per judge, mode, pair and order or story (later runs win)."""
    latest: dict[tuple[str, str, int, str | None], JudgeRecord] = {}
    for record in records:
        latest[record.judge, record.mode, record.pair, record.order or record.story] = record
    return list(latest.values())


def _percent(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "-"


@dataclass(frozen=True)
class _Judge:
    family: str
    label: str
    records: list[JudgeRecord]


def _pairwise_row(judge: _Judge, pairs: Sequence[PairLabels], names: Sequence[str]) -> str:
    verdicts: dict[int, dict[str, str]] = {}
    for record in judge.records:
        if record.mode == "pairwise" and record.preferred is not None and record.order:
            verdicts.setdefault(record.pair, {})[record.order] = record.preferred
    wins = dict.fromkeys(names, 0)
    judged: list[str] = []
    owner: list[str] = []
    both_orders = consistent = 0
    for number, labels in enumerate(pairs, start=1):
        orders = verdicts.get(number, {})
        if len(orders) < len(ORDERS):
            verdict = INCOMPLETE
        else:
            both_orders += 1
            if orders["AB"] == orders["BA"]:
                consistent += 1
                verdict = labels.condition(orders["AB"])
                wins[verdict] += 1
            else:
                verdict = INCONSISTENT
        judged.append(verdict)
        owner.append(labels.owner)
    agreed = sum(a == b for a, b in zip(judged, owner, strict=True))
    kappa = cohens_kappa(owner, judged)
    win_cells = " | ".join(str(wins[name]) for name in names)
    return (
        f"| {judge.family} | {judge.label} | {len(pairs)} | {win_cells} "
        f"| {both_orders - consistent} | {len(pairs) - both_orders} "
        f"| {_percent(agreed, len(pairs))} "
        f"| {'n/a' if kappa is None else f'{kappa:.2f}'} | {_percent(consistent, both_orders)} |"
    )


def _rubric_rows(
    judge: _Judge, pairs: Sequence[PairLabels], names: Sequence[str], rubric: Rubric
) -> list[str]:
    scores: dict[str, dict[str, list[int]]] = {
        name: {dimension: [] for dimension in rubric.dimension_ids} for name in names
    }
    for record in judge.records:
        if record.mode == "rubric" and record.scores and record.story:
            condition = pairs[record.pair - 1].condition(record.story)
            for dimension, value in record.scores.items():
                scores[condition][dimension].append(value)
    rows: list[str] = []
    for dimension in rubric.dimension_ids:
        cells = [
            f"{sum(v) / len(v):.2f}" if (v := scores[name][dimension]) else "-" for name in names
        ]
        rows.append(f"| {judge.family} | {judge.label} | {dimension} | {' | '.join(cells)} |")
    return rows


def judges_section(
    records: Sequence[JudgeRecord],
    pairs: Sequence[PairLabels],
    names: Sequence[str],
    rubric: Rubric | None,
) -> list[str]:
    if not records:
        return []
    latest = _latest(records)
    judges = [
        _Judge(family, label, [record for record in latest if record.judge == label])
        for family, label in sorted({(record.family, record.judge) for record in latest})
    ]
    errors = sum(record.error is not None for record in latest)
    lines = [
        "## LLM judges",
        "",
        "Judges never pass a gate: they screen and diagnose, and only the owner's blind "
        "preferences decide a gate. Agreement and kappa compare each judge with the owner's "
        "verdicts in this closed batch; position-inconsistent pairs count as disagreement.",
        "",
        f"| Family | Judge | Pairs | {' | '.join(f'{n} wins' for n in names)} "
        "| Position-inconsistent | Incomplete | Agreement with owner | Cohen's kappa "
        "| Position consistency |",
        f"| --- | --- | --- |{' --- |' * len(names)} --- | --- | --- | --- | --- |",
        *(_pairwise_row(judge, pairs, names) for judge in judges),
        "",
    ]
    if errors:
        lines += [f"{errors} judge verdicts were still invalid after their repairs.", ""]
    if rubric is not None and any(record.mode == "rubric" for record in latest):
        lines += [
            "### Judge rubric scores",
            "",
            "Mean score per condition from each judge's rubric-mode scoring.",
            "",
            f"| Family | Judge | Dimension | {' | '.join(names)} |",
            f"| --- | --- | --- |{' --- |' * len(names)}",
        ]
        for judge in judges:
            lines += _rubric_rows(judge, pairs, names, rubric)
        lines.append("")
    return lines
