"""The report of a closed blind batch: the key revealed, preferences and leakage checks.

Every batch reports win counts with an exact binomial test. Batches scored with
a rubric add rubric means per condition with bootstrap confidence intervals,
failure-tag counts and the accuracy of the reader's condition guesses. Batches
with judge-panel runs add the judges' results against the owner's verdicts.
"""

from collections import Counter
from pathlib import Path

from nde.eval.batch import (
    Judgement,
    Key,
    StorySource,
    View,
    labels,
    read_closed,
    read_judgements,
    read_key,
    read_view,
)
from nde.eval.calibration import render_calibration_report
from nde.eval.judge_report import PairLabels, judges_section
from nde.eval.judges import read_records
from nde.eval.rubric import Rubric
from nde.eval.sections import BOOTSTRAP_SEED, TIE, mean_ci, pick, warnings_section
from nde.eval.stats import binomial_two_sided
from nde.files import write_text_lf

REPORT = "report.md"
# Guesses right in more than this share of pairs flag the batch as possibly unblinded.
UNBLINDED_THRESHOLD = 0.70


class _Revealed:
    """A closed batch with its key: each shown story's condition label."""

    def __init__(self, batch: Path) -> None:
        self.key: Key = read_key(batch)
        self.view: View = read_view(batch)
        self.judgements: dict[int, Judgement] = read_judgements(batch)
        self._labels = labels(self.key.sources)
        self.names = [self.label(source) for source in self.key.sources]

    def label(self, source: StorySource) -> str:
        return self._labels[source.run, source.condition]

    def numbered(self) -> list[tuple[int, Judgement, str, str]]:
        """(pair number, judgement, condition of A, condition of B) for every pair."""
        return [
            (number, self.judgements[number], self.label(pair.a), self.label(pair.b))
            for number, pair in enumerate(self.key.pairs, start=1)
        ]


def _table(names: list[str], first: str) -> list[str]:
    return [f"| {first} | {' | '.join(names)} |", f"| --- |{' --- |' * len(names)}"]


def _owner_pick(judgement: Judgement, a: str, b: str, primary: str | None) -> str | None:
    """The condition whose story the owner scored higher on the primary dimension."""
    if primary is None or judgement.a is None or judgement.b is None:
        return None
    return pick(a, b, judgement.a.scores.get(primary), judgement.b.scores.get(primary))


def _primary_section(revealed: _Revealed, primary: str) -> list[str]:
    """The gate's primary measure: in each pair, which story scored higher on `primary`."""
    picks = [_owner_pick(j, a, b, primary) for _, j, a, b in revealed.numbered()]
    counts = {name: picks.count(name) for name in [*revealed.names, TIE]}
    total = len(picks)
    first = revealed.names[0]
    p_value = binomial_two_sided(counts[first], total)
    return [
        f"## Primary measure: {primary}",
        "",
        f"For each pair, the condition whose story the owner scored higher on {primary}; "
        "equal scores are a tie. This leads the report because the gate is decided on it.",
        "",
        "| Pick | Pairs |",
        "| --- | --- |",
        *(f"| {name} | {counts[name]} |" for name in [*revealed.names, TIE]),
        "",
        f"Two-sided exact binomial test against no difference (p = 0.5), ties counting "
        f"against {first}: p = {p_value:.4g} ({counts[first]} of {total} pairs for {first}).",
        "",
    ]


def _preferences(revealed: _Revealed) -> list[str]:
    wins = dict.fromkeys(revealed.names, 0)
    for _, judgement, a, b in revealed.numbered():
        wins[a if judgement.preferred == "A" else b] += 1
    total = len(revealed.key.pairs)
    first = revealed.names[0]
    p_value = binomial_two_sided(wins[first], total)
    return [
        "## Preferences",
        "",
        "| Condition | Wins | Share |",
        "| --- | --- | --- |",
        *(f"| {name} | {wins[name]} | {wins[name] / total:.0%} |" for name in revealed.names),
        "",
        f"Two-sided exact binomial test against no preference (p = 0.5): "
        f"p = {p_value:.4g} ({wins[first]} of {total} pairs for {first}).",
        "",
    ]


def _rubric_section(revealed: _Revealed, rubric: Rubric) -> list[str]:
    scores: dict[str, dict[str, list[int]]] = {
        name: {dimension: [] for dimension in rubric.dimension_ids} for name in revealed.names
    }
    for _, judgement, a, b in revealed.numbered():
        for condition, story in ((a, judgement.a), (b, judgement.b)):
            if story is not None:
                for dimension, value in story.scores.items():
                    scores[condition][dimension].append(value)
    lines = [
        f"## Rubric scores (rubric v{rubric.version})",
        "",
        f"Mean score per condition on the {rubric.scale.min}-{rubric.scale.max} scale, with a "
        f"95% percentile bootstrap interval (10,000 resamples, seed {BOOTSTRAP_SEED}).",
        "",
        *_table(revealed.names, "Dimension"),
    ]
    for dimension in rubric.dimension_ids:
        cells = [mean_ci(scores[name][dimension]) for name in revealed.names]
        lines.append(f"| {dimension} | {' | '.join(cells)} |")
    return [*lines, ""]


def _tags_section(revealed: _Revealed, rubric: Rubric) -> list[str]:
    counts = {name: Counter[str]() for name in revealed.names}
    free = {name: Counter[str]() for name in revealed.names}
    for _, judgement, a, b in revealed.numbered():
        for condition, story in ((a, judgement.a), (b, judgement.b)):
            if story is not None:
                counts[condition].update(story.tags)
                free[condition].update(tag.lower() for tag in story.free_tags)
    lines = [
        "## Failure tags",
        "",
        "Stories tagged with each failure, per condition.",
        "",
        *_table(revealed.names, "Tag"),
        *(
            f"| {tag} | {' | '.join(str(counts[name][tag]) for name in revealed.names)} |"
            for tag in rubric.tag_ids
        ),
        "",
    ]
    free_tags = sorted({tag for counter in free.values() for tag in counter})
    if free_tags:
        lines += [
            "Free-text tags:",
            "",
            *_table(revealed.names, "Tag"),
            *(
                f"| {tag} | {' | '.join(str(free[name][tag]) for name in revealed.names)} |"
                for tag in free_tags
            ),
            "",
        ]
    return lines


def _guess_section(revealed: _Revealed) -> list[str]:
    guesses = [(j.guess, a) for _, j, a, _ in revealed.numbered() if j.guess is not None]
    right = sum(guess == actual for guess, actual in guesses)
    accuracy = right / len(guesses)
    lines = [
        "## Condition guesses",
        "",
        f"{right} of {len(guesses)} guesses right ({accuracy:.0%}) about which condition story "
        "A came from. Chance is 50%.",
        "",
    ]
    if accuracy > UNBLINDED_THRESHOLD:
        lines += [
            f"**Possibly unblinded:** guesses were right in more than "
            f"{UNBLINDED_THRESHOLD:.0%} of pairs, so the reader may have recognised the "
            "conditions. Treat the preferences with caution.",
            "",
        ]
    return lines


def _pairs_section(revealed: _Revealed) -> list[str]:
    lines = [
        "## Pairs (key revealed)",
        "",
        "| Pair | Brief | Rep | Story A | Story B | Preferred | Strength | Guess for A |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (number, judgement, a, b), pair in zip(
        revealed.numbered(), revealed.key.pairs, strict=True
    ):
        winner = a if judgement.preferred == "A" else b
        lines.append(
            f"| {number} | {pair.brief} | {pair.repetition + 1} | {a} | {b} | {winner} "
            f"| {judgement.strength} | {judgement.guess or '-'} |"
        )
    return [*lines, ""]


def render_report(batch: Path) -> str:
    """The batch report. Raises BatchError while the batch is open."""
    revealed = _Revealed(batch)
    closed = read_closed(batch)
    key = revealed.key
    sources = "; ".join(f"{revealed.label(s)} from `{s.run}`" for s in key.sources)
    lines = [
        f"# Blind batch report: {batch.name}",
        "",
        f"- **Pairs:** {len(key.pairs)}",
        f"- **Sources:** {sources}",
        f"- **Rubric:** {f'v{key.rubric_version}' if key.rubric_version else 'none'}",
        "- **Marks:** preference and strength required; rubric scores, tags and the guess "
        "optional, so the secondary sections summarise only what was given"
        if revealed.view.optional_scores
        else "- **Marks:** all required",
        f"- **Closed:** {closed.closed_at if closed else '-'}",
        "",
    ]
    lines += warnings_section(batch)
    primary = revealed.view.primary_dimension
    if primary is not None:
        lines += _primary_section(revealed, primary)
    lines += _preferences(revealed)
    rubric = revealed.view.rubric
    if rubric is not None:
        lines += _rubric_section(revealed, rubric)
        lines += _tags_section(revealed, rubric)
    if any(judgement.guess is not None for judgement in revealed.judgements.values()):
        lines += _guess_section(revealed)
    pairs = [
        PairLabels(
            a=a,
            b=b,
            owner=a if judgement.preferred == "A" else b,
            owner_primary=_owner_pick(judgement, a, b, primary),
        )
        for _, judgement, a, b in revealed.numbered()
    ]
    lines += judges_section(read_records(batch), pairs, revealed.names, rubric, primary)
    lines += _pairs_section(revealed)
    return "\n".join(lines)


def write_report(batch: Path) -> str:
    view = read_view(batch)
    report = (
        render_calibration_report(batch, view) if view.kind == "single" else render_report(batch)
    )
    write_text_lf(batch / REPORT, report)
    return report
