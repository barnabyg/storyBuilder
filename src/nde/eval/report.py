"""The report of a closed blind batch: the key revealed, win counts and an exact binomial test."""

from pathlib import Path

from nde.eval.batch import Key, StorySource, read_closed, read_judgements, read_key
from nde.eval.stats import binomial_two_sided
from nde.files import write_text_lf

REPORT = "report.md"


def _labels(key: Key) -> dict[tuple[str, str], str]:
    """A label per source; the run is added when both sources share a condition name."""
    first, second = key.sources
    if first.condition != second.condition:
        return {(s.run, s.condition): s.condition for s in key.sources}
    return {(s.run, s.condition): f"{s.condition} ({Path(s.run).name})" for s in key.sources}


def _label(labels: dict[tuple[str, str], str], source: StorySource) -> str:
    return labels[source.run, source.condition]


def render_report(batch: Path) -> str:
    """The batch report. Raises BatchError while the batch is open."""
    key = read_key(batch)
    judgements = read_judgements(batch)
    labels = _labels(key)
    names = [_label(labels, source) for source in key.sources]
    wins = dict.fromkeys(names, 0)
    rows: list[str] = []
    for number, pair in enumerate(key.pairs, start=1):
        judgement = judgements[number]
        chosen = pair.a if judgement.preferred == "A" else pair.b
        winner = _label(labels, chosen)
        wins[winner] += 1
        rows.append(
            f"| {number} | {pair.brief} | {pair.repetition + 1} | {_label(labels, pair.a)} "
            f"| {_label(labels, pair.b)} | {winner} | {judgement.strength} |"
        )
    closed = read_closed(batch)
    total = len(key.pairs)
    first = names[0]
    p_value = binomial_two_sided(wins[first], total)
    lines = [
        f"# Blind batch report: {batch.name}",
        "",
        f"- **Pairs:** {total}",
        "- **Sources:** " + "; ".join(f"{_label(labels, s)} from `{s.run}`" for s in key.sources),
        f"- **Closed:** {closed.closed_at if closed else '-'}",
        "",
        "## Preferences",
        "",
        "| Condition | Wins | Share |",
        "| --- | --- | --- |",
        *(f"| {name} | {wins[name]} | {wins[name] / total:.0%} |" for name in names),
        "",
        f"Two-sided exact binomial test against no preference (p = 0.5): "
        f"p = {p_value:.4g} ({wins[first]} of {total} pairs for {first}).",
        "",
        "## Pairs (key revealed)",
        "",
        "| Pair | Brief | Rep | Story A | Story B | Preferred | Strength |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
    ]
    return "\n".join(lines)


def write_report(batch: Path) -> str:
    report = render_report(batch)
    write_text_lf(batch / REPORT, report)
    return report
