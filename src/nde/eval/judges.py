"""The LLM judge panel: models that judge a blind batch, for screening and diagnosis.

Judges see only the batch's normalised story text, exactly as the reader does.

- **Pairwise mode:** every judge gives a forced-choice preference with strength
  for every pair, in both orders: as shown (story A first, `AB`) and swapped
  (`BA`). A pair whose two orders disagree is position-inconsistent.
- **Rubric mode:** every judge scores every story against the batch's rubric
  and assigns failure tags.

A judge's model family must differ from the families of the planner and
prose models that generated the batch, and the panel needs at least two
families. Results are recorded by pair number and shown position, never by
condition, in the run directory (`judgements.jsonl`) and in the batch
(`judges/<run>.jsonl`), so they cannot reveal the key to the reader.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from nde.eval.batch import SIDES, BatchError, Side, View, batch_dir, read_generators, read_view
from nde.eval.rubric import Rubric
from nde.files import write_text_lf
from nde.llm.config import ModelBinding
from nde.llm.gateway import Gateway, Prompt
from nde.llm.structured import StructuredOutputError
from nde.runs.config import ConfigError, JudgeMode, LoadedConfig, PromptRef
from nde.runs.prompts import PromptTemplate, require_template
from nde.runs.records import StructuredMetrics
from nde.runs.state import Attribution, RunState, record_calls_since

JUDGES_DIR = "judges"
JUDGEMENTS = "judgements.jsonl"

type Order = Literal["AB", "BA"]
ORDERS: tuple[Order, Order] = ("AB", "BA")

PAIRWISE_VARIABLES = frozenset({"story_1", "story_2"})
RUBRIC_VARIABLES = frozenset({"rubric", "story", "scale_min", "scale_max"})


class PairwiseVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred: int = Field(ge=1, le=2, description="1 or 2: the better story")
    strength: int = Field(ge=1, le=3, description="1: slightly, 2: clearly, 3: much better")
    reason: str = Field(min_length=1, description="one sentence")


def rubric_verdict_model(rubric: Rubric) -> type[BaseModel]:
    """A response model with one score field per rubric dimension and the taxonomy's tags."""
    scale = rubric.scale
    # The taxonomy's tag IDs as an enum, so the schema lists them and others are invalid.
    tag = StrEnum("FailureTag", [(tag_id.replace("-", "_"), tag_id) for tag_id in rubric.tag_ids])
    fields: dict[str, Any] = {
        dimension: (int, Field(ge=scale.min, le=scale.max)) for dimension in rubric.dimension_ids
    }
    fields["failure_tags"] = (list[tag], Field(default_factory=list[tag]))
    return create_model("RubricVerdict", __config__=ConfigDict(extra="forbid"), **fields)


class JudgeRecord(BaseModel):
    """One judge's verdict on one pair (pairwise) or one story (rubric). Never names a condition."""

    model_config = ConfigDict(extra="forbid")

    judge: str
    family: str
    mode: JudgeMode
    pair: int
    # Pairwise: the order the stories were given in; the preferred shown story and strength.
    order: Order | None = None
    preferred: Side | None = None
    strength: int | None = None
    # Rubric: which shown story was scored, its scores and failure tags.
    story: Side | None = None
    scores: dict[str, int] | None = None
    tags: list[str] | None = None
    # Set when the judge's output stayed invalid after its repairs.
    error: str | None = None
    # The judge shares a family with the batch's generators (a waived family rule).
    shares_generator_family: bool = False


@dataclass(frozen=True)
class PreparedJudges:
    batch: Path
    view: View
    panel: list[ModelBinding]
    modes: list[JudgeMode]
    pairwise_prompt: PromptRef
    rubric_prompt: PromptRef
    # Judges allowed to share a generator's family (judges.allow_generator_families).
    shared_family: frozenset[str] = frozenset()

    @property
    def waivers(self) -> list[str]:
        """One warning per judge that shares a family with the batch's generators."""
        return [
            f"family separation waived: judge {judge.label} shares the family "
            f"{judge.family!r} with a model that generated batch {self.batch.name!r}, so "
            "self-preference bias is not controlled for it"
            for judge in self.panel
            if judge.label in self.shared_family
        ]


def prepare_judges(
    loaded: LoadedConfig, templates: dict[str, PromptTemplate]
) -> PreparedJudges | None:
    """Check the batch, the family rule and the templates; raises ConfigError before any call."""
    judges = loaded.config.judges
    batches = loaded.batches_dir
    if judges is None or batches is None:
        return None
    try:
        batch = batch_dir(batches, judges.batch)
        view = read_view(batch)
        if view.kind != "pairs":
            raise ConfigError(f"judges: batch {judges.batch!r} is not a pair batch")
        generators = read_generators(batch)
    except (BatchError, OSError, ValidationError) as error:
        raise ConfigError(f"judges: batch {judges.batch!r} cannot be read: {error}") from error
    if not generators:
        raise ConfigError(f"judges: batch {judges.batch!r} records no generator models")
    families: set[str] = set()
    for generator in generators:
        if generator.family is None:
            raise ConfigError(
                f"judges: the family of generator {generator.label} is not recorded; set "
                "`family` on its role in the generating config, re-run it (the cache makes "
                "that free) and rebuild the batch"
            )
        families.add(generator.family.lower())
    shared = frozenset(j.label for j in judges.panel if j.family in families)
    for judge in judges.panel:
        if judge.label in shared and not judges.allow_generator_families:
            raise ConfigError(
                f"judges: judge {judge.label} is in the family {judge.family!r}, like a model "
                f"that generated the batch ({', '.join(sorted(families))}); judges must come "
                "from other families"
            )
    if "rubric" in judges.modes and view.rubric is None:
        raise ConfigError(f"judges: batch {judges.batch!r} has no rubric for rubric mode")
    needs = [(judges.pairwise_prompt, PAIRWISE_VARIABLES), (judges.rubric_prompt, RUBRIC_VARIABLES)]
    for ref, variables in needs:
        require_template(templates, loaded.prompts_dir, ref, dict.fromkeys(variables, "x"))
    return PreparedJudges(
        batch,
        view,
        judges.panel,
        judges.modes,
        judges.pairwise_prompt,
        judges.rubric_prompt,
        shared,
    )


@dataclass
class JudgeRun:
    gateway: Gateway
    prepared: PreparedJudges
    templates: dict[str, PromptTemplate]
    state: RunState
    seed: int
    max_repairs: int
    records: list[JudgeRecord]

    def _ask[T: BaseModel](
        self, judge: ModelBinding, call: str, step: JudgeMode, prompt: Prompt, model: type[T]
    ) -> T | str:
        """One structured judge call; the validated verdict, or the error once repairs run out."""
        attribution = Attribution(call=call, repetition=0, step=step)
        self.state.current_call = call
        metrics = self.state.structured.setdefault(judge.label, StructuredMetrics())
        first = len(self.gateway.history)
        try:
            result = self.gateway.structured(
                judge, prompt, model, seed=self.seed, max_repairs=self.max_repairs
            )
        except StructuredOutputError as failure:
            metrics.record(attempts=len(failure.attempts), valid=False)
            return "; ".join(failure.errors)
        finally:
            record_calls_since(self.gateway, self.state, first, attribution)
        metrics.record(attempts=len(result.attempts), valid=True)
        return result.value

    def _prompt(self, ref: PromptRef, variables: dict[str, str]) -> Prompt:
        template = self.templates[ref.label]
        return Prompt(template.name, template.version, template.render(variables))

    def pairwise(self, judge: ModelBinding, number: int) -> None:
        pair = self.prepared.view.pairs[number - 1]
        shown = {"A": pair.a, "B": pair.b}
        for order in ORDERS:
            first, second = SIDES if order == "AB" else SIDES[::-1]
            prompt = self._prompt(
                self.prepared.pairwise_prompt,
                {"story_1": shown[first], "story_2": shown[second]},
            )
            call = f"judge/{judge.label}/pair-{number}/{order}"
            verdict = self._ask(judge, call, "pairwise", prompt, PairwiseVerdict)
            record = JudgeRecord(
                judge=judge.label,
                family=judge.family or "",
                mode="pairwise",
                shares_generator_family=judge.label in self.prepared.shared_family,
                pair=number,
                order=order,
            )
            if isinstance(verdict, str):
                record.error = verdict
            else:
                record.preferred = first if verdict.preferred == 1 else second
                record.strength = verdict.strength
            self.records.append(record)

    def rubric(self, judge: ModelBinding, number: int, rubric: Rubric) -> None:
        pair = self.prepared.view.pairs[number - 1]
        model = rubric_verdict_model(rubric)
        for side, text in (("A", pair.a), ("B", pair.b)):
            prompt = self._prompt(
                self.prepared.rubric_prompt,
                {
                    "rubric": rubric.describe(),
                    "story": text,
                    "scale_min": str(rubric.scale.min),
                    "scale_max": str(rubric.scale.max),
                },
            )
            call = f"judge/{judge.label}/pair-{number}/{side}"
            verdict = self._ask(judge, call, "rubric", prompt, model)
            record = JudgeRecord(
                judge=judge.label,
                family=judge.family or "",
                mode="rubric",
                shares_generator_family=judge.label in self.prepared.shared_family,
                pair=number,
                story=cast("Side", side),
            )
            if isinstance(verdict, str):
                record.error = verdict
            else:
                values = verdict.model_dump()
                record.scores = {d: int(values[d]) for d in rubric.dimension_ids}
                record.tags = sorted(str(tag) for tag in values["failure_tags"])
            self.records.append(record)


def run_judges(run: JudgeRun) -> None:
    """Every pair, judges taking turns, so a budget stop leaves comparable verdicts."""
    prepared = run.prepared
    rubric = prepared.view.rubric
    for number in range(1, len(prepared.view.pairs) + 1):
        for judge in prepared.panel:
            if "pairwise" in prepared.modes:
                run.pairwise(judge, number)
            if "rubric" in prepared.modes and rubric is not None:
                run.rubric(judge, number, rubric)


def write_records(run_dir: Path, batch: Path, records: list[JudgeRecord]) -> None:
    text = "".join(record.model_dump_json() + "\n" for record in records)
    write_text_lf(run_dir / JUDGEMENTS, text)
    (batch / JUDGES_DIR).mkdir(exist_ok=True)
    write_text_lf(batch / JUDGES_DIR / f"{run_dir.name}.jsonl", text)


def read_records(batch: Path) -> list[JudgeRecord]:
    """Every judge run's records, oldest run first (run directories sort by start time)."""
    folder = batch / JUDGES_DIR
    if not folder.is_dir():
        return []
    return [
        JudgeRecord.model_validate_json(line)
        for path in sorted(folder.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
