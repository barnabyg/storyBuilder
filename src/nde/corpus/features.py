"""Deterministic counts and explicitly limited textual heuristics; no NLP downloads."""

import html
import json
import re
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Literal

from pydantic import Field, TypeAdapter

from nde.corpus.acquisition import CorpusError, json_digest, load_work, sha256
from nde.corpus.artifacts import publish
from nde.corpus.segmentation import load_segmentation
from nde.llm.config import Strict

_WORD = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*|\d+", re.UNICODE)
_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "st", "sr", "jr", "etc"}
_FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
_THIRD_PERSON = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}
_ALIASES = TypeAdapter(dict[str, list[str]])


class Observation(Strict):
    start: int
    end: int
    text: str


class Mention(Observation):
    entity_id: str | None
    candidates: list[str]


class Measurements(Strict):
    tokens: int
    sentence_lengths: list[int]
    mean_sentence_tokens: float | None
    median_sentence_tokens: float | None
    quotation_status: Literal["balanced", "unbalanced"]
    quotations: list[Observation]
    dialogue_tokens: int
    dialogue_proportion: float | None
    entity_status: Literal["measured", "unsupported-no-aliases"]
    mentions: list[Mention]
    cooccurrence: dict[str, int] | None
    first_person_tokens: int
    third_person_tokens: int
    viewpoint: Literal["first-person-cue", "third-person-cue", "mixed-cues", "unknown"]
    viewpoint_evidence: list[Observation]


class SceneFeatures(Strict):
    scene_id: str
    start: int
    end: int
    measurements: Measurements


class Features(Strict):
    version: Literal["1"] = "1"
    work_id: str
    source_id: str
    text_sha256: str
    segmentation_sha256: str
    method: Literal["text-features@1"] = "text-features@1"
    offset_basis: Literal["unicode-codepoints-lf"] = "unicode-codepoints-lf"
    aliases_path: str | None
    aliases_sha256: str | None
    aliases: dict[str, list[str]] = Field(default_factory=dict[str, list[str]])
    work: Measurements
    scenes: list[SceneFeatures]
    sha256: str


def _quotations(text: str) -> tuple[list[tuple[int, int]], bool]:
    spans: list[tuple[int, int]] = []
    balanced = True
    for opener, closer in [('"', '"'), ("“", "”"), ("\u2018", "\u2019")]:
        opening: int | None = None
        for index, character in enumerate(text):
            if character == opener and opening is None:
                opening = index + 1
            elif character == closer:
                if opening is None:
                    # A curly apostrophe inside a word is punctuation, not a closing quote.
                    if closer != "\u2019" or not index or not text[index - 1].isalnum():
                        balanced = False
                elif closer != "\u2019" or index + 1 == len(text) or not text[index + 1].isalnum():
                    spans.append((opening, index))
                    opening = None
        balanced = balanced and opening is None
    spans.extend(
        (match.start(1), match.end(1))
        for match in re.finditer(r"(?<!\w)'(?!\s)(.*?)'(?!\w)", text, re.DOTALL)
    )
    for opener in re.finditer(r"(?<!\w)'(?=\w)", text):
        if not any(start - 1 <= opener.start() <= end for start, end in spans):
            balanced = False
    return sorted(spans), balanced


def _sentence_lengths(text: str) -> list[int]:
    lengths: list[int] = []
    start = 0
    for match in re.finditer(r"[.!?]+", text):
        previous = list(_WORD.finditer(text[start : match.start()]))
        if match.group() == "." and previous and previous[-1].group().casefold() in _ABBREVIATIONS:
            continue
        if (
            match.group() == "."
            and match.start()
            and text[match.start() - 1].isdigit()
            and match.end() < len(text)
            and text[match.end()].isdigit()
        ):
            continue
        count = len(_WORD.findall(text[start : match.end()]))
        if count:
            lengths.append(count)
        start = match.end()
    remaining = len(_WORD.findall(text[start:]))
    if remaining:
        lengths.append(remaining)
    return lengths


def _mentions(text: str, offset: int, aliases: dict[str, list[str]]) -> list[Mention]:
    identities: dict[str, set[str]] = {}
    for entity, names in aliases.items():
        for name in names:
            if not entity.strip() or not name.strip() or "|" in entity:
                raise CorpusError(
                    "entity identities and aliases must be nonempty; ids cannot contain |"
                )
            identities.setdefault(name.casefold(), set()).add(entity)
    candidates: list[tuple[int, int, list[str]]] = []
    for name, entities in sorted(identities.items()):
        for match in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE):
            candidates.append((match.start(), match.end(), sorted(entities)))
    result: list[Mention] = []
    cursor = 0
    for start, end, entities in sorted(candidates, key=lambda row: (row[0], -row[1])):
        if start < cursor:
            continue
        result.append(
            Mention(
                start=start + offset,
                end=end + offset,
                text=text[start:end],
                entity_id=entities[0] if len(entities) == 1 else None,
                candidates=entities,
            )
        )
        cursor = end
    return result


def _measure(text: str, offset: int, aliases: dict[str, list[str]]) -> Measurements:
    words = list(_WORD.finditer(text))
    quotations, balanced = _quotations(text)
    dialogue = sum(
        any(start <= word.start() and word.end() <= end for start, end in quotations)
        for word in words
    )
    mentions = _mentions(text, offset, aliases)
    entities = sorted({m.entity_id for m in mentions if m.entity_id is not None})
    lengths = _sentence_lengths(text)
    first = sum(word.group().casefold() in _FIRST_PERSON for word in words)
    third = sum(word.group().casefold() in _THIRD_PERSON for word in words)
    viewpoint = (
        "mixed-cues"
        if first and third
        else "first-person-cue"
        if first
        else "third-person-cue"
        if third
        else "unknown"
    )
    return Measurements(
        tokens=len(words),
        sentence_lengths=lengths,
        mean_sentence_tokens=statistics.mean(lengths) if lengths else None,
        median_sentence_tokens=statistics.median(lengths) if lengths else None,
        quotation_status="balanced" if balanced else "unbalanced",
        quotations=[
            Observation(start=start + offset, end=end + offset, text=text[start:end])
            for start, end in quotations
        ],
        dialogue_tokens=dialogue,
        dialogue_proportion=dialogue / len(words) if words and balanced else None,
        entity_status="measured" if aliases else "unsupported-no-aliases",
        mentions=mentions,
        cooccurrence={"|".join(pair): 1 for pair in combinations(entities, 2)} if aliases else None,
        first_person_tokens=first,
        third_person_tokens=third,
        viewpoint=viewpoint,
        viewpoint_evidence=[
            Observation(start=w.start() + offset, end=w.end() + offset, text=w.group())
            for w in words
            if w.group().casefold() in _FIRST_PERSON | _THIRD_PERSON
        ],
    )


def _build(work_dir: Path, segmentation_path: Path, aliases_path: Path | None) -> Features:
    work, text = load_work(work_dir)
    segments = load_segmentation(segmentation_path, work_dir)
    aliases = (
        _ALIASES.validate_python(json.loads(aliases_path.read_text("utf-8")))
        if aliases_path
        else {}
    )
    scenes = [
        SceneFeatures(
            scene_id=s.id,
            start=s.start,
            end=s.end,
            measurements=_measure(text[s.start : s.end], s.start, aliases),
        )
        for s in segments.scenes
    ]
    measured = _measure(text, 0, aliases)
    pairs: Counter[str] = Counter()
    for scene in scenes:
        pairs.update(scene.measurements.cooccurrence or {})
    measured = measured.model_copy(
        update={"cooccurrence": dict(sorted(pairs.items())) if aliases else None}
    )
    value = Features(
        work_id=work.work_id,
        source_id=work.source_id,
        text_sha256=work.text_sha256,
        segmentation_sha256=segments.sha256,
        aliases_path=str(aliases_path.resolve()) if aliases_path else None,
        aliases_sha256=sha256(aliases_path.read_bytes()) if aliases_path else None,
        aliases=aliases,
        work=measured,
        scenes=scenes,
        sha256="",
    )
    return value.model_copy(update={"sha256": json_digest(value)})


def extract_features(
    work_dir: Path, segmentation_path: Path, out: Path, aliases_path: Path | None = None
) -> Features:
    value = _build(work_dir, segmentation_path, aliases_path)
    publish(
        out,
        {
            "features.json": value.model_dump_json(indent=2) + "\n",
            "report.html": "<!doctype html><meta charset='utf-8'>"
            "<h1>Deterministic textual features</h1><p>Unicode word tokens include internal "
            "apostrophes; sentences split on .!? except registered common abbreviations and "
            "decimals. Dialogue counts tokens inside balanced straight/curly quotations, "
            "not proven speech. Entity aliases use case-insensitive longest whole-name "
            "matching; ambiguity stays unresolved. Cooccurrence counts shared scenes. "
            "Pronouns are viewpoint heuristic cues, including quoted speech, not literary "
            "truth. Missing denominators are null; observed zeros remain zero.</p><pre>"
            + html.escape(value.model_dump_json(indent=2))
            + "</pre>",
        },
    )
    return value


def load_features(path: Path, work_dir: Path, segmentation_path: Path) -> Features:
    file = path / "features.json" if path.is_dir() else path
    value = Features.model_validate_json(file.read_text("utf-8"))
    if value != _build(
        work_dir, segmentation_path, Path(value.aliases_path) if value.aliases_path else None
    ):
        raise CorpusError("changed features, aliases, source or segmentation identity")
    return value
