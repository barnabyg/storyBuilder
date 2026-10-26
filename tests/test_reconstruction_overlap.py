"""Public overlap command known-answer fixtures."""

from pathlib import Path

import pytest

from nde.corpus.records_io import create_record, read_sealed
from nde.reconstruction.overlap import audit_overlap
from tests.fake_openai import FakeOpenAI
from tests.reconstruction_support import paired_datasets


def test_audit_accepts_public_encoding_and_rechecks_transitive_sources(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    train, held = paired_datasets(tmp_path, fake_llm)
    result = audit_overlap(tmp_path / "encoded", held, tmp_path / "audit")
    assert result["unresolved_overlap"] is False
    assert len(result["index_members"]) == 2
    annotation = Path(read_sealed(train)["examples"][0]["record"]["provenance"]["annotation_path"])
    annotation.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        audit_overlap(train, held, tmp_path / "stale")


def dataset(path: Path, *, query: bool = False, renamed: bool = False) -> Path:
    scene = "renamed" if renamed else "s1"
    other = "renamed-two" if renamed else "s2"
    record = {
        "view": {
            "scenes": [{"id": scene, "title": "Start"}, {"id": other, "title": "End"}],
            "causal_links": [{"id": "edge", "cause_scene_id": scene, "effect_scene_id": other}],
        },
        "provenance": {
            "work_id": "edition-b" if query else "work",
            "author_id": "author",
            "source_sha256": "same",
            "split": "test" if query else "train",
        },
    }
    create_record(
        path,
        {
            "role": "evaluator-only" if query else "optimizer",
            "split": "test" if query else "train",
            "codec_id": "codec",
            "test_only": True,
            "examples": [{"id": "q" if query else "a", "record": record}],
            "inputs": {},
        },
    )
    return path


def test_overlap_reports_renamed_duplicates_and_author_overlap_without_mutation(
    tmp_path: Path,
) -> None:
    train = dataset(tmp_path / "train.json")
    query = dataset(tmp_path / "query.json", query=True, renamed=True)
    before = train.read_bytes(), query.read_bytes()
    report = audit_overlap(train, query, tmp_path / "audit")
    match = report["queries"][0]["matches"][0]
    assert match["distance"] == 0
    assert match["exact_source"] is True
    assert match["author_overlap"] is True
    assert report["unresolved_overlap"] is True
    assert (train.read_bytes(), query.read_bytes()) == before
    assert read_sealed(tmp_path / "audit" / "overlap.json")["id"] == report["id"]


def test_overlap_rejects_stale_index_inputs(tmp_path: Path) -> None:
    train = dataset(tmp_path / "train.json")
    query = dataset(tmp_path / "query.json", query=True)
    train.write_text(train.read_text().replace('"same"', '"changed"'))
    with pytest.raises(ValueError, match="identity"):
        audit_overlap(train, query, tmp_path / "audit")


def test_overlap_distinguishes_changed_structure_and_unavailable_channels(tmp_path: Path) -> None:
    train = dataset(tmp_path / "train.json")
    query = dataset(tmp_path / "query.json", query=True)
    value = read_sealed(query)
    value.pop("id")
    value["examples"][0]["record"]["view"]["causal_links"] = []
    query.unlink()
    create_record(query, value)
    report = audit_overlap(train, query, tmp_path / "audit")
    assert report["queries"][0]["matches"][0]["distance"] == pytest.approx(1 / 3)


def test_overlap_ties_are_stable_and_index_never_contains_queries(tmp_path: Path) -> None:
    train = dataset(tmp_path / "train.json")
    query = dataset(tmp_path / "query.json", query=True)
    value = read_sealed(train)
    value.pop("id")
    value["examples"].append({**value["examples"][0], "id": "b"})
    value["examples"].reverse()
    train.unlink()
    create_record(train, value)
    report = audit_overlap(train, query, tmp_path / "audit")
    assert [item["record_id"] for item in report["queries"][0]["matches"]] == ["a", "b"]
    assert "q" not in report["index_members"]
