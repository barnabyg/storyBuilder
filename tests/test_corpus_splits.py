import json
from pathlib import Path

import pytest

from nde.cli import main
from tests.test_corpus_acquisition import acquire


def make_work(root: Path, work_id: str, author_id: str, author: str = "Ann Author") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return acquire(
        root,
        f"BEGIN LAMP\n{work_id} waited.\nEND LAMP",
        work_id=work_id,
        source_id=f"{work_id}-edition",
        author_id=author_id,
        author=author,
    )


def freeze(root: Path, works: list[Path], **changes: object) -> tuple[Path, int]:
    config = root / "cohort-config.json"
    config.write_text(
        json.dumps({"id": root.name, "works": [str(w) for w in works], **changes}), encoding="utf-8"
    )
    out = root / "cohort"
    return out, main(["corpus", "split", "freeze", str(config), "--out", str(out)])


def test_frozen_cohort_is_author_disjoint_reproducible_and_protects_aliases(tmp_path: Path) -> None:
    first = make_work(tmp_path / "first", "first", "ann")
    second = make_work(tmp_path / "second", "second", "ann-alias")
    other = make_work(tmp_path / "other", "other", "bea", "Bea Author")
    out, result = freeze(
        tmp_path,
        [first, second, other],
        author_aliases={"ann-alias": "ann"},
        protected_work_ids=["first"],
        train_percent=100,
        validation_percent=0,
    )
    assert result == 0
    data = json.loads((out / "cohort.json").read_text("utf-8"))
    assert [(a["work_id"], a["split"]) for a in data["assignments"]] == [
        ("first", "gold"),
        ("other", "train"),
        ("second", "gold"),
    ]
    original = (out / "cohort.json").read_bytes()
    assert main(["corpus", "split", "inspect", str(out)]) == 0
    assert (
        freeze(
            tmp_path,
            [first, second, other],
            author_aliases={"ann-alias": "ann"},
            protected_work_ids=["first"],
            train_percent=100,
            validation_percent=0,
        )[1]
        == 0
    )
    assert (out / "cohort.json").read_bytes() == original


def test_same_author_name_cannot_be_split_by_unknown_alias(tmp_path: Path) -> None:
    first = make_work(tmp_path / "first", "first", "ann")
    second = make_work(tmp_path / "second", "second", "unregistered-ann")
    out, result = freeze(tmp_path, [first, second])
    assert result == 1
    assert not out.exists()


def test_changed_records_and_frozen_assignments_are_detected(tmp_path: Path) -> None:
    work = make_work(tmp_path / "one", "one", "ann")
    out, result = freeze(tmp_path, [work])
    assert result == 0
    path = out / "cohort.json"
    data = json.loads(path.read_text("utf-8"))
    data["assignments"][0]["split"] = (
        "test" if data["assignments"][0]["split"] != "test" else "train"
    )
    path.write_text(json.dumps(data), encoding="utf-8")
    assert main(["corpus", "split", "inspect", str(out)]) == 1


def test_expansion_preserves_existing_author_assignments_and_rejects_missing_works(
    tmp_path: Path,
) -> None:
    first = make_work(tmp_path / "one", "one", "ann")
    base = tmp_path / "base"
    base.mkdir()
    old, result = freeze(base, [first], author_splits={"ann": "test"})
    assert result == 0
    second = make_work(tmp_path / "two", "two", "ann")
    expanded = tmp_path / "expanded"
    expanded.mkdir()
    out, result = freeze(
        expanded, [first, second], previous=str(old), train_percent=100, validation_percent=0
    )
    assert result == 0
    assert {
        a["split"] for a in json.loads((out / "cohort.json").read_text("utf-8"))["assignments"]
    } == {"test"}
    missing = tmp_path / "missing"
    missing.mkdir()
    assert freeze(missing, [second], previous=str(old))[1] == 1


@pytest.mark.parametrize("changed", ["config", "source", "provenance"])
def test_inspection_rejects_changed_frozen_inputs(tmp_path: Path, changed: str) -> None:
    work = make_work(tmp_path / "one", "one", "ann")
    out, result = freeze(tmp_path, [work])
    assert result == 0
    if changed == "source":
        (work / "source.raw").write_bytes(b"new source")
    elif changed == "config":
        config = tmp_path / "cohort-config.json"
        config.write_text(
            config.read_text("utf-8").replace('"id":', '"seed": 99, "id":'), encoding="utf-8"
        )
    else:
        record = work / "work.json"
        record.write_text(
            record.read_text("utf-8").replace('"edition 1"', '"edition 2"'), encoding="utf-8"
        )
    assert main(["corpus", "split", "inspect", str(out)]) == 1


@pytest.mark.parametrize(("field", "value"), [("work_id", "one"), ("author_id", "unknown")])
def test_duplicates_and_unresolved_identities_cannot_be_frozen(
    tmp_path: Path, field: str, value: str
) -> None:
    first = make_work(tmp_path / "one", "one", "ann")
    second = make_work(tmp_path / "two", "two", "bea", "Bea Author")
    record = second / "work.json"
    data = json.loads(record.read_text("utf-8"))
    data[field] = value
    record.write_text(json.dumps(data), encoding="utf-8")
    assert freeze(tmp_path, [first, second])[1] == 1


def test_expansion_preserves_work_exclusion_without_excluding_the_author(tmp_path: Path) -> None:
    excluded = make_work(tmp_path / "one", "one", "ann")
    included = make_work(tmp_path / "two", "two", "ann")
    additional = make_work(tmp_path / "three", "three", "ann")
    base = tmp_path / "base"
    base.mkdir()
    old, result = freeze(
        base, [excluded, included], excluded_work_ids=["one"], author_splits={"ann": "train"}
    )
    assert result == 0
    expansion = tmp_path / "expansion"
    expansion.mkdir()
    out, result = freeze(expansion, [excluded, included, additional], previous=str(old))
    assert result == 0
    assignments = json.loads((out / "cohort.json").read_text("utf-8"))["assignments"]
    assert {a["work_id"]: a["split"] for a in assignments} == {
        "one": "excluded",
        "two": "train",
        "three": "train",
    }
    assert main(["corpus", "split", "inspect", str(out)]) == 0


def test_expansion_cannot_use_work_exclusion_to_bypass_protected_author(tmp_path: Path) -> None:
    excluded = make_work(tmp_path / "one", "one", "ann")
    protected = make_work(tmp_path / "two", "two", "ann")
    additional = make_work(tmp_path / "three", "three", "ann")
    base = tmp_path / "base"
    base.mkdir()
    old, result = freeze(
        base, [excluded, protected], excluded_work_ids=["one"], protected_work_ids=["two"]
    )
    assert result == 0
    expansion = tmp_path / "expansion"
    expansion.mkdir()
    out, result = freeze(
        expansion,
        [excluded, protected, additional],
        previous=str(old),
        train_percent=100,
        validation_percent=0,
    )
    assert result == 0
    assignments = json.loads((out / "cohort.json").read_text("utf-8"))["assignments"]
    assert {a["work_id"]: a["split"] for a in assignments} == {
        "one": "excluded",
        "two": "gold",
        "three": "gold",
    }


def test_protected_author_remains_protected_when_only_prior_work_is_excluded(
    tmp_path: Path,
) -> None:
    excluded = make_work(tmp_path / "one", "one", "ann")
    additional = make_work(tmp_path / "two", "two", "ann")
    base = tmp_path / "base"
    base.mkdir()
    old, result = freeze(base, [excluded], excluded_work_ids=["one"], protected_work_ids=["one"])
    assert result == 0
    expansion = tmp_path / "expansion"
    expansion.mkdir()
    out, result = freeze(
        expansion,
        [excluded, additional],
        previous=str(old),
        train_percent=100,
        validation_percent=0,
    )
    assert result == 0
    assignments = json.loads((out / "cohort.json").read_text("utf-8"))["assignments"]
    assert {a["work_id"]: a["split"] for a in assignments} == {"one": "excluded", "two": "gold"}
