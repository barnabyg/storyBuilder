"""Source overlap and inspectable structural neighbours; similarity is not memorisation."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

from nde.conditional.data import load_dataset as load_conditional_dataset
from nde.corpus.records_io import check_snapshot, create_record, read_sealed, report_files, snapshot
from nde.reconstruction.codec import load_encoding
from nde.reconstruction.dataset import load_dataset
from nde.reconstruction.training_data import load_training_dataset

METRIC = "canonical-typed-record-multiset-jaccard-v1"


def view_path(path: Path) -> Path:
    if path.is_dir():
        return path / ("dataset.json" if (path / "dataset.json").is_file() else "encoding.json")
    return path


def load_view(path: Path) -> dict[str, Any]:
    """Read an identified view without fitting or changing either dataset."""
    path = view_path(path)
    value = read_sealed(path)
    if value.get("kind") == "reconstruction-dataset":
        return load_dataset(path)
    if value.get("kind") in {"reconstruction-subset", "reconstruction-augmentation"}:
        return load_training_dataset(path)
    if value.get("kind") == "conditional-dataset":
        # Conditional fine-tuning examples, with their complete source lineage rechecked.
        return load_conditional_dataset(path)
    if "specification" in value and "records" in value:
        return load_encoding(path)
    check_snapshot(value.get("inputs", {}))
    return value


def records(view: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in view.get("examples", view.get("records", [])):
        # Conditional examples keep their encoded target view under `target`.
        record = example.get("record", example.get("target", example))
        rows.append({"id": example.get("id", record.get("id")), **record})
    if not rows:
        raise ValueError("identified view has no records")
    return rows


def canonical_view(view: dict[str, Any]) -> dict[str, Any]:
    """Normalize only record identities, retaining typed references and uncertainty."""
    identities: dict[str, str] = {}

    def identify(value: Any, path: str) -> None:
        if isinstance(value, dict):
            mapping = cast(dict[str, Any], value)
            if isinstance(mapping.get("id"), str):
                identities[mapping["id"]] = path
            for key, child in mapping.items():
                identify(child, f"{path}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(cast(list[Any], value)):
                identify(child, f"{path}/{index}")

    def normalize(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {
                name: normalize(child, name)
                for name, child in cast(dict[str, Any], value).items()
                if name not in {"id", "sources", "evidence"}
            }
        if isinstance(value, list):
            return [normalize(child, key) for child in cast(list[Any], value)]
        if isinstance(value, str) and key.endswith(("_id", "_ids")):
            return identities.get(value, f"unresolved:{value}")
        return value

    identify(view, "")
    return cast(dict[str, Any], normalize(view))


def structural_features(view: dict[str, Any]) -> dict[str, Counter[str]]:
    canonical = canonical_view(view)
    return {
        field: Counter(
            json.dumps(item, sort_keys=True, ensure_ascii=False) for item in cast(list[Any], value)
        )
        for field, value in canonical.items()
        if isinstance(value, list)
    }


def view_distance(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Multiset Jaccard distance over the canonical typed records of two views' shared fields."""
    a, b = structural_features(left), structural_features(right)
    common = sorted(a.keys() & b.keys())
    if not common:
        return {"distance": None, "comparison": "unavailable", "common_fields": []}
    intersection = sum(sum((a[field] & b[field]).values()) for field in common)
    union = sum(sum((a[field] | b[field]).values()) for field in common)
    return {
        "distance": 1 - intersection / union if union else 0.0,
        "comparison": "available" if union else "empty-shared-channels",
        "common_fields": common,
        "unavailable_fields": sorted(a.keys() ^ b.keys()),
        "intersection": intersection,
        "union": union,
        "uncertain": any(
            marker in json.dumps([left, right]) for marker in ('"unknown"', '"ambiguous"')
        ),
    }


def neighbour_rows(
    index: list[dict[str, Any]], queries: list[dict[str, Any]], neighbours: int
) -> tuple[list[dict[str, Any]], bool]:
    """Each query's nearest index records, and whether any source/work/author overlaps.

    Queries are only compared; nothing is ever added to the index.
    """
    rows: list[dict[str, Any]] = []
    unresolved = False
    for item in queries:
        provenance = item.get("provenance", {})
        matches: list[dict[str, Any]] = []
        for candidate in index:
            source = candidate.get("provenance", {})
            exact_source = bool(provenance.get("source_sha256")) and (
                provenance.get("source_sha256") == source.get("source_sha256")
            )
            exact_work = bool(provenance.get("work_id")) and (
                provenance.get("work_id") == source.get("work_id")
            )
            author_overlap = bool(provenance.get("author_id")) and (
                provenance.get("author_id") == source.get("author_id")
            )
            unresolved |= exact_source or exact_work or author_overlap
            matches.append(
                {
                    "record_id": candidate["id"],
                    "provenance": source,
                    "exact_source": exact_source,
                    "exact_work": exact_work,
                    "author_overlap": author_overlap,
                    **view_distance(item["view"], candidate["view"]),
                }
            )
        matches.sort(
            key=lambda row: (row["distance"] is None, row["distance"] or 0, row["record_id"])
        )
        rows.append(
            {"query_id": item["id"], "provenance": provenance, "matches": matches[:neighbours]}
        )
    return rows, unresolved


def check_index(train: dict[str, Any]) -> None:
    """Only optimizer training records may form a nearest-neighbour index."""
    split = train.get("split")
    if split is None and train.get("records"):
        assignments = {record.get("provenance", {}).get("split") for record in train["records"]}
        split = "train" if assignments == {"train"} else None
    if split != "train" or train.get("role", "optimizer") != "optimizer":
        raise ValueError("nearest-neighbour index requires optimizer training records")


def audit_overlap(training: Path, query: Path, out: Path, *, neighbours: int = 5) -> dict[str, Any]:
    """Save deterministic matches without ever adding query material to the index."""
    if neighbours < 1:
        raise ValueError("neighbours must be positive")
    train, queries = load_view(training), load_view(query)
    check_index(train)
    if train.get("codec_id") != queries.get("codec_id"):
        raise ValueError("incompatible codec identities")
    rows, unresolved = neighbour_rows(records(train), records(queries), neighbours)
    training_file = view_path(training)
    query_file = view_path(query)
    report = create_record(
        out / "overlap.json",
        {
            "kind": "reconstruction-overlap",
            "training_id": train["id"],
            "query_id": queries["id"],
            "codec_id": train.get("codec_id"),
            "metric": METRIC,
            "index_members": [item["id"] for item in records(train)],
            "queries": rows,
            "unresolved_overlap": unresolved,
            "test_only": train.get("test_only", False) or queries.get("test_only", False),
            "limitations": (
                "Structural similarity is a candidate match, not proof of memorisation "
                "or its absence. Missing channels are unavailable comparisons; "
                "uncertain labels remain uncertain."
            ),
            "inputs": snapshot([training_file, query_file]),
        },
    )
    report_files(out, "Training overlap and structural neighbours", report)
    return report


def configure_overlap(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("training", type=Path)
    parser.add_argument("query", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--neighbours", type=int, default=5)


def run_overlap(args: argparse.Namespace) -> int:
    try:
        audit_overlap(args.training, args.query, args.out, neighbours=args.neighbours)
        return 0
    except (OSError, ValueError) as error:
        print(f"nde reconstruction-overlap: {error}", file=sys.stderr)
        return 2
