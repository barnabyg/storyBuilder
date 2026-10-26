"""A fixed byte codec for filtered structural views, never original source tensors."""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, cast

from nde.corpus.channels import CHANNEL_FIELDS
from nde.corpus.export import validate_export
from nde.corpus.records_io import (
    check_snapshot,
    create_record,
    identity,
    read_sealed,
    report_files,
    snapshot,
)
from nde.files import write_text_lf
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan

PAD, MASK, BOS, EOS = 0, 1, 2, 3
BYTE_OFFSET = 4
VOCABULARY_SIZE = 260
SPECIFICATION = {
    "name": "nde-filtered-structural-byte",
    "version": "1",
    "vocabulary_size": VOCABULARY_SIZE,
    "special_tokens": {"pad": PAD, "mask": MASK, "bos": BOS, "eos": EOS},
    "byte_offset": BYTE_OFFSET,
    "serialization": "canonical JSON UTF-8; sorted object fields; explicit list lengths",
    "references": "typed field names; IDs canonicalized by collection and list position",
    "retained": "registered channel fields, referenced character identities, reading uncertainty",
    "excluded": "source text, evidence, source checksums, original IDs, plan title, constraints",
    "roundtrip": "exact retained canonical view; not a complete tier-L annotation",
    "fitting": "none: all UTF-8 bytes have fixed identities",
}
CODEC_ID = identity(SPECIFICATION)
VOCABULARY_ID = identity({"size": VOCABULARY_SIZE, "offset": BYTE_OFFSET, "special": [0, 1, 2, 3]})
SCHEMA_ID = identity(NarrativePlan.model_json_schema())
DEFAULT_MAX_SCENES = 256
DEFAULT_MAX_TOKENS = 65536
type Record = dict[str, Any]


def seal(record: Record) -> Record:
    result = {**record, "version": "1"}
    result["id"] = identity(result)
    return result


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def tokenize(value: Any) -> list[int]:
    return [BOS, *(byte + BYTE_OFFSET for byte in canonical_json(value).encode("utf-8")), EOS]


def detokenize(tokens: list[int]) -> Any:
    if not tokens or tokens[0] != BOS or tokens[-1] != EOS:
        raise ValueError("token sequence requires BOS/EOS delimiters")
    if any(token < BYTE_OFFSET or token >= VOCABULARY_SIZE for token in tokens[1:-1]):
        raise ValueError("invalid byte token")
    return json.loads(bytes(token - BYTE_OFFSET for token in tokens[1:-1]).decode("utf-8"))


def _retained(plan: NarrativePlan, policy: dict[str, str]) -> tuple[Record, dict[str, str]]:
    if any(value not in {"hard", "soft", "excluded"} for value in policy.values()):
        raise ValueError("unsupported target policy")
    original = plan.model_dump(mode="json")
    selected: Record = {}
    for channel, fields in CHANNEL_FIELDS.items():
        if policy.get(channel, "excluded") != "excluded":
            selected.update({field: original[field] for field in fields})
    states = [
        state
        for state in original["states"]
        if state["kind"] in {"goal", "belief"}
        and policy.get(state["kind"], "excluded") != "excluded"
    ]
    if any(policy.get(channel, "excluded") != "excluded" for channel in ("goal", "belief")):
        selected["states"] = states
    if "scenes" not in selected:
        # References require chronology, but an excluded plot cannot expose summaries.
        selected["scenes"] = [{"id": scene.id} for scene in plan.scenes]
    selected["characters"] = [{"id": item.id} for item in plan.characters]
    references: dict[str, str] = {}

    def collect(value: Any, path: str) -> None:
        if isinstance(value, dict):
            obj = cast(Record, value)
            if "id" in obj:
                references[obj["id"]] = path
            for key, item in obj.items():
                if key != "evidence":
                    collect(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(cast(list[Any], value)):
                collect(item, f"{path}.{index}" if path else str(index))

    for field, values in selected.items():
        collect(values, field)

    def convert(value: Any, field: str = "") -> Any:
        if isinstance(value, dict):
            return {
                key: convert(item, key)
                for key, item in sorted(cast(Record, value).items())
                if key != "evidence"
            }
        if isinstance(value, list):
            return [convert(item, field) for item in cast(list[Any], value)]
        if isinstance(value, str) and (field == "id" or field.endswith(("_id", "_ids"))):
            if value not in references:
                raise ValueError(
                    f"retained reference points outside registered view: {field}:{value}"
                )
            return references[value]
        return value

    return convert(selected), references


def _channel(path: list[str | int], view: Record) -> str:
    field = str(path[0])
    if field == "states" and len(path) > 1:
        return str(view["states"][int(path[1])]["kind"])
    return next(
        (channel for channel, fields in CHANNEL_FIELDS.items() if field in fields), "context"
    )


def flatten_view(
    view: Record, policy: dict[str, str], ineligible_channels: tuple[str, ...] = ()
) -> list[Record]:
    slots: list[Record] = []

    def visit(value: Any, path: list[str | int], uncertain: bool = False) -> None:
        if isinstance(value, dict):
            obj = cast(Record, value)
            uncertain = uncertain or obj.get("status") in {"unknown", "ambiguous"}
            for key, item in sorted(obj.items()):
                visit(item, [*path, key], uncertain)
            return
        channel = _channel(path, view)
        target_policy = policy.get(channel, "excluded")
        eligible = (
            target_policy != "excluded"
            and not uncertain
            and path[-1] != "id"
            and channel not in ineligible_channels
        )
        if isinstance(value, list):
            slots.append(
                {
                    "path": [*path, "$length"],
                    "type": "length",
                    "value": len(cast(list[Any], value)),
                    "channel": channel,
                    "policy": target_policy,
                    "eligible": False,
                }
            )
            for index, item in enumerate(cast(list[Any], value)):
                visit(item, [*path, index], uncertain)
            return
        key = next((str(part) for part in reversed(path) if isinstance(part, str)), "")
        kind = "reference" if key.endswith(("_id", "_ids")) else type(value).__name__
        slots.append(
            {
                "path": path,
                "type": kind,
                "value": value,
                "channel": channel,
                "policy": target_policy,
                "eligible": eligible,
            }
        )

    for field, value in sorted(view.items()):
        visit(value, [field])
    return slots


def view_from_slots(slots: list[Record]) -> Record:
    result: Record = {}
    for slot in slots:
        path = slot["path"]
        container: Any = result
        for index, key in enumerate(path):
            if key == "$length":
                if (
                    not isinstance(container, list)
                    or len(cast(list[Any], container)) != slot["value"]
                ):
                    raise ValueError("inconsistent structural length")
                break
            last = index == len(path) - 1
            if last:
                container[key] = copy.deepcopy(slot["value"])
                break
            following = path[index + 1]
            if isinstance(container, list):
                container = cast(list[Any], container)
                if container[key] is None:
                    container[key] = [None] * slot["value"] if following == "$length" else {}
            elif key not in container:
                container[key] = [None] * slot["value"] if following == "$length" else {}
            container = container[key]
    return result


# PLR0913: explicit public capacity/policy controls; revisit if another codec needs shared options.
def encode_plan(  # noqa: PLR0913
    plan: NarrativePlan,
    policy: dict[str, str],
    provenance: Record,
    *,
    max_scenes: int = DEFAULT_MAX_SCENES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    ineligible_channels: tuple[str, ...] = (),
) -> Record:
    if max_scenes < 1 or max_tokens < 1 or len(plan.scenes) > max_scenes:
        raise ValueError("unsupported scene capacity")
    if not validate_plan(plan).valid:
        raise ValueError("invalid source annotation or graph references")
    view, original_ids = _retained(plan, policy)
    tokens = tokenize(view)
    if len(tokens) > max_tokens:
        raise ValueError("unsupported token capacity; no truncation performed")
    return seal(
        {
            "codec_id": CODEC_ID,
            "vocabulary_id": VOCABULARY_ID,
            "schema_id": SCHEMA_ID,
            "target_policy": dict(sorted(policy.items())),
            "view": view,
            "slots": flatten_view(view, policy, ineligible_channels),
            "ineligible_channels": sorted(set(ineligible_channels)),
            "tokens": tokens,
            "provenance": copy.deepcopy(provenance),
            "original_ids": original_ids,
            "capacity": {"max_scenes": max_scenes, "max_tokens": max_tokens},
        }
    )


def decode_record(record: Record) -> Record:
    if record.get("id") != identity(record) or record.get("codec_id") != CODEC_ID:
        raise ValueError("stale encoded record or codec")
    if record.get("vocabulary_id") != VOCABULARY_ID or record.get("schema_id") != SCHEMA_ID:
        raise ValueError("stale vocabulary or schema")
    view = detokenize(record["tokens"])
    if view != record["view"] or view_from_slots(record["slots"]) != view:
        raise ValueError("encoded view/slot/token disagreement")
    if (
        flatten_view(view, record["target_policy"], tuple(record["ineligible_channels"]))
        != record["slots"]
    ):
        raise ValueError("encoded slot policy disagreement")
    return cast(Record, copy.deepcopy(view))


def materialize_view(view: Record) -> NarrativePlan:
    """Validate retained structure with empty excluded channels; not completeness evidence."""
    content = copy.deepcopy(view)
    for character in content.get("characters", []):
        character.setdefault("name", character["id"])
    for scene in content.get("scenes", []):
        scene.setdefault("title", "retained scene")
    return NarrativePlan.model_validate(
        {"id": "retained-plan", "title": "Retained view", **content}
    )


def create_encoding(export_path: Path, out: Path) -> Record:
    source = validate_export(export_path)
    export_file = export_path / "dataset.json" if export_path.is_dir() else export_path
    records: list[Record] = []
    paths = [export_file]
    for row in source["records"]:
        path = Path(row["annotation_path"])
        paths.append(path)
        plan = NarrativePlan.model_validate_json(path.read_text(encoding="utf-8"))
        records.append(encode_plan(plan, source["target_policy"], row))
    result = create_record(
        out / "encoding.json",
        {
            "codec_id": CODEC_ID,
            "vocabulary_id": VOCABULARY_ID,
            "schema_id": SCHEMA_ID,
            "specification": SPECIFICATION,
            "export_path": str(export_file.resolve()),
            "export_id": source["id"],
            "test_only": source["test_only"],
            "records": records,
            "inputs": snapshot(paths),
        },
    )
    report_files(out, "Filtered structural encoding", result)
    return result


def load_encoding(path: Path) -> Record:
    path = path / "encoding.json" if path.is_dir() else path
    result = read_sealed(path)
    check_snapshot(result["inputs"])
    if validate_export(Path(result["export_path"]))["id"] != result["export_id"]:
        raise ValueError("stale approved export identity")
    if result["codec_id"] != CODEC_ID or result["specification"] != SPECIFICATION:
        raise ValueError("stale codec specification")
    for record in result["records"]:
        decode_record(record)
    return result


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="codec_command", required=True)
    encode = commands.add_parser("encode")
    encode.add_argument("path", type=Path)
    encode.add_argument("--out", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("path", type=Path)
    decode = commands.add_parser("decode")
    decode.add_argument("path", type=Path)
    decode.add_argument("--record", type=int, default=0)
    decode.add_argument("--out", type=Path, required=True)


def run(args: argparse.Namespace) -> int:
    try:
        if args.codec_command == "encode":
            result = create_encoding(args.path, args.out)
        else:
            result = load_encoding(args.path)
            if args.codec_command == "decode":
                if not 0 <= args.record < len(result["records"]):
                    raise ValueError("record index outside encoding")
                write_text_lf(
                    args.out,
                    json.dumps(
                        {
                            "representation": SPECIFICATION["roundtrip"],
                            "retained_view": decode_record(result["records"][args.record]),
                        },
                        indent=2,
                    ),
                )
        print(json.dumps({"id": result["id"], "codec_id": CODEC_ID}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"nde reconstruction codec: {error}", file=sys.stderr)
        return 2
