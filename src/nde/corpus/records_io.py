"""Content identities and immutable records for the additive corpus workflows."""

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from nde.files import write_text_lf


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort_file(path: Path) -> Path:
    return path / "cohort.json" if path.is_dir() else path


def identity(record: dict[str, Any]) -> str:
    content = {key: value for key, value in record.items() if key != "id"}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return cast(dict[str, Any], value)


def read_sealed(path: Path) -> dict[str, Any]:
    record = read_json(path)
    if record.get("version") != "1" or record.get("id") != identity(record):
        raise ValueError(f"record identity missing or changed: {path}")
    return record


def save_json(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    write_text_lf(temporary, json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def create_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    record = {**record, "version": "1"}
    record["id"] = identity(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): file_sha(path) for path in sorted(set(paths))}


def check_snapshot(files: dict[str, str]) -> None:
    for name, checksum in files.items():
        path = Path(name)
        if not path.is_file() or file_sha(path) != checksum:
            raise ValueError(f"bound evidence missing or changed: {name}")


def evidence_files(directory: Path) -> list[Path]:
    return [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix in {".json", ".txt", ".raw"}
        and path.name not in {"report.json"}
    ]


def report_files(directory: Path, title: str, result: dict[str, Any]) -> None:
    save_json(directory / "report.json", result)
    write_text_lf(
        directory / "report.md",
        f"# {title}\n\n```json\n{json.dumps(result, indent=2, sort_keys=True)}\n```\n",
    )
