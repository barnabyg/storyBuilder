import json
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from tests.conftest import REPO


def test_saved_example_renders_ordered_escaped_cards(tmp_path: Path) -> None:
    out = tmp_path / "cards"
    assert (
        main(["plan", "inspect", str(REPO / "examples/harbour.plan.json"), "--out", str(out)]) == 0
    )
    markdown = (out / "cards.md").read_text(encoding="utf-8")
    html = (out / "cards.html").read_text(encoding="utf-8")
    assert markdown.index("s1") < markdown.index("s2")
    assert "Mara" in markdown
    assert "&lt;harbour&gt;" in html
    assert "<harbour>" not in html
    saved = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == "1"
    assert main(["plan", "validate", str(out / "plan.json")]) == 0


def test_round_trip_schema_and_empty_content(tmp_path: Path) -> None:
    file = tmp_path / "input.json"
    file.write_text(
        '{"schema_version":"1","id":"story","title":"Empty","scenes":[{"id":"s1","title":""}]}',
        encoding="utf-8",
    )
    out = tmp_path / "cards"
    assert main(["plan", "inspect", str(file), "--out", str(out)]) == 0
    first = (out / "plan.json").read_bytes()
    assert main(["plan", "inspect", str(out / "plan.json"), "--out", str(out)]) == 0
    assert (out / "plan.json").read_bytes() == first
    schema = tmp_path / "schema.json"
    assert main(["plan", "schema", "--out", str(schema)]) == 0
    first_schema = schema.read_bytes()
    assert main(["plan", "schema", "--out", str(schema)]) == 0
    assert schema.read_bytes() == first_schema
    assert json.loads(first_schema)["properties"]["schema_version"]["const"] == "1"


@pytest.mark.parametrize(
    ("mutation", "expected", "exit_code"),
    [
        ("duplicate", "duplicate_id", 1),
        ("dangling", "character_reference", 1),
        ("version", "unsupported schema version", 2),
        ("malformed", "Expecting", 2),
    ],
)
def test_bad_saved_plans_have_useful_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
    expected: str,
    exit_code: int,
) -> None:
    data: dict[str, Any] = json.loads((REPO / "examples/harbour.plan.json").read_text())
    if mutation == "duplicate":
        data["scenes"][1]["id"] = "s1"
    elif mutation == "dangling":
        data["scenes"][0]["character_ids"] = ["ghost"]
    elif mutation == "version":
        data["schema_version"] = "999"
    file = tmp_path / "bad.json"
    file.write_text("{" if mutation == "malformed" else json.dumps(data), encoding="utf-8")
    assert main(["plan", "validate", str(file)]) == exit_code
    assert expected in capsys.readouterr().err
