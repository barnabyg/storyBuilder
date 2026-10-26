import hashlib
import json
from pathlib import Path

import pytest

from nde.cli import main

pytest_plugins = ["tests.test_calibration"]


def spec(tmp_path: Path, **changes: object) -> Path:
    value: dict[str, object] = {
        "work_id": "lamp",
        "source_id": "edition-1",
        "title": "The Lamp",
        "author_id": "ann",
        "author": "Ann Author",
        "source": "https://example.org/lamp",
        "edition": "edition 1",
        "licence_basis": "supplied public-domain claim",
        "licence_evidence": "Publisher statement, checked by owner",
        "start": "BEGIN LAMP",
        "end": "END LAMP",
    }
    value.update(changes)
    path = tmp_path / "work-spec.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def acquire(tmp_path: Path, text: str, **changes: object) -> Path:
    source = tmp_path / "input.txt"
    source.write_bytes(text.encode("utf-8"))
    out = tmp_path / "lamp"
    assert (
        main(["corpus", "import", str(spec(tmp_path, **changes)), str(source), "--out", str(out)])
        == 0
    )
    return out


def test_import_inspect_preserves_exact_raw_selection_and_structure(tmp_path: Path) -> None:
    raw = (
        "Other work\r\nBEGIN LAMP\r\n\r\nCHAPTER I\r\n\r\nÉva waited.\r\n\r\n"
        "***\r\n\r\nShe left.\r\nEND LAMP\r\nOther work"
    )
    out = acquire(tmp_path, raw)
    assert (out / "source.raw").read_bytes() == raw.encode()
    assert (
        out / "text.txt"
    ).read_bytes() == "CHAPTER I\n\nÉva waited.\n\n***\n\nShe left.\n".encode()
    saved = json.loads((out / "work.json").read_text("utf-8"))
    assert saved["raw_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert saved["period"] == "unknown"
    assert saved["selection"]["start"] > 0
    assert main(["corpus", "inspect", str(out)]) == 0
    assert "does not establish legal eligibility" in (out / "report.html").read_text("utf-8")


def test_changed_source_or_duplicate_identity_cannot_replace_verified_work(tmp_path: Path) -> None:
    out = acquire(tmp_path, "BEGIN LAMP\nA lamp.\nEND LAMP")
    original = (out / "work.json").read_bytes()
    source = tmp_path / "input.txt"
    source.write_text("BEGIN LAMP\nChanged.\nEND LAMP", encoding="utf-8")
    assert main(["corpus", "import", str(spec(tmp_path)), str(source), "--out", str(out)]) == 1
    assert (out / "work.json").read_bytes() == original
    (out / "source.raw").write_bytes(b"changed")
    assert main(["corpus", "inspect", str(out)]) == 1


def test_fetch_uses_local_server_and_verified_copy(
    tmp_path: Path, text_server: dict[str, str]
) -> None:
    text_server["/lamp"] = "BEGIN LAMP\nA lamp.\nEND LAMP"
    config = spec(tmp_path, source=text_server["base"] + "/lamp")
    out = tmp_path / "fetched"
    assert main(["corpus", "fetch", str(config), "--out", str(out)]) == 0
    text_server.pop("/lamp")
    assert main(["corpus", "fetch", str(config), "--out", str(out)]) == 0


@pytest.mark.parametrize(
    "body", ["missing markers", "END LAMP\nBEGIN LAMP", "BEGIN LAMP\n\nEND LAMP"]
)
def test_invalid_selection_publishes_nothing(tmp_path: Path, body: str) -> None:
    source = tmp_path / "input.txt"
    source.write_text(body, encoding="utf-8")
    out = tmp_path / "bad"
    assert main(["corpus", "import", str(spec(tmp_path)), str(source), "--out", str(out)]) == 1
    assert not out.exists()


def test_xhtml_selects_only_the_requested_work_and_rejects_malformed_xml(tmp_path: Path) -> None:
    body = (
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<section id="front"><p>Credits</p></section>'
        '<section id="lamp"><h2>Chapter I</h2><p>Éva waited.</p><hr/><p>She left.</p></section>'
        '<section id="other"><p>Other</p></section></body></html>'
    )
    out = acquire(tmp_path, body, format="xhtml", section_id="lamp", start=None, end=None)
    assert (out / "text.txt").read_text("utf-8") == "Chapter I\n\nÉva waited.\n\n***\n\nShe left.\n"
    (tmp_path / "input.txt").write_text("<broken", encoding="utf-8")
    assert (
        main(
            [
                "corpus",
                "import",
                str(spec(tmp_path, format="xhtml", section_id="lamp", start=None, end=None)),
                str(tmp_path / "input.txt"),
                "--out",
                str(tmp_path / "bad"),
            ]
        )
        == 1
    )


def test_interrupted_publication_does_not_create_ready_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.txt"
    source.write_text("BEGIN LAMP\nA lamp.\nEND LAMP", encoding="utf-8")
    out = tmp_path / "interrupted"

    def fail_rename(self: Path, target: Path) -> Path:
        raise OSError(f"interrupted publication of {self} to {target}")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail_rename)
        assert main(["corpus", "import", str(spec(tmp_path)), str(source), "--out", str(out)]) == 1
    assert not out.exists()
    assert main(["corpus", "import", str(spec(tmp_path)), str(source), "--out", str(out)]) == 0
