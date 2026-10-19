import hashlib
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, override

import pytest
import yaml

from nde.cli import main
from tests.conftest import REPO
from tests.eval_helpers import DIMENSIONS, reader, two_runs
from tests.fake_openai import FakeOpenAI

GUTENBERG_TEXT = """\
The Project Gutenberg eBook of A Test Story

*** START OF THE PROJECT GUTENBERG EBOOK A TEST STORY ***

Produced by A Volunteer

A TEST STORY

By Ann Author

The keeper climbed the stair at dusk, as she had
every evening for thirty years, and _counted_ the steps.

The lamp would not light. She tried
it twice, then sat down in the dark to wait.

*** END OF THE PROJECT GUTENBERG EBOOK A TEST STORY ***

Licence text that must not appear.
"""

STORY = (
    "The keeper climbed the stair at dusk, as she had every evening for thirty years, "
    "and counted the steps.\n\n"
    "The lamp would not light. She tried it twice, then sat down in the dark to wait.\n"
)


class _Texts(ThreadingHTTPServer):
    daemon_threads = True


@pytest.fixture
def text_server() -> Iterator[dict[str, str]]:
    """Serves the texts in the yielded dict by path, on 127.0.0.1."""
    texts: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = texts.get(self.path)
            if body is None:
                self.send_error(404)
                return
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        @override
        def log_message(self, format: str, *args: object) -> None:
            return

    server = _Texts(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    texts["base"] = f"http://{host!s}:{port}"
    yield texts
    server.shutdown()
    server.server_close()


def _manifest(tmp_path: Path, base: str, sha256: str | None = None) -> Path:
    path = tmp_path / "data" / "manifest.calibration.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    story: dict[str, Any] = {
        "id": "test-story",
        "title": "A Test Story",
        "author": "Ann Author",
        "source": f"{base}/story.txt",
        "edition": "Project Gutenberg eBook #0",
        "licence_basis": "public domain in the USA",
        "sha256": sha256,
    }
    path.write_text(yaml.safe_dump({"stories": [story]}, sort_keys=False), encoding="utf-8")
    return path


def _fetch(manifest: Path) -> int:
    return main(["calibration", "fetch", "--manifest", str(manifest)])


def test_fetching_writes_the_text_and_records_its_checksum(
    tmp_path: Path, text_server: dict[str, str]
) -> None:
    text_server["/story.txt"] = GUTENBERG_TEXT
    manifest = _manifest(tmp_path, text_server["base"])

    assert _fetch(manifest) == 0

    recorded = yaml.safe_load(manifest.read_text(encoding="utf-8"))["stories"][0]
    assert recorded["sha256"] == hashlib.sha256(GUTENBERG_TEXT.encode("utf-8")).hexdigest()
    texts = manifest.parent / "calibration"
    assert (texts / "test-story.md").read_text(encoding="utf-8") == STORY
    assert (texts / "test-story.source.txt").read_text(encoding="utf-8") == GUTENBERG_TEXT


def test_a_recorded_checksum_is_verified_on_reuse(
    tmp_path: Path, text_server: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    text_server["/story.txt"] = GUTENBERG_TEXT
    manifest = _manifest(tmp_path, text_server["base"])
    assert _fetch(manifest) == 0
    source = manifest.parent / "calibration" / "test-story.source.txt"

    assert _fetch(manifest) == 0
    source.write_text(GUTENBERG_TEXT.replace("dusk", "dawn"), encoding="utf-8")

    assert _fetch(manifest) == 1
    assert "test-story: checksum mismatch" in capsys.readouterr().err


def test_a_changed_download_is_rejected_against_its_recorded_checksum(
    tmp_path: Path, text_server: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    text_server["/story.txt"] = GUTENBERG_TEXT.replace("dusk", "dawn")
    digest = hashlib.sha256(GUTENBERG_TEXT.encode("utf-8")).hexdigest()

    assert _fetch(_manifest(tmp_path, text_server["base"], digest)) == 1

    assert "checksum mismatch" in capsys.readouterr().err
    assert not (tmp_path / "data" / "calibration" / "test-story.md").exists()


def test_a_story_can_be_imported_from_a_local_file(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, "http://unused.invalid")
    local = tmp_path / "download.txt"
    local.write_text(GUTENBERG_TEXT, encoding="utf-8")

    assert (
        main(["calibration", "import", "test-story", str(local), "--manifest", str(manifest)]) == 0
    )

    assert (manifest.parent / "calibration" / "test-story.md").read_text("utf-8") == STORY
    assert yaml.safe_load(manifest.read_text("utf-8"))["stories"][0]["sha256"]


def test_the_committed_provenance_manifest_is_valid() -> None:
    manifest = yaml.safe_load(
        (REPO / "data" / "manifest.calibration.yaml").read_text(encoding="utf-8")
    )

    stories = manifest["stories"]
    assert 3 <= len(stories) <= 5
    for story in stories:
        assert {"id", "title", "author", "source", "edition", "licence_basis", "sha256"} <= set(
            story
        )


@pytest.fixture
def calibration(
    fake_llm: FakeOpenAI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text_server: dict[str, str],
) -> Path:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    text_server["/story.txt"] = GUTENBERG_TEXT
    manifest = _manifest(tmp_path, text_server["base"])
    assert _fetch(manifest) == 0
    first, second = two_runs(fake_llm, tmp_path)
    code = main(
        [
            "eval",
            "calibration",
            "calib",
            "--humans",
            str(manifest),
            "--from",
            str(first),
            "intact",
            "--from",
            str(second),
            "control",
            "--per-source",
            "2",
            "--seed",
            "3",
            "--batches-dir",
            str(tmp_path / "batches"),
            "--rubric-dir",
            str(REPO / "benchmark" / "rubric"),
        ]
    )
    assert code == 0
    return tmp_path / "batches" / "calib"


def test_a_calibration_batch_mixes_human_and_generated_stories(calibration: Path) -> None:
    view = json.loads((calibration / "view.json").read_text(encoding="utf-8"))
    key = json.loads((calibration / "sealed" / "key.json").read_text(encoding="utf-8"))

    assert view["kind"] == "single"
    assert len(view["stories"]) == 5
    assert sorted(item["source"] for item in key["items"]) == [
        "control",
        "control",
        "human",
        "intact",
        "intact",
    ]
    assert STORY in view["stories"]


def test_the_reader_shows_one_story_at_a_time_with_no_source(calibration: Path) -> None:
    client = reader(calibration)
    forbidden = [
        "human",
        "intact",
        "control",
        "Ann Author",
        "A Test Story",
        "Gutenberg",
        "secret-model-id",
        "brief-0",
        "test-story",
    ]

    pages = [
        client.get("/", follow_redirects=True),
        client.get("/overview"),
        client.get("/api/progress"),
    ]
    for number in range(1, 6):
        pages += [client.get(f"/stories/{number}"), client.get(f"/api/stories/{number}")]

    for page in pages:
        for word in forbidden:
            assert word.lower() not in page.text.lower(), f"{page.request.url} leaks {word!r}"
    story_page = client.get("/stories/1").text
    assert 'name="score-story-engagement"' in story_page
    assert 'name="preferred"' not in story_page
    assert 'name="guess"' not in story_page
    assert "Story 1 of 5" in story_page


def _score_all(calibration: Path, by_source: dict[str, int]) -> None:
    key = json.loads((calibration / "sealed" / "key.json").read_text(encoding="utf-8"))
    client = reader(calibration)
    for number, item in enumerate(key["items"], start=1):
        form: dict[str, str | list[str]] = {
            f"score-story-{dimension}": str(by_source[item["source"]]) for dimension in DIMENSIONS
        }
        form["tags-story"] = ["generic"] if item["source"] != "human" else []
        response = client.post(f"/stories/{number}", data=form)
        assert response.status_code in {200, 303}


def test_an_incomplete_score_is_rejected(calibration: Path) -> None:
    client = reader(calibration)

    response = client.post("/stories/1", data={"score-story-engagement": "3"})

    assert response.status_code == 422
    assert "score-story-pacing" in response.text
    assert client.get("/api/progress").json()["judged"] == 0


def test_a_closed_calibration_batch_reports_scores_by_source(
    calibration: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _score_all(calibration, {"human": 5, "intact": 3, "control": 2})
    client = reader(calibration)
    assert client.post("/close", follow_redirects=False).status_code == 303
    capsys.readouterr()

    assert main(["eval", "report", "calib", "--batches-dir", str(tmp_path / "batches")]) == 0

    report = capsys.readouterr().out
    assert "# Calibration report: calib" in report
    assert "| Dimension | human | intact | control |" in report
    assert "| engagement | 5.00 [5.00, 5.00] | 3.00 [3.00, 3.00] | 2.00 [2.00, 2.00] |" in report
    assert "| generic | 0 | 2 | 2 |" in report
    assert "A Test Story" in report  # the key is revealed once closed


def test_the_calibration_report_refuses_an_open_batch(
    calibration: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval", "report", "calib", "--batches-dir", str(tmp_path / "batches")]) == 2
    assert "still open" in capsys.readouterr().err
    assert calibration.exists()


def test_building_a_batch_verifies_the_fetched_sources(
    fake_llm: FakeOpenAI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text_server: dict[str, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    text_server["/story.txt"] = GUTENBERG_TEXT
    manifest = _manifest(tmp_path, text_server["base"])
    assert _fetch(manifest) == 0
    texts = manifest.parent / "calibration"
    (texts / "test-story.md").write_text("A different story entirely.\n", encoding="utf-8")
    first, _ = two_runs(fake_llm, tmp_path)

    code = main(
        [
            "eval",
            "calibration",
            "calib",
            "--humans",
            str(manifest),
            "--from",
            str(first),
            "intact",
            "--batches-dir",
            str(tmp_path / "batches"),
            "--rubric-dir",
            str(REPO / "benchmark" / "rubric"),
        ]
    )

    assert code == 0
    view = json.loads((tmp_path / "batches" / "calib" / "view.json").read_text("utf-8"))
    assert STORY in view["stories"]
    assert "A different story entirely.\n" not in view["stories"]

    (texts / "test-story.source.txt").write_text(GUTENBERG_TEXT + "x", encoding="utf-8")
    code = main(
        [
            "eval",
            "calibration",
            "calib2",
            "--humans",
            str(manifest),
            "--from",
            str(first),
            "intact",
            "--batches-dir",
            str(tmp_path / "batches"),
            "--rubric-dir",
            str(REPO / "benchmark" / "rubric"),
        ]
    )
    assert code == 2
    assert "checksum mismatch" in capsys.readouterr().err


def test_calibration_sources_need_distinct_condition_names(
    fake_llm: FakeOpenAI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text_server: dict[str, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    text_server["/story.txt"] = GUTENBERG_TEXT
    manifest = _manifest(tmp_path, text_server["base"])
    assert _fetch(manifest) == 0
    first, _ = two_runs(fake_llm, tmp_path)

    code = main(
        [
            "eval",
            "calibration",
            "calib",
            "--humans",
            str(manifest),
            "--from",
            str(first),
            "intact",
            "--from",
            str(first),
            "intact",
            "--batches-dir",
            str(tmp_path / "batches"),
            "--rubric-dir",
            str(REPO / "benchmark" / "rubric"),
        ]
    )

    assert code == 2
    assert "distinct condition names" in capsys.readouterr().err


def test_importing_a_missing_file_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _manifest(tmp_path, "http://unused.invalid")

    code = main(
        [
            "calibration",
            "import",
            "test-story",
            str(tmp_path / "nope.txt"),
            "--manifest",
            str(manifest),
        ]
    )

    assert code == 1
    assert "cannot read" in capsys.readouterr().err
