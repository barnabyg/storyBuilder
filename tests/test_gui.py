import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from nde.gui.app import create_app, parse_idea
from nde.gui.studio import SavedStory, Studio, describe_step
from nde.runs.runner import RunOptions
from tests.conftest import REPO, write_config
from tests.continuity_support import continuity_config, fake_continuity
from tests.fake_openai import FakeOpenAI, Json, Reply, error

LOCAL = "http://127.0.0.1:8771"

IDEA = {
    "genre": "mystery",
    "premise": "A lighthouse keeper finds the lamp <lit> though nobody climbed the tower.",
    "protagonist_name": "Elsie Marr",
    "protagonist_description": "a keeper who trusts her logbook",
    "central_conflict": "The log says one thing and the islanders another.",
    "themes": "trust, memory",
    "ending_kind": "twist",
    "ending_detail": "The keeper learns she lit it herself.",
    "target_words": "2000",
}


@pytest.fixture
def studio(fake_llm: FakeOpenAI, tmp_path: Path) -> Studio:
    fake_llm.default = fake_continuity()
    config = continuity_config(fake_llm.base_url)
    return Studio(
        write_config(tmp_path, config),
        RunOptions(
            runs_dir=tmp_path / "runs",
            # pragma: allowlist nextline secret - a fake key for the fake server
            environ={"FAKE_API_KEY": "test-key"},
            ledger_path=tmp_path / "ledger.sqlite",
        ),
        tmp_path / "ideas",
    )


@pytest.fixture
def client(studio: Studio) -> TestClient:
    return TestClient(create_app(studio), base_url=LOCAL, follow_redirects=False)


def _write_story(client: TestClient, idea: dict[str, str] = IDEA) -> str:
    """Submit the form and wait for the story; returns the address of its page."""
    job = client.post("/stories", data=idea)
    assert job.status_code == 303
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        page = client.get(job.headers["location"])
        if page.status_code == 303:
            return page.headers["location"]
        assert "Writing your story" in page.text
        time.sleep(0.02)
    raise AssertionError("the story did not finish")


def _story_at(studio: Studio, address: str) -> SavedStory:
    _, _, run, brief = address.split("/")
    story = studio.story(run, brief)
    assert story is not None
    return story


def test_typing_an_idea_writes_a_story_that_can_be_read_and_saved(
    client: TestClient, studio: Studio, fake_llm: FakeOpenAI
) -> None:
    address = _write_story(client)

    page = client.get(address)
    assert page.status_code == 200
    assert "The Last Lamp" in page.text
    assert "Scene1 first" in page.text
    assert "Elsie Marr" in page.text
    # What was typed reached the model, and is shown back safely escaped.
    assert any(
        "A lighthouse keeper finds the lamp <lit>" in str(request) for request in fake_llm.requests
    )
    assert "the lamp &lt;lit&gt; though" in page.text
    assert "the lamp <lit>" not in page.text

    saved = client.get(f"{address}/story.md")
    assert saved.headers["content-disposition"] == 'attachment; filename="the-last-lamp.md"'
    assert saved.text.startswith("# The Last Lamp\n\nScene1 first")

    home = client.get("/")
    assert address in home.text
    assert "The Last Lamp" in home.text
    assert studio.current is None


def test_a_typed_idea_stays_out_of_the_benchmark(
    client: TestClient, tmp_path: Path, studio: Studio
) -> None:
    before = sorted(path.name for path in (REPO / "benchmark" / "dev").iterdir())

    _write_story(client)

    assert sorted(path.name for path in (REPO / "benchmark" / "dev").iterdir()) == before
    [saved] = (tmp_path / "ideas" / "dev").iterdir()
    assert saved.name == "brief-1001.yaml"
    brief: dict[str, Any] = yaml.safe_load(saved.read_text(encoding="utf-8"))
    assert brief["themes"] == ["trust", "memory"]
    assert brief["split"] == "dev"
    assert studio.find_idea("brief-1001") is not None


def test_writing_again_gives_a_new_story_rather_than_a_stored_one(
    client: TestClient, fake_llm: FakeOpenAI
) -> None:
    first = _write_story(client)
    requests = len(fake_llm.requests)

    second = _write_story(client)

    assert second != first
    assert len(fake_llm.requests) == 2 * requests


def test_an_example_fills_in_the_form_and_a_held_out_brief_is_never_offered(
    client: TestClient,
) -> None:
    home = client.get("/")
    assert 'value="brief-09"' in home.text
    assert 'value="brief-11"' not in home.text

    filled = client.get("/", params={"idea": "brief-09"})
    assert 'value="Kari Lund"' in filled.text
    assert '<option value="brief-09" selected>' in filled.text
    held_out = client.get("/", params={"idea": "brief-11"})
    assert 'name="genre" value=""' in held_out.text


def test_a_missing_answer_keeps_what_was_typed_and_spends_nothing(
    client: TestClient, fake_llm: FakeOpenAI, tmp_path: Path
) -> None:
    page = client.post("/stories", data=IDEA | {"premise": "   "})

    assert page.status_code == 422
    assert "Please fill in: The situation." in page.text
    assert "Elsie Marr" in page.text
    assert fake_llm.requests == []
    assert not (tmp_path / "ideas" / "dev").exists()


def test_too_many_themes_are_explained() -> None:
    content, problems = parse_idea(IDEA | {"themes": "a, b, c, d, e"})

    assert content is None
    assert problems == ["Themes: please give between one and four, separated by commas."]


def test_a_missing_api_key_is_explained_before_any_spend(
    fake_llm: FakeOpenAI, tmp_path: Path, studio: Studio
) -> None:
    keyless = Studio(
        studio.config,
        RunOptions(runs_dir=tmp_path / "runs", environ={}, ledger_path=tmp_path / "ledger.sqlite"),
        tmp_path / "ideas",
    )

    page = TestClient(create_app(keyless), base_url=LOCAL).post("/stories", data=IDEA)

    assert page.status_code == 422
    assert "FAKE_API_KEY" in page.text
    assert "Elsie Marr" in page.text
    assert fake_llm.requests == []
    assert list((tmp_path / "ideas" / "dev").iterdir()) == []


def test_a_second_story_waits_for_the_first(
    client: TestClient, fake_llm: FakeOpenAI, studio: Studio
) -> None:
    reply = fake_continuity()
    released = threading.Event()

    def slow(body: Json) -> Reply:
        released.wait(timeout=30)
        return reply(body)

    fake_llm.default = slow
    first = client.post("/stories", data=IDEA)
    second = client.post("/stories", data=IDEA)
    assert second.headers["location"] == first.headers["location"]
    assert "A story is being written" in client.get("/").text
    page = client.get(first.headers["location"])
    assert 'http-equiv="refresh"' in page.text
    job = studio.current
    assert job is not None
    assert job.thread is not None
    released.set()
    job.thread.join(timeout=30)
    assert client.get(first.headers["location"]).status_code == 303


def test_a_failed_story_says_so_and_what_it_cost(client: TestClient, fake_llm: FakeOpenAI) -> None:
    fake_llm.default = lambda _: error(401, "bad key")
    job = client.post("/stories", data=IDEA)
    deadline = time.monotonic() + 30
    text = ""
    while time.monotonic() < deadline and "could not be written" not in text:
        text = client.get(job.headers["location"]).text
        time.sleep(0.02)
    assert "could not be written" in text
    assert "This cost" in text
    assert "bad key" in text


def test_a_story_stopped_early_is_shown_as_unfinished(client: TestClient, studio: Studio) -> None:
    address = _write_story(client)
    [story] = studio.stories()
    record = story.folder / "continuity.json"
    saved = json.loads(record.read_text(encoding="utf-8"))
    record.write_text(json.dumps(saved | {"finished": False}), encoding="utf-8")

    assert "stopped before its checks and repairs finished" in client.get(address).text
    assert "unfinished" in client.get("/").text


def test_deleting_a_story_asks_first_then_removes_only_that_story(
    client: TestClient, studio: Studio, tmp_path: Path
) -> None:
    keep = _write_story(client)
    doomed = _write_story(client)
    kept_story = _story_at(studio, keep)
    doomed_run = _story_at(studio, doomed).folder.parents[3]
    assert keep != doomed
    assert doomed_run.is_dir()

    # Asking changes nothing, and offers a way back.
    question = client.get(f"{doomed}/delete")
    assert "The Last Lamp" in question.text
    assert "cannot be undone" in question.text
    assert f'href="{doomed}"' in question.text
    assert doomed_run.is_dir()
    assert client.get(doomed).status_code == 200

    gone = client.post(f"{doomed}/delete")

    assert gone.status_code == 303
    assert "Deleted" in client.get(gone.headers["location"]).text
    assert not doomed_run.exists()
    assert client.get(doomed).status_code == 404
    assert client.get(keep).status_code == 200
    assert kept_story.text
    assert [story.run for story in studio.stories()] == [kept_story.run]
    # The ledger and the typed ideas are untouched, so the idea can be used again.
    assert (tmp_path / "ledger.sqlite").is_file()
    assert sorted(p.name for p in (tmp_path / "ideas" / "dev").iterdir()) == [
        "brief-1001.yaml",
        "brief-1002.yaml",
    ]
    assert 'value="Elsie Marr"' in client.get("/", params={"idea": "brief-1002"}).text


def test_deleting_keeps_the_rest_of_a_run_that_holds_other_stories(
    client: TestClient, studio: Studio
) -> None:
    address = _write_story(client)
    [story] = studio.stories()
    other = story.folder.parent.parent / "brief-1999" / "r1"
    other.mkdir(parents=True)
    (other / "story.md").write_text("Another story.", encoding="utf-8")

    assert client.post(f"{address}/delete").status_code == 303

    assert not story.folder.exists()
    assert (other / "story.md").is_file()


def test_a_story_being_written_cannot_be_deleted_and_others_can(
    client: TestClient, fake_llm: FakeOpenAI, studio: Studio
) -> None:
    still_running = _write_story(client)
    finished = _write_story(client)
    manifest = _story_at(studio, still_running).folder.parents[3] / "manifest.json"
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps(recorded | {"status": "running"}), encoding="utf-8")
    reply = fake_continuity()
    released = threading.Event()

    def slow(body: Json) -> Reply:
        released.wait(timeout=30)
        return reply(body)

    fake_llm.default = slow
    client.post("/stories", data=IDEA)

    refused = client.post(f"{still_running}/delete")
    allowed = client.post(f"{finished}/delete")

    job = studio.current
    assert job is not None
    assert job.thread is not None
    released.set()
    job.thread.join(timeout=30)
    assert refused.status_code == 409
    assert "still being written" in refused.text
    assert client.get(still_running).status_code == 200
    assert allowed.status_code == 303
    assert client.get(finished).status_code == 404
    assert job.run_dir is not None
    assert json.loads((job.run_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == (
        "succeeded"
    )


def test_a_failed_delete_says_so(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    address = _write_story(client)

    def locked(_: Path) -> None:
        raise PermissionError("the file is open in another program")

    monkeypatch.setattr("nde.gui.studio.shutil.rmtree", locked)
    page = client.post(f"{address}/delete")

    assert page.status_code == 500
    assert "could not be fully deleted" in page.text
    assert "open in another program" in page.text


def test_another_website_cannot_delete_a_story(client: TestClient, studio: Studio) -> None:
    address = _write_story(client)

    page = client.post(f"{address}/delete", headers={"origin": "https://evil.example"})

    assert page.status_code == 403
    assert len(studio.stories()) == 1
    assert client.get(address).status_code == 200


def test_opening_a_finished_jobs_page_after_deleting_its_story_returns_home(
    client: TestClient,
) -> None:
    job = client.post("/stories", data=IDEA).headers["location"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and client.get(job).status_code != 303:
        time.sleep(0.02)
    address = client.get(job).headers["location"]

    client.post(f"{address}/delete")

    assert client.get(job).headers["location"] == "/"


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "https://evil.example"},
        {"origin": "null"},
        {"host": "evil.example"},
    ],
)
def test_another_website_cannot_start_a_story(
    client: TestClient, fake_llm: FakeOpenAI, headers: dict[str, str]
) -> None:
    page = client.post("/stories", data=IDEA, headers=headers)

    assert page.status_code == 403
    assert fake_llm.requests == []


def test_another_address_cannot_read_stories(client: TestClient) -> None:
    assert client.get("/", headers={"host": "evil.example"}).status_code == 403


def test_a_page_of_this_app_may_submit(client: TestClient) -> None:
    assert client.post("/stories", data=IDEA, headers={"origin": LOCAL}).status_code == 303


@pytest.mark.parametrize(
    "path",
    [
        "/stories/nothing/brief-01",
        "/stories/nothing/brief-01/delete",
        "/stories/..%2F..%2Fsecret/brief-01",
        "/jobs/nothing",
    ],
)
def test_unknown_addresses_are_not_found(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (None, "Getting started"),
        ("continuity/brief-1001/plan", "Planning the characters and scenes"),
        ("continuity/brief-1001/scene-03", "Writing scene 3"),
        (
            "continuity/brief-1001/scene-03 (attempt 2)",
            "Writing scene 3, trying again to reach the length",
        ),
        ("continuity/brief-1001/audit-clock-round-0", "Checking the story for problems"),
        ("continuity/brief-1001/repair-1-p004", "Repairing the passages the checks flagged"),
    ],
)
def test_progress_is_described_in_plain_words(call: str | None, expected: str) -> None:
    assert describe_step(call) == expected


def _benchmark_files() -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in (REPO / "benchmark").rglob("*") if path.is_file()}


def test_an_added_example_is_offered_edited_and_deleted(client: TestClient) -> None:
    added = client.post("/examples", data=IDEA)
    assert added.status_code == 303
    assert "Added" in client.get(added.headers["location"]).text
    assert '<option value="brief-1001">' in client.get("/").text
    filled = client.get("/", params={"idea": "brief-1001"})
    assert 'value="Elsie Marr"' in filled.text
    assert '<option value="brief-1001" selected>' in filled.text

    form = client.get("/examples/brief-1001/edit").text
    assert "lamp &lt;lit&gt;" in form

    changed = client.post("/examples/brief-1001/edit", data={**IDEA, "genre": "horror"})
    assert changed.status_code == 303
    assert "horror: A lighthouse" in client.get("/").text
    assert 'value="brief-1002"' not in client.get("/").text

    assert "cannot be undone" in client.get("/examples/brief-1001/delete").text
    assert 'value="brief-1001"' in client.get("/").text
    deleted = client.post("/examples/brief-1001/delete")
    assert deleted.status_code == 303
    assert 'value="brief-1001"' not in client.get("/").text
    assert client.get("/examples/brief-1001/edit").status_code == 404
    assert client.post("/examples/brief-1001/delete").status_code == 404


def test_an_example_with_a_missing_answer_keeps_what_was_typed_and_saves_nothing(
    client: TestClient, tmp_path: Path
) -> None:
    page = client.post("/examples", data={**IDEA, "genre": " "})
    assert page.status_code == 422
    assert "Please fill in: Genre." in page.text
    assert 'value="Elsie Marr"' in page.text
    assert not list((tmp_path / "ideas").rglob("*.yaml"))

    edited = client.post("/examples/brief-09/edit", data={**IDEA, "genre": ""})
    assert edited.status_code == 422
    assert 'value="brief-09"' in client.get("/").text


def test_changing_benchmark_examples_leaves_the_benchmark_alone_and_can_be_undone(
    client: TestClient,
) -> None:
    before = _benchmark_files()
    original = client.get("/", params={"idea": "brief-09"})
    assert 'value="Kari Lund"' in original.text

    assert client.post("/examples/brief-09/edit", data=IDEA).status_code == 303
    home = client.get("/")
    assert 'value="brief-09"' not in home.text
    assert '<option value="brief-1001">' in home.text
    assert client.get("/examples/brief-09/edit").status_code == 404

    assert client.post("/examples/brief-1001/delete").status_code == 303
    manage = client.get("/examples")
    assert "Restore the original examples (1)" in manage.text

    assert "Restore the original" in client.get("/examples/brief-01/delete").text
    assert client.post("/examples/brief-01/delete").status_code == 303
    assert "Restore the original examples (2)" in client.get("/examples").text

    assert client.post("/examples/restore").status_code == 303
    home = client.get("/")
    assert 'value="brief-09"' in home.text
    assert 'value="brief-01"' in home.text
    assert "Restore the original" not in client.get("/examples").text
    assert _benchmark_files() == before


def test_a_story_keeps_its_idea_when_its_example_changes(
    client: TestClient, studio: Studio
) -> None:
    assert client.post("/examples", data=IDEA).status_code == 303
    address = _write_story(client, IDEA)
    assert _story_at(studio, address).brief == "brief-1002"

    client.post("/examples/brief-1001/edit", data={**IDEA, "protagonist_name": "Someone Else"})
    story = client.get(address).text
    assert "Elsie Marr" in story
    assert "Someone Else" not in story


def test_a_benchmark_brief_stays_readable_on_a_story_after_its_example_is_removed(
    client: TestClient, studio: Studio
) -> None:
    assert client.post("/examples/brief-09/delete").status_code == 303
    idea = studio.find_idea("brief-09")
    assert idea is not None
    assert idea.protagonist.name == "Kari Lund"


def test_typed_ideas_and_examples_never_share_an_id(client: TestClient, studio: Studio) -> None:
    address = _write_story(client, IDEA)
    assert _story_at(studio, address).brief == "brief-1001"
    assert client.post("/examples", data=IDEA).status_code == 303
    assert 'value="brief-1002"' in client.get("/").text
    assert _story_at(studio, _write_story(client, IDEA)).brief == "brief-1003"


def test_deleting_every_example_still_leaves_a_way_back(client: TestClient, studio: Studio) -> None:
    for example in studio.examples():
        assert client.post(f"/examples/{example.id}/delete").status_code == 303
    home = client.get("/")
    assert "No example ideas." in home.text
    assert 'href="/examples"' in home.text
    assert "Restore the original examples" in client.get("/examples").text


def test_a_failed_example_save_says_so_and_keeps_what_was_typed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_path: Path, _text: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("nde.gui.studio._replace_text", refuse)
    page = client.post("/examples", data=IDEA)
    assert page.status_code == 500
    assert "disk full" in page.text
    assert 'value="Elsie Marr"' in page.text


@pytest.mark.parametrize(
    "path",
    ["/examples", "/examples/brief-09/edit", "/examples/brief-09/delete", "/examples/restore"],
)
def test_another_website_cannot_change_examples(
    client: TestClient, tmp_path: Path, path: str
) -> None:
    page = client.post(path, data=IDEA, headers={"origin": "https://evil.example"})
    assert page.status_code == 403
    assert not (tmp_path / "ideas").exists()


@pytest.mark.parametrize(
    "path",
    ["/examples/brief-11/edit", "/examples/brief-11/delete", "/examples/nonsense/edit"],
)
def test_a_held_out_or_unknown_brief_is_no_example(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404
    assert client.post(path, data=IDEA).status_code == 404
