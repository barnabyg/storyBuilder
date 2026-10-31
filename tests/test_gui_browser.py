"""Opt-in local browser acceptance: no browser or package downloads in default tests."""

import os
import socket
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import Page, expect, sync_playwright

from nde.gui.app import create_app
from nde.gui.studio import Studio
from nde.runs.runner import RunOptions
from tests.conftest import write_config
from tests.continuity_support import continuity_config, fake_continuity
from tests.fake_openai import FakeOpenAI

_browser_only = pytest.mark.skipif(
    os.environ.get("NDE_BROWSER") != "1",
    reason="set NDE_BROWSER=1 to run acceptance with installed Chrome (or NDE_BROWSER_CHANNEL)",
)


@contextmanager
def _serving(fake_llm: FakeOpenAI, tmp_path: Path) -> Generator[str]:
    """The app on a free local port, using the fake provider; yields its address."""
    fake_llm.default = fake_continuity()
    studio = Studio(
        write_config(tmp_path, continuity_config(fake_llm.base_url)),
        RunOptions(
            runs_dir=tmp_path / "runs",
            # pragma: allowlist nextline secret - a fake key for the fake server
            environ={"FAKE_API_KEY": "test-key"},
            ledger_path=tmp_path / "ledger.sqlite",
        ),
        tmp_path / "ideas",
    )
    server = uvicorn.Server(uvicorn.Config(create_app(studio), log_level="error"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive()


@_browser_only
def test_an_example_idea_becomes_a_story_read_saved_and_deleted_in_the_browser(
    fake_llm: FakeOpenAI, tmp_path: Path
) -> None:
    with _serving(fake_llm, tmp_path) as base:
        _write_and_read(base, tmp_path)


@_browser_only
def test_examples_are_added_edited_deleted_and_restored_in_the_browser(
    fake_llm: FakeOpenAI, tmp_path: Path
) -> None:
    with _serving(fake_llm, tmp_path) as base, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, channel=os.environ.get("NDE_BROWSER_CHANNEL", "chrome")
        )
        try:
            _manage_examples(browser.new_page(), base)
        finally:
            browser.close()


def _manage_examples(page: Page, base: str) -> None:
    page.goto(base)
    page.get_by_role("link", name="Manage examples").click()

    page.get_by_role("link", name="Add an example").click()
    page.get_by_label("Genre").fill("mystery")
    page.get_by_label("The situation").fill("A keeper finds the lamp lit.")
    page.get_by_label("Main character's name").fill("Elsie Marr")
    page.get_by_label("Who they are").fill("a keeper who trusts her logbook")
    page.get_by_label("The central conflict").fill("The log and the islanders disagree.")
    page.get_by_label("Themes").fill("trust, memory")
    page.get_by_label("What the ending must do").fill("She learns she lit it herself.")
    page.get_by_role("button", name="Save example").click()
    expect(page.get_by_role("status")).to_contain_text("mystery: A keeper finds the lamp lit.")

    page.get_by_role("link", name="Edit: mystery: A keeper").click()
    expect(page.get_by_label("Main character's name")).to_have_value("Elsie Marr")
    page.get_by_label("Genre").fill("horror")
    page.get_by_role("button", name="Save example").click()
    expect(page.get_by_role("status")).to_contain_text("horror: A keeper finds the lamp lit.")

    page.get_by_role("link", name="Use: horror: A keeper").click()
    expect(page.get_by_label("Main character's name")).to_have_value("Elsie Marr")
    expect(page.get_by_label("Example idea")).to_have_value("brief-1001")

    page.get_by_role("link", name="Manage examples").click()
    page.get_by_role("link", name="Delete: horror: A keeper").click()
    page.get_by_role("link", name="Keep this example").click()
    page.get_by_role("link", name="Delete: horror: A keeper").click()
    page.get_by_role("button", name="Delete example").click()
    expect(page.get_by_role("link", name="Edit: horror: A keeper")).to_have_count(0)

    page.locator('a[href="/examples/brief-09/edit"]').click()
    expect(page.get_by_label("Main character's name")).to_have_value("Kari Lund")
    page.get_by_label("Main character's name").fill("Kari Lund-Berg")
    page.get_by_role("button", name="Save example").click()
    expect(page.locator('a[href="/examples/brief-09/edit"]')).to_have_count(0)
    page.goto(f"{base}/?idea=brief-1001")
    expect(page.get_by_label("Main character's name")).to_have_value("Kari Lund-Berg")

    page.get_by_role("link", name="Manage examples").click()
    page.get_by_role("button", name="Restore the original examples (1)").click()
    expect(page.locator('a[href="/examples/brief-09/edit"]')).to_have_count(1)
    expect(page.get_by_role("button", name="Restore the original")).to_have_count(0)


def _write_and_read(base: str, tmp_path: Path) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, channel=os.environ.get("NDE_BROWSER_CHANNEL", "chrome")
        )
        try:
            page = browser.new_page(accept_downloads=True)
            page.goto(base)
            page.get_by_label("Example idea").select_option("brief-09")
            page.get_by_role("button", name="Fill in").click()
            expect(page.get_by_label("Main character's name")).to_have_value("Kari Lund")
            expect(page.get_by_label("Example idea")).to_have_value("brief-09")

            # The browser itself stops an unanswered question, before the server sees it.
            genre = page.locator('input[name="genre"]').input_value()
            page.locator('input[name="genre"]').fill("")
            page.get_by_role("button", name="Write this story").click()
            expect(page).to_have_url(f"{base}/?idea=brief-09")
            page.locator('input[name="genre"]').fill(genre)

            page.get_by_role("button", name="Write this story").click()
            expect(page.get_by_role("heading", name="The Last Lamp")).to_be_visible(timeout=30_000)
            expect(page.locator("article")).to_contain_text("Scene1 first")

            page.get_by_text("The characters").click()
            expect(page.get_by_text("Elsie Marr is tall")).to_be_visible()

            with page.expect_download() as download:
                page.get_by_role("link", name="Save as a file").click()
            assert download.value.suggested_filename == "the-last-lamp.md"
            saved = tmp_path / "saved.md"
            download.value.save_as(saved)
            assert saved.read_text(encoding="utf-8").startswith("# The Last Lamp")

            page.get_by_role("link", name="Use this idea again").click()
            expect(page.get_by_label("Main character's name")).to_have_value("Kari Lund")
            library = page.get_by_role("link", name="The Last Lamp")
            expect(library).to_be_visible()

            library.click()
            page.get_by_role("link", name="Delete this story").click()
            page.get_by_role("link", name="Keep this story").click()
            expect(page.get_by_role("heading", name="The Last Lamp")).to_be_visible()
            page.get_by_role("link", name="Delete this story").click()
            page.get_by_role("button", name="Delete permanently").click()
            expect(page.get_by_role("status")).to_contain_text("The Last Lamp")
            expect(page.get_by_role("link", name="The Last Lamp")).to_have_count(0)
            page.reload()
            expect(page.get_by_text("Stories you write will appear here.")).to_be_visible()
            assert not list((tmp_path / "runs").iterdir())
        finally:
            browser.close()
