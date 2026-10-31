"""Opt-in local browser acceptance: no browser or package downloads in default tests."""

import os
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import expect, sync_playwright

from nde.gui.app import create_app
from nde.gui.studio import Studio
from nde.runs.runner import RunOptions
from tests.conftest import write_config
from tests.continuity_support import continuity_config, fake_continuity
from tests.fake_openai import FakeOpenAI


@pytest.mark.skipif(
    os.environ.get("NDE_BROWSER") != "1",
    reason="set NDE_BROWSER=1 to run acceptance with installed Chrome (or NDE_BROWSER_CHANNEL)",
)
def test_an_example_idea_becomes_a_story_read_saved_and_deleted_in_the_browser(
    fake_llm: FakeOpenAI, tmp_path: Path
) -> None:
    fake_llm.default = fake_continuity()
    studio = Studio(
        write_config(tmp_path, continuity_config(fake_llm.base_url)),
        RunOptions(
            runs_dir=tmp_path / "runs",
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
            _write_and_read(f"http://127.0.0.1:{port}", tmp_path)
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive()


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
