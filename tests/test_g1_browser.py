"""Opt-in local browser acceptance: no browser or package downloads in default tests."""

import os
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import sync_playwright

from nde.eval.batch import (
    Key,
    SealedPair,
    ShownPair,
    StorySource,
    View,
    read_judgements,
    write_batch,
)
from nde.eval.reader import create_app
from nde.eval.rubric import load_rubric
from tests.conftest import REPO


@pytest.mark.skipif(
    os.environ.get("NDE_BROWSER") != "1",
    reason="set NDE_BROWSER=1 to run acceptance with installed Chrome (or NDE_BROWSER_CHANNEL)",
)
@pytest.mark.parametrize("binding_file", ["g1.json", "gd.json"])
def test_synthetic_gate_browser_controls_api_storage_and_blinding(
    tmp_path: Path, binding_file: str
) -> None:
    batch = tmp_path / "synthetic-test-only"
    source_c = StorySource(run="SECRET-RUN-C", condition="SECRET-CANDIDATE", story="plan-ledger-C")
    source_b = StorySource(run="SECRET-RUN-B", condition="SECRET-COMPARATOR", story="plan-ledger-B")
    write_batch(
        batch,
        View(
            pairs=[ShownPair(a="Synthetic story one", b="Synthetic story two")],
            rubric=load_rubric(REPO / "benchmark/rubric", 1),
            optional_scores=True,
            primary_dimension="coherence",
            required_score_dimensions=["coherence", "originality"],
        ),
        Key(
            created_at="synthetic-test-only",
            seed=7,
            sources=[source_c, source_b],
            pairs=[SealedPair(brief="brief-01", repetition=0, a=source_c, b=source_b)],
        ),
    )
    (batch / "sealed" / binding_file).write_text("SECRET-PLAN-CRITIC-LEDGER")
    server = uvicorn.Server(uvicorn.Config(create_app(batch), log_level="error"))
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
            _read_in_browser(f"http://127.0.0.1:{port}", binding_file)
            assert read_judgements(batch)[1].a is not None
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive()


def _read_in_browser(base: str, binding_file: str) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, channel=os.environ.get("NDE_BROWSER_CHANNEL", "chrome")
        )
        try:
            page = browser.new_page()
            page.goto(f"{base}/pairs/1")
            assert "SECRET-" not in page.content()
            for side in ("a", "b"):
                for dimension in ("coherence", "originality"):
                    control = page.locator(f'input[name="score-{side}-{dimension}"][value="1"]')
                    assert control.get_attribute("required") == ""
            assert (
                page.request.post(
                    f"{base}/pairs/1", form={"preferred": "A", "strength": "2"}
                ).status
                == 422
            )
            assert page.request.post(f"{base}/close").status == 409
            assert page.request.get(f"{base}/sealed/key.json").status == 404
            assert page.request.get(f"{base}/sealed/{binding_file}").status == 404
            assert page.request.get(f"{base}/api/progress").status == 200
            for side in ("a", "b"):
                page.locator(f'input[name="score-{side}-coherence"][value="4"]').check()
                page.locator(f'input[name="score-{side}-originality"][value="3"]').check()
            page.locator('input[name="preferred"][value="A"]').check()
            page.locator('input[name="strength"][value="2"]').check()
            page.get_by_role("button", name="Save and continue").click()
            saved = page.request.get(f"{base}/api/pairs/1").json()
            assert saved["judgement"]["a"]["scores"] == {"coherence": 4, "originality": 3}
            assert not any(word in str(saved) for word in ("sealed", "condition", "SECRET-"))
            page.get_by_role("button", name="Close the batch").click()
            page.reload()
            progress = page.request.get(f"{base}/api/progress").json()
            assert progress == {"pairs": 1, "judged": 1, "closed": True}
        finally:
            browser.close()
