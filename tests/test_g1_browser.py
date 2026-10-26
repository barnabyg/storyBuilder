"""Opt-in local browser acceptance: no browser or package downloads in default tests."""

import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import uvicorn

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
    not os.environ.get("NDE_BROWSER_NODE"),
    reason="set NDE_BROWSER_NODE and NODE_PATH to installed Node/Playwright for browser acceptance",
)
def test_synthetic_g1_browser_controls_api_storage_and_blinding(tmp_path: Path) -> None:
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
            # S603: fixed test script; installed Node path is an explicit test prerequisite.
            subprocess.run(  # noqa: S603
                [
                    os.environ["NDE_BROWSER_NODE"],
                    str(REPO / "tests/browser/g1-reader.cjs"),
                    f"http://127.0.0.1:{port}",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=45,
            )
            assert read_judgements(batch)[1].a is not None
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive()
