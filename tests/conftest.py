from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.fake_openai import FakeOpenAI

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Commands read `.env` and default state paths from the working directory;
    # run every test in its own empty directory so a developer's files never leak in.
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def fake_llm() -> Iterator[FakeOpenAI]:
    server = FakeOpenAI()
    server.start()
    yield server
    server.stop()


def base_config(base_url: str) -> dict[str, Any]:
    """A minimal valid experiment config pointing at a fake server."""
    return {
        "name": "test",
        "seed": 7,
        "prompts_dir": str(REPO / "prompts"),
        # pragma: allowlist nextline secret - names a variable that holds a fake key
        "providers": {"fake": {"base_url": base_url, "api_key_env": "FAKE_API_KEY"}},
        "roles": {"prose": {"provider": "fake", "model": "fake-model", "params": {}}},
        "calls": [
            {
                "id": "line",
                "role": "prose",
                "prompt": {"template": "smoke", "version": 1},
                "variables": {"setting": "a quiet harbour"},
            }
        ],
    }


def write_config(directory: Path, config: dict[str, Any], name: str = "config.yaml") -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path
