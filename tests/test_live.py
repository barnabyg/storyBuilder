"""Opt-in tests against real providers: `uv run pytest -m live`.

They may spend real money (a fraction of a cent) and are excluded from `nde-verify`.
Each test skips itself when its provider is not available.
"""

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from nde.cli import main
from nde.env import load_dotenv
from tests.conftest import REPO, write_config

pytestmark = pytest.mark.live

LOCAL_CONFIG = REPO / "configs" / "local-ollama.yaml"


def test_smoke_config_against_the_real_provider(tmp_path: Path) -> None:
    load_dotenv(REPO / ".env")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is not set")

    code = main(["run", str(REPO / "configs" / "smoke.yaml"), "--runs-dir", str(tmp_path)])

    assert code == 0


def _local_models(base_url: str) -> list[str]:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=2.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return []
    data: list[dict[str, Any]] = response.json().get("data", [])
    return [str(model["id"]) for model in data]


def test_a_local_server_works_through_config_alone(tmp_path: Path) -> None:
    config: dict[str, Any] = yaml.safe_load(LOCAL_CONFIG.read_text(encoding="utf-8"))
    provider = config["providers"]["local"]
    base_url = os.environ.get(provider["base_url_env"]) or provider["base_url"]
    models = _local_models(base_url)
    if not models:
        pytest.skip(f"no local OpenAI-compatible server with models at {base_url}")
    # Use whichever model the server has, rather than requiring a specific download.
    model = config["roles"]["prose"]["model"]
    if model not in models:
        model = models[0]
        config["roles"]["prose"]["model"] = model
        config["prices"] = {f"local/{model}": {"input_per_mtok": 0, "output_per_mtok": 0}}
    config["prompts_dir"] = str(REPO / "prompts")
    runs = tmp_path / "runs"

    code = main(["run", str(write_config(tmp_path, config)), "--runs-dir", str(runs)])

    assert code == 0
    [run_dir] = list(runs.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["models"][0]["model"] == model
