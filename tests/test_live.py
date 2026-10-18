"""Opt-in tests against real providers: `uv run pytest -m live`.

They spend real money (a fraction of a cent) and are excluded from `nde-verify`.
"""

import os
from pathlib import Path

import pytest

from nde.cli import main
from nde.env import load_dotenv
from tests.conftest import REPO

pytestmark = pytest.mark.live


def test_smoke_config_against_the_real_provider(tmp_path: Path) -> None:
    load_dotenv(REPO / ".env")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is not set")

    code = main(["run", str(REPO / "configs" / "smoke.yaml"), "--runs-dir", str(tmp_path)])

    assert code == 0
