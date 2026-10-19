"""Helpers for story-run tests: configs, fake replies and run-directory readers."""

import json
from pathlib import Path
from typing import Any

from nde.cli import main
from tests.conftest import REPO, base_config, write_config
from tests.fake_openai import Json, Reply, completion


def story_text(words: int, marker: str = "word") -> str:
    """A plain story of exactly `words` words in paragraphs of 50."""
    tokens = [marker] * words
    paragraphs = [" ".join(tokens[start : start + 50]) + "." for start in range(0, words, 50)]
    return "\n\n".join(paragraphs)


def story_config(base_url: str, conditions: list[dict[str, Any]], **stories: Any) -> dict[str, Any]:
    config = base_config(base_url)
    del config["calls"]
    config["roles"]["planner"] = {"provider": "fake", "model": "fake-model", "params": {}}
    config["stories"] = {
        "briefs_dir": str(REPO / "benchmark"),
        "briefs": {"ids": ["brief-01", "brief-02"]},
        "conditions": conditions,
    } | stories
    return config


BASELINE_A = {"name": "A", "pipeline": "baseline_a"}


def run(tmp_path: Path, config: dict[str, Any], expect: int = 0) -> Path:
    runs = tmp_path / "runs"
    before: set[Path] = set(runs.iterdir()) if runs.exists() else set()
    code = main(["run", str(write_config(tmp_path, config)), "--runs-dir", str(runs)])
    assert code == expect
    [run_dir] = set(runs.iterdir()) - before
    return run_dir


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def stories(run_dir: Path) -> list[dict[str, Any]]:
    return jsonl(run_dir / "stories.jsonl")


def story_files(run_dir: Path) -> dict[str, bytes]:
    """Every file under the run's stories folder, plus the story index."""
    files = {
        path.relative_to(run_dir).as_posix(): path.read_bytes()
        for path in sorted((run_dir / "stories").rglob("*"))
        if path.is_file()
    }
    return files | {"stories.jsonl": (run_dir / "stories.jsonl").read_bytes()}


def last_user_message(body: Json) -> str:
    messages: list[Json] = body["messages"]
    return str(messages[-1]["content"])


def reply_with_story(words: int) -> Reply:
    return completion(story_text(words), completion_tokens=words)
