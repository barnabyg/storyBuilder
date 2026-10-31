"""Helpers for continuity tests: a plan, fixture stories and a fake provider that serves them."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.fake_openai import FakeOpenAI, Json, Reply, completion
from tests.stories import jsonl, last_user_message, run, stories, story_config

CONTINUITY: dict[str, Any] = {"name": "continuity", "pipeline": "continuity"}
BRIEF = "brief-01"


def sheet(name: str, enters: int, how: str = "Walks in and says who they are.") -> dict[str, Any]:
    return {
        "name": name,
        "appearance": f"{name} is tall and wears a grey coat.",
        "back_story": f"{name} has lived on the island all their life.",
        "enters_scene": enters,
        "introduction": how,
        "knows_at_start": [f"{name} knows the tide tables."],
    }


def plan_reply(scenes: int = 5) -> dict[str, Any]:
    return {
        "title": "The Last Lamp",
        "characters": [sheet("Elsie Marr", 1), sheet("Tom Reyes", 2)],
        "scenes": [
            {
                "summary": f"Summary {i}.",
                "beats": [f"beat {i}a", f"beat {i}b"],
                "elapsed_minutes": 60,
                "key_events": [
                    {
                        "event": "Food in the fridge smells sour",
                        "expected_min_minutes": 720,
                        "expected_max_minutes": 4320,
                    }
                ]
                if i == 3
                else [],
            }
            for i in range(1, scenes + 1)
        ],
    }


def scene_text(number: int, words: int = 400) -> str:
    """Two paragraphs a scene, so a story's paragraph numbers are easy to work out."""
    half = words // 2
    return (
        f"Scene{number} first " + "word " * (half - 3) + "end.\n\n"
        f"Scene{number} second " + "word " * (half - 3) + "end."
    )


def person(
    name: str, paragraph: int, *, introduced: bool = True, described: bool = True
) -> dict[str, Any]:
    return {
        "name": name,
        "first_paragraph": paragraph,
        "introduced": introduced,
        "described": described,
        "why": "The narrator says who they are." if introduced else "Nothing is said of them.",
    }


CLEAN_PEOPLE = [person("Elsie Marr", 1), person("Tom", 3)]

type Audits = dict[str, Any]


def clean_audits() -> Audits:
    """What each audit answers for a story with nothing wrong, by the phrase in its prompt."""
    return {
        "List every named person": {"people": CLEAN_PEOPLE},
        "List every event whose time": {"events": [], "scenes": []},
        "record each specific fact": {"entries": []},
    }


def fake_continuity(audits: Audits | None = None) -> Callable[[Json], Reply]:
    answers = clean_audits() if audits is None else audits

    def reply(body: Json) -> Reply:
        system = str(body["messages"][0]["content"])
        message = last_user_message(body)
        if "story planner" in system:
            return completion(json.dumps(plan_reply()))
        if "Now write scene" in message:
            number = int(message.split("Now write scene ")[1].split(" ")[0])
            return completion(scene_text(number), completion_tokens=400)
        if "Rewrite only this passage" in message:
            return completion(json.dumps({"replacement": "A repaired paragraph."}))
        for phrase, answer in answers.items():
            if phrase in message:
                return completion(json.dumps(answer))
        raise AssertionError(f"unexpected request: {message[:200]}")

    return reply


def continuity_config(base_url: str, *, repair_tries: int = 0) -> dict[str, Any]:
    condition = CONTINUITY | {"params": {"repair_tries": repair_tries}}
    config = story_config(base_url, [condition], briefs={"split": "dev", "ids": [BRIEF]})
    config["roles"]["critic"] = {"provider": "fake", "model": "fake-model", "params": {}}
    return config


def run_continuity(
    tmp_path: Path, fake: FakeOpenAI, *, repair_tries: int = 0, expect: int = 0
) -> tuple[Path, Path]:
    """Run the pipeline on the fake; returns the run directory and the story's folder."""
    config = continuity_config(fake.base_url, repair_tries=repair_tries)
    run_dir = run(tmp_path, config, expect=expect)
    [record] = stories(run_dir)
    return run_dir, run_dir / record["path"]


def result(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "continuity.json").read_text(encoding="utf-8"))


def flags_of(folder: Path, problem_type: str) -> list[dict[str, Any]]:
    return [f for f in result(folder)["flagged_problems"] if f["problem_type"] == problem_type]


def calls(run_dir: Path) -> list[dict[str, Any]]:
    return jsonl(run_dir / "calls.jsonl")
