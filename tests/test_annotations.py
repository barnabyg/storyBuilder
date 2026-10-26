import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from nde.cli import main
from tests.conftest import REPO
from tests.fake_openai import FakeOpenAI, completion


def annotation_inputs(tmp_path: Path, fake: FakeOpenAI) -> tuple[Path, Path]:
    raw = b"START\n\nThe keeper saw the boat. She lit the lamp.\n\nEND\n"
    source = tmp_path / "source.txt"
    source.write_bytes(raw)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "stories": [
                    {
                        "id": "keeper",
                        "title": "Keeper",
                        "author": "Fixture author",
                        "source": "https://example.org/fixture",
                        "edition": "Test fixture",
                        "licence_basis": "Public domain fixture",
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "start": "START",
                        "end": "END",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert main(["calibration", "import", "keeper", str(source), "--manifest", str(manifest)]) == 0
    config = tmp_path / "annotation.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 7,
                "prompts_dir": str(REPO / "prompts"),
                "providers": {"fake": {"base_url": fake.base_url}},
                "annotator": {
                    "provider": "fake",
                    "model": "fake-model",
                    "params": {"max_tokens": 1000},
                },
                "prices": {"fake/fake-model": {"input_per_mtok": 0.1, "output_per_mtok": 0.2}},
                "budget": {"per_run_usd": 1, "per_day_usd": 5},
                "max_repairs": 1,
            }
        ),
        encoding="utf-8",
    )
    return manifest, config


def draft_args(manifest: Path, config: Path, out: Path, tmp_path: Path) -> list[str]:
    return [
        "annotate",
        "draft",
        "keeper",
        "--manifest",
        str(manifest),
        "--config",
        str(config),
        "--out",
        str(out),
        "--cache",
        str(tmp_path / "cache.sqlite"),
        "--ledger",
        str(tmp_path / "ledger.sqlite"),
    ]


def test_draft_is_persisted_source_linked_and_explicitly_unreviewed(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    manifest, config = annotation_inputs(tmp_path, fake_llm)
    data: dict[str, Any] = json.loads((REPO / "examples/harbour.plan.json").read_text())
    fake_llm.queue(completion(json.dumps(data)))
    out = tmp_path / "annotation"
    assert main(draft_args(manifest, config, out, tmp_path)) == 0
    state = json.loads((out / "session.json").read_text())
    assert state["status"] == "drafted"
    assert state["owner_reviews"] == []
    assert (
        state["source"]["sha256"]
        == hashlib.sha256(b"The keeper saw the boat. She lit the lamp.\n").hexdigest()
    )
    assert state["spent_usd"] > 0
    assert (out / "draft/cards.html").is_file()
    assert (out / "original.txt").read_text() == json.dumps(data)


def test_semantic_repair_and_cache_reuse_preserve_original(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    manifest, config = annotation_inputs(tmp_path, fake_llm)
    good: dict[str, Any] = json.loads((REPO / "examples/harbour.plan.json").read_text())
    bad = json.loads(json.dumps(good))
    bad["scenes"][0]["character_ids"] = ["ghost"]
    fake_llm.queue(completion(json.dumps(bad)), completion(json.dumps(good)))
    first, second = tmp_path / "first", tmp_path / "second"
    assert main(draft_args(manifest, config, first, tmp_path)) == 0
    assert main(draft_args(manifest, config, second, tmp_path)) == 0
    saved = json.loads((second / "session.json").read_text())
    assert saved["spent_usd"] == 0
    assert len(saved["attempts"]) == 2
    assert "character_reference" in saved["attempts"][0]["errors"][0]
    assert fake_llm.call_count == 2
    assert (first / "original.txt").read_text() == json.dumps(bad)


@pytest.mark.parametrize("failure", ["exhausted", "budget_aborted"])
def test_bounded_failures_are_saved(tmp_path: Path, fake_llm: FakeOpenAI, failure: str) -> None:
    manifest, config = annotation_inputs(tmp_path, fake_llm)
    if failure == "budget_aborted":
        data = yaml.safe_load(config.read_text())
        data["budget"]["per_run_usd"] = 0
        config.write_text(yaml.safe_dump(data), encoding="utf-8")
    fake_llm.queue(completion("{}"), completion("{}"))
    out = tmp_path / "failed"
    assert main(draft_args(manifest, config, out, tmp_path)) == 1
    state = json.loads((out / "session.json").read_text())
    assert state["status"] == failure
    assert fake_llm.call_count == (0 if failure == "budget_aborted" else 2)
    assert state["owner_reviews"] == []


def test_revisions_timer_and_review_are_separate_from_draft(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, config = annotation_inputs(tmp_path, fake_llm)
    good = (REPO / "examples/harbour.plan.json").read_text()
    fake_llm.queue(completion(good))
    out = tmp_path / "work"
    assert main(draft_args(manifest, config, out, tmp_path)) == 0

    monkeypatch.setattr(time, "time", lambda: 100.0)
    assert main(["annotate", "timer", str(out), "start"]) == 0
    monkeypatch.setattr(time, "time", lambda: 160.0)
    assert main(["annotate", "timer", str(out), "pause"]) == 0
    correction = tmp_path / "correction.json"
    correction.write_text(good, encoding="utf-8")
    assert main(["annotate", "revise", str(out), str(correction)]) == 0
    state = json.loads((out / "session.json").read_text())
    assert state["owner_reviews"] == []
    assert state["revisions"][0]["valid"] is True
    assert (
        main(
            [
                "annotate",
                "review",
                str(out),
                "1",
                "--owner",
                "test-owner",
                "--notes",
                "Explicit test of evidence recording only.",
                "--accept",
                "--meaningful-trajectories",
            ]
        )
        == 0
    )
    assert main(["annotate", "status", str(out)]) == 0
    events = [json.loads(line) for line in (out / "timer.jsonl").read_text().splitlines()]
    assert events[-1]["seconds"] == 60
    assert (out / "revisions/0001/cards.html").is_file()


def test_resume_keeps_progress_and_recovers_unfinished_intervals(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, config = annotation_inputs(tmp_path, fake_llm)
    good = (REPO / "examples/harbour.plan.json").read_text()
    fake_llm.queue(completion(good))
    out = tmp_path / "work"
    args = draft_args(manifest, config, out, tmp_path)
    assert main(args) == 0
    original = (out / "original.txt").read_bytes()
    state = json.loads((out / "session.json").read_text())
    state["status"] = "running"  # interruption after saving response, before rendering completion
    (out / "session.json").write_text(json.dumps(state), encoding="utf-8")
    assert main(args) == 0
    assert fake_llm.call_count == 1
    assert (out / "original.txt").read_bytes() == original
    pending = out / "revisions/0001/submitted.json"
    pending.parent.mkdir(parents=True)
    pending.write_text(good, encoding="utf-8")
    assert main(["annotate", "revise", str(out), str(pending)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["annotate", "revise", str(out), str(bad)]) == 1
    assert pending.read_text() == good
    assert (
        main(
            [
                "annotate",
                "review",
                str(out),
                "2",
                "--owner",
                "test",
                "--notes",
                "Invalid corrections cannot be approved.",
                "--accept",
            ]
        )
        == 2
    )
    monkeypatch.setattr(time, "time", lambda: 100.0)
    assert main(["annotate", "timer", str(out), "start"]) == 0
    monkeypatch.setattr(time, "time", lambda: 10000.0)
    assert main(["annotate", "timer", str(out), "recover"]) == 0
    events = [json.loads(line) for line in (out / "timer.jsonl").read_text().splitlines()]
    assert events[-1]["seconds"] == 0


def test_changed_source_is_rejected_before_provider_call(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    manifest, config = annotation_inputs(tmp_path, fake_llm)
    (tmp_path / "calibration/keeper.source.txt").write_text("changed", encoding="utf-8")
    assert main(draft_args(manifest, config, tmp_path / "work", tmp_path)) == 2
    assert fake_llm.call_count == 0
