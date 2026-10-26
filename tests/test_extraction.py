import argparse
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nde.corpus import acquisition, extraction_cli, segmentation
from tests.fake_openai import FakeOpenAI, completion


def extraction_inputs(tmp_path: Path, fake: FakeOpenAI) -> tuple[Path, Path, Path]:
    raw = tmp_path / "story.txt"
    raw.write_text("START\n\nMara saw the boat. She lit the lamp.\n\nEND\n", encoding="utf-8")
    spec = tmp_path / "work.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "work_id": "keeper",
                "source_id": "keeper-source",
                "title": "Keeper",
                "author_id": "fixture-author",
                "author": "Fixture author",
                "source": "https://example.org/fixture",
                "edition": "Fixture",
                "licence_basis": "Test fixture",
                "licence_evidence": "Test-only authored text",
                "format": "text",
                "start": "START",
                "end": "END",
            }
        ),
        encoding="utf-8",
    )
    work, segments = tmp_path / "work", tmp_path / "segments"
    acquisition.acquire(spec, work, raw)
    segmentation.segment_work(work, segments)
    config = tmp_path / "extract.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "providers": {"fake": {"base_url": fake.base_url}},
                "annotator": {"provider": "fake", "model": "fake-model", "family": "one"},
                "prices": {"fake/fake-model": {"input_per_mtok": 0.1, "output_per_mtok": 0.2}},
                "budget": {"per_run_usd": 1, "per_day_usd": 5},
                "max_repairs": 1,
            }
        ),
        encoding="utf-8",
    )
    return work, segments, config


def plan_fixture(work: Path, segments: Path) -> dict[str, Any]:
    record, text = acquisition.load_work(work)
    bounds = segmentation.load_segmentation(segments, work)
    return {
        "schema_version": "1",
        "id": "keeper-plan",
        "title": "Keeper",
        "tier": "L",
        "sources": [
            {
                "id": record.source_id,
                "title": record.title,
                "text": text,
                "sha256": record.text_sha256,
                "provenance": record.source,
            }
        ],
        "characters": [{"id": "mara", "name": "Mara"}],
        "scenes": [
            {
                "id": scene.id,
                "title": "Lamp",
                "summary": "Mara lights the lamp.",
                "character_ids": ["mara"],
                "evidence": [
                    {
                        "source_id": record.source_id,
                        "source_sha256": record.text_sha256,
                        "start": scene.start,
                        "end": scene.end,
                    }
                ],
            }
            for scene in bounds.scenes
        ],
    }


def extract_command(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    extraction_cli.configure(parser)
    return extraction_cli.run(parser.parse_args(arguments))


def command_args(work: Path, segments: Path, config: Path, out: Path) -> list[str]:
    return [
        "run",
        str(work),
        "--segments",
        str(segments),
        "--config",
        str(config),
        "--out",
        str(out),
        "--cache",
        str(out.parent / "cache.sqlite"),
        "--ledger",
        str(out.parent / "ledger.sqlite"),
    ]


def test_extraction_passes_persist_exact_source_and_conditioning(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    work, segments, config = extraction_inputs(tmp_path, fake_llm)
    good = plan_fixture(work, segments)
    fake_llm.queue(*(completion(json.dumps(good)) for _ in range(3)))
    out = tmp_path / "extraction"
    assert extract_command(command_args(work, segments, config, out)) == 0
    saved = json.loads((out / "extraction.json").read_text())
    assert saved["status"] == "complete"
    assert [row["name"] for row in saved["passes"]] == ["structural", "state", "interpretive"]
    assert all(row["status"] == "complete" for row in saved["passes"])
    assert saved["owner_approved"] is False
    assert (out / "annotation/cards.html").is_file()
    assert "accepted_plan" in fake_llm.requests[1].body["messages"][1]["content"]
    assert saved["spent_usd"] > 0
    assert extract_command(command_args(work, segments, config, out)) == 0
    assert fake_llm.call_count == 3


def test_report_rejects_checkpoint_claims_not_in_original_response(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    work, segments, config = extraction_inputs(tmp_path, fake_llm)
    good = plan_fixture(work, segments)
    fake_llm.queue(*(completion(json.dumps(good)) for _ in range(3)))
    out = tmp_path / "extraction"
    assert extract_command(command_args(work, segments, config, out)) == 0
    state = json.loads((out / "extraction.json").read_text())
    state["passes"][-1]["accepted_plan"]["themes"] = [
        {"id": "invented", "reading": {"status": "unknown"}}
    ]
    (out / "extraction.json").write_text(json.dumps(state), encoding="utf-8")
    assert extract_command(["report", str(out)]) == 2


@pytest.mark.parametrize("fault", ["span", "locked"])
def test_pass_repairs_keep_originals_and_accepted_channels(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    fault: str,
) -> None:
    work, segments, config = extraction_inputs(tmp_path, fake_llm)
    good = plan_fixture(work, segments)
    bad = json.loads(json.dumps(good))
    if fault == "span":
        bad["scenes"][0]["evidence"][0]["end"] = 999
        sequence = [bad, good, good, good]
        pass_name, diagnostic = "structural", "span_bounds"
    else:
        bad["scenes"][0]["summary"] = "Unrequested rewrite."
        sequence = [good, bad, good, good]
        pass_name, diagnostic = "state", "locked_channel"
    fake_llm.queue(*(completion(json.dumps(plan)) for plan in sequence))
    out = tmp_path / "extraction"
    assert extract_command(command_args(work, segments, config, out)) == 0
    state = json.loads((out / "extraction.json").read_text())
    row = next(item for item in state["passes"] if item["name"] == pass_name)
    assert diagnostic in " ".join(row["attempts"][0]["errors"])
    assert (out / f"{pass_name}-0000.txt").read_text() == json.dumps(bad)
    assert len(row["attempts"]) == 2


@pytest.mark.parametrize("cache_available", [True, False])
def test_interrupted_call_recovers_only_from_cache_without_repeating_paid_work(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    cache_available: bool,
) -> None:
    work, segments, config = extraction_inputs(tmp_path, fake_llm)
    good = plan_fixture(work, segments)
    fake_llm.queue(*(completion(json.dumps(good)) for _ in range(3)))
    out = tmp_path / "extraction"
    replace = Path.replace

    def fail_response_write(path: Path, target: Path) -> Path:
        if path.name == "extraction.tmp" and json.loads(path.read_text())["passes"][0]["attempts"]:
            raise OSError("simulated interrupted response publication")
        return replace(path, target)

    with monkeypatch.context() as context:
        context.setattr(Path, "replace", fail_response_write)
        assert extract_command(command_args(work, segments, config, out)) == 2
    assert fake_llm.call_count == 1
    if not cache_available:
        (tmp_path / "cache.sqlite").unlink()
    assert extract_command(command_args(work, segments, config, out)) == (
        0 if cache_available else 1
    )
    state = json.loads((out / "extraction.json").read_text())
    assert state["spent_usd"] > 0
    assert fake_llm.call_count == (3 if cache_available else 1)
    assert state["status"] == ("complete" if cache_available else "recovery_required")


def test_exhausted_or_changed_inputs_cannot_reset_attempts(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    work, segments, config = extraction_inputs(tmp_path, fake_llm)
    fake_llm.queue(completion("{}"), completion("{}"))
    out = tmp_path / "extraction"
    args = command_args(work, segments, config, out)
    assert extract_command(args) == 1
    assert extract_command(args) == 1
    assert fake_llm.call_count == 2
    changed = yaml.safe_load(config.read_text())
    changed["max_repairs"] = 3
    config.write_text(yaml.safe_dump(changed), encoding="utf-8")
    assert extract_command(args) == 2
    assert fake_llm.call_count == 2
