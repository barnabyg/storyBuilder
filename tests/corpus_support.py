"""Small synthetic corpus with real persisted source, cohort and fake-provider annotations."""

import json
from pathlib import Path
from typing import Any

from nde.cli import main
from nde.corpus.acquisition import load_work
from nde.corpus.segmentation import load_segmentation
from tests.fake_openai import FakeOpenAI, completion
from tests.test_corpus_acquisition import acquire


def corpus_inputs(directory: Path, fake: FakeOpenAI, *, work_id: str = "lamp") -> dict[str, Path]:
    work = acquire(
        directory,
        "BEGIN LAMP\nAnn saw the lamp.\n\n***\n\nAnn lit the lamp.\nEND LAMP",
        work_id=work_id,
    )
    segments = directory / "segments"
    assert main(["corpus", "segment", str(work), "--out", str(segments)]) == 0
    config = directory / "cohort-config.json"
    config.write_text(
        json.dumps({"id": "fixture", "works": [str(work)], "author_splits": {"ann": "train"}})
    )
    cohort = directory / "cohort"
    assert main(["corpus", "split", "freeze", str(config), "--out", str(cohort)]) == 0
    family = directory / "families-config.json"
    family.write_text(
        json.dumps(
            {
                "providers": {"fake": {"base_url": fake.base_url, "retry": {"max_retries": 0}}},
                "annotators": [
                    {
                        "provider": "fake",
                        "model": name,
                        "family": name,
                        "params": {"max_tokens": 2000},
                    }
                    for name in ("alpha", "beta")
                ],
                "prices": {
                    f"fake/{name}": {"input_per_mtok": 1, "output_per_mtok": 1}
                    for name in ("alpha", "beta")
                },
                "budget": {"per_run_usd": 1, "per_day_usd": 10},
                "max_repairs": 0,
            }
        )
    )
    return {
        "work": work,
        "segments": segments / "segmentation.json",
        "cohort": cohort / "cohort.json",
        "family_config": family,
    }


def queue_annotation(inputs: dict[str, Path], fake: FakeOpenAI) -> None:
    work, text = load_work(inputs["work"])
    segments = load_segmentation(inputs["segments"], inputs["work"])
    scene_ids = [s.id for s in segments.scenes]
    plan: dict[str, Any] = {
        "id": "fixture-plan",
        "title": "The Lamp",
        "tier": "L",
        "sources": [
            {
                "id": work.source_id,
                "title": work.title,
                "text": text,
                "sha256": work.text_sha256,
                "provenance": work.source,
            }
        ],
        "characters": [{"id": "ann", "name": "Ann"}],
        "scenes": [
            {
                "id": s.id,
                "title": f"Moment {i}",
                "character_ids": ["ann"],
                "evidence": [
                    {
                        "source_id": work.source_id,
                        "source_sha256": work.text_sha256,
                        "start": s.start,
                        "end": s.end,
                    }
                ],
            }
            for i, s in enumerate(segments.scenes)
        ],
        "causal_links": [
            {"id": "cause", "cause_scene_id": scene_ids[0], "effect_scene_id": scene_ids[1]}
        ],
        "knowledge_facts": [
            {"id": "lamp-fact", "description": "The lamp is present", "initial_holder_ids": ["ann"]}
        ],
    }
    structural = json.dumps(plan)
    plan["states"] = [
        {
            "id": kind,
            "scene_id": scene_ids[0],
            "character_id": "ann",
            "kind": kind,
            "key": "lamp",
            "before": {"status": "unknown"},
            "after": {"value": value},
        }
        for kind, value in (("goal", "light the lamp"), ("belief", "the lamp works"))
    ]
    state = json.dumps(plan)
    for _ in ("alpha", "beta"):
        fake.queue(completion(structural), completion(state), completion(state))


def review_config(directory: Path, inputs: dict[str, Path], families: Path, features: Path) -> Path:
    path = directory / "review-config.json"
    path.write_text(
        json.dumps(
            {
                "cohort": str(inputs["cohort"]),
                "test_only": True,
                "min_owner_per_channel": 1,
                "min_gold_per_channel": 0,
                "channels": ["plot", "causality", "information", "goal", "belief"],
                "inputs": [
                    {
                        "work_id": "lamp",
                        "work_dir": str(inputs["work"]),
                        "segmentation": str(inputs["segments"]),
                        "features": str(features),
                        "families": str(families),
                    }
                ],
            }
        )
    )
    return path


def label_packet(directory: Path, packet: Path, *, judgement: str = "defensible") -> None:
    saved = json.loads((packet / "packet.json").read_text())
    for sample in saved["samples"]:
        label = directory / "label.json"
        label.write_text(
            json.dumps(
                {
                    "sample_id": sample["id"],
                    "owner": "synthetic fixture",
                    "notes": "Test-only known answer; no literary judgement.",
                    "judgement": judgement,
                    "test_only": True,
                }
            )
        )
        assert main(["corpus-review", "label", str(packet), str(label)]) == 0
