"""Synthetic scientific evidence built through public commands for G1 acceptance."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from nde.cli import main
from tests.conftest import REPO
from tests.fake_openai import FakeOpenAI, completion


def synthetic_schema_packet(root: Path, fake: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> Path:
    root.mkdir(parents=True)
    source = root / "source.txt"
    raw = ("START\n\n" + "The keeper waited by the shore. " * 400 + "\n\nEND\n").encode()
    source.write_bytes(raw)
    manifest = root / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "stories": [
                    {
                        "id": f"synthetic-{index}",
                        "title": f"Synthetic test-only story {index}",
                        "author": "Synthetic fixture",
                        "source": f"fixture://synthetic-{index}",
                        "edition": "Test-only",
                        "licence_basis": "Synthetic test-only text",
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "start": "START",
                        "end": "END",
                    }
                    for index in range(3)
                ]
            }
        ),
        encoding="utf-8",
    )
    config = root / "annotation.yaml"
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
    annotations = root / "annotations"
    for index in range(3):
        story_id = f"synthetic-{index}"
        assert (
            main(["calibration", "import", story_id, str(source), "--manifest", str(manifest)]) == 0
        )
        values = [0, 2, 4] if index == 0 else [4, 2, 0]
        fake.queue(
            completion(
                json.dumps(
                    {
                        "id": story_id,
                        "title": "Synthetic test-only annotation",
                        "scenes": [{"id": f"s{n}", "title": f"Scene {n}"} for n in range(3)],
                        "tension": [
                            {
                                "id": "t1",
                                "reading": {"value": "Synthetic reading"},
                                "trajectory": {
                                    "shape": "rising" if index == 0 else "falling",
                                    "points": [
                                        {"scene_id": f"s{n}", "value": values[n]} for n in range(3)
                                    ],
                                },
                            }
                        ],
                    }
                )
            )
        )
        folder = annotations / story_id
        assert (
            main(
                [
                    "annotate",
                    "draft",
                    story_id,
                    "--manifest",
                    str(manifest),
                    "--config",
                    str(config),
                    "--out",
                    str(folder),
                    "--cache",
                    str(root / "cache.sqlite"),
                    "--ledger",
                    str(root / "ledger.sqlite"),
                ]
            )
            == 0
        )
        assert main(["annotate", "revise", str(folder), str(folder / "draft/plan.json")]) == 0
        with monkeypatch.context() as timing:
            timing.setattr("nde.annotations.corrections.time.time", lambda: 1000.0)
            assert main(["annotate", "timer", str(folder), "start"]) == 0
            timing.setattr("nde.annotations.corrections.time.time", lambda: 1120.0)
            assert main(["annotate", "timer", str(folder), "pause"]) == 0
        assert (
            main(
                [
                    "annotate",
                    "review",
                    str(folder),
                    "1",
                    "--owner",
                    "SYNTHETIC-TEST-ONLY",
                    "--notes",
                    "TEST ONLY: synthetic review, no actual owner assessment.",
                    "--accept",
                    "--meaningful-trajectories",
                ]
            )
            == 0
        )
    out = root / "evidence"
    assert (
        main(
            [
                "schema-evidence",
                "--manifest",
                str(manifest),
                "--annotations",
                str(annotations),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    return out
