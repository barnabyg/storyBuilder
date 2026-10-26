"""Pilot registration fixture shared by coverage and G2 tests."""

import json
from pathlib import Path


def pilot_config(directory: Path, inputs: dict[str, Path], *, cap: float = 1) -> Path:
    path = directory / "pilot-config.json"
    path.write_text(
        json.dumps(
            {
                "cohort": str(inputs["cohort"]),
                "extraction_config": str(inputs["family_config"]),
                "test_only": True,
                "per_run_usd": cap,
                "per_day_usd": 10,
                "max_stories": 1,
                "works": [
                    {
                        "work_id": "lamp",
                        "work_dir": str(inputs["work"]),
                        "segmentation": str(inputs["segments"]),
                    }
                ],
            }
        )
    )
    return path
