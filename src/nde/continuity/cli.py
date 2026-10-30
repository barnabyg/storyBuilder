"""`nde continuity`: one development brief to a saved story, sheets, flags and cost."""

import json
from pathlib import Path

from nde.continuity.pipeline import RESULT_FILE
from nde.runs.config import BriefSelection
from nde.runs.records import read_manifest, read_stories
from nde.runs.runner import RunOptions, execute, prepare

DEFAULT_CONFIG = Path("configs/continuity-dev.yaml")


def run_brief(brief: str, config: Path, options: RunOptions) -> tuple[str, str]:
    """Run one dev brief; returns the run's status and a report for the terminal.

    Raises ConfigError before any call if the brief is not in the dev split.
    """
    selection = BriefSelection(split="dev", ids=[brief])
    run_dir = execute(prepare(config, options.environ, briefs=selection), options)
    status = read_manifest(run_dir).status
    lines = [f"run {status}: {run_dir}"]
    for story in read_stories(run_dir):
        folder = run_dir / story.path
        lines.append(f"story folder: {folder}")
        if story.error:
            lines.append(f"story failed: {story.error}")
        result_path = folder / RESULT_FILE
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            lines.append(f"flagged problems: {len(result['flagged_problems'])}")
            lines.append(f"unresolved after repair: {len(result['unresolved_problems'])}")
            lines.append(f"cost: US${result['cost']['usd']:.4f}")
    return status, "\n".join(lines)
