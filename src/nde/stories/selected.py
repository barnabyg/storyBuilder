"""Read final prose only when it still matches the run's recorded selection."""

from pathlib import Path

from nde.files import checksum_lf
from nde.runs.records import STORY_FILE, StoryRecord
from nde.stories.normalise import normalise


def selected_text(run: Path, record: StoryRecord) -> str:
    folder = (run / record.path).resolve()
    if not folder.is_relative_to(run.resolve()):
        raise ValueError("selected story path escapes run directory")
    data = (folder / STORY_FILE).read_bytes()
    if record.sha256 is not None and checksum_lf(data) != record.sha256:
        raise ValueError("final story checksum changed from the recorded selection")
    text = data.decode("utf-8").replace("\r\n", "\n")
    raw = (folder / "story.raw.md").read_text(encoding="utf-8")
    if normalise(raw) != text:
        raise ValueError("final story differs from selected raw output")
    return text
