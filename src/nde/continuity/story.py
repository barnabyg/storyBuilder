"""A finished story as numbered paragraphs, each belonging to a planned scene.

Checks read and cite paragraph numbers rather than quotations, so a citation
cannot be misquoted and order in the story can be compared exactly.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from nde.stories.normalise import normalise


def _split(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]


@dataclass(frozen=True)
class NumberedStory:
    paragraphs: tuple[str, ...]
    # The 1-based planned scene of each paragraph.
    scenes: tuple[int, ...]

    @classmethod
    def from_scenes(cls, story: str, scene_texts: Sequence[str]) -> "NumberedStory":
        """Number `story`, already normalised, using the scene texts it was joined from.

        Normalising the joined story can drop a paragraph the per-scene pass kept, such
        as a title-like first line, so each paragraph is matched by its text, in order.
        A paragraph with no match stays in the scene before it.
        """
        paragraphs = _split(story)
        written = [
            (paragraph, number)
            for number, text in enumerate(scene_texts, 1)
            for paragraph in _split(normalise(text))
        ]
        scenes: list[int] = []
        position = 0
        current = 1
        for paragraph in paragraphs:
            for index in range(position, len(written)):
                if written[index][0] == paragraph:
                    current = written[index][1]
                    position = index + 1
                    break
            scenes.append(current)
        return cls(tuple(paragraphs), tuple(scenes))

    def __len__(self) -> int:
        return len(self.paragraphs)

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs) + "\n"

    def paragraph(self, number: int) -> str:
        return self.paragraphs[number - 1]

    def scene(self, number: int) -> int:
        return self.scenes[number - 1]

    def first_of_scene(self, scene: int) -> int:
        """The first paragraph of `scene`; paragraph 1 if the scene wrote no paragraphs."""
        return self.scenes.index(scene) + 1 if scene in self.scenes else 1

    def render(self) -> str:
        """The story as a checker reads it, with scene headings and paragraph numbers."""
        lines: list[str] = []
        current = 0
        for number, (paragraph, scene) in enumerate(
            zip(self.paragraphs, self.scenes, strict=True), 1
        ):
            if scene != current:
                lines.append(f"--- Scene {scene} ---")
                current = scene
            lines.append(f"[{number}] {paragraph}")
        return "\n\n".join(lines)

    def with_paragraph(self, number: int, text: str) -> "NumberedStory":
        """A copy with one paragraph replaced and every other paragraph untouched."""
        paragraphs = list(self.paragraphs)
        paragraphs[number - 1] = text
        return replace(self, paragraphs=tuple(paragraphs))
