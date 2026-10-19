import pytest

from nde.stories.normalise import normalise, word_count

MARKDOWN_STORY = """\
Here is a 2,000-word story based on your brief:

# The Last Lamp

## Scene 1: The Storm

**Ada** climbed the *ninety* steps, as she had every night for __thirty__ years.
She did not look down.

* * *

**Scene 2**

> "You're late," said Bram.

Chapter Three

- She lit the lamp.

---

The End.

I hope you enjoyed this story! Let me know if you'd like any changes.
"""

MARKDOWN_EXPECTED = """\
Ada climbed the ninety steps, as she had every night for thirty years.

She did not look down.

"You're late," said Bram.

She lit the lamp.
"""


def test_markup_headers_scene_titles_preamble_and_sign_off_are_removed() -> None:
    assert normalise(MARKDOWN_STORY) == MARKDOWN_EXPECTED


def test_quotation_marks_and_ellipses_are_made_consistent() -> None:
    text = "\u201cWait\u2026\u201d she said. \u2018It\u2019s fine,\u2019 he lied."

    assert normalise(text) == "\"Wait...\" she said. 'It's fine,' he lied.\n"


def test_paragraphs_are_separated_by_one_blank_line_with_spacing_collapsed() -> None:
    text = "  First   paragraph.\r\n\r\n\r\n\tSecond\u00a0paragraph.  \nThird.\n\n\n"

    assert normalise(text) == "First paragraph.\n\nSecond paragraph.\n\nThird.\n"


def test_a_leading_title_line_is_removed_but_a_short_first_sentence_is_kept() -> None:
    assert normalise("The Keeper of Light\n\nIt began at dusk.") == "It began at dusk.\n"
    assert normalise("Run.\n\nShe ran.") == "Run.\n\nShe ran.\n"
    assert normalise('"Now!"\n\nShe ran.') == '"Now!"\n\nShe ran.\n'


def test_title_and_word_count_labels_are_removed() -> None:
    text = "Title: The Keeper\nWord count: 2,013\n\nIt began at dusk.\n\n(Word count: 2,013)"

    assert normalise(text) == "It began at dusk.\n"


def test_links_inline_code_html_and_fences_are_reduced_to_their_text() -> None:
    text = "```\nShe read [the letter](http://x.example) and `wept`.<br>\n```"

    assert normalise(text) == "She read the letter and wept.\n"


def test_underscores_and_asterisks_inside_words_are_kept() -> None:
    assert normalise("The file was snake_case_name and 2*3*4 was 24.") == (
        "The file was snake_case_name and 2*3*4 was 24.\n"
    )


@pytest.mark.parametrize("text", [MARKDOWN_STORY, MARKDOWN_EXPECTED, "One.\n\nTwo."])
def test_normalisation_is_idempotent(text: str) -> None:
    once = normalise(text)

    assert normalise(once) == once


def test_words_are_counted_on_whitespace() -> None:
    assert word_count(MARKDOWN_EXPECTED) == 26
    assert word_count("") == 0
