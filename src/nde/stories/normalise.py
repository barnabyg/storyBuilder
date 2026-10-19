"""Story normalisation: strip everything but the prose, so format cannot reveal the condition.

Every story is stored raw and normalised; blind batches and judges only ever
see the normalised form. The rules are deterministic:

1. Line endings become LF; non-breaking and other special spaces become spaces.
2. Curly quotation marks become straight ones, and an ellipsis character
   becomes three full stops.
3. Code fences, Markdown headings, horizontal rules and scene-break lines
   (`***`, `* * *`, `---`, `#`) are removed, as are scene or chapter titles
   ("Scene 2", "Chapter Three: The Storm") and labels such as "Title:" or
   "Word count:".
4. Markdown and HTML markup is reduced to its text: emphasis, inline code,
   links, block quotes, list bullets and tags.
5. A model preamble ("Here is your story:") and a leading title line are
   removed, as are sign-offs at the end ("The End", "I hope you enjoyed the
   story"): a closing line that starts like one but is not about the story or
   the request, or is dialogue, is part of the story and stays.
6. Every remaining non-empty line is a paragraph, with runs of whitespace
   collapsed; paragraphs are separated by one blank line.

Scene breaks are dropped rather than standardised: some conditions produce
them and others do not, so even a uniform marker could reveal the condition.
"""

import re

# Non-breaking, figure, thin and narrow no-break spaces, and tabs, become spaces.
_SPACES = str.maketrans(dict.fromkeys(map(chr, (0x00A0, 0x2007, 0x2009, 0x202F, 0x09)), " "))
# Curly and angled double quotes, then curly single quotes, become straight ones.
_QUOTES = str.maketrans(
    dict.fromkeys(map(chr, (0x201C, 0x201D, 0x201E, 0x201F, 0x00AB, 0x00BB)), '"')
    | dict.fromkeys(map(chr, (0x2018, 0x2019, 0x201A, 0x201B)), "'")
)
_ELLIPSIS = chr(0x2026)

_FENCE = re.compile(r"^(```|~~~)")
_HEADING = re.compile(r"^#{1,6}(\s|$)")
_BREAK = re.compile(r"^[\s*\-_=~#\u2022\u00b7]+$")
_QUOTE_PREFIX = re.compile(r"^(>\s?)+")
_BULLET = re.compile(r"^([-*+\u2022]|\d+[.)])\s+")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_TAG = re.compile(r"<[^>]+>")
_CODE = re.compile(r"`([^`]*)`")
_STRONG = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_EMPHASIS = re.compile(
    r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])|(?<![\w_])_(?=\S)(.+?)(?<=\S)_(?![\w_])"
)
_WHITESPACE = re.compile(r"\s+")

_NUMBER = r"([0-9]+|[ivxlc]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_SCENE_TITLE = re.compile(rf"^(scene|chapter|part|act)\s+{_NUMBER}\b[^.!?]*$", re.IGNORECASE)
_LABEL = re.compile(r"^\(?\s*(title|word count|words|length)\s*:", re.IGNORECASE)
_MAX_TITLE_CHARS = 80

_CHATTER = re.compile(
    r"^(sure|certainly|of course|absolutely|okay|ok|here('s| is| are)|below is|what follows)\b",
    re.IGNORECASE,
)
_CHATTER_TOPIC = re.compile(r"\b(story|brief|words|request)\b", re.IGNORECASE)
_END_MARK = re.compile(r"^((the )?end|fin|finis)[.!]?$", re.IGNORECASE)
_NOTE = re.compile(r"^(note:|author'?s note)", re.IGNORECASE)
# A closing paragraph that starts like a model talking to the user is a sign-off only
# if it also talks about the story or the request, and is not dialogue: "I can still
# hear the gulls, she thought." is a story's last line.
_SIGN_OFF_START = re.compile(
    r"^(i hope|hope you|let me know|feel free|would you like|if you'?d like|i can )",
    re.IGNORECASE,
)
_SIGN_OFF_TOPIC = re.compile(
    r"\b(story|stories|scene|works|enjoy\w*|changes?|revis\w*|adjust\w*|edit\w*|"
    r"versions?|expand\w*|shorten\w*|longer|shorter|tone|feedback|request|help)\b",
    re.IGNORECASE,
)


def _is_sign_off(paragraph: str) -> bool:
    if _END_MARK.match(paragraph) or _NOTE.match(paragraph):
        return True
    return (
        bool(_SIGN_OFF_START.match(paragraph))
        and '"' not in paragraph
        and bool(_SIGN_OFF_TOPIC.search(paragraph))
    )


_MAX_TITLE_WORDS = 12
# A first line ending in one of these is prose, not a title (0x2014 is an em dash).
_TERMINAL = (*".!?'\":;,)-", chr(0x2014))


def _strip_markup(line: str) -> str:
    line = _QUOTE_PREFIX.sub("", line)
    line = _BULLET.sub("", line)
    line = _LINK.sub(r"\1", line)
    line = _TAG.sub("", line)
    line = _CODE.sub(r"\1", line)
    line = _STRONG.sub(r"\2", line)
    line = _EMPHASIS.sub(lambda match: match.group(1) or match.group(2), line)
    return _WHITESPACE.sub(" ", line).strip()


def _paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or _FENCE.match(line) or _HEADING.match(line) or _BREAK.match(line):
            continue
        line = _strip_markup(line)
        if not line or _LABEL.match(line):
            continue
        if len(line) <= _MAX_TITLE_CHARS and _SCENE_TITLE.match(line):
            continue
        paragraphs.append(line)
    return paragraphs


def _is_preamble(paragraph: str) -> bool:
    return bool(_CHATTER.match(paragraph)) and (
        paragraph.endswith(":") or bool(_CHATTER_TOPIC.search(paragraph))
    )


def _is_title(paragraph: str) -> bool:
    return len(paragraph.split()) <= _MAX_TITLE_WORDS and not paragraph.endswith(_TERMINAL)


def normalise(text: str) -> str:
    """The story's prose alone, in a consistent format. See the module docstring."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").translate(_SPACES)
    text = text.translate(_QUOTES).replace(_ELLIPSIS, "...")
    paragraphs = _paragraphs(text)
    while paragraphs and _is_preamble(paragraphs[0]):
        paragraphs.pop(0)
    if len(paragraphs) > 1 and _is_title(paragraphs[0]):
        paragraphs.pop(0)
    while paragraphs and _is_sign_off(paragraphs[-1]):
        paragraphs.pop()
    return "\n\n".join(paragraphs) + "\n" if paragraphs else ""


def word_count(text: str) -> int:
    return len(text.split())
