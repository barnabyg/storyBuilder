"""The blind reader: a small local web app for judging a batch.

A pair batch shows one pair at a time and asks for a preference (plus rubric
scores, tags and a condition guess when the batch has a rubric). A single-story
batch, such as a calibration batch, shows one story at a time and asks only
for rubric scores and tags.

It serves server-rendered HTML on localhost. It only ever reads the batch's
view, judgements or scores, warnings and closed marker (see `nde.eval.batch`),
never the sealed key, so no page or API response can reveal which condition,
or which source, a story came from.
"""

from html import escape
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from nde.eval.batch import (
    SIDES,
    SINGLE,
    BatchError,
    InvalidJudgementError,
    Judgement,
    StoryScores,
    close_batch,
    is_closed,
    judged,
    parse_judgement,
    parse_scores,
    read_judgements,
    read_scores,
    read_view,
    read_warnings,
    record_judgement,
    record_scores,
)
from nde.eval.rubric import Rubric

HOST = "127.0.0.1"
DEFAULT_PORT = 8770

_STRENGTHS = {1: "slightly better", 2: "clearly better", 3: "much better"}

_STYLE = """
body { font-family: Georgia, serif; margin: 0 auto; max-width: 75rem; padding: 1rem;
       color: #1d1d1f; background: #fbfaf7; }
header, nav, form, .notice { font-family: system-ui, sans-serif; }
header { display: flex; justify-content: space-between; border-bottom: 1px solid #ccc; }
.columns { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
@media (max-width: 50rem) { .columns { grid-template-columns: 1fr; } }
.single { max-width: 42rem; margin: 0 auto; }
article { line-height: 1.55; font-size: 1.05rem; }
form { background: #f0ede6; padding: 1rem; border-top: 1px solid #ccc; }
fieldset { border: none; margin: 0 0 0.75rem; padding: 0; }
legend { font-weight: 600; }
details { font-size: 0.85rem; color: #444; }
.notice { background: #f6e7c8; padding: 0.75rem; }
.problems { background: #f8d7d3; padding: 0.75rem; }
"""


def _page(title: str, body: str, status: int = 200) -> HTMLResponse:
    html = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body>{body}</body></html>"
    )
    return HTMLResponse(html, status_code=status)


def _story(heading: str, text: str) -> str:
    paragraphs = "".join(f"<p>{escape(p)}</p>" for p in text.split("\n\n") if p.strip())
    return f"<article><h2>{escape(heading)}</h2>{paragraphs}</article>"


def _radio(name: str, value: str, label: str, *, checked: bool) -> str:
    return (
        f'<label><input type="radio" name="{name}" value="{escape(value)}" required'
        f"{' checked' if checked else ''}> {escape(label)}</label> "
    )


def _scoring(rubric: Rubric, prefix: str, heading: str, scores: StoryScores | None) -> str:
    """Rubric scores, failure tags and free-text tags for one story."""
    parts = [f"<div><h3>{escape(heading)}</h3>"]
    for dimension in rubric.dimensions:
        chosen = scores.scores.get(dimension.id) if scores else None
        radios = "".join(
            _radio(
                f"score-{prefix}-{dimension.id}", str(value), str(value), checked=chosen == value
            )
            for value in range(rubric.scale.min, rubric.scale.max + 1)
        )
        anchors = dimension.anchors
        parts.append(
            f"<fieldset><legend>{escape(dimension.name)}</legend>{radios}"
            f"<details><summary>What the scale means</summary><p>{escape(dimension.definition)}"
            f"</p><p><b>{rubric.scale.min}:</b> {escape(anchors.low)}</p><p><b>Middle:</b> "
            f"{escape(anchors.middle)}</p><p><b>{rubric.scale.max}:</b> {escape(anchors.high)}"
            "</p></details></fieldset>"
        )
    ticked = set(scores.tags) if scores else set[str]()
    tags = "".join(
        f'<label title="{escape(tag.definition)}"><input type="checkbox" '
        f'name="tags-{prefix}" value="{tag.id}"{" checked" if tag.id in ticked else ""}> '
        f"{escape(tag.name)}</label> "
        for tag in rubric.failure_tags
    )
    free = ", ".join(scores.free_tags) if scores else ""
    parts.append(
        f"<fieldset><legend>Failure tags</legend>{tags}<br><label>Other tags (comma separated): "
        f'<input type="text" name="free-tags-{prefix}" value="{escape(free)}"></label>'
        "</fieldset></div>"
    )
    return "".join(parts)


def _pair_form(
    number: int, rubric: Rubric | None, guesses: list[str], current: Judgement | None
) -> str:
    preferred = "".join(
        _radio(
            "preferred",
            side,
            f"Story {side}",
            checked=current is not None and current.preferred == side,
        )
        for side in SIDES
    )
    strength = "".join(
        _radio(
            "strength",
            str(value),
            f"{value}: {label}",
            checked=current is not None and current.strength == value,
        )
        for value, label in _STRENGTHS.items()
    )
    sections = [
        f'<form method="post" action="/pairs/{number}">',
        f"<fieldset><legend>Which story is better?</legend>{preferred}</fieldset>",
        f"<fieldset><legend>By how much?</legend>{strength}</fieldset>",
    ]
    if rubric is not None:
        story_a = _scoring(rubric, "a", "Story A", current.a if current else None)
        story_b = _scoring(rubric, "b", "Story B", current.b if current else None)
        sections.append(f'<div class="columns">{story_a}{story_b}</div>')
        options = "".join(
            _radio("guess", option, option, checked=current is not None and current.guess == option)
            for option in guesses
        )
        sections.append(
            f'<fieldset class="guess"><legend>Which condition do you think story A came '
            f"from?</legend>{options}</fieldset>"
        )
    sections.append("<button>Save and continue</button></form>")
    return "".join(sections)


def _judgement_json(judgement: Judgement | None) -> dict[str, object] | None:
    if judgement is None:
        return None
    return judgement.model_dump(mode="json", exclude={"pair", "at"})


class _Reader:
    """Pages for one batch, rendered from its files on every request."""

    def __init__(self, batch: Path) -> None:
        self.batch = batch
        self.view = read_view(batch)
        self.single = self.view.kind == "single"
        self.path = "/stories" if self.single else "/pairs"
        self.noun = "Story" if self.single else "Pair"

    def header(self, number: int | None) -> str:
        done = len(judged(self.batch, self.view))
        where = f"{self.noun} {number} of {self.view.size}" if number else "Overview"
        warnings = "".join(
            f'<p class="notice">Warning: {escape(w.warning)}</p>' for w in read_warnings(self.batch)
        )
        return (
            f"<header><strong>{where}</strong><span>{done} of {self.view.size} judged"
            f' &middot; <a href="/overview">overview</a></span></header>{warnings}'
        )

    def item(self, number: int) -> int:
        if not 1 <= number <= self.view.size:
            raise HTTPException(status_code=404, detail="no such item")
        return number

    def next_page(self) -> RedirectResponse:
        """The first unjudged item, or the overview when none is left or the batch is closed."""
        done = judged(self.batch, self.view)
        unjudged = (
            None
            if is_closed(self.batch)
            else next((n for n in range(1, self.view.size + 1) if n not in done), None)
        )
        return RedirectResponse(f"{self.path}/{unjudged}" if unjudged else "/overview", 303)

    def _problems(self, problems: list[str] | None) -> str:
        if not problems:
            return ""
        items = "".join(f"<li>{escape(problem)}</li>" for problem in problems)
        return f'<div class="problems">Not saved: complete these first.<ul>{items}</ul></div>'

    def pair_page(self, number: int, problems: list[str] | None = None) -> HTMLResponse:
        pair = self.view.pairs[number - 1]
        current = read_judgements(self.batch).get(number)
        if is_closed(self.batch):
            form = '<p class="notice">This batch is closed.</p>'
        else:
            form = _pair_form(number, self.view.rubric, self.view.guess_options, current)
        body = (
            f'{self.header(number)}<main class="columns">{_story("Story A", pair.a)}'
            f"{_story('Story B', pair.b)}</main>{self._problems(problems)}{form}"
        )
        return _page(f"Pair {number}", body, 422 if problems else 200)

    def story_page(self, number: int, problems: list[str] | None = None) -> HTMLResponse:
        text = self.view.stories[number - 1]
        current = read_scores(self.batch).get(number)
        if is_closed(self.batch) or self.view.rubric is None:
            form = '<p class="notice">This batch is closed.</p>'
        else:
            scoring = _scoring(
                self.view.rubric, SINGLE, "Your scores", current.story if current else None
            )
            form = (
                f'<form method="post" action="/stories/{number}">{scoring}'
                "<button>Save and continue</button></form>"
            )
        body = (
            f'{self.header(number)}<main class="single">{_story(f"Story {number}", text)}'
            f"</main>{self._problems(problems)}{form}"
        )
        return _page(f"Story {number}", body, 422 if problems else 200)

    def overview_page(self) -> HTMLResponse:
        done = judged(self.batch, self.view)
        total = self.view.size
        rows = "".join(
            f'<li><a href="{self.path}/{n}">{self.noun} {n}</a>: '
            f"{'judged' if n in done else 'not judged'}</li>"
            for n in range(1, total + 1)
        )
        if is_closed(self.batch):
            action = '<p class="notice">This batch is closed. Judgements can no longer change.</p>'
        elif len(done) == total:
            action = (
                '<form method="post" action="/close"><p>Everything is judged. Closing the batch '
                "makes the judgements final.</p><button>Close the batch</button></form>"
            )
        else:
            action = f"<p>{total - len(done)} left to judge.</p>"
        return _page("Blind reader", f"{self.header(None)}<ul>{rows}</ul>{action}")

    def judge_pair(self, number: int, fields: dict[str, list[str]]) -> Response:
        try:
            record_judgement(self.batch, parse_judgement(self.view, number, fields))
        except InvalidJudgementError as error:
            return self.pair_page(number, error.problems)
        except BatchError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return self.next_page()

    def score_story(self, number: int, fields: dict[str, list[str]]) -> Response:
        try:
            record_scores(self.batch, parse_scores(self.view, number, fields))
        except InvalidJudgementError as error:
            return self.story_page(number, error.problems)
        except BatchError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return self.next_page()

    def close(self) -> Response:
        try:
            close_batch(self.batch)
        except BatchError as error:
            return _page("Not closed", f"{self.header(None)}<p>{escape(str(error))}</p>", 409)
        return RedirectResponse("/overview", 303)

    def progress(self) -> dict[str, object]:
        return {
            "stories" if self.single else "pairs": self.view.size,
            "judged": len(judged(self.batch, self.view)),
            "closed": is_closed(self.batch),
        }

    def pair_json(self, number: int) -> dict[str, object]:
        pair = self.view.pairs[self.item(number) - 1]
        judgement = read_judgements(self.batch).get(number)
        return {"pair": number, "a": pair.a, "b": pair.b, "judgement": _judgement_json(judgement)}

    def story_json(self, number: int) -> dict[str, object]:
        text = self.view.stories[self.item(number) - 1]
        current = read_scores(self.batch).get(number)
        scores = None if current is None else current.story.model_dump(mode="json")
        return {"story": number, "text": text, "scores": scores}


async def _fields(request: Request) -> dict[str, list[str]]:
    form = await request.form()
    return {key: [str(value) for value in form.getlist(key)] for key in form}


def create_app(batch: Path) -> FastAPI:
    """The reader for one batch. State lives on disk, so a restart resumes where it left off."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    reader = _Reader(batch)

    def pairs_only() -> None:
        if reader.single:
            raise HTTPException(status_code=404, detail="this batch has single stories")

    def stories_only() -> None:
        if not reader.single:
            raise HTTPException(status_code=404, detail="this batch has pairs")

    @app.get("/")
    def index() -> RedirectResponse:
        return reader.next_page()

    @app.get("/overview", response_class=HTMLResponse)
    def overview() -> HTMLResponse:
        return reader.overview_page()

    @app.get("/pairs/{number}", response_class=HTMLResponse)
    def show_pair(number: int) -> HTMLResponse:
        pairs_only()
        return reader.pair_page(reader.item(number))

    @app.post("/pairs/{number}", response_model=None)
    async def judge_pair(number: int, request: Request) -> Response:
        pairs_only()
        reader.item(number)
        return reader.judge_pair(number, await _fields(request))

    @app.get("/stories/{number}", response_class=HTMLResponse)
    def show_story(number: int) -> HTMLResponse:
        stories_only()
        return reader.story_page(reader.item(number))

    @app.post("/stories/{number}", response_model=None)
    async def score_story(number: int, request: Request) -> Response:
        stories_only()
        reader.item(number)
        return reader.score_story(number, await _fields(request))

    @app.post("/close", response_model=None)
    def close() -> Response:
        return reader.close()

    @app.get("/api/progress")
    def progress() -> dict[str, object]:
        return reader.progress()

    @app.get("/api/pairs/{number}")
    def pair_json(number: int) -> dict[str, object]:
        pairs_only()
        return reader.pair_json(number)

    @app.get("/api/stories/{number}")
    def story_json(number: int) -> dict[str, object]:
        stories_only()
        return reader.story_json(number)

    return app
