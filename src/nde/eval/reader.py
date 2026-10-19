"""The blind reader: a small local web app for judging a batch, one pair at a time.

It serves server-rendered HTML on localhost. It only ever reads the batch's
view, judgements, warnings and closed marker (see `nde.eval.batch`), never
the sealed key, so no page or API response can reveal which condition wrote a
story.
"""

from html import escape
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from nde.eval.batch import (
    SIDES,
    BatchError,
    InvalidJudgementError,
    Judgement,
    close_batch,
    is_closed,
    parse_judgement,
    read_judgements,
    read_view,
    read_warnings,
    record_judgement,
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


def _story(label: str, text: str) -> str:
    paragraphs = "".join(f"<p>{escape(p)}</p>" for p in text.split("\n\n") if p.strip())
    return f"<article><h2>Story {label}</h2>{paragraphs}</article>"


def _radio(name: str, value: str, label: str, *, checked: bool) -> str:
    return (
        f'<label><input type="radio" name="{name}" value="{escape(value)}" required'
        f"{' checked' if checked else ''}> {escape(label)}</label> "
    )


def _scoring(rubric: Rubric, side: str, current: Judgement | None) -> str:
    """Rubric scores, failure tags and free-text tags for one story."""
    scores = getattr(current, side.lower(), None) if current else None
    parts = [f"<div><h3>Story {side}</h3>"]
    for dimension in rubric.dimensions:
        chosen = scores.scores.get(dimension.id) if scores else None
        radios = "".join(
            _radio(
                f"score-{side.lower()}-{dimension.id}",
                str(value),
                str(value),
                checked=chosen == value,
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
        f'name="tags-{side.lower()}" value="{tag.id}"{" checked" if tag.id in ticked else ""}> '
        f"{escape(tag.name)}</label> "
        for tag in rubric.failure_tags
    )
    free = ", ".join(scores.free_tags) if scores else ""
    parts.append(
        f"<fieldset><legend>Failure tags</legend>{tags}<br><label>Other tags (comma separated): "
        f'<input type="text" name="free-tags-{side.lower()}" value="{escape(free)}"></label>'
        "</fieldset></div>"
    )
    return "".join(parts)


def _form(
    number: int, view_rubric: Rubric | None, guesses: list[str], current: Judgement | None
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
    if view_rubric is not None:
        sections.append(
            f'<div class="columns">{_scoring(view_rubric, "A", current)}'
            f"{_scoring(view_rubric, 'B', current)}</div>"
        )
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
        self.pairs = self.view.pairs

    def header(self, number: int | None) -> str:
        judged = len(read_judgements(self.batch))
        where = f"Pair {number} of {len(self.pairs)}" if number else "Overview"
        warnings = "".join(
            f'<p class="notice">Warning: {escape(w.warning)}</p>' for w in read_warnings(self.batch)
        )
        return (
            f"<header><strong>{where}</strong><span>{judged} of {len(self.pairs)} judged"
            f' &middot; <a href="/overview">overview</a></span></header>{warnings}'
        )

    def pair_number(self, number: int) -> int:
        if not 1 <= number <= len(self.pairs):
            raise HTTPException(status_code=404, detail="no such pair")
        return number

    def next_page(self) -> RedirectResponse:
        """The first unjudged pair, or the overview when none is left or the batch is closed."""
        judged = read_judgements(self.batch)
        unjudged = (
            None
            if is_closed(self.batch)
            else next((n for n in range(1, len(self.pairs) + 1) if n not in judged), None)
        )
        return RedirectResponse(f"/pairs/{unjudged}" if unjudged else "/overview", 303)

    def pair_page(self, number: int, problems: list[str] | None = None) -> HTMLResponse:
        pair = self.pairs[number - 1]
        current = read_judgements(self.batch).get(number)
        if is_closed(self.batch):
            form = '<p class="notice">This batch is closed.</p>'
        else:
            form = _form(number, self.view.rubric, self.view.guess_options, current)
        shown = ""
        if problems:
            items = "".join(f"<li>{escape(problem)}</li>" for problem in problems)
            shown = f'<div class="problems">Not saved: complete these first.<ul>{items}</ul></div>'
        body = (
            f'{self.header(number)}<main class="columns">{_story("A", pair.a)}'
            f"{_story('B', pair.b)}</main>{shown}{form}"
        )
        return _page(f"Pair {number}", body, 422 if problems else 200)

    def overview_page(self) -> HTMLResponse:
        judged = read_judgements(self.batch)
        total = len(self.pairs)
        rows = "".join(
            f'<li><a href="/pairs/{n}">Pair {n}</a>: '
            f"{'judged' if n in judged else 'not judged'}</li>"
            for n in range(1, total + 1)
        )
        if is_closed(self.batch):
            action = '<p class="notice">This batch is closed. Judgements can no longer change.</p>'
        elif len(judged) == total:
            action = (
                '<form method="post" action="/close"><p>Every pair is judged. Closing the batch '
                "makes the judgements final.</p><button>Close the batch</button></form>"
            )
        else:
            action = f"<p>{total - len(judged)} pairs left to judge.</p>"
        return _page("Blind reader", f"{self.header(None)}<ul>{rows}</ul>{action}")

    def judge(self, number: int, fields: dict[str, list[str]]) -> Response:
        try:
            record_judgement(self.batch, parse_judgement(self.view, number, fields))
        except InvalidJudgementError as error:
            return self.pair_page(number, error.problems)
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
            "pairs": len(self.pairs),
            "judged": len(read_judgements(self.batch)),
            "closed": is_closed(self.batch),
        }

    def pair_json(self, number: int) -> dict[str, object]:
        pair = self.pairs[self.pair_number(number) - 1]
        judgement = read_judgements(self.batch).get(number)
        return {"pair": number, "a": pair.a, "b": pair.b, "judgement": _judgement_json(judgement)}


def create_app(batch: Path) -> FastAPI:
    """The reader for one batch. State lives on disk, so a restart resumes where it left off."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    reader = _Reader(batch)

    @app.get("/")
    def index() -> RedirectResponse:
        return reader.next_page()

    @app.get("/overview", response_class=HTMLResponse)
    def overview() -> HTMLResponse:
        return reader.overview_page()

    @app.get("/pairs/{number}", response_class=HTMLResponse)
    def show_pair(number: int) -> HTMLResponse:
        return reader.pair_page(reader.pair_number(number))

    @app.post("/pairs/{number}", response_model=None)
    async def judge(number: int, request: Request) -> Response:
        reader.pair_number(number)
        form = await request.form()
        return reader.judge(number, {key: [str(v) for v in form.getlist(key)] for key in form})

    @app.post("/close", response_model=None)
    def close() -> Response:
        return reader.close()

    @app.get("/api/progress")
    def progress() -> dict[str, object]:
        return reader.progress()

    @app.get("/api/pairs/{number}")
    def pair_json(number: int) -> dict[str, object]:
        return reader.pair_json(number)

    return app
