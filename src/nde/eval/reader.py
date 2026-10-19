"""The blind reader: a small local web app for judging a batch, one pair at a time.

It serves server-rendered HTML on localhost. It only ever reads the batch's
view, judgements and closed marker (see `nde.eval.batch`), never the sealed
key, so no page or API response can reveal which condition wrote a story.
"""

from html import escape
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from nde.eval.batch import (
    BatchError,
    Judgement,
    Side,
    close_batch,
    is_closed,
    read_judgements,
    read_view,
    record_judgement,
)

HOST = "127.0.0.1"
DEFAULT_PORT = 8770

_STRENGTHS = {1: "slightly better", 2: "clearly better", 3: "much better"}

_STYLE = """
body { font-family: Georgia, serif; margin: 0 auto; max-width: 75rem; padding: 1rem;
       color: #1d1d1f; background: #fbfaf7; }
header, nav, form { font-family: system-ui, sans-serif; }
header { display: flex; justify-content: space-between; border-bottom: 1px solid #ccc; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
@media (max-width: 50rem) { .pair { grid-template-columns: 1fr; } }
article { line-height: 1.55; font-size: 1.05rem; }
form { position: sticky; bottom: 0; background: #f0ede6; padding: 1rem;
       border-top: 1px solid #ccc; }
fieldset { border: none; display: inline-block; margin-right: 2rem; }
.notice { background: #f6e7c8; padding: 0.75rem; }
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


def _judgement_json(judgement: Judgement | None) -> dict[str, object] | None:
    if judgement is None:
        return None
    return {"preferred": judgement.preferred, "strength": judgement.strength}


def create_app(batch: Path) -> FastAPI:
    """The reader for one batch. State lives on disk, so a restart resumes where it left off."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    pairs = read_view(batch).pairs

    def header(number: int | None) -> str:
        judged = len(read_judgements(batch))
        where = f"Pair {number} of {len(pairs)}" if number else "Overview"
        return (
            f"<header><strong>{where}</strong><span>{judged} of {len(pairs)} judged"
            f' &middot; <a href="/overview">overview</a></span></header>'
        )

    def pair_number(number: int) -> int:
        if not 1 <= number <= len(pairs):
            raise HTTPException(status_code=404, detail="no such pair")
        return number

    def next_unjudged() -> int | None:
        judged = read_judgements(batch)
        return next((n for n in range(1, len(pairs) + 1) if n not in judged), None)

    @app.get("/")
    def index() -> RedirectResponse:
        unjudged = None if is_closed(batch) else next_unjudged()
        return RedirectResponse(f"/pairs/{unjudged}" if unjudged else "/overview", 303)

    @app.get("/overview", response_class=HTMLResponse)
    def overview() -> HTMLResponse:
        judged = read_judgements(batch)
        rows = "".join(
            f'<li><a href="/pairs/{n}">Pair {n}</a>: '
            f"{'judged' if n in judged else 'not judged'}</li>"
            for n in range(1, len(pairs) + 1)
        )
        if is_closed(batch):
            action = '<p class="notice">This batch is closed. Judgements can no longer change.</p>'
        elif len(judged) == len(pairs):
            action = (
                '<form method="post" action="/close"><p>Every pair is judged. Closing the batch '
                "makes the judgements final.</p><button>Close the batch</button></form>"
            )
        else:
            action = f"<p>{len(pairs) - len(judged)} pairs left to judge.</p>"
        return _page("Blind reader", f"{header(None)}<ul>{rows}</ul>{action}")

    @app.get("/pairs/{number}", response_class=HTMLResponse)
    def show_pair(number: int) -> HTMLResponse:
        pair = pairs[pair_number(number) - 1]
        current = read_judgements(batch).get(number)
        if is_closed(batch):
            form = '<p class="notice">This batch is closed.</p>'
        else:
            choices = "".join(
                f'<label><input type="radio" name="preferred" value="{side}" required'
                f"{' checked' if current and current.preferred == side else ''}> "
                f"Story {side}</label> "
                for side in ("A", "B")
            )
            strengths = "".join(
                f'<label><input type="radio" name="strength" value="{value}" required'
                f"{' checked' if current and current.strength == value else ''}> "
                f"{value}: {label}</label> "
                for value, label in _STRENGTHS.items()
            )
            form = (
                f'<form method="post" action="/pairs/{number}">'
                f"<fieldset><legend>Which story is better?</legend>{choices}</fieldset>"
                f"<fieldset><legend>By how much?</legend>{strengths}</fieldset>"
                f"<button>Save and continue</button></form>"
            )
        body = (
            f'{header(number)}<main class="pair">{_story("A", pair.a)}{_story("B", pair.b)}'
            f"</main>{form}"
        )
        return _page(f"Pair {number}", body)

    @app.post("/pairs/{number}")
    def judge(
        number: int,
        preferred: Annotated[Side, Form()],
        strength: Annotated[int, Form(ge=1, le=3)],
    ) -> RedirectResponse:
        pair_number(number)
        try:
            record_judgement(batch, number, preferred, strength)
        except BatchError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        unjudged = next_unjudged()
        return RedirectResponse(f"/pairs/{unjudged}" if unjudged else "/overview", 303)

    @app.post("/close", response_model=None)
    def close() -> RedirectResponse | HTMLResponse:
        try:
            close_batch(batch)
        except BatchError as error:
            return _page("Not closed", f"{header(None)}<p>{escape(str(error))}</p>", 409)
        return RedirectResponse("/overview", 303)

    @app.get("/api/progress")
    def progress() -> dict[str, object]:
        return {
            "pairs": len(pairs),
            "judged": len(read_judgements(batch)),
            "closed": is_closed(batch),
        }

    @app.get("/api/pairs/{number}")
    def pair_json(number: int) -> dict[str, object]:
        pair = pairs[pair_number(number) - 1]
        judgement = read_judgements(batch).get(number)
        return {"pair": number, "a": pair.a, "b": pair.b, "judgement": _judgement_json(judgement)}

    return app
