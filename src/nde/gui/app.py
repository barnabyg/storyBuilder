"""The GUI's web pages, server-rendered on localhost.

Writing a story spends money, so the app refuses any request that does not come from
its own pages: another website open in the same browser must not be able to start one.
"""

import re
from collections.abc import Awaitable, Callable, Mapping
from html import escape
from typing import Any
from urllib.parse import urlencode, urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from pydantic import ValidationError

from nde.continuity.pipeline import QUALITY_NOTE
from nde.gui.studio import Job, SavedStory, Studio, StudioBusyError, problems_in
from nde.runs.config import ConfigError
from nde.schema.brief import DEFAULT_TARGET_WORDS, Brief, BriefContent, EndingKind

_LOCAL_HOSTS = {"127.0.0.1", "localhost"}
_REFRESH_SECONDS = 3
_LENGTHS = (1000, 2000, 4000)
_EXAMPLE_LABEL_CHARS = 80

_STYLE = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0 auto; max-width: 72rem; padding: 1rem;
       color: #1d1d1f; background: #fbfaf7; line-height: 1.45; }
header { border-bottom: 1px solid #ccc; margin-bottom: 1.25rem; }
header a { color: inherit; text-decoration: none; }
header h1 { font-family: Georgia, serif; margin: 0.25rem 0 0.5rem; }
.columns { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 2.5rem; }
@media (max-width: 50rem) { .columns { grid-template-columns: minmax(0, 1fr); } }
h2 { font-size: 1.15rem; margin: 0 0 0.75rem; }
label { display: block; font-weight: 600; margin: 0.9rem 0 0.2rem; }
.hint { display: block; font-weight: 400; font-size: 0.85rem; color: #555; }
input, select, textarea { width: 100%; padding: 0.5rem; font: inherit; border: 1px solid #aaa;
                          border-radius: 4px; background: #fff; color: inherit; }
textarea { min-height: 4.5rem; resize: vertical; }
.pair { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 2fr); gap: 1rem; }
@media (max-width: 30rem) { .pair { grid-template-columns: minmax(0, 1fr); } }
button, .button { display: inline-block; padding: 0.6rem 1.1rem; font: inherit; font-weight: 600;
                  border: none; border-radius: 4px; background: #2f4f4f; color: #fff;
                  cursor: pointer; text-decoration: none; }
button.quiet, .button.quiet { background: #e6e2d8; color: #1d1d1f; }
button.danger, .button.danger { background: #8b2e2e; color: #fff; }
.inline { display: flex; gap: 0.5rem; align-items: stretch; }
.inline select { flex: 1; min-width: 0; }
.notice { background: #f6e7c8; padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem; }
.problems { background: #f8d7d3; padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem; }
.problems ul { margin: 0.25rem 0 0; padding-left: 1.25rem; }
.cost { font-size: 0.9rem; color: #444; margin: 1rem 0; }
.library { list-style: none; margin: 0; padding: 0; }
.library li { padding: 0.6rem 0; border-bottom: 1px solid #e3e0d8; }
.library a { font-family: Georgia, serif; font-size: 1.1rem; }
.meta { display: block; font-size: 0.85rem; color: #555; }
.reading { max-width: 42rem; margin: 0 auto; }
.reading h1 { font-family: Georgia, serif; margin-bottom: 0.25rem; }
article { font-family: Georgia, serif; font-size: 1.1rem; line-height: 1.6; margin: 1.5rem 0; }
.actions { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0; }
details { margin: 0.75rem 0; font-size: 0.95rem; }
summary { cursor: pointer; font-weight: 600; }
blockquote { margin: 0.25rem 0 0.75rem 1rem; color: #444; font-family: Georgia, serif; }
.activity { font-size: 1.25rem; margin: 1rem 0 0.5rem; }
"""

_PLACEHOLDERS = {
    "genre": "adventure",
    "premise": "Two estranged sisters try to finish their late father's winter crossing "
    "of a frozen lake.",
    "protagonist_name": "Kari Lund",
    "protagonist_description": "the younger sister, a reckless guide",
    "central_conflict": "The ice is weaker than the maps say, and the sisters disagree "
    "about turning back.",
    "themes": "sibling rivalry, risk, inheritance",
    "ending_detail": "They do not reach the far shore, but they are reconciled.",
}

# The brief's fields, as a person would name them in a message.
_LABELS = {
    "genre": "Genre",
    "premise": "The situation",
    "protagonist.name": "Main character's name",
    "protagonist.description": "Who the main character is",
    "central_conflict": "The central conflict",
    "themes": "Themes",
    "ending_constraint.kind": "Kind of ending",
    "ending_constraint.detail": "What the ending must do",
    "target_words": "Length",
}


def _page(title: str, body: str, *, status: int = 200, refresh: int | None = None) -> HTMLResponse:
    meta = "" if refresh is None else f'<meta http-equiv="refresh" content="{refresh}">'
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">{meta}'
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        '<body><header><h1><a href="/">Story Builder</a></h1></header>'
        f"{body}</body></html>"
    )
    return HTMLResponse(html, status_code=status)


def _usd(amount: float) -> str:
    return f"US${amount:.2f}"


def _values_of(brief: BriefContent) -> dict[str, str]:
    return {
        "genre": brief.genre,
        "premise": brief.premise,
        "protagonist_name": brief.protagonist.name,
        "protagonist_description": brief.protagonist.description,
        "central_conflict": brief.central_conflict,
        "themes": ", ".join(brief.themes),
        "ending_kind": brief.ending_constraint.kind.value,
        "ending_detail": brief.ending_constraint.detail,
        "target_words": str(brief.target_words),
    }


def parse_idea(form: Mapping[str, str]) -> tuple[BriefContent | None, list[str]]:
    """The form as a brief, or the problems to show the person who filled it in."""

    def value(name: str) -> str:
        return form.get(name, "").strip()

    words = value("target_words")
    try:
        content = BriefContent.model_validate(
            {
                "genre": value("genre"),
                "premise": value("premise"),
                "protagonist": {
                    "name": value("protagonist_name"),
                    "description": value("protagonist_description"),
                },
                "central_conflict": value("central_conflict"),
                "themes": [theme.strip() for theme in value("themes").split(",") if theme.strip()],
                "ending_constraint": {
                    "kind": value("ending_kind"),
                    "detail": value("ending_detail"),
                },
                "target_words": int(words) if words.isdecimal() else DEFAULT_TARGET_WORDS,
            }
        )
    except ValidationError as error:
        return None, [_problem(name, message) for name, message in problems_in(error)]
    return content, []


def _problem(name: str, message: str) -> str:
    label = _LABELS.get(name, name)
    if "at least 1" in message:
        return f"Please fill in: {label}."
    if name == "themes":
        return "Themes: please give between one and four, separated by commas."
    if name == "ending_constraint.kind":
        return "Please choose a kind of ending."
    return f"{label}: {message}."


def _field(
    name: str, label: str, hint: str, values: Mapping[str, str], *, long: bool = False
) -> str:
    value = escape(values.get(name, ""))
    example = escape(_PLACEHOLDERS[name])
    control = (
        f'<textarea id="{name}" name="{name}" placeholder="{example}" required>{value}</textarea>'
        if long
        else f'<input id="{name}" name="{name}" value="{value}" placeholder="{example}" required>'
    )
    return (
        f'<div><label for="{name}">{escape(label)}'
        f'<span class="hint">{escape(hint)}</span></label>{control}</div>'
    )


def _select(name: str, label: str, options: list[tuple[str, str]], chosen: str) -> str:
    items = "".join(
        f'<option value="{escape(value)}"{" selected" if value == chosen else ""}>'
        f"{escape(text)}</option>"
        for value, text in options
    )
    return f'<div><label for="{name}">{escape(label)}</label>' + (
        f'<select id="{name}" name="{name}">{items}</select></div>'
    )


def _example_label(brief: Brief) -> str:
    premise = brief.premise
    if len(premise) > _EXAMPLE_LABEL_CHARS:
        premise = premise[: _EXAMPLE_LABEL_CHARS - 1].rstrip() + "…"
    return f"{brief.genre}: {premise}"


def _examples(examples: list[Brief], chosen: str | None) -> str:
    options = "".join(
        f'<option value="{escape(brief.id)}"{" selected" if brief.id == chosen else ""}>'
        f"{escape(_example_label(brief))}</option>"
        for brief in examples
    )
    return (
        '<form method="get" action="/" class="inline" aria-label="Start from an example">'
        f'<select name="idea" aria-label="Example idea">{options}</select>'
        '<button class="quiet">Fill in</button></form>'
    )


def _form(studio: Studio, values: Mapping[str, str], problems: list[str]) -> str:
    listed = "".join(f"<li>{escape(problem)}</li>" for problem in problems)
    alert = f'<div class="problems" role="alert"><ul>{listed}</ul></div>' if problems else ""
    endings = [(kind.value, kind.value.capitalize()) for kind in EndingKind]
    lengths = [(str(words), f"About {words:,} words") for words in _LENGTHS]
    cap = studio.cap_usd
    limit = "" if cap is None else f" It stops at {_usd(cap)} at the very most."
    return f"""
    <h2>Write a new story</h2>
    {_examples(studio.examples(), values.get("idea"))}
    {alert}
    <form method="post" action="/stories">
      {_field("genre", "Genre", "What kind of story is it?", values)}
      {_field("premise", "The situation", "Where does the story start?", values, long=True)}
      <div class="pair">
        {_field("protagonist_name", "Main character's name", "Their full name", values)}
        {_field("protagonist_description", "Who they are", "In one sentence", values)}
      </div>
      {
        _field(
            "central_conflict",
            "The central conflict",
            "What stands in their way?",
            values,
            long=True,
        )
    }
      {_field("themes", "Themes", "One to four, separated by commas", values)}
      <div class="pair">
        {_select("ending_kind", "Kind of ending", endings, values.get("ending_kind", "happy"))}
        {_field("ending_detail", "What the ending must do", "In one sentence", values)}
      </div>
      {
        _select(
            "target_words", "Length", lengths, values.get("target_words", str(DEFAULT_TARGET_WORDS))
        )
    }
      <p class="cost">A 2,000-word story has taken 5 to 12 minutes and cost about US$0.25 to
      US$0.60. Longer stories take longer and cost more.{limit}</p>
      <button>Write this story</button>
    </form>
    """


def _library(stories: list[SavedStory]) -> str:
    if not stories:
        return "<h2>Your stories</h2><p>Stories you write will appear here.</p>"
    items = "".join(
        f'<li><a href="/stories/{escape(story.run)}/{escape(story.brief)}">'
        f"{escape(story.title)}</a>"
        f'<span class="meta">{escape(story.written_on)}, {story.words:,} words, '
        f"{_usd(story.cost_usd)}{'' if story.finished else ', unfinished'}</span></li>"
        for story in stories
    )
    return f'<h2>Your stories</h2><ul class="library">{items}</ul>'


def _home(
    studio: Studio,
    values: Mapping[str, str] | None = None,
    problems: list[str] | None = None,
    *,
    status: int = 200,
    deleted: str | None = None,
) -> HTMLResponse:
    running = studio.current
    banner = (
        ""
        if running is None
        else f'<div class="notice">A story is being written. '
        f'<a href="/jobs/{running.id}">See how it is going</a>.</div>'
    )
    if deleted is not None:
        banner += f'<div class="notice" role="status">Deleted “{escape(deleted)}”.</div>'
    columns = (
        f'<div class="columns"><section>{_form(studio, values or {}, problems or [])}</section>'
        f"<section>{_library(studio.stories())}</section></div>"
    )
    return _page("Story Builder", banner + columns, status=status)


def _problem_type(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _problem_list(problems: list[dict[str, Any]]) -> str:
    return "".join(
        f"<p><strong>{escape(_problem_type(str(problem['problem_type'])))}:</strong> "
        f"{escape(str(problem['reason']))}</p>"
        f"<blockquote>{escape(str(problem['passage']))}</blockquote>"
        for problem in problems
    )


def _characters(story: SavedStory) -> str:
    return "".join(
        f"<p><strong>{escape(str(sheet['name']))}:</strong> {escape(str(sheet['appearance']))}</p>"
        for sheet in story.characters()
    )


def _repairs(story: SavedStory) -> str:
    return "".join(
        f"<p>Paragraph {change['paragraph']}, before:</p>"
        f"<blockquote>{escape(str(change['before']))}</blockquote><p>After:</p>"
        f"<blockquote>{escape(str(change['after']))}</blockquote>"
        for change in story.repairs()
    )


def _section(heading: str, content: str) -> str:
    return f"<details><summary>{escape(heading)}</summary>{content}</details>" if content else ""


def _idea(idea: Brief | None) -> str:
    if idea is None:
        return ""
    return "".join(f"<p>{escape(line)}</p>" for line in idea.describe().splitlines())


def _address(story: SavedStory) -> str:
    return f"/stories/{escape(story.run)}/{escape(story.brief)}"


def _story_page(studio: Studio, story: SavedStory) -> HTMLResponse:
    paragraphs = "".join(f"<p>{escape(p)}</p>" for p in story.text.split("\n\n") if p.strip())
    idea = studio.find_idea(story.brief)
    warning = (
        ""
        if story.finished
        else '<div class="notice">This story stopped before its checks and repairs finished.</div>'
    )
    again = (
        ""
        if idea is None
        else f'<a class="button quiet" href="/?idea={escape(story.brief)}">Use this idea again</a>'
    )
    unresolved = f"Problems the checks could not fix ({len(story.unresolved)})"
    address = _address(story)
    link = f"{address}/story.md"
    written = f"{escape(story.written_on)}, {story.words:,} words, {_usd(story.cost_usd)}"
    body = f"""
    <div class="reading">
      <h1>{escape(story.title)}</h1>
      <span class="meta">{written}</span>
      {warning}
      <article>{paragraphs}</article>
      <div class="actions">
        <a class="button" href="{link}">Save as a file</a>
        {again}
        <a class="button quiet" href="/">Back to all stories</a>
        <a class="button danger" href="{address}/delete">Delete this story</a>
      </div>
      <h2>Behind the story</h2>
      <p class="meta">{escape(QUALITY_NOTE)}</p>
      {_section("The idea", _idea(idea))}
      {_section("The characters", _characters(story))}
      {_section(unresolved, _problem_list(story.unresolved))}
      {_section("Passages repaired", _repairs(story))}
    </div>
    """
    return _page(story.title, body)


def _delete_page(
    story: SavedStory, problem: str | None = None, *, status: int = 200
) -> HTMLResponse:
    alert = "" if problem is None else f'<div class="problems" role="alert">{escape(problem)}</div>'
    address = _address(story)
    body = f"""
    <div class="reading">
      {alert}
      <h2>Delete “{escape(story.title)}”?</h2>
      <p>This permanently deletes the story and everything saved with it. It cannot be undone.
      The money it cost stays spent, and its idea stays available.</p>
      <form method="post" action="{address}/delete" class="actions">
        <button class="danger">Delete permanently</button>
        <a class="button quiet" href="{address}">Keep this story</a>
      </form>
    </div>
    """
    return _page("Delete this story?", body, status=status)


def _job_page(job: Job) -> HTMLResponse:
    cap = "" if job.cap_usd is None else f" of at most {_usd(job.cap_usd)}"
    minutes = "under a minute" if job.minutes < 1 else f"{job.minutes} min"
    body = f"""
    <h2>Writing your story</h2>
    <p class="activity" role="status">{escape(job.activity)}…</p>
    <p class="meta">Running for {minutes}. Spent so far: {_usd(job.spent_usd)}{cap}.</p>
    <p>This usually takes 5 to 12 minutes, and the story opens here when it is ready. You can
    leave this page and come back, but keep the Story Builder window open, because closing it
    stops the writing and loses the story.</p>
    <a class="button quiet" href="/">Back to all stories</a>
    """
    return _page("Writing your story", body, refresh=_REFRESH_SECONDS)


def _failed_page(job: Job, reason: str | None) -> HTMLResponse:
    detail = _section("What went wrong", f"<p>{escape(reason)}</p>") if reason else ""
    body = f"""
    <div class="problems" role="alert">The story could not be written. This cost
    {_usd(job.spent_usd)}. You can try again.</div>
    {detail}
    <a class="button" href="/">Back to Story Builder</a>
    """
    return _page("The story could not be written", body)


def _from_this_app(request: Request) -> bool:
    """Whether a request came to the local address, and a write from its own pages.

    The Host check stops another site reaching the app by pointing a name at this
    machine; the Origin check stops another site submitting a form to it.
    """
    host = request.headers.get("host", "")
    if urlsplit(f"//{host}").hostname not in _LOCAL_HOSTS:
        return False
    if request.method in {"GET", "HEAD"}:
        return True
    origin = request.headers.get("origin")
    return origin is None or urlsplit(origin).netloc == host


def _file_name(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"{slug or 'story'}.md"


def create_app(studio: Studio) -> FastAPI:
    app = FastAPI(title="Story Builder", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def _guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not _from_this_app(request):
            return PlainTextResponse("Requests from other sites are not accepted.", status_code=403)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def home(idea: str | None = None, deleted: str | None = None) -> HTMLResponse:
        chosen = None if idea is None else studio.find_idea(idea)
        values = None if chosen is None else {**_values_of(chosen), "idea": chosen.id}
        return _home(studio, values, deleted=deleted)

    @app.post("/stories")
    async def write(request: Request) -> Response:
        form = await request.form()
        values = {name: value for name, value in form.items() if isinstance(value, str)}
        content, problems = parse_idea(values)
        if content is None:
            return _home(studio, values, problems, status=422)
        try:
            job = studio.start(content)
        except StudioBusyError as busy:
            job = busy.job
        except ConfigError as error:
            return _home(studio, values, [str(error)], status=422)
        return RedirectResponse(f"/jobs/{job.id}", status_code=303)

    @app.get("/jobs/{job_id}")
    def job_page(job_id: str) -> Response:
        job = studio.job(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if not job.finished:
            return _job_page(job)
        if job.run_dir is not None and not job.run_dir.exists():
            # Its story has since been deleted.
            return RedirectResponse("/", status_code=303)
        story, reason = studio.outcome(job)
        if story is None:
            return _failed_page(job, reason)
        return RedirectResponse(f"/stories/{story.run}/{story.brief}", status_code=303)

    @app.get("/stories/{run}/{brief}", response_class=HTMLResponse)
    def read(run: str, brief: str) -> HTMLResponse:
        story = studio.story(run, brief)
        if story is None:
            raise HTTPException(404, "no such story")
        return _story_page(studio, story)

    @app.get("/stories/{run}/{brief}/delete", response_class=HTMLResponse)
    def confirm_delete(run: str, brief: str) -> HTMLResponse:
        story = studio.story(run, brief)
        if story is None:
            raise HTTPException(404, "no such story")
        return _delete_page(story)

    @app.post("/stories/{run}/{brief}/delete")
    def delete(run: str, brief: str) -> Response:
        story = studio.story(run, brief)
        if story is None:
            raise HTTPException(404, "no such story")
        try:
            studio.delete(story)
        except StudioBusyError:
            problem = "This story is still being written. Try again when it has finished."
            return _delete_page(story, problem, status=409)
        except OSError as error:
            problem = f"The story could not be fully deleted: {error}"
            return _delete_page(story, problem, status=500)
        return RedirectResponse(f"/?{urlencode({'deleted': story.title})}", status_code=303)

    @app.get("/stories/{run}/{brief}/story.md")
    def download(run: str, brief: str) -> PlainTextResponse:
        story = studio.story(run, brief)
        if story is None:
            raise HTTPException(404, "no such story")
        return PlainTextResponse(
            f"# {story.title}\n\n{story.text}",
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{_file_name(story.title)}"'},
        )

    return app
