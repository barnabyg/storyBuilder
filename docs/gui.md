# Story Builder GUI

A local web page for writing and reading [continuity stories](continuity.md), with no
command line beyond starting it. Run this from the repository folder:

```powershell
uv run nde gui
```

The page opens in your browser. Keep the window that started it open, and press Ctrl+C
there to stop it. If a story is being written, stopping loses it, and the window says so. It needs `OPENAI_API_KEY` in the environment or in `.env`, and says so
on the page if the key is missing.

## What it does

- **Write a new story.** Fill in the idea, or choose an example and press **Fill in** to
  start from one of the development briefs. Then press **Write this story**.
- **Watch it being written.** The page shows the current step and what has been spent so
  far. It opens the story when it is ready. Only one story is written at a time.
- **Read it.** The story page shows the story, and under **Behind the story** the idea,
  the characters, the problems the checks could not fix and the passages they repaired.
  **Save as a file** downloads it, and **Use this idea again** returns to the form with
  the idea filled in, ready to change.

Every story in the runs folder is listed, including those written with `nde continuity`.

## Cost and time

Stories use the models and budget in
[`configs/continuity-dev.yaml`](../configs/continuity-dev.yaml), so a story stops at that
file's `budget.per_run_usd` (US$3). A 2,000-word story has taken 5 to 12 minutes and cost
about US$0.25 to US$0.60. There is no way to cancel from the page: closing the window
stops the writing, loses the story and keeps what was spent. A story that stops at the
budget is still shown if it got as far as its checks, marked as unfinished.

## Where things are saved

| What | Where |
| --- | --- |
| Stories | `runs/<run>/stories/continuity/<brief>/r1/`, the same files as [`nde continuity`](continuity.md#what-it-saves) |
| Typed ideas | `.nde/gui-briefs/dev/brief-1001.yaml` and upward |

Both folders are ignored by git, so keep your own copy of anything you want to last.

Typed ideas never go into `benchmark/`, so the development and held-out splits stay as
they are. Held-out briefs are never opened or offered. A story from a typed idea is not
benchmark evidence.

Unlike the command line, the GUI never uses the response cache: writing the same idea
again gives a new story, at full cost.

## Options

`nde gui --help` lists them: `--port` (8771 by default), `--no-browser`, `--config`,
`--runs-dir`, `--ledger` and `--ideas-dir`.

## Safety

The page is served only on `127.0.0.1`. Because writing a story spends money, it refuses
requests addressed to any other host name, and form submissions from any other site.

## Testing

`tests/test_gui.py` runs the whole path from the form to the saved story against a fake
provider. `tests/test_gui_browser.py` does the same in real Chrome and only runs when
asked:

```powershell
$env:NDE_BROWSER = "1"; uv run pytest tests/test_gui_browser.py
```
