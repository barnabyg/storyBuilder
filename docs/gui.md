# Story Builder GUI

A local web page for writing and reading [continuity stories](continuity.md), with no
command line needed. To start it, double-click `run.bat` in the repository folder, or
type `.\run` in PowerShell in that folder.

The page opens in your browser. Story Builder runs in its own window: keep that window
open, and press Ctrl+C there or close it to stop. If a story is being written, stopping loses it, and the window says so. It needs
`OPENAI_API_KEY` in the environment or in `.env`, and says so on the page if the key is
missing. The page follows your computer's light or dark setting.

`run.bat` needs the `.venv` folder that `uv sync` creates. It is the same as
`uv run nde gui`, without needing the pinned version of `uv`. Started from PowerShell, it
gives the prompt back straight away, and that terminal plays no further part.

## What it does

- **Write a new story.** Enter your idea, or choose an example, read its premise and press
  **Use this example**. If you have edited the form, choose **Keep my draft** or
  **Replace draft**. Press **Write this story** when ready. If a field needs correcting,
  its error links directly to the field and your answers stay in the form.
- **Choose a model.** Open **Settings** in the header. Story Builder fetches the models
  available to your OpenAI API key and shows supported text models, including dated
  versions when available. Choose a model and a thinking level, then press **Save settings**.
  Select **Use configuration defaults** to go back to the models and thinking levels in
  the configuration file. Saving applies to stories you start after the change.
- **Manage the examples.** **Manage examples** lists them, starting with the development
  briefs. **Add an example** saves a new idea without writing a story, and each example
  has **Use**, **Edit** and **Delete**. Deleting asks you to confirm. Editing an original
  example saves your changed copy in its place, at the end of the list. Deleting an
  original, or replacing it by editing, can be undone with **Restore the original
  examples**, which keeps your own examples as they are. Deleting your own example is
  permanent. Stories already written keep their idea whatever happens to the example.
- **Watch it being written.** The page shows the current step, elapsed time and cost so far.
  The budget bar shows how much of the spending limit has been used. The story opens
  when ready. You can draft another idea while a story is writing. **View story in
  progress** opens a separate tab, and **Write this story** becomes available when the
  current run finishes.
- **Read it.** The story page shows the story, and under **Behind the story** the idea,
  the characters, the problems the checks could not fix and the passages they repaired.
  **Save as a file** downloads it, and **Use this idea again** returns to the form with
  the idea filled in, ready to change. **Your stories** in the header and **Back to all
  stories** take you directly to the library.
- **Delete it.** **Delete this story** asks you to confirm, then permanently removes the
  story and everything saved with it. There is no undo, and git cannot bring it back. What
  it cost stays in the cost ledger, and its idea stays available.

Every story in the runs folder is listed, including those written with `nde continuity`.

## Cost and time

Stories use the budget in [`configs/continuity-dev.yaml`](../configs/continuity-dev.yaml)
(US$3 per run by default). The model chosen in Settings is used for OpenAI planning, prose,
and checks and repairs, with a known token price for spending limits. The cost and time
can vary with the model and thinking level. A 2,000-word story has taken 5 to 12 minutes and cost
about US$0.25 to US$0.60. There is no way to cancel from the page: closing the window
stops the writing, loses the story and keeps what was spent. A story that stops at the
budget is still shown if it got as far as its checks, marked as unfinished.

## Where things are saved

| What | Where |
| --- | --- |
| Stories | `runs/<run>/stories/continuity/<brief>/r1/`, the same files as [`nde continuity`](continuity.md#what-it-saves) |
| Typed ideas | `.nde/gui-briefs/dev/brief-1001.yaml` and upward |
| Your examples | `.nde/gui-briefs/examples/dev/`, numbered along with the typed ideas |
| Original examples you removed | `.nde/gui-briefs/examples/removed-benchmark-examples.yaml` |
| Model and thinking preferences | `.nde/gui-briefs/settings.json` |

These folders are ignored by git, so keep your own copy of anything you want to last.

Typed ideas and changes to examples never go into `benchmark/`, so the development and
held-out splits stay as they are. Held-out briefs are never opened or offered. A story from a typed idea or a
changed example is not benchmark evidence.

Unlike the command line, the GUI never uses the response cache: writing the same idea
again gives a new story, at full cost.

## Options

`nde gui --help` lists them: `--port` (8771 by default), `--no-browser`, `--config`,
`--runs-dir`, `--ledger` and `--ideas-dir`, which also holds your examples and settings.

## Safety

The page is served only on `127.0.0.1`. Because writing a story spends money, it refuses
requests addressed to any other host name, and form submissions from any other site.

## Testing

From the repository folder, use PowerShell with the local environment set up and
`OPENAI_API_KEY` configured:

```powershell
.\run.bat
```

Open **Settings** and switch between two models. If both are available, `gpt-6.2-sol`
offers **Maximum** and `gpt-5.4-mini` offers **None**. Save a model and thinking level,
then revisit Settings to confirm both choices are still selected. To restore the settings
from the configuration file, select **Use configuration defaults** and save. This check
makes model-list calls only.

`tests/test_gui.py` runs the whole path from the form to the saved story against a fake
provider. `tests/test_gui_browser.py` does the same in real Chrome and only runs when
asked:

```powershell
$env:NDE_BROWSER = "1"; uv run pytest tests/test_gui_browser.py
```

`tests/test_gui_exit.py` checks that Ctrl+C stops `nde gui` calmly and promptly in a
real process. `tests/test_run_bat.py` starts Story Builder through `run.bat` and stops it with Ctrl+C. It
opens a real window, so it also only runs when asked, and only on Windows:

```powershell
$env:NDE_CONSOLE = "1"; uv run pytest tests/test_run_bat.py
```
