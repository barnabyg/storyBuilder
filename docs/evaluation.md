# Blind evaluation

Stories from two conditions are compared blind: the owner reads them in pairs
in a local reader app without knowing which condition wrote which, and the key
is revealed only after the batch is closed. See §3.3 and §4 of
[the implementation plan](IMPLEMENTATION_PLAN.md).

## The whole loop

```powershell
uv run nde run configs/stories-ab.yaml                     # prints the run directory
uv run nde eval batch demo --from <run dir> A --from <run dir> B
uv run nde eval serve --batch demo                         # open http://127.0.0.1:8770/
# judge every pair in the browser, then close the batch on the overview page
uv run nde eval report demo
```

## Building a batch

```powershell
uv run nde eval batch <name> --from <run dir> <condition> --from <run dir> <condition>
```

- `--from` names a run directory and one of its conditions, exactly twice. The
  two may come from the same run (as above) or from two runs.
- Stories are paired by brief and repetition. Pairs where either story failed
  are left out.
- Within each pair, which story is shown as **A** is randomised, and so is the
  order of the pairs. `--seed` fixes the shuffle; by default it is random and
  recorded only in the sealed key.
- Only the normalised story text (`story.md`) goes into what the reader sees.
- The batch records the rubric version it scores with (`--rubric-version`,
  default 1, read from `--rubric-dir`, default `benchmark/rubric`). See
  [rubric.md](rubric.md).

A batch lives in `batches/<name>/` (gitignored; change it with
`--batches-dir` or `NDE_BATCHES`):

| File | Contents |
| --- | --- |
| `view.json` | the reader's view: each pair's two normalised stories in shown order, the rubric, and the two condition names offered for the guess (sorted by name) |
| `sealed/key.json` | the key: for each shown story, its brief, repetition, run and condition, plus the shuffle seed and rubric version |
| `judgements.jsonl` | every judgement, appended as it is given; the latest for a pair counts |
| `reader-warnings.jsonl` | leakage warnings raised when the reader started (the 48-hour rule) |
| `closed.json` | written when the batch is closed |
| `report.md` | written by `nde eval report` |

The reader app never opens `sealed/`, so no page or API response can reveal
the key. The only condition names it shows are the two guess options, always
both and always in the same order. Do not open `sealed/key.json` yourself
before closing the batch.

## Reading a batch

```powershell
uv run nde eval serve --batch <name>
```

The reader serves `http://127.0.0.1:8770/` (`--port` changes the port) and
binds to localhost only. It shows one pair at a time, with no condition
markers: the two stories side by side (stacked on narrow screens), then a
form for:

- **Which story is better?** A or B (a forced choice).
- **By how much?** 1 (slightly), 2 (clearly) or 3 (much better).
- **Rubric scores** for both stories on every rubric dimension. Each
  dimension's definition and anchors are one click away ("What the scale
  means").
- **Failure tags** for each story, from the rubric's taxonomy, plus optional
  free-text tags (comma separated).
- **Which condition do you think story A came from?** One of the two
  conditions. This measures whether blinding has leaked.

Everything except the tags is required: a pair cannot be saved incomplete,
and the page lists what is missing. Each judgement is saved to disk as soon as it is
given, and the reader moves on to the next unjudged pair. Stopping the server
loses nothing: started again, it resumes at the first unjudged pair. Until the
batch is closed, a pair can be revisited from the overview and re-judged.

**Closing.** When every pair is judged, the overview page offers **Close the
batch**. After that, judgements cannot change, and the reader shows the
stories without a form.

### The 48-hour rule

Reading should start at least 48 hours after you last worked on the code that
generates the stories (§4 of the plan). When the reader starts, it checks the
last commit touching `src/nde/stories`, `src/nde/plan`, `src/nde/prose`,
`src/nde/llm` and `prompts`. If that is less than 48 hours ago, or those paths
have uncommitted changes, or the working directory is not a git repository,
`nde eval serve` prints a warning, every reader page shows it, and the batch
report records it. The reader still works: the warning is for the record.

## Reporting a batch

```powershell
uv run nde eval report <name>
```

It refuses an open batch. On a closed batch it reveals the key and prints a
Markdown report (also written to `batches/<name>/report.md`):

- win counts and shares per condition;
- a two-sided exact binomial test of the win count against no preference
  (p = 0.5), with its p-value;
- per-dimension **rubric means** per condition, each with a 95% percentile
  bootstrap confidence interval (10,000 resamples, fixed seed 0, so the
  report regenerates identically);
- **failure-tag counts** per condition, for the taxonomy and for free-text
  tags;
- **condition-guess accuracy**: how often the guess for story A was right.
  If it was right in more than 70% of pairs, the batch is flagged
  **possibly unblinded**;
- any reader **warnings**, such as the 48-hour rule;
- every pair with its brief, which condition was A and which was B, the
  preferred condition, the strength and the guess.

Batches built before rubric scoring (no rubric in `view.json`) are read with
the preference questions only and report without the rubric, tag and guess
sections.

The statistics live in `src/nde/eval/stats.py` and are tested against exactly
computed values.
