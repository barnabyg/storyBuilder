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

## LLM judge panel

LLM judges judge the same blind batch the owner reads. They are for
screening and diagnosis, and every batch report checks them against the
owner's verdicts. **Judges never pass a gate.** See P1.7 and §4 of the plan.

### Configuring the panel

A judge run is an ordinary experiment config with a `judges` section, run
with `nde run`, so it is cached, cost-capped and reproducible like any run:

```yaml
providers: { ... }          # the judges' OpenAI-compatible endpoints

judges:
  batch: p1-instrument      # the batch to judge
  batches_dir: ../batches   # relative to the config file
  modes: [pairwise, rubric] # the default: both
  panel:                    # the judge role: one model per panel member
    - {provider: anthropic, model: some-model, family: anthropic, params: {max_tokens: 400}}
    - {provider: mistral, model: other-model, family: mistral, params: {max_tokens: 400}}
  # pairwise_prompt: {template: judge_pairwise, version: 1}   (the defaults)
  # rubric_prompt: {template: judge_rubric, version: 1}

prices: { ... }             # every judge model needs a price
```

The panel needs **at least two model families**. Each model's `family` is
recorded explicitly, not inferred from its provider (one provider can serve
several families).

**Family separation.** A judge must not share a family with the `planner` or
`prose` models that generated the batch's stories, to reduce self-preference
bias. Batches record their generating models, with families, from the source
runs' manifests, so set `family` on the `planner` and `prose` roles of every
story config (the committed ones do). A panel with fewer than two families, a
judge without a family, a judge in a generator's family, or a batch whose
generators have no recorded family is a config error, raised before any call.

### What judges do

Judges see only the batch's normalised story text, exactly as the reader
does: no brief, no condition, no run.

- **Pairwise mode** (`judge_pairwise@1`): for every pair, each judge gives a
  forced-choice preference with strength (1 to 3) twice, once with the
  stories in their shown order (`AB`) and once swapped (`BA`), to cancel
  position bias. A pair whose two orders disagree is **position-inconsistent**.
- **Rubric mode** (`judge_rubric@1`): each judge scores every story on every
  rubric dimension and assigns failure tags from the taxonomy, as structured
  output validated against the batch's rubric.

Judges can run before the owner has closed the batch. Their verdicts are
recorded by pair number and shown position, never by condition, in the run
directory (`judgements.jsonl`) and in the batch (`judges/<run>.jsonl`). The
reader never reads them. A verdict still invalid after its repairs is recorded
with its error and counted in the run report's structured-output validity.

### Reading the metrics

Once the batch is closed, `nde eval report` adds an **LLM judges** table,
one row per judge (so per family on a typical panel):

| Column | Meaning |
| --- | --- |
| Wins per condition | position-consistent verdicts only |
| Position-inconsistent | pairs whose two orders disagreed; reported, not averaged away |
| Agreement with owner | share of pairs where the judge's consistent verdict matched the owner's preference; inconsistent pairs count as disagreement |
| Cohen's kappa | agreement beyond chance between the owner's preferred condition and the judge's verdict ("inconsistent" is its own category); `n/a` when undefined |
| Position consistency | consistent pairs as a share of the pairs judged in both orders |

A **Judge rubric scores** table gives each judge's mean score per dimension
and condition. If a judge is run again, its latest verdicts count.

## Calibration batch

Before any gate reading, the owner scores a **calibration batch** to anchor
the rubric scale: 3 to 5 public-domain human short stories mixed with a few
Baseline A and B stories from dev-brief runs, shuffled and read one at a
time. See P1.9 of the plan.

### Human stories and provenance

`data/manifest.calibration.yaml` is the committed provenance manifest. Each
story has an `id`, `title`, `author`, `source` (download URL), `edition`,
`licence_basis` and `sha256`, plus the `start` and `end` markers its text lies
between (Project Gutenberg's `*** START OF` and `*** END OF` by default). The
committed stories are from Project Gutenberg's single-story ebooks:

| Story | Author | eBook |
| --- | --- | --- |
| An Occurrence at Owl Creek Bridge | Ambrose Bierce | #375 |
| The Gift of the Magi | O. Henry | #7256 |
| The Yellow Wallpaper | Charlotte Perkins Gilman | #1952 (about 6,000 words, above the 2,000 to 5,000 aim) |

To add a story from a collection, set `start` and `end` to lines that bracket
it. Standard Ebooks texts are preferred where a clean single-story source
exists; download one by hand and use `nde calibration import`.

```powershell
uv run nde calibration fetch
```

downloads each source into `data/calibration/` (gitignored), or reuses the
copy already there. The first fetch records each source's SHA-256 in the
manifest: **commit the manifest afterwards**. Every later fetch verifies the
source against it and stops with `checksum mismatch` if it changed. The story
text is extracted from between the markers, unwrapped into paragraphs,
stripped of its title and credit lines, and normalised exactly like a
generated story. Check the extracted `data/calibration/<id>.md` files once:
any byline or heading left in them would reveal a human source.

`uv run nde calibration import <id> <file>` uses a local copy of a source
instead of downloading it, with the same checksum rules.

### The command sequence

```powershell
uv run nde calibration fetch
uv run nde run configs/stories-ab.yaml
$run = (Get-ChildItem runs -Directory -Filter '*-stories-ab-*' | Sort-Object Name | Select-Object -Last 1).FullName
uv run nde eval calibration p1-calibration --from $run A --from $run B
uv run nde eval serve --batch p1-calibration
```

`nde eval calibration` mixes every fetched human story with `--per-source`
stories (default 2, sampled with the seed) from each `--from` run and
condition, and shuffles them. The reader then shows **one story at a time**
with only the rubric scores and failure tags: no pairing, preference or
condition guess. Every dimension must be scored. The source of each story
(human, A or B) is kept in the sealed key and never shown before the batch is
closed.

When every story is scored, close the batch on the overview page, then:

```powershell
uv run nde eval report p1-calibration
```

The calibration report gives each rubric dimension's mean score by source
(human, then each condition), with bootstrap intervals, failure-tag counts by
source, and the revealed source of every story.
