# Running experiments

Every LLM experiment is a YAML config run with:

```powershell
uv run nde run configs/smoke.yaml
```

The command prints the run directory it created. It exits with 0 when the run
succeeds, 1 when the run fails or is aborted (the run directory still records
why), and 2 when the config is invalid. An invalid config never reaches the
network.

## API keys

API keys come from the environment, never from committed files. `nde run`
also reads a `.env` file in the working directory (gitignored); variables
already set in the environment take precedence. Copy `.env.example` to `.env`
and fill in the keys the config's providers need.

## Config

```yaml
name: smoke                 # used in the run directory name
seed: 20261018              # base seed; repetition i of a call uses seed + i
prompts_dir: ../prompts     # relative to the config file (default ../prompts)

providers:                  # OpenAI-compatible endpoints
  deepseek:
    base_url: https://api.deepseek.com/v1
    base_url_env: NDE_DEEPSEEK_BASE_URL   # optional override from the environment
    api_key_env: DEEPSEEK_API_KEY         # omit for servers that need no key
    timeout_s: 60

roles:                      # annotator | planner | prose | critic | judge
  prose:
    provider: deepseek
    model: deepseek-flash
    params: {thinking: {type: disabled}, temperature: 0.7, max_tokens: 200}
    # params pass straight through to the API, including provider-specific ones

calls:
  - id: opening-line
    role: prose
    prompt: {template: smoke, version: 1}
    variables: {setting: a lighthouse in a storm}
    repeat: 1               # optional; number of repetitions
```

Code and configs refer to roles; only the `roles` section names concrete
models. Unknown roles, missing models, unknown providers, missing template
versions or variables, and unset API-key variables are all rejected before
any call.

A config may also have a `benchmark` section that runs one role's calls once
per candidate model and compares them; the role then needs no entry under
`roles`. See [benchmark.md](benchmark.md).

## Story-generation runs

A config with a `stories` section generates **stories**: one per brief, per
**condition**, per repetition. A condition names a registered generation
**pipeline** and its parameters. `configs/stories-a.yaml` runs Baseline A on
two dev briefs:

```yaml
roles:
  prose:
    provider: deepseek
    model: deepseek-flash
    family: deepseek       # the model family, recorded per model (judges must differ)
    params: {max_tokens: 5000}

stories:
  briefs_dir: ../benchmark          # relative to the config file (default ../benchmark)
  briefs: {ids: [brief-01, brief-02]}   # or {split: dev}, or a split and ids in it
  repeat: 1                         # stories per brief and condition
  conditions:
    - name: A                       # a label for reports and blind batches
      pipeline: baseline_a          # registered in nde.stories.PIPELINES
      params: {}                    # validated by the pipeline; optional
```

`calls` and `stories` can appear in the same config; the calls run first. A
config needs at least one of them.

Before any call, `nde run` rejects unknown briefs, unknown pipelines, invalid
condition parameters, missing or changed prompt templates, a role a pipeline
uses that has no model, and unpriced models (see [briefs.md](briefs.md) for
the briefs themselves).

Stories are written brief by brief: for each brief and repetition, every
condition in turn, so a budget stop leaves comparable stories. Repetition *i*
uses seed + *i* for every condition.

### Normalisation and length control

These are the story-level leakage controls from §4 of the
[implementation plan](IMPLEMENTATION_PLAN.md). They apply to every condition,
so neither format nor length can reveal which condition wrote a story.

**Normalisation.** Every story is stored raw (`story.raw.md`) and normalised
(`story.md`). Blind batches and judges only see the normalised form. The rules
(`src/nde/stories/normalise.py`) are deterministic:

- Curly quotation marks become straight ones; an ellipsis character becomes
  three full stops; non-breaking and other special spaces become spaces.
- Markdown headings, code fences, horizontal rules and scene-break lines
  (`***`, `* * *`, `---`) are removed, as are scene and chapter titles
  ("Scene 2", "Chapter Three: The Storm"), bare section numbers ("II", "3."),
  and labels such as "Title:" or "Word count:".
- Markdown and HTML markup (emphasis, inline code, links, block quotes, list
  bullets, tags) is reduced to its text.
- A model preamble ("Here is your story:"), a leading title line and
  sign-offs ("The End", "I hope you enjoyed the story") are removed. A closing
  line that only starts like a sign-off but is not about the story or the
  request, or is dialogue ("I can still hear the gulls, she thought."), stays.
- Scene-by-scene conditions normalise each scene before joining them, so
  chatter around a scene never lands mid-story.
- Every remaining line is a paragraph, with whitespace collapsed; paragraphs
  are separated by one blank line.

Scene breaks are dropped rather than standardised, because some conditions
produce them and others do not.

**Length control.** A story must be within a tolerance of its brief's
`target_words`, counted in words on the normalised story:

```yaml
stories:
  length:
    tolerance: 0.10     # the default: within 10% of the target
    max_attempts: 3     # the default: the first generation and two regenerations
```

An out-of-range story's prose is regenerated on a new seed, up to
`max_attempts` generations in all. The first attempt uses the story's seed;
regeneration *k* uses a seed derived deterministically from it, so a re-run is
still free and byte-identical. Only the prose phase is regenerated: a
pipeline's plan (such as an outline) is kept. Every attempt is cached and
costed, and its files are kept under `attempts/<k>/` in the story's folder.
`stories.jsonl` records each attempt's seed, word count and whether it was in
range, and which one was kept.

A story still out of range after the last attempt is kept, not dropped: the
attempt closest to the target becomes the story, and it is flagged in the
manifest (`stories.out_of_range`) and in the report's **Out of length range**
list. The Stories table shows each story's words, target, attempts and whether
it is in range.

### Generation budget and equal-budget mode

A comparison between conditions should measure the method, not the compute
spent on it (proposal §9.8, plan §8 decision 9). So every story records its
**generation budget**: the provider calls it made and their input and output
tokens. Everything attributed to the story counts: the plan, every prose
call, structured-output repairs, length regenerations and revision passes.
Cache hits count at their recorded tokens, so a re-run reports the same
budgets. Each `stories.jsonl` line has a `budget` (`calls`, `input_tokens`,
`output_tokens`), the Stories table shows them per story, and the report's
**Generation budget** table gives calls and tokens per condition, in total and
per story.

**Equal-budget mode** makes two conditions comparable at equal budget:

```yaml
stories:
  equal_budget:
    conditions: [A, B]        # two conditions of this run
    tolerance: 0.10           # the default: within 10% of the larger budget
    max_revision_passes: 4    # the default
    revision_prompt: {template: story_revision, version: 1}   # the default
```

Matching is per brief and repetition, not on average. Once both conditions
have written their story for a brief, the cheaper one gets **revision passes**
until the two generation-token budgets are within the tolerance of the larger.
A revision pass rewrites the whole story on the `prose` role with the shared
prose contract's revision prompt (`story_revision@1`), from the brief and the
story so far. Its output goes through normalisation and length control like
any other story, and every attempt is cached and costed.

The revised story replaces the story. The original is kept as
`story.original.md`, and each pass's files are under `revisions/<n>/`. The
story's record counts its `revision_passes`.

Matching fails, and is reported for that brief rather than accepted, when:

- the budgets are still apart after `max_revision_passes` passes;
- a pass overshoots, taking the cheaper story above the tolerance of the
  other (a pass's cost is only known after it is made); or
- either story failed.

The manifest's `stories.equal_budget.matches` and the report's **Equal
budget** table give, per brief, each condition's budget before and after
matching, which condition was revised, the passes added, whether it matched,
and why not. `configs/stories-ab-equal-budget.yaml` runs A against B this way.

### Pipelines

| Pipeline | Condition | Steps |
| --- | --- | --- |
| `baseline_a` | Baseline A | one call on `prose` with `story_a@1`: brief to complete story |
| `baseline_b` | Baseline B | a beat outline on `planner` (`beat_outline@1`, structured, with repair), then one call per scene on `prose` through the shared prose contract |
| `baseline_b_scrambled` | B-scrambled (gate G-instr) | B's outline (the same cached call), scenes shuffled with a seeded permutation that is never the original order, then fresh prose through the shared prose contract; see [instrument-check.md](instrument-check.md) |

Parameters: `baseline_a` takes `prompt` (default `{template: story_a,
version: 1}`); `baseline_b` takes `outline_prompt` (default
`{template: beat_outline, version: 1}`).

`configs/stories-ab.yaml` runs A and B on the same two dev briefs in one run.

### Baseline B

The `planner` role turns the brief into a `BeatOutline`
(`src/nde/plan/beats.py`): a title and, in story order, between the brief's
`min_scenes` and `max_scenes` scenes, each with a one-sentence summary and one
to six beats. It is validated structured output with bounded repair, so
per-model validity appears in the report's **Structured output** table as for
any structured call. The outline is kept as `outline.json` beside the story.
An outline still invalid after its repairs fails that story only.

The outline's seed is the story's seed, so another condition that asks for the
same outline (B-scrambled) gets it from the cache. Length control regenerates
B's prose, never its outline: each attempt's scenes are kept as
`attempts/<k>/scene-NN.md`.

### The shared prose contract

Every structured condition writes its prose through
`nde.prose.scenes.write_scenes(ctx, scenes, seed=...)`, with the scene-prose
template `scene_prose@1` (`SCENE_PROSE`) on the `prose` role (`PROSE_ROLE`).
These are the **shared prose contract**: B, B-scrambled and the later
conditions C, D and E must reuse them unchanged, so that conditions differ
only in their plans (a leakage control in §4 of the plan). A new template
version is a change to every condition at once.

A condition supplies a list of `SceneSpec`s: a summary, beats and optional
free-text `notes` for whatever else its planner wants realised. Each scene is
written in order from:

- the brief;
- the whole plan, as the numbered list of scene summaries;
- the current scene's beats and notes, with a word target of the brief's
  `target_words` divided by the number of scenes;
- a compact summary of the story so far: the planned summaries of the scenes
  already written and the last paragraph of the previous scene.

Each scene is a step named `scene-01`, `scene-02`, ... in `calls.jsonl`. The
scenes are joined with blank lines, and normalisation makes them one
continuous story. A different planner feeds the same prose steps simply by
passing its own scene list; the tests drive them with a hand-written one.

### What a story run records

| File | Contents |
| --- | --- |
| `stories/<condition>/<brief>/r<n>/story.md` | the normalised story; `story.raw.md` beside it is the model's text, plan outputs (such as an outline) sit beside it too, and each length-control attempt's files are under `attempts/<k>/` |
| `stories.jsonl` | one line per story: brief, condition, pipeline, repetition, seed, status, word count, target, length attempts and its files |
| `calls.jsonl` | as for every run; each line also names the `brief`, `condition`, pipeline `step` and length-control `story_attempt` it served |
| `manifest.json` | the `stories` section: brief IDs, the checksum of each frozen split they come from, conditions, repetitions, the length settings and any stories still out of range |

The report's **Stories** section lists every story with its word count,
calls, tokens and cost, and a **Per condition** table gives the totals and the
cost per story. A re-run reads every response from the cache and reproduces
`stories/` and `stories.jsonl` byte for byte.

A step whose structured output is still invalid after its repairs fails that
story, not the run: the story is recorded with `status: failed` and the error,
and the run goes on. A budget stop aborts the run and a provider error fails
it, as for any call; both still leave the manifest, the stories so far and the
report.

### Registering a pipeline

A pipeline is a subclass of `nde.stories.pipeline.Pipeline` with:

- `roles`: the roles it calls, checked against the config's `roles`;
- `templates()`: every prompt template it uses and the variables it fills in,
  so templates are loaded and checked before any call;
- `plan(ctx)`: calls that are never regenerated, such as an outline (return
  `None` if there are none);
- `write(ctx, plan, seed=...)`: the prose calls; returns the raw story text.
  Length control may call it several times with different seeds, so it must
  not change the plan.

Every call goes through the `StoryContext`: `ctx.complete(step, seed=...)` and
`ctx.structured(step, Model, seed=...)` take a `Step` (its name in
`calls.jsonl`, role, template and variables), and `ctx.keep(name, text)` keeps
an intermediate output as a file beside the story. A pipeline's parameters are
a Pydantic model it validates in a `from_params` class method. Register the
class in `PIPELINES` in `src/nde/stories/__init__.py`. Baseline A
(`src/nde/stories/baseline_a.py`) is the smallest example.

## Prompt templates

Templates live at `prompts/<name>/v<version>.yaml`:

```yaml
system: You are a concise fiction writer.          # optional
user: Write one opening sentence set in $setting.  # $variables from the config
```

`prompts/checksums.json` records the SHA-256 of every released version. A
template whose text no longer matches its checksum is rejected: to change a
prompt, save the new text as a new version. After adding a version, record it:

```powershell
uv run nde prompts lock
```

`nde prompts lock` never re-records a changed version; it reports it and
exits 2. `uv run nde prompts lock --check` reports unrecorded and changed
versions without writing anything, and exits 2 if there are any.

## Provider failures and retries

Each provider has a retry policy for rate limits and transient failures:

```yaml
providers:
  deepseek:
    base_url: https://api.deepseek.com/v1
    timeout_s: 60
    retry:
      max_retries: 3          # retries after the first attempt
      base_delay_s: 1.0
      max_delay_s: 30.0
```

Retry *n* (from 0) waits a random time up to
`min(max_delay_s, base_delay_s * 2**n)` (exponential backoff with full
jitter); a rate limit's `Retry-After` header sets a minimum wait, still capped
at `max_delay_s`. Every provider failure is mapped to one of these types:

| Type | Retried | Typical cause |
| --- | --- | --- |
| `rate_limited` | yes | HTTP 429 |
| `transient` | yes | HTTP 408, 409 or 5xx; timeouts; dropped or refused connections; garbled replies |
| `authentication` | no | HTTP 401 or 403 |
| `invalid_request` | no | other HTTP 4xx, such as an unknown model or a bad parameter |
| `content_refused` | no | a provider content filter (error code or `finish_reason`) |

A call that still fails after its retries, or fails with a non-retryable
error, stops the run. The run directory records `status: failed` with
`error.type`, the message, the call that failed and the HTTP status. The
outputs completed so far and the report are still written, and `nde run`
exits 1 without a stack trace. `calls.jsonl` records how many retries each
successful call needed.

## Local models

A local OpenAI-compatible server is just another provider.
`configs/local-ollama.yaml` runs the smoke experiment against Ollama:

```powershell
ollama pull llama3.1:8b
uv run nde run configs/local-ollama.yaml
```

The provider has no `api_key_env`, a long `timeout_s` for the first call
(which loads the model), and a zero price. For llama.cpp, start
`llama-server -m model.gguf --port 8080`, set
`NDE_LOCAL_BASE_URL=http://localhost:8080/v1`, and set the role's model to the
name the server reports at `/v1/models`.

`uv run pytest -m live` includes a test that runs this config against a local
server when one answers at the configured URL (using whichever model it has),
and skips otherwise.

## Structured output

A call can ask for a response that conforms to a Pydantic model:

```yaml
providers:
  deepseek:
    base_url: https://api.deepseek.com/v1
    json_schema: false        # does the provider accept JSON-schema response mode?

roles:
  planner:
    provider: deepseek
    model: deepseek-flash
    json_schema: null         # optional per-model override of the provider's flag

calls:
  - id: premise
    role: planner
    prompt: {template: premise, version: 1}
    response_model: story_premise   # registered in nde.schema.RESPONSE_MODELS

structured:
  max_repairs: 2              # repair requests after an invalid first response
```

- When the model's provider (or the model's own override) has `json_schema:
  true`, the request sets `response_format` to the model's JSON schema.
  Otherwise the schema is appended to the final user message.
- The response is validated with Pydantic (Markdown code fences are
  stripped first). If it fails, the invalid response and the specific
  validation errors (for example `protagonist.goal: Field required`) are sent
  back in a repair request, up to `max_repairs` times.
- In code, `Gateway.structured(binding, prompt, Model, seed=...)` returns a
  `StructuredResult` holding the validated instance and every attempt, or
  raises `StructuredOutputError` with the last errors once repairs run out.
- In a run, a call whose repairs run out does not fail the run: its output
  records the error and counts against the model's validity.
- Every attempt is an ordinary gateway call: it is cached, costed and capped,
  so a re-run makes no calls. `calls.jsonl` has one line per attempt
  (`attempt` 0 is the first request).
- The manifest's `structured` section and the report's **Structured output**
  table give, per model: first-attempt validity, validity after repair,
  repair rate (calls that needed at least one repair) and failed calls. The
  report shows each parsed object.

To add a response model, define it under `src/nde/schema/` and register it in
`RESPONSE_MODELS`.

`configs/structured-smoke.yaml` asks for a small nested `StoryPremise` three
times.

## Response cache

Every call is cached by everything that can change its response: provider and
its resolved base URL, model, parameters, prompt template version, rendered
messages, seed and
response format. Running the same config again makes no network calls and
reproduces `outputs.jsonl` byte for byte. The manifest records the cache mode
and the hit and miss counts.

| Option | Effect |
| --- | --- |
| *(default)* | read and write the cache |
| `--no-cache` | neither read nor write the cache |
| `--refresh-cache` | call the provider for every request and replace the stored responses |
| `--cache PATH` | cache file; default `.nde/cache.sqlite` (gitignored), or `NDE_CACHE` |

The cache is a SQLite database in write-ahead-log mode, so concurrent runs can
share one file safely. Because repetition *i* of a call uses seed + *i*, each
repetition is cached separately.

## Cost ledger and budget caps

Every config carries a price table and, optionally, caps:

```yaml
prices:                     # USD per million tokens, keyed by provider/model
  deepseek/deepseek-flash: {input_per_mtok: 0.30, output_per_mtok: 1.20}
  local/llama3.1:8b: {input_per_mtok: 0, output_per_mtok: 0}

budget:
  per_run_usd: 0.50         # optional
  per_day_usd: 5.00         # optional; UTC calendar day, across all runs
  allow_unpriced_models: false
  assumed_output_tokens: 1024
```

Prices are supplied by you because they vary by provider and change over time;
check them against the provider's price list.

- Every call records its input and output tokens and its cost in
  `calls.jsonl`. The manifest's `cost` section and the report's **Cost**
  section show the totals, per model.
- A cache hit costs nothing. The report also shows the *token cost at list
  price*, which counts cache hits, so a cached re-run still shows what its
  outputs cost to produce.
- Before each provider call, the gateway reserves a conservative estimate of
  its cost: input tokens are estimated at 3 characters per token plus 8 per
  message, and output tokens at the call's `max_tokens` (or
  `assumed_output_tokens` when it sets none). If the reservation would take the
  run or the day over its cap, the call is not made. The run stops with
  `status: aborted`, `abort_reason: budget`, and still writes its manifest,
  outputs so far and report. `nde run` then exits 1. Set `max_tokens` on every
  role: providers stop there, so it bounds the output cost. The input
  estimate is generous for English but is not a hard bound, so a request can
  cost slightly more than its reservation, and the cap can be overshot by that
  difference on the last request. Leave some headroom under a hard limit.
- The daily total lives in `.nde/ledger.sqlite` (gitignored; change it with
  `--ledger PATH` or `NDE_LEDGER`), so it persists across runs and is shared
  by concurrent runs. Reservations are atomic, so concurrent runs cannot
  jointly overshoot the daily cap.
- **A model missing from the price table fails the config before any call.**
  Set `budget.allow_unpriced_models: true` to run it anyway: its calls are
  recorded at $0, are invisible to the caps, and the manifest and report carry
  a warning naming the model. Prefer an explicit zero price for local models.

## Run directories

Runs are written to `runs/` (gitignored; change it with `--runs-dir`):

| File | Contents |
| --- | --- |
| `manifest.json` | git SHA (and whether the tree was dirty), config path and hash, seed, models per role, prompt template versions and checksums, timestamps and run status |
| `outputs.jsonl` | one line per call: call ID, repetition, model, seed and the raw output text |
| `calls.jsonl` | one line per provider call (including repair attempts): the reported model ID, template, tokens, cost, latency, retries and whether it was a cache hit |
| `stories.jsonl`, `stories/` | story runs only; see [Story-generation runs](#story-generation-runs) |
| `judgements.jsonl` | judge-panel runs only: every judge verdict, by pair and shown position; see [evaluation.md](evaluation.md#llm-judge-panel) |

The manifest records each model's role, provider, model and `family`. Story
runs add a `stories` section (brief IDs, frozen-split checksums, conditions,
length settings, out-of-range stories and any equal-budget matches) and
judge-panel runs a `judges` section (the batch, modes, panel, rubric version
and verdict counts).
| `report.md` | generated from the files above |

Regenerate a report with `uv run nde report <run dir>`.

## Live tests

`uv run pytest -m live` runs the smoke config against the real provider when
`DEEPSEEK_API_KEY` is set (in the environment or `.env`), and skips otherwise.
It is excluded from `nde-verify`. Unit tests use a fake OpenAI-compatible server
on `127.0.0.1` and never call a real provider.
