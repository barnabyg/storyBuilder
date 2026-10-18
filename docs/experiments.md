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
    model: deepseek-chat
    params: {temperature: 0.7, max_tokens: 200}   # passed through to the API

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
exits 2.

## Response cache

Every call is cached by everything that can change its response: provider,
model, parameters, prompt template version, rendered messages, seed and
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
  deepseek/deepseek-chat: {input_per_mtok: 0.27, output_per_mtok: 1.10}
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
  role to keep the estimate tight.
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
| `calls.jsonl` | one line per call: the provider's reported model ID, template, tokens, cost, latency and whether it was a cache hit |
| `report.md` | generated from the files above |

Regenerate a report with `uv run nde report <run dir>`.

## Live tests

`uv run pytest -m live` runs the smoke config against the real provider when
`DEEPSEEK_API_KEY` is set (in the environment or `.env`), and skips otherwise.
It is excluded from `nde-verify`. Unit tests use a fake OpenAI-compatible server
on `127.0.0.1` and never call a real provider.
