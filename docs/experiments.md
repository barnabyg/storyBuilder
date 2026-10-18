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

A config may also have a `benchmark` section that runs one role's calls once
per candidate model and compares them; the role then needs no entry under
`roles`. See [benchmark.md](benchmark.md).

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
    model: deepseek-chat
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
| `calls.jsonl` | one line per provider call (including repair attempts): the reported model ID, template, tokens, cost, latency, retries and whether it was a cache hit |
| `report.md` | generated from the files above |

Regenerate a report with `uv run nde report <run dir>`.

## Live tests

`uv run pytest -m live` runs the smoke config against the real provider when
`DEEPSEEK_API_KEY` is set (in the environment or `.env`), and skips otherwise.
It is excluded from `nde-verify`. Unit tests use a fake OpenAI-compatible server
on `127.0.0.1` and never call a real provider.
