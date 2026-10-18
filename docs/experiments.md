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

## Run directories

Runs are written to `runs/` (gitignored; change it with `--runs-dir`):

| File | Contents |
| --- | --- |
| `manifest.json` | git SHA (and whether the tree was dirty), config path and hash, seed, models per role, prompt template versions and checksums, timestamps and run status |
| `outputs.jsonl` | one line per call: call ID, repetition, model, seed and the raw output text |
| `calls.jsonl` | one line per call: the provider's reported model ID, template, tokens and latency |
| `report.md` | generated from the files above |

Regenerate a report with `uv run nde report <run dir>`.

## Live tests

`uv run pytest -m live` runs the smoke config against the real provider when
`DEEPSEEK_API_KEY` is set (in the environment or `.env`), and skips otherwise.
It is excluded from `nde-verify`. Unit tests use a fake OpenAI-compatible server
on `127.0.0.1` and never call a real provider.
