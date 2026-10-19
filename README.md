# Narrative Diffusion Engine

Research tooling for structured story planning. See
[the proposal](Narrative_Diffusion_Writing_Engine_Proposal.md) and
[the implementation plan](docs/IMPLEMENTATION_PLAN.md).

## Setup

Install [uv](https://docs.astral.sh/uv/) 0.8.14 (pinned by `required-version` in
`pyproject.toml`), then:

```powershell
uv sync
```

`uv sync` creates `.venv` with the runtime and development dependencies. The
optional ML dependencies (`torch`) are not installed by default; install them
with `uv sync --group ml` when a phase needs them.

```powershell
uv run nde --version
```

## Running experiments

```powershell
Copy-Item .env.example .env   # then add your API key
uv run nde run configs/smoke.yaml
```

See [docs/experiments.md](docs/experiments.md) for configs, prompt templates and
run directories.
The G0 model benchmark is described in [docs/benchmark.md](docs/benchmark.md).
Benchmark briefs (format, drafting, freezing) are described in
[docs/briefs.md](docs/briefs.md), and the rubric in [docs/rubric.md](docs/rubric.md).
Blind evaluation (batches, the reader and reports) is described in
[docs/evaluation.md](docs/evaluation.md), and the Phase 1 instrument check in
[docs/instrument-check.md](docs/instrument-check.md).

## Verification

One canonical, non-mutating command runs every gate in order:

```powershell
uv run nde-verify
```

| # | Gate | Tool |
| --- | --- | --- |
| 1 | Formatting check | `ruff format --check` |
| 2 | Lint and style | `ruff check` (the full rule selection in `pyproject.toml`) |
| 3 | Type checking | `pyright` in strict mode, warnings as errors |
| 4 | Static bug analysis | `ruff check --select B,S` (flake8-bugbear and bandit rules) |
| 5 | Automated tests | `pytest`, with warnings as errors; `live` tests excluded |
| 6 | Dependencies and secrets | `uv lock --check`, `pip-audit` over every locked group, `detect-secrets` |
| 7 | Build | `uv build` into a temporary directory |

A gate fails on a non-zero exit **or** on any warning in its output. Later
gates still run after a failure, so one run shows every problem. The command
exits non-zero if any gate failed.

Gate 4 has no separate analyser: ruff's bugbear and bandit rule sets cover
static bug and security analysis for this codebase. Every inline suppression
names what it suppresses (`# noqa: <rule>`, or `# pragma: allowlist secret` for
a secret-scan false positive) and has its justification beside it.
The only broader exceptions are the per-file ignores for `tests/**` in
`pyproject.toml` (assert statements and literal expected values), justified
there, and the secret-scan exclusions for checksum files (`prompts/checksums.json`,
`benchmark/frozen.json`, `benchmark/rubric/checksums.json`,
`data/manifest.calibration.yaml`), justified in `src/nde/verify/steps.py`. The code has no type-checker suppressions.

CI (`.github/workflows/verify.yml`) runs the same command headless on every
push and pull request.

Run locally, `nde-verify` also serves a live dashboard on localhost and prints
its address as `TEST_DASHBOARD_URL=...`. Turn it off with `--no-dashboard` or
`NDE_DASHBOARD=0`; it never runs in CI. See
[docs/verification.md](docs/verification.md).

Line endings are fixed to LF by `.gitattributes`, overriding
`core.autocrlf` on Windows checkouts.

### Live tests

Tests that call a real LLM provider are marked `live` and are excluded by
default. Run them explicitly:

```powershell
uv run pytest -m live
```
