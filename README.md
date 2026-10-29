# Narrative Diffusion Engine

Research tooling for structured story planning. See
[the proposal](Narrative_Diffusion_Writing_Engine_Proposal.md) and
[the implementation plan](docs/IMPLEMENTATION_PLAN.md).

Inspect saved narrative plans with the [plan workflow](docs/narrative-plans.md).
Replay [planned scene state](docs/state-ledger.md) before writing prose.
Generate a reproducible [G-schema evidence packet](docs/schema-evidence.md) from saved inputs.
Compare [development tiers and freeze an owner selection](docs/tier-selection.md).
Generate [validated C plans](docs/generated-plans.md) and write them through
the [shared prose pipeline](docs/baseline-c.md). Inspect
[equal-budget comparisons](docs/bc-comparisons.md) and
[final-prose realisation evidence](docs/realisation.md).
Require [strict gate readiness](docs/gate-readiness.md) before blind admission,
then produce a [protocol-bound G1 decision](docs/g1-decision.md) after closure.

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

## Phase 6 reconstruction

Phase 6 adds versioned structural encodings, corruption tasks, protected datasets,
local denoising and matched AR training, paired scoring, overlap audits, bounded
learning curves, G3 reporting, and a separate synthetic augmentation ablation.
Existing narrative-plan and corpus-export formats remain readable.

Start with the [codec and corruption workflows](docs/reconstruction-codec.md),
then [prepare protected datasets](docs/reconstruction-data.md) and
[train local models](docs/reconstruction-training.md). See the
[experiment workflow](docs/reconstruction-experiments.md),
[evaluation and G3 protocol](docs/reconstruction-evaluation.md), and
[augmentation ablation](docs/reconstruction-augmentation.md) for comparisons.

The optional ML runtime uses a supported native local backbone; it never downloads
weights automatically. Tiny CPU fixtures establish software mechanics. Scientific
G3 requires genuine passed G1/G2 evidence, registered budgets, identified models,
and any registered owner meaning review. Synthetic success remains pending.

## Phase 7 conditional planning

Prepare [traceable conditions and plan targets](docs/conditional-data.md),
then [train and sample variable layouts](docs/conditional-layouts.md) using the
local CPU runtime. The conditional denoiser learns structural layouts from fully
masked output, with classifier-free guidance and bounded work accounting.

[Full generation](docs/conditional-generation.md) accepts conditions and explicit
locks without a target annotation. It saves rejected attempts and validated
structural proposals. The [restricted text-completion workflow](docs/conditional-completion.md)
fills authorised names, descriptions and subtext through the gateway, preserving
structure and saving validated plans and cards. [Candidate E story runs](docs/candidate-e.md)
use both in an ordinary `nde run` experiment, then C/D's shared prose, ledger and
final-story selection, recording local model work separately from provider calls.
[Inpainting](docs/conditional-inpainting.md) offers E behind D's `nde plan revise`
mask/lock contract, with honest reporting of duplicate alternatives.
[`nde conditional-report`](docs/conditional-report.md) reports E's requested versus
realised conditions with documented predicates and complete denominators, keeps
genre, theme and ending adherence pending until the owner assesses it, and measures
ID-independent structural diversity across seeds and similarity to known training
populations and the permitted reference baseline. Real E experiments require
genuine passed G3 evidence; tiny fixtures establish software mechanics only.

[`nde ge`](docs/ge.md) prepares the registered G-E development comparison. A
no-spend readiness check replays the checkpoint's genuine G3 pass and lineage;
a committed protocol freezes E, the identified C/D reference, the exact dev
cohort and seeds, resource limits, the validator and structural criteria, the
owner sample and budget, and the single permitted development iteration. The
registered run records complete denominators, failures, costs and evidence, and
the owner's blinded structural review packet stays pending until the owner
completes it. `nde ge decide` then revalidates all of it and seals an immutable
G-E decision: complete pass, complete failure, pending or inconsistent. It
enforces the single development iteration across the repository's committed G-E
history and keeps the identified reference planner on failure. A pass records
eligibility to prepare Phase 8, not proof that E improves stories, and nothing is
launched. Fixture decisions stay pending; a real one needs genuine G3 evidence.

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
`benchmark/frozen.json`, `tests/fixtures/frozen-split/frozen.json`,
`benchmark/rubric/checksums.json`,
`data/manifest.calibration.yaml`, and the synthetic source/span checksum fixture
`src/nde/evidence/controls/evidence.json`), justified in `src/nde/verify/steps.py`.
The code has no type-checker suppressions.

CI (`.github/workflows/verify.yml`) runs the same command headless on every
push and pull request.

A separate CPU ML job installs the pinned optional group and sets
`NDE_REQUIRE_ML=1`, making missing ML support a failure. Run its checks locally:

```powershell
uv sync --locked --group ml
$env:NDE_REQUIRE_ML = "1"
uv run --locked --group ml nde-verify
```

The ordinary dependency set stays lightweight; ML tests may skip when the
optional group is absent. The CPU lane runs actual model updates, checkpoint
resume, matched reconstruction, sweeps and augmentation checks.

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
