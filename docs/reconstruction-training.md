# Local reconstruction training

**Status:** G1 and its one retest did not pass, so [the diffusion track has stopped](g1-four-pairs.md). Real training needs passed G1 and G2 results, so the code refuses it. The tooling remains, tested on fixtures only.

`nde reconstruction-train initialize --out BASE --width 16 --max-positions 4096`
creates deterministic local random **test-only** weights. It never downloads a model.
`register-local WEIGHTS CONFIG --out BASE` imports pretrained native backbone
weights. `CONFIG` must give `width`, `max_positions`, `model_id`,
`pretraining_description` and `artifacts`, a list of local provenance files to
checksum. The sealed weights object contains `weights` and an explicit `test_only`
boolean; fixture weights cannot be registered as real weights.

The supported native architecture is one residual output-query attention/MLP block,
with learned byte and position embeddings and 260 fixed codec tokens. Output queries
attend to the complete visible byte sequence and bounded output positions. Attention
memory grows with output capacity times total sequence length. Existing model bundles
with another architecture require an explicit adapter; they are rejected here.
The artifact records weights, architecture, parameter count and initialization identity.

A JSON training config supplies `dataset`, `base_model`, `objective: "denoising"`,
`seed`, `learning_rate`, `max_steps`, `max_tokens`, `max_attention_cells`,
`max_seconds`, `max_output_tokens`, `inference_max_tokens` and `test_only`.
Run `preflight CONFIG`, `train CONFIG --out RUN`, or `resume CONFIG --out RUN`.
`--steps-this-call N` deliberately stops at a usable checkpoint for bounded work.
Real training additionally requires `g1_batch` (plus any `g1_realisation` packets) and `g2_gate`:
genuine passed G1/G2 outcomes and the approved export must match. G-D is not required.

The denoiser receives exactly the saved visible-input whitelist, followed by a fixed
capacity MASK block. Its outputs are JSON replacements for allowed slots. Gold output
lengths are never exposed at inference. Syntax has unit loss; value bytes inherit
hard/soft channel weights. Padding has zero loss. Unknown and excluded fields cannot
become authoritative value targets. Oversize inputs and targets are rejected.

Each step atomically publishes an immutable JSON checkpoint containing actual model,
Adam optimizer, RNG and deterministic sampler progress, input identities, losses and
consumed resources. A separate latest pointer is atomically updated. Resume recovers
fully published work after an interrupted pointer update; pending files never replace
the last usable checkpoint. Changed inputs/configuration reject resume. Token,
attention-cell and elapsed training budgets continue from saved progress.

`reconstruct CHECKPOINT DATASET --out RESULTS` consumes only visible slots during
inference. It records exact selected checkpoint/dataset/task identities, work,
replacement values and parse/mask diagnostics for every expected task, including
capacity failures and resource stops. A tiny random model commonly produces invalid
JSON; these failures remain evidence rather than disappearing from denominators.

Ordinary imports do not load torch. Install optional pinned dependencies with
`uv sync --locked --group ml`. The dedicated CPU ML CI job runs canonical verification
with `NDE_REQUIRE_ML=1`, so absent runtime support fails rather than skipping training
mechanics. Tests perform actual CPU parameter updates, optimizer/RNG resume,
checkpoint publication recovery and reconstruction without downloads or a GPU.


## Matched autoregressive infilling

Set `objective` to `ar` for a causal infilling baseline using the same native
weights, byte vocabulary, parameter count, saved tasks and resource caps.
`match DENOISER_CONFIG AR_CONFIG` rejects changes in seed, initialization,
optimizer settings, data/tasks, vocabulary or budget policy. Equal epoch counts
alone are not used to assert compute equality.

Both objectives see the same complete visible prefix. AR training shifts gold
replacement bytes by one position; causal attention hides future target bytes.
At inference only generated prior bytes enter the target stream. Completion stops
at EOS or the fixed registered capacity. The denoiser predicts its masked output
block in one pass. Their different actual token and attention work
is recorded separately, alongside their common registered caps. This is a declared
budget comparison, not a claim that unequal realized inference costs are equal.
The public `inspect_attention` API allows controlled verification of causal future
invariance. CPU integration tests perform real AR updates, checkpoint resume and
reconstruction on exactly the saved denoiser tasks.


`inference_max_seconds` defaults to 3600 for standalone reconstruction. Training and
inference write bound phase-timer journals; their elapsed allowances persist across
resume. Inference checks before every forward pass, reports resource-stopped tasks
and records `inference_elapsed_seconds`. If result publication is interrupted, the
unknown remaining inference allowance is conservatively charged rather than granted
again. A registered sweep freezes smaller matched per-phase caps when necessary to
fit its aggregate time budget.
