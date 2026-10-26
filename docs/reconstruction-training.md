# Local reconstruction training

`nde reconstruction-train initialize --out BASE --width 16 --max-positions 4096`
creates deterministic local random **test-only** weights. It never downloads a model.
`register-local WEIGHTS CONFIG --out BASE` imports pretrained native backbone
weights with a model ID, pretraining description and checksummed local provenance
artifacts. The sealed weights object contains `weights` and an explicit `test_only`
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
Real training additionally requires `g1_batch`, `g1_realisation` and `g2_gate`:
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
