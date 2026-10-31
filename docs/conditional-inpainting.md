# Conditional inpainting behind the revision contract

Candidate E can revise a saved plan through the same `uv run nde plan revise
config.yaml` workflow, mask, locks and artifacts as [D's operator](revisions.md).
Select it with an additive `backend` section beside the unchanged version-1
revision request:

```yaml
plans:
  briefs_dir: ../benchmark
  briefs: {ids: [brief-01]}
  repeat: 3                      # three independent alternatives
  params:
    saved_plan: path/to/plan.json     # a plan saved by nde plan generate; the IDs below are illustrative
    revision:
      version: '1'
      mask: {units: [states/belief/key, knowledge_facts/lamp-fact/description]}
      locks: [states/goal]
      instruction: Give the belief a new key and describe the fact.
    backend:
      version: '1'
      kind: e
      checkpoint: ../e/checkpoints/step-00000100.json
      target_policy: {plot: hard, causality: hard, information: hard, goal: hard, belief: hard}
      conditions: {genre: parable}
      guidance_scale: 1
      max_attempts: 3
```

`checkpoint` resolves relative to the config. `target_policy` names the approved
channel policy of the checkpoint's training targets and must include `plot`.
Channels it excludes are not represented by E and stay as the source has them.
Preflight checks it against the policy of the training dataset the checkpoint is
bound to, found among the checkpoint's recorded inputs by its sealed identity;
any difference in a non-excluded channel fails before any call or inference. If
that dataset is no longer available unchanged, a real checkpoint is refused and a
test-only fixture checkpoint proceeds with the policy recorded as declared and
unverified (`target_policy_check` in the provenance `backend` block).
`conditions` uses the registered [condition semantics](conditional-data.md) and
defaults to unknown. `max_tokens`, `max_attention_cells` and `max_seconds` bound
local work per alternative; `text_prompt` defaults to `plan_inpainting_text@1`.
`kind: d` selects the released D operator explicitly. Omitting `backend` is D with
byte-for-byte unchanged provenance.

## How E edits

Before any call, the source must pass D's mask/lock preflight, the checkpoint
must be trained for the requested guidance, a real (non-fixture) checkpoint
needs current genuine passed G3 evidence, and the source layout must fit the
registered capacity. E then sees the source as the checkpoint's textless
canonical layout. Only the serialized values of selected, unlocked structural
leaves are masked; identities, list order and counts, locked and unselected
content stay observed and are clamped. Masked values keep their serialized width,
so version 1 cannot add, remove or reorder records even where a whole-channel
mask would let D do so. Selections E cannot represent at all fail preflight.
Selected, unlocked structural values outside the target policy or the textless
layout stay as the source has them; provenance and `inpainting/observation.json`
list them per selected unit under `unrepresented`. When no selected structural
value is representable but selected text is, as for a mask of only
`knowledge_facts/lamp-fact/description`, E runs no learned inference: the planner
completes the text manifest under the same checks, and provenance records backend
status `text-only` with no attempts and no local work.

Preflight hashes the checkpoint file once and binds its identity. Each attempt
uses its own derived seed and is reserved before inference, and must run on that
same checkpoint: if the file becomes unreadable or its identity changes, the
attempt is recorded as rejected, the alternative fails with status
`checkpoint-changed` and its provenance and boundary are still saved. Any other
sampler failure, such as the local model running out of memory, marks the
attempt `failed` with the exception named in its diagnostics and fails the
alternative with status `sampler-failed`; the traceback is kept as
`inpainting/sampler-error.txt` beside the same provenance, boundary and
`last-accepted` plan, and the run continues with its remaining alternatives. The
complete learned layout is mapped back onto the source by record identity, and
selected unlocked text (titles, summaries, descriptions and interpretive reading
text, including channels outside E's representation) is reset for completion.
That structural candidate must pass the same complete-plan validation and
scope/lock/identity boundary as D before the planner role completes the exact
text manifest through the gateway, cache and budget caps. The completed plan is
checked again. Argmax checkpoints stop after one rejected attempt because a
retry would repeat it exactly; sampled checkpoints retry up to `max_attempts`.

Every candidate is checked as a whole plan, so a locally plausible span edit
cannot break incoming or outgoing causal, setup/payoff or knowledge timing, or a
hard constraint. A learned candidate that changes an identity, order, count,
lock or unselected unit is rejected, never spliced.

## Artifacts and alternatives

The run keeps D's layout: `plan.json`, cards, `revisions/revision/provenance.json`,
`boundary.json`, `change-summary.md`, `source-plan.json` and `last-accepted/`.
Provenance adds a versioned `backend` block with the checkpoint/parameter/
representation identities, test-only status, target policy, conditions, guidance,
sampling temperature, target-policy check, masked leaves, unrepresented values, text manifest, every attempt's seed, work
reservation, generated masked values and diagnostics, and total local
tokens/attention cells/passes/time. LLM completion calls, tokens and costs remain
in `calls`. `inpainting/observation.json` and `inpainting/attempt-NNN.json` keep
the observed layout and each attempt; text completion attempts are under
`planning/revision-revision-text/`. Rejected, resource-stopped and provider-failed
alternatives keep the source and `last-accepted` plan and never publish `plan.json`.

`plans.repeat` produces independent alternatives from the same recorded original;
each is reloaded and checked against the original's recorded checksum. The run's
`alternatives.json` and `alternatives.md` list every expected alternative with its
backend, seed, checkpoint/guidance, status, identities, costs and local work. Only
distinct accepted changes count: no-ops, exact duplicates and structural
duplicates (identical apart from descriptive text) are reported, not counted.
`uv run nde plan alternatives RUN_DIR` rebuilds the report from the saved
artifacts and exits non-zero when a plan, provenance record or source changed.

Current conditional checkpoints are trained with every output position masked
(`training_condition_policy` in the checkpoint binding), so a partially observed
layout is outside their training distribution. Inpainting such a layout is
mechanically valid and checked like any other candidate, but the quality of the
masked values it proposes is unvalidated.

Fixture checkpoints establish software mechanics only. An accepted E revision is
a mechanically valid generated plan; literary adherence, owner review and G-E
remain separate. Real E experiments require genuine passed G3 evidence.
