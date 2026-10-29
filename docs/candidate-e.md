# Candidate E story runs

Use `pipeline: candidate_e` in an ordinary `nde run CONFIG.yaml` story
experiment. E plans with a local conditional checkpoint and the established
conditional workflows, then writes prose exactly as C and D do. It has no prose
realiser of its own: the planned-state ledger, scene notes, `scene_prose@1` on
the `prose` role, normalisation, length control, equal-budget revisions and
final-story selection are C's, unchanged.

```yaml
roles:
  planner: {provider: local, model: completion-model}   # E's hybrid text completion
  prose: {provider: local, model: prose-model}          # the shared prose contract
stories:
  briefs: {ids: [brief-01]}
  conditions:
    - name: E
      pipeline: candidate_e
      params:
        checkpoint: .nde/e-training/checkpoints/step-00001000.json
        condition_fields: [genre, themes, ending_intent, target_words, scene_range]
        guidance_scale: 1.5
        locks: {}            # optional ID-keyed scalar selectors, as for generation
        max_attempts: 3      # local-model limits, as for `nde conditional-generate`
        max_tokens: 1000000
        max_attention_cells: 100000000
        max_seconds: 60
        completion_max_tokens: 1024
```

This example selects no model, checkpoint or scientific setting; supply real
identities, prices and hard caps. A real (non-fixture) checkpoint must be bound
to a current genuine passed G3 decision, or preparation fails before any call.
Tiny fixture checkpoints run only as test-only mechanics.

## What E does per story

1. `condition_fields` chooses which brief attributes become
   [conditions](conditional-data.md): genre, themes, the ending (kind and detail
   as one natural-language intent), the target length as a one-value range and
   the brief's scene range. Omitted attributes stay unknown; omitting
   `scene_range` means the brief's scene bounds are not requested.
2. [Full generation](conditional-generation.md) samples a structural proposal
   with the configured guidance, locks and limits, using a seed derived from the
   story's seed. Its request, every attempt, durable reservations and report
   are kept under `conditional/generation/`.
3. [Restricted completion](conditional-completion.md) fills the proposal's whole
   text manifest on the `planner` binding through the run's own gateway, so the
   run's cache, prices, cost ledger and caps apply. Its records are under
   `conditional/completion/`. Only a complete, validated completion becomes
   the selected plan.
4. The selected plan is saved as `plan.json`, `validation.json` and cards, with
   `conditional-selection.json` naming the checkpoint, proposal, completion and
   plan identities. C's prose step then writes `ledger.json` and each attempt's
   `scene-NN.inputs.json` and `scene-NN.md`.

`planning-provenance.json` (`pipeline: candidate_e`, version 1) binds the brief,
schema, parameters, conditions, checkpoint identities (training configuration,
condition schema, guidance policy, dataset, base and scientific evidence),
generation request and report, sealed proposal and completion records, the
completion calls and the local work. It is written whether planning succeeds or
fails.

## Resource accounting

Local model work is never a provider call. Each generation adds one line to
`local-work.jsonl` (a `LocalWorkRecord`): checkpoint, guidance scale and policy,
seed, outcome, and per attempt the forward passes actually run (conditional,
unconditional or both), model tokens, attention cells and seconds, with the
operation's total wall time. Interrupted attempts keep their reserved work.
Provider calls stay in `calls.jsonl`; the completion's calls are attributed to
the step `conditional-completion`.

The report's **Resource accounting** section groups each story's provider calls
by what they served (planning, completion, prose, revision) beside its local
passes, model tokens, attention cells and seconds. It lists every local
operation and each selected E plan with its proposal, completion, plan and final
story identities. The units are never converted into one another or summed.

## Comparisons and the resource policy

An `equal_budget` comparison that includes E must register a resource policy:

```yaml
  equal_budget:
    conditions: [D, E]
    resource_policy: {version: "1"}  # provider_measure: generation_tokens,
                                     # local_compute: report-separately
```

Without it, preparation fails before any call. Under policy version 1, matching
still uses provider generation tokens. Each match records both conditions' local
work (`local_work`), and `comparison.json` adds `resource_policy`, a
`budget_basis` and, per pair, each side's resources and `local_compute_equal`.
A matched pair is therefore matched on provider tokens only; it is never an
equal total budget. A pair in which a condition has zero attributable provider
tokens is unresolved. Configs without the policy keep their released identities
and files.

## Failures

A missing, unusable or relabelled checkpoint, malformed lock authority or a
streaming planner binding fails preparation. During a run, a rejected or
resource-stopped generation, or an incomplete completion, fails that story
only: `stories.jsonl` keeps the failed record, the local work and attempted
calls stay attributed, the conditional records and failed provenance are saved,
and comparisons report the pair as unresolved rather than omitting it. A run
budget stop or provider error during completion aborts the run as for any step,
after the completion record and local work are saved.

## Importing a saved E plan

Set `saved_plan` and `provenance` to a previous E story's `plan.json` and
`planning-provenance.json`. The checkpoint must be identical (same content
identity); brief, schema, conditions, guidance, locks, limits, planner binding,
seed and repair bound must match the run. Every generation attempt must lead
to the sealed proposal, the local work must equal the generation report, and
the sealed completion must fill exactly the proposal's manifest to give the
saved plan, with consistent call identities and list-price accounting. Any
difference fails the story before prose.

An accepted import attributes the original completion calls once (step
`imported-conditional-completion`, zero new spend, `imported_from`) and the
original local work once (`imported-conditional-generation`), so reuse never
hides prior work or cost.

## Evidence status

Fixture runs establish software mechanics only. They are not condition
adherence, literary quality, owner review or a G-E result. Real E experiments
require genuine passed G3 evidence and a separately registered comparison.
Tests in `tests/test_candidate_e.py` run a real tiny CPU checkpoint and
fake-provider calls.
