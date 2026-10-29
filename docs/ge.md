# Registered G-E development comparison

G-E asks whether Candidate E is competitive enough with the reference planner
to justify Phase 8's owner reading. `nde ge` prepares and runs that comparison
on frozen **dev** briefs and assembles the owner's structural review packet. It
never applies the criteria, decides G-E, selects held-out briefs or launches
another gate; the decision report is separate.

```powershell
uv run nde ge readiness registration.json --out ge-readiness
uv run nde ge register registration.json --out prereg/ge-1.json
git add prereg/ge-1.json; git commit -m "research: preregister G-E dev comparison"
uv run nde ge check prereg/ge-1.json --out ge-check
uv run nde ge run prereg/ge-1.json --runs-dir runs
uv run nde ge verify prereg/ge-1.json runs/RUN
uv run nde ge packet prereg/ge-1.json runs/RUN --out review/ge-1
uv run nde ge review review/ge-1 owner-review.json
```

Exit codes: **0** ready, current or complete; **1** blocked, stale, pending, or a
run that did not succeed; **2** malformed or unsupported input. `readiness`,
`register`, `check`, `verify`, `packet` and `review` make no provider call and
run no model.

## Readiness: genuine G3 only

`readiness` resolves a registration without spending and writes
`ge-readiness.json` and `.md`. Real E experiments need a **genuine passed G3
decision**: the E checkpoint's G3 protocol is replayed now with
`reconstruction-g3`'s own assessment, which must be a non-test-only `passed`
outcome whose ID the checkpoint recorded. A test-only checkpoint, a relabelled
fixture without G3 evidence, a synthetic G3 criteria pass, a pending or failed
G3 decision, or a checkpoint whose recorded decision no longer replays is
blocked. No registration field accepts a closed issue or a declared pass in
place of that replay.

Lineage is rechecked too: the checkpoint's conditional fine-tuning dataset must
reload with its complete source lineage, and with genuine authority the
`g3_population` must be the G3 protocol's registered training population (the
[evidence report's](conditional-report.md) `g3_lineage` rule). Test-only
registrations accept test-only checkpoints and record `authority: test-only`;
they are restricted to local fake providers and produce test-only evidence.

## Registration

Paths are relative to the registration file. All fields are required unless
noted; unknown fields are rejected, so no threshold can be added that the
protocol does not register.

```json
{
  "id": "ge-dev-1",
  "test_only": false,
  "config": "../configs/ge-dev.yaml",
  "reference": {"condition": "C", "c_selection": "../selection.json"},
  "cohort": {"briefs": ["brief-01", "brief-02"], "repeat": 3},
  "resources": {
    "max_attempts": 3, "max_tokens": 1000000, "max_attention_cells": 100000000,
    "max_seconds": 60, "completion_max_tokens": 1024, "max_repairs": 2,
    "max_length_attempts": 3, "max_revision_passes": 4, "budget_tolerance": 0.1,
    "per_run_usd": 2.0, "per_day_usd": 5.0
  },
  "g3_population": "../data/g3-train",
  "protected_populations": [{"name": "g3-test", "path": "../data/g3-test"}],
  "validator_comparison": {
    "analysis": "pass_rate_difference", "margin": 0.05,
    "minimum_candidate_rate": 0.8, "aggregation": "pooled"
  },
  "structural_criterion": {
    "analysis": "owner_paired_competitive_share",
    "question": "Which plan has the more coherent long-range structure for this brief?",
    "minimum_share": 0.5
  },
  "owner_review": {
    "reviewer": "Owner name", "sample_size": 6, "sample_seed": 17,
    "minutes_per_item": 10, "max_minutes": 60
  },
  "iteration": {"attempt": 1},
  "prior_exposure": "Which dev briefs and outputs the owner has already read."
}
```

This example registers nothing real; supply genuine identities and limits.

- `config` is an ordinary `nde run` story configuration with exactly two
  conditions: `E` (`candidate_e`, see [E story runs](candidate-e.md)) and the
  reference `C` (`baseline_c`) or `D` (`baseline_d`), equal-budget matching of
  the two under `resource_policy: {version: "1"}`, explicit hard spending caps,
  no unpriced models and no other calls, judges or plan runs.
- `reference` must be the permitted, identified reference. C is identified by
  the owner's frozen C tier selection; the configuration's C condition must
  match that selection's reviewed contract (as `nde compare-cd` checks). D is
  permitted only by a `gd_decision` folder that replays from its `gd_batch` (and
  `gd_realisation` packets) as an actual pass naming D, and the D condition must
  satisfy the evidence report's planner contract against that reference baseline:
  pipeline, parameters, schema, the baseline's own role bindings and shared
  prompts. Its configuration checksum is not compared, because dev outputs
  necessarily come from another configuration than G-D's test run; the report
  applies the same contract, so it identifies the same D. A registered C is
  refused when a replayed G-D decision permits D.
- `cohort` must equal the configuration's dev briefs, in order, and its `repeat`.
  The configuration must select `briefs: {split: dev}` (optionally with `ids`).
  Each brief must be in the frozen `dev` split, unchanged. Configurations that
  select the `test` split, list IDs without `split: dev`, or name a brief outside
  `dev` are refused.
- `resources` must equal E's local-model limits and completion budget and the
  run's repair, length, revision, budget-tolerance and dollar limits.
- `validator_comparison`: E's plan-validity rate minus the reference's must be at
  least minus `margin`, and E's rate at least `minimum_candidate_rate` (0 for no
  absolute target); `aggregation` is `pooled` (every slot) or `brief_macro` (mean
  of per-brief rates). The denominator is every expected slot; a slot without a
  plan counts as not valid, unknown evidence is pending and boundaries are
  inclusive. These defaults are written into the protocol; no other analysis is
  supported.
- `structural_criterion`: the owner compares E's and the reference's plans for
  each sampled slot under `question`, blinded as A and B. Competitive pairs
  (E better, or `equivalent`) over the owner denominator must reach
  `minimum_share`. `unknown` ratings and slots where E has no plan are not
  competitive; slots where only the reference lacks a plan are excluded and
  reported; missing evidence is pending.
- `owner_review` fixes the reviewer, sample size and seed and the time budget;
  `sample_size × minutes_per_item` may not exceed `max_minutes`, nor the sample
  the cohort. The sample is drawn from the cohort slots, which exist before any
  output, and frozen in the protocol.
- `judges` defaults to and only accepts `diagnostic_only`: model judges are never
  validated for this task here, and their agreement cannot replace owner review.
- `iteration`: attempt 1, or the single development iteration (attempt 2) with
  `previous_protocol` and a `change`. Registration binds attempt 2 to an attempt
  1 that was run (its run claim exists) and records that protocol's and claim's
  IDs. The iteration must keep the first attempt's whole frozen cohort (briefs
  and checksums, split, repeat, run seed and slots), its reference identity, its
  validator and structural criteria and its owner review, must change E, and
  cannot point at another iteration. Registration checks one protocol and the
  attempt it names; it cannot see other protocols, so it does not stop a fresh
  attempt 1 or a second attempt 2 naming the same first attempt. Enforcing the
  history across protocols (at most one iteration per first attempt, no reset to
  a new attempt 1) belongs to the G-E decision.

`register` refuses anything readiness blocks and writes a sealed `ge-protocol`
record (`protocol_version` 1) with its paths rebased beside the protocol. Its
`frozen` section holds E's checkpoint (ID, file checksum, parameters, training
configuration, representation, condition schema, dataset and base), conditioning
fields and locks, guidance scale and policy, the hybrid completion binding,
prompt and limit, lineage, the dev cohort with brief and split checksums, run
seed and slots, resources, configuration, model, prompt and schema identities,
the reference identity, protected populations, the owner sample, the iteration
binding and the checksum of every input. `criteria` repeats the registered
rules. Commit the protocol before running.

`check` replays all of this against current evidence and requires the protocol
to be committed and unchanged. A changed checkpoint, dataset, brief, selection,
G-D or G3 replay, configuration, resource limit or protocol blocks it, and each
reason is listed in `ge-check.json`.

## Registered run

`run` rechecks the protocol and validates the configuration without spending.
It then claims the protocol's single run by creating the sealed
`PROTOCOL.run-claim.json` (kind `ge-run-claim`) beside the protocol, and only
then runs the bound configuration through the shared experiment machinery
(`--runs-dir`, `--cache`, `--ledger` as for `nde run`). It builds the [evidence report](conditional-report.md) over E's
stories and the reference outputs in `ge-evidence/`, then writes the sealed
`ge-run.json` (kind `ge-dev-run`, `run_version` 1) and `ge-run.md` in the run
folder:

- `units`: every registered slot for E and the reference, with status, seed,
  plan and final story identities and plan validity (`success`, `failure`,
  `no-output` or `unknown`). E's validity is the evidence report's `validity`
  predicate; the reference's is complete validation of its saved plan.
- `validator`: counts, pooled and brief-macro rates and the measured difference
  under the registered aggregation, with `complete` false while any unit is
  unknown. Measured, never judged.
- `failures`: each slot that did not succeed, with its error.
- `costs`: run spend and list price, and per condition the provider calls by
  purpose and local model work, never converted or summed, plus equal-budget
  matching.
- `evidence`: the report identity and its adherence, diversity, training
  similarity, protected overlap, reference, realisation and pending sections.
- `claim`: the run claim's path and ID; `owner_review: pending`, `judges:
  diagnostic_only`, `problems` and the checksum of every run and evidence
  artifact.

A protocol is run once, wherever its outputs go: `run` refuses whenever the
claim exists, and `verify` requires the run to carry the claim's ID. Keep the
claim with the protocol; deleting it to rerun is a protocol violation the
decision must treat as such. If the evidence report cannot be built after
generation, `ge-run.json` is still sealed with `evidence: {status:
unavailable, error}`, the run's costs, failures and complete denominators, and
E's saved plans counted `unknown` because they cannot be verified. Such a run
is never current, so no packet can be built from it, and it cannot be replaced
by regenerating. The command exits 1 when the run did not succeed or its
evidence is unavailable; failed stories remain in the denominators either way. `verify` replays a recorded run: the protocol check,
every artifact checksum, the evidence report's own replay and every measured
section. Any changed or missing plan, story, record or input makes it stale.

## Owner structural review

`packet` refuses a stale run and writes `packet.json` (kind `ge-review-packet`,
`packet_version` 1), a sealed `key.json`, `packet.md` and `review-template.json`.
The packet lists exactly the protocol's sample. For each slot with two plans the
owner sees both plans' cards as A and B, chosen by a seeded coin, under the
registered question; titles and plan identity lines are omitted, but structure
can still reveal the condition. The sealed key holds which condition and run file
each label shows. Slots without a plan are settled by the registered rules and
need no reading. The packet states the owner items and the
estimated and budgeted minutes. Do not open `key.json` while reviewing.

A review names the packet and, per owner item, the two plan checksums it read:

```json
{
  "packet_id": "ID from packet.json",
  "reviewer": "Owner name",
  "test_only": false,
  "ratings": [{
    "item": "item-01",
    "plans": {"A": "plan A sha256 from packet.json", "B": "plan B sha256"},
    "rating": "A",
    "note": "Why, in a sentence."
  }]
}
```

`rating` is `A`, `B`, `equivalent` or `unknown`. `nde ge review PACKET REVIEW`
rejects reviews of another packet, of items that are not owner items or of other
plan identities, then saves the review immutably under `reviews/`. Later ratings
of an item supersede earlier ones; history is kept. A review with
`diagnostic_provenance` (`model` and `prompt` required) is diagnostic, a
`test_only` review or any review of a test-only packet is synthetic, and a
different reviewer is recorded as such. Only the registered reviewer's actual
ratings of every owner item complete the review (`status: complete`, `evidence:
actual`). Synthetic ratings can finish the mechanics (`evidence: synthetic`) but
the status stays `pending`. A missing review stays pending, and a changed plan,
run record, run artifact or key makes the packet `stale`.

## For the G-E decision

The decision report consumes, without re-deciding anything here:

- the committed protocol (`nde.conditional.ge.read_protocol`, `check`): its
  `criteria`, `frozen` identities, `owner_review.sample`, `iteration` and
  `test_only`;
- the run claim (`read_claim`) and the one recorded run carrying its ID
  (`read_run`, `verify`): `validator`, `units`, `failures`, `costs`, `evidence`
  (including its `status`) and the artifact checksums;
- the packet status (`nde.conditional.ge_review.status`): `status`, `evidence`
  and the unblinded per-item `outcomes` (`competitive`, `not_competitive`,
  `excluded` or `pending`), which are populated only once ratings cover every
  owner item.

## Limits and evidence status

Because the configuration selects `split: dev`, the run, the evidence report and
every `nde ge` command check and read only the dev split's briefs and frozen
record; held-out brief files are never opened (a dev/test ID collision is still
found from file names). Keep Phase 8's held-out cohort in its own frozen
benchmark version all the same. Fixture runs use test-only checkpoints and selections and are
software evidence only: they never establish G3, G-E, condition adherence or
literary quality. Tests in `tests/test_ge.py` use a real tiny CPU checkpoint and
fake-provider calls.
