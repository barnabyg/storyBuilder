# Registered G-E development comparison

G-E asks whether Candidate E is competitive enough with the reference planner
to justify Phase 8's owner reading. `nde ge` prepares and runs that comparison
on frozen **dev** briefs, assembles the owner's structural review packet and,
as a separate final step, seals the [G-E decision](#g-e-decision). It never
selects held-out briefs, trains, generates outside the registered run, or
launches another gate or Phase 8.

```powershell
uv run nde ge readiness registration.json --out ge-readiness
uv run nde ge register registration.json --out prereg/ge-1.json
git add prereg/ge-1.json; git commit -m "research: preregister G-E dev comparison"
uv run nde ge check prereg/ge-1.json --out ge-check
uv run nde ge run prereg/ge-1.json --runs-dir runs
uv run nde ge verify prereg/ge-1.json runs/RUN
uv run nde ge packet prereg/ge-1.json runs/RUN --out review/ge-1
uv run nde ge review review/ge-1 owner-review.json
git add prereg/ge-1.run-claim.json; git commit -m "research: record the G-E dev run"
uv run nde ge decide prereg/ge-1.json runs/RUN --packet review/ge-1 --out decisions/ge-1
git add prereg/ge-1.*-claim.json decisions/ge-1; git commit -m "research: decide G-E"
uv run nde ge replay decisions/ge-1
```

Exit codes: **0** ready, current or complete (for `decide`, a complete actual pass
or failure); **1** blocked, stale, pending, inconsistent, or a run that did not
succeed; **2** malformed or unsupported input. Every command except `run` makes
no provider call and runs no model.

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
  pipeline, parameters, schema, the baseline's own role bindings, and every
  prompt D uses, recorded by the baseline and loaded now with the same checksum. Its configuration checksum is not compared, because dev outputs
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
  1 that was run (its run claim exists) and **decided**: the first attempt's
  settled [decision claim](#settled-decisions-and-replay) must be committed and
  record a `failed` criteria outcome, and, for a real iteration, an actual
  `failed` outcome. No iteration follows a pass or a pending first attempt.
  Registration records the first attempt's protocol, claim and decision IDs, and
  `check` revalidates them. The iteration must keep the first attempt's whole
  frozen cohort (briefs and checksums, split, repeat, run seed and slots), its
  reference identity, its validator and structural criteria and its owner review,
  must change E, and cannot point at another iteration. Registration checks one
  protocol and the attempt it names; it cannot see other protocols, so it does
  not stop a fresh attempt 1 or a second attempt 2 naming the same first
  attempt. The [decision](#development-iteration-history) enforces that history
  across protocols.


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
reason is listed in `ge-check.json`. It also rebuilds every other protocol
section (criteria, owner sample and budget, rules) from the registration and the
frozen evidence, so a protocol re-sealed with edited sections does not check.

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
owner sees both plans' cards as A and B under the registered question; titles
and plan identity lines are omitted. The A/B coin uses a random salt kept only
in the sealed key, not the protocol's public sample seed, and plans are
identified by salted tokens, so nothing in `packet.json`, `packet.md` or the
template matches the run's condition-named checksums. The key holds the salt and
which condition, run file and checksum each label shows. Slots without a plan
are settled by the registered rules and need no reading. The packet states the
owner items and the estimated and budgeted minutes.

Blinding is procedural, not enforced: the key and the run folder stay on disk,
and plan structure itself can still reveal the condition. Do not open
`key.json`, `ge-run.json` or the run folder while reviewing.

A review names the packet and, per owner item, the two plan tokens it read:

```json
{
  "packet_id": "ID from packet.json",
  "reviewer": "Owner name",
  "test_only": false,
  "ratings": [{
    "item": "item-01",
    "plans": {"A": "plan A token from packet.json", "B": "plan B token"},
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

Reviewer identity is self-declared, as for every other owner review in this
repository ([realisation](realisation.md), tier and corpus reviews): `actual`
means a non-synthetic, non-diagnostic review whose `reviewer` names the
registered reviewer and whose plan tokens match the packet. Nothing proves that
the owner, rather than someone or something writing their name, made the
ratings, or that the key stayed closed. The saved review keeps the checksum of
the submitted file; keep that file under version control or with the owner's
other review records, so the decision can be audited against it. Owner review
evidence rests on this procedure, not on cryptographic proof.

## G-E decision

`nde ge decide PROTOCOL RUN [--packet PACKET] --out NEW_FOLDER` applies the
registered criteria and writes a sealed `decision.json` (kind `ge-decision`,
`decision_version` 1) and `decision.md` in a new folder. It first revalidates
everything, with no provider call:

- the protocol `check`: committed and unchanged, the genuine G3 replay, and the
  checkpoint, data and reference lineage;
- the run `verify`: the claimed run, every artifact checksum and the evidence
  report's replay, plus the run's own `problems` and the report's reference
  identification;
- the packet: it must belong to this protocol and run record, show exactly the
  registered owner sample with the registered reviewer and criterion, settle each
  item as the run's outputs require, and its key must map each owner item's A and
  B to that slot's E and reference plans. Its review status must be current.

Any failure makes the decision **inconsistent**. An inconsistent decision never
applies the structural criterion: its item outcomes and share are withheld, so no
owner rating is unblinded until the evidence is consistent.

The criteria are applied in the candidate direction, E against the identified
reference:

- **Validator comparison.** E's rate minus the reference's, under the
  registered aggregation, must be at least minus `margin`, and E's rate at least
  `minimum_candidate_rate`. Every expected slot is in each denominator and a slot
  without a plan counts as not valid. Any unknown unit leaves it pending. Counts,
  unknown units, rates and the difference are recorded.
- **Structural competitiveness.** From the packet's unblinded outcomes,
  competitive pairs (E better, or equivalent) over the owner denominator must
  reach `minimum_share`. Unknown ratings and slots where E has no plan are not
  competitive. Slots where only the reference lacks a plan are excluded and
  listed. A missing packet, missing ratings or evidence, or a sample with no
  owner-comparable pair leaves it pending.

Boundaries are inclusive: a floating-point rate within 1e-12 of its threshold
meets it. Both criteria are required. Any pending criterion leaves the whole
`criteria_outcome` pending, even when the other has already failed.

Owner evidence counts only as the registered reviewer's **actual** ratings of
every owner item of a current packet. Diagnostic (model) ratings never count.
Synthetic ratings count only for a test-only packet, and then only as software
evidence. A test-only protocol, run or packet keeps the `outcome` **pending**
even when its `criteria_outcome` is passed or failed.

### Frozen owner ratings

The first decision that applies the structural criterion (consistent evidence and
a complete owner review) freezes the ratings it used. `decide` creates the sealed
`PROTOCOL.review-claim.json` (kind `ge-review-claim`) beside the protocol, naming
the packet and each saved review file with its ID. From then on every decision
and replay applies exactly those reviews: a later review, superseding or
diagnostic, is reported but never applied, and a frozen review that is missing or
changed makes the packet stale. Commit the review claim straight away: until it
is committed, decisions that rely on it are inconsistent. Any other frozen set for
the protocol, beside it or in history, is inconsistent too. Before a structural
outcome has been shown, the owner may still revise ratings; afterwards the
decision cannot be changed by re-rating.

| `outcome` | Meaning | `action.result` |
| --- | --- | --- |
| `passed` | Both criteria met on actual, consistent evidence | `eligible_to_prepare_phase_8` |
| `failed` (attempt 1) | A criterion not met on actual, consistent evidence | `iterate_once_or_stop` |
| `failed` (attempt 2) | The single iteration also fell short | `stop` |
| `pending` | Evidence incomplete or synthetic | `none` |
| `inconsistent` | Stale, changed, mismatched or history-violating evidence | `none` |

Every decision records the identified C/D reference, which stays the planner.
A pass records eligibility to *prepare* Phase 8: freeze a fresh held-out
benchmark and pre-register that comparison separately. It is not evidence that E
improves final stories. No outcome starts training, paid generation, another
gate or Phase 8.

From the verified run the decision preserves every unit's plan and final-story
identity, failures and costs. It also keeps the evidence report's
condition-adherence, diversity, overlap, reference and realisation findings and
pending items. It records the protocol, run, claim, packet, key and review
identities, G3 authority, lineage, prior exposure, limitations and the
diagnostic-only judge status. Adherence judgement, realisation review and overlap
are reported; they are not G-E criteria. Paths are relative to the decision
folder, so a decision moved with its evidence still replays. `decide` refuses
(exit 2) evidence that can only be named by an absolute path, such as a protocol,
run or packet on another drive.

### Settled decisions and replay

A decision **settles** its protocol when its criteria were completely applied to
consistent evidence: passed or failed, actual or synthetic. `decide` then creates
the sealed `PROTOCOL.decision-claim.json` (kind `ge-decision-claim`) beside the
protocol, naming the decision. A settled protocol is never decided again, even
if the claim is deleted, while any commit still holds the claim or the complete
decision itself. Pending and
inconsistent decisions do not settle, so they can be written again to new
folders once the evidence changes. Commit the decision claim and the decision
folder.

`nde ge replay DECISION` recomputes the decision from the recorded protocol, run
and packet, and lists every section that differs and every new problem. A settled
decision must also be its protocol's only settled decision. A changed plan, story
or frozen review, or a history violation, makes a decision stale, and replay
exits 1. Reviews saved after the ratings were frozen do not: replay lists each one
as not applied and stays current, so a later model-judge review or re-rating
cannot disturb a settled decision or the development iteration bound to it.


### Development iteration history

The decision enforces the single development iteration across protocols. Its
history is every G-E protocol, run claim, review claim, decision claim, decision
and review packet committed in the protocol's repository, in any commit of any
local branch, deleted or superseded versions included, whatever characters their
paths use. The claims currently kept beside those protocols count too. A history
record that cannot be read back, or is not a sealed record of its kind, makes the
decision inconsistent rather than being skipped. A **comparison** is every protocol with the same test-only
status and the same canonical reference identity: the reference condition, its
planner pipeline and parameters, and the owner's C selection ID or the G-D
decision ID. The dev briefs do not matter, and neither do the bytes or path of
the file holding the selection or decision. Only protocols that were run
(claimed) count, and a run attempt counts even if it is never decided or stays
pending.

- The decided protocol's run claim must be committed, so the repository records
  the run.
- A comparison has one run first attempt. Another run attempt 1 against the same
  reference, on any dev briefs and including one registered after deleting the
  first, is a **history reset**: every decision in the comparison becomes
  inconsistent.

- It has at most one run development iteration, bound to that first attempt. A
  second one is inconsistent, as is an iteration of another first attempt.
- An iteration is registered, and later decided, only against the first
  attempt's **settled, committed decision**, found through the claim beside the
  first protocol. When the iteration is decided, that decision must still replay
  and must be the one the iteration was registered on, and its
  `criteria_outcome` must be `failed`. No iteration follows a pass. A real
  iteration also needs an actual failed first attempt, not a synthetic one. The
  iteration's decision records the first attempt's protocol, claim and decision
  IDs, so replaying it replays the first decision too.

- Another review packet committed for the same protocol is inconsistent: one
  packet per run.

History cannot see records that were never committed here: a run claim deleted
before it was committed, packets kept outside the repository, or protocols in
another clone. Like reviewer identity, this rests on following the procedure:
commit each protocol before running, its run claim after running, and the
review and decision claims after deciding.


## Limits and evidence status

Because the configuration selects `split: dev`, the run, the evidence report and
every `nde ge` command check and read only the dev split's briefs and frozen
record; held-out brief files are never opened (a dev/test ID collision is still
found from file names). Keep Phase 8's held-out cohort in its own frozen
benchmark version all the same. Fixture runs use test-only checkpoints and selections and are
software evidence only: they never establish G3, G-E, condition adherence or
literary quality, and every fixture decision stays pending. No genuine G3 pass
exists yet, so no real G-E protocol can be registered. Tests in `tests/test_ge.py`
and `tests/test_ge_decision.py` use a real tiny CPU checkpoint and fake-provider
calls. Genuine outcomes are exercised only by promotion inside a test: the G3
authority replay and the evidence report's test-only flag are promoted in memory,
and an actual failure, its settlement, the bound real iteration and an actual pass
then run through the public commands.
