# Condition adherence, diversity and training similarity

`nde conditional-report build REQUEST.json --out NEW_FOLDER` reports what Candidate E
was asked for against what it produced, how different its outputs are across
seeds, and how close they are to known training plans and to the reference
baseline. `nde conditional-report check FOLDER` replays a saved report against the
current evidence. Neither command calls a provider, runs a model, writes training
data or decides G-E.

```json
{
  "story_runs": [{"run": "runs/e-dev", "condition": "E", "label": "e-dev"}],
  "revision_runs": [{"run": "runs/e-alternatives", "label": "e-alternatives"}],
  "reference": {
    "gd_decision": "gd/decision-1",
    "gd_batch": "eval/batches/gd-1",
    "gd_realisation": [],
    "runs": [{"run": "runs/cd-dev", "condition": "C"}],
    "c_selection": "selection-1.json"
  },
  "training_populations": [{"name": "g3", "path": "data/g3-train", "role": "g3"}],
  "protected_populations": [{"name": "test", "path": "data/g3-test"}],
  "realisation": ["review/e-dev-brief-01"],
  "judgements": ["owner-adherence.json"],
  "neighbours": 3
}
```

Paths resolve relative to the request file. `story_runs` are ordinary `nde run`
folders with a `candidate_e` condition ([E story runs](candidate-e.md));
`revision_runs` are `nde plan revise` folders using the E backend
([inpainting](conditional-inpainting.md)). Every unit is a story repetition or
revision alternative, with ID `story/LABEL/BRIEF/rN` or `revision/LABEL/BRIEF/rN`;
`label` defaults to the run folder name and must be distinct. All other fields
are optional. `target_policy` is only used when the checkpoint's training dataset
is unavailable, and is then recorded as declared and unverified.

The output folder must be new. It holds `report.json` (kind
`conditional-evidence-report`, record version 1 with `report_version` 2, sealed with a
content `id`), a readable `report.md` and an unsealed `location.json` locating the
request. Every path in `report.json`, including the `inputs` checksums, is relative
to the request file's folder, so the content and its `id` do not depend on where the
evidence lives. Write the request's own paths relative to it as well. Build exits 0 when a report is written, including one
with failures, unknowns, problems or pending review, and 2 for malformed or
unsupported inputs.

## What the report binds

The expected units come from each run's saved configuration: every selected
brief times `repeat`, so a story or alternative that left no record is still
counted as `missing`. Runs that select held-out (`test`) briefs are refused; this
report reads development evidence only. All units must use one E checkpoint.

For each unit the report records the brief and its checksum, the registered
conditions and allowed constraints (E story locks, or the revision mask and
locks), checkpoint identity, generation or revision history, seed, status and the
final plan and story identities. It checks that the plan matches its
`conditional-selection.json`/planning provenance or revision provenance, the
final story matches the run's selection and the revision run's `alternatives.json`
still rebuilds; any mismatch is listed under `problems` and makes `consistent`
false. `inputs` holds the checksum of every file read, including run
configurations, briefs, `stories.jsonl`, `local-work.jsonl`, `comparison.json`
(whose `resource_policy` is carried into `runs`), plans, stories, provenance,
the checkpoint and the training datasets.

## Deterministic compliance

Only explicitly representable requests are evaluated mechanically. Each row is
`success`, `failure`, `unknown` or `no-output` with its reason, and
`by_predicate`/`total` count each status; the denominators include every expected
unit. `failure` always means a produced output did not satisfy the request.

| Predicate | Success when |
| --- | --- |
| `scene_range` | the selected plan's scene count is in the requested inclusive range |
| `target_words` | the final selected prose's word count is in the requested range widened by the run's `length.tolerance`; `unknown` for plan-only revisions |
| `lock` | an E story lock's ID-keyed selector holds exactly the locked value; `failure` if absent |
| `revision_lock` | a revision lock's canonical JSON is unchanged from the recorded source |
| `revision_scope` | the plan changes only units the mask authorises (the shared revision boundary) |
| `validity` | the selected plan passes complete plan validation |

Operational failure is kept apart from non-adherence. A story that failed after
planning keeps its saved plan when `conditional-selection.json` and the planning
provenance verify it, so its plan predicates are evaluated and only
`target_words` is `no-output`. A unit with no saved plan, such as a rejected
generation, a sampler or provider failure or a failed revision alternative, is
`no-output` for every predicate: it stays in the denominator but is neither success
nor failure. A missing unit, or one whose saved provenance, plan or story cannot be
read (status `unreadable`), is `unknown`, never success or failure. So is a
story whose plan does not match its selection or provenance, and a revision
alternative whose saved source plan or recorded `before_sha256` is not the run's
recorded original (status `unverified`): locks and scope cannot be judged against a
substituted baseline. Both are also listed under `problems`. Model diagnostics (attempts,
rejections, guidance, local work and completion calls) are reported separately and
never count as adherence.

## Literary judgement

Genre, themes and the natural-language ending (`ending_intent`) need actual
judgement. They stay `pending` until the owner assesses the current final output;
units without final output (no prose for a story, no plan for a revision) are
`no-output`, and missing, unreadable or unverified units `missing`. A judgement file is:

```json
{
  "reviewer": "Owner name",
  "test_only": false,
  "assessments": [{
    "unit": "story/e-dev/brief-01/r1",
    "output_sha256": "SHA-256 of the final selected story (the plan identity for a revision)",
    "condition": "genre",
    "rating": "adheres",
    "note": "Why, in a sentence."
  }]
}
```

`rating` is `adheres`, `does_not_adhere` or `unknown`. A file with
`diagnostic_provenance` (a model estimate) is counted as diagnostic only, a
`test_only` file is counted as synthetic only, and an assessment of any other
output identity is counted as stale. None of these changes the status. Units that
request nothing for a condition have no row for it.

## Structural diversity

Each run and brief forms a group of seeds. The report compares every pair of
units with output, so duplicates stay in the evidence as zero distances and are
also marked `exact_duplicate_of` (same plan identity) or `structural_duplicate_of`
(same retained structure).

`retained-structure-multiset-jaccard-v1` uses the checkpoint's retained codec view
under its training target policy. Identities and references become typed list
positions, and titles, summaries, descriptions, names and interpretive reading
text become null. Each channel's canonical records form a multiset; the distance
is one minus shared over combined counts across the channels both views retain,
as in the [overlap audit](reconstruction-evaluation.md). Renaming identities alone
gives distance 0. Unknown and uncertain readings stay in the features, are counted
per sample (`unknown_labels`) and mark the pair `uncertain`. The edge distance is
the [diversity report's](controllability.md) ordinal typed-edge metric and is null
for invalid plans.

## Training similarity and lineage

The checkpoint's conditional fine-tuning dataset is found from its binding and
reloaded with its complete source lineage. Any `training_populations`, such as the
reconstruction population behind the checkpoint's G3 evidence (`role: g3`), are
added; each must be optimizer training records in the same codec. Every unit's
retained structure is compared with every record (nearest `neighbours`, with
distance, work and uncertainty). `exact_structural_matches` counts distance 0.

`lineage` is complete only when the fine-tuning dataset is verified and, for a
checkpoint with passed G3 evidence, a supplied `g3` population is verified against
that evidence: the checkpoint's G3 protocol must still assess as the genuine pass
whose decision ID the checkpoint recorded, and the population's identity must be the
protocol's registered training population. Any other population tagged `g3` makes
`g3_population` a `mismatch`. A checkpoint without passed G3 evidence leaves supplied
`g3` populations `unverified` and does not require one. A missing or changed
population is listed, not omitted. The base model's pretraining data is
reported as `none` only for a random fixture initialization; otherwise overlap
with pretraining material is an explicit unknown limitation. Structural similarity
is a candidate match, not proof of memorisation or its absence.

`protected_populations` must be evaluator-only or validation/test views. They are
only ever queried: each one is audited against every record of every training
index for shared source, work or author (`protected_overlap`); `overlaps` lists every
overlapping pair, however many there are, independent of `neighbours`. E outputs'
nearest protected records are listed. A protected view cannot be supplied as a training population.

## Reference baseline

D is the reference only when an actual G-D pass permits it. `gd_decision` names the
`nde gd-report` output folder and `gd_batch` the closed batch it was reported from
(both or neither), with the same `--realisation` packets as `gd_realisation`. The
report rebuilds the decision from that batch exactly as `gd-report` does; it must
equal the saved `decision.json` (and `reference-baseline.json`), or the decision does
not replay and C stays the reference. A replayed decision permits D only when its
outcome is `passed`, it is not test-only and its reference baseline names D; a
synthetic, pending or failed decision keeps the identified C baseline. The reason is
always recorded under `reference.reason`, with the decision's identity, outcome and
`replays` under `reference.gd_decision`.

When a replayed decision names the permitted reference, each reference run must
match the planner contract of its full recorded baseline
(`planner_contract_problems`, shared with [G-E](ge.md)): the pipeline, its
parameters, the plan schema, the baseline's own role bindings, and every prompt
the D planner uses, which the baseline must record and the run must have loaded
with the same checksum (other prompts both record must also agree; C's own
prompts from G-D's configuration are not required). The configuration checksum is not part of the contract: dev runs, which
select dev briefs and may run E beside D, necessarily differ from G-D's frozen
test configuration. Any of the decision's planning artifacts inside the run must
also be unchanged. Each difference is listed under `reference.identity_problems`. Reference runs of
the other condition are not used and are also listed there. Each E story is
compared with the reference outputs for the same brief and brief checksum.

A reference is `identified` only when its identity is established: by a replayed
actual G-D decision naming it, or, for C, by `c_selection`, the owner's frozen C
tier selection. Each C reference run's original configuration is then checked
against that selection's reviewed C contract (tier, planning parameters, planner and
prose bindings, schema and prompts), as `nde compare-cd` does. Any difference is
listed under `reference.identity_problems`. Comparisons with an unidentified
reference are still reported but have status `unidentified`, not `compared`, and the
missing identification is listed under `pending`.

## Realisation evidence

Packets from the [realisation workflow](realisation.md) are matched to E stories
by run, condition, brief and repetition. A packet counts only while it binds the
current final selected prose and plan; changing the story makes it `stale`.
Every successful E story without a current packet stays `missing`, and an
incomplete owner review stays `pending`.

## Checking a saved report

`nde conditional-report check FOLDER [--request REQUEST.json]` rebuilds the report
from its request and compares it with the saved content. The request is found from
`location.json` (relative to the report folder) unless `--request` names it. It exits
0 when the report is current, 1 when any bound input or section has changed (each
reason is printed), and 2 when the saved report is unreadable. Stale reports are
never edited; build a new folder.

A report can be committed and replayed elsewhere when the request, the report and
every input move together with their relative layout preserved. Absolute paths that
runs recorded (configuration, brief, checkpoint and dataset locations) are then
found at the same place relative to the request. Upstream sealed records keep their
own absolute bindings, however: a conditional dataset rechecks its export, cohort
and annotations, a realisation packet its run, a G-D decision its batch and a G3
protocol its evidence at the paths they recorded. Where those are not available,
`check` reports the affected sections as changed rather than assuming they hold.

## Use in the G-E comparison

A [registered G-E comparison](ge.md) binds the report `id` and reads `adherence`,
`diversity`, `training_similarity`, `reference`, `realisation`, `pending` and
`test_only`. Reports from fixture checkpoints are software evidence only. The
report records evidence; the G-E decision, owner structural review and any
scientific claim remain separate. Tests in `tests/test_conditional_report.py`
run a real tiny CPU checkpoint with fake-provider calls.
