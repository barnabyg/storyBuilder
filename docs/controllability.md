# Plan controllability and structural diversity

`nde control-report INPUT.json --out REPORT` measures explicit plan changes.
Input paths resolve relative to `INPUT.json`. The input contains `before`,
`after`, a revision operator `revision` specification, and `targets` with
distinct IDs. JSON, Markdown and HTML reports include checksums of all inputs,
canonical before/after plan identities, actual changed units, validator
diagnostics, mask compliance and byte-stability of the locked subtrees.
The reports are deterministic and use the revision operator's canonical JSON.

```json
{
  "before": "source/plan.json",
  "after": "revision/plan.json",
  "revision_provenance": "revision/provenance.json",
  "revision": {
    "mask": {"units": ["scenes/s3/summary"]},
    "locks": ["scenes/s1"],
    "instruction": "Change the third scene's intention"
  },
  "targets": [
    {"id": "changed-scene", "kind": "changed", "selector": "scenes/s3/summary"},
    {"id": "literary-effect", "kind": "literary"}
  ]
}
```

`changed` and `unchanged` compare a stable ID selector's canonical JSON.
An unknown selector produces an unknown target result; it cannot count as
success. `literary` always remains unknown pending actual assessment.

`renewed_hope_middle` selects one reader-affect record by `annotation_id`;
the caller explicitly registers which ordinal trajectory represents hope.
The report does not infer that interpretation from keywords. The default
threshold is 2 on the existing 0–4 ordinal scale, and `min_episodes` defaults
to 1. Scene positions are zero-based list positions. For n scenes the middle
third is `[floor(n/3), ceil(2n/3))`. A renewal occurs at an observed position
in that interval only when its value is at least the threshold and the
immediately preceding adjacent scene's observed value is below the threshold.
A high plateau counts once, separated low-to-high transitions count
separately, a renewal just outside the interval does not count, and an
initial high value without a preceding low value is not a renewal.
Every scene in the interval and its immediate predecessor must have an
explicit ordinal observation; missing/uncertain observations produce unknown.
There is no interpolation between the schema's coarse trajectory points.

An overall success requires every requested target to succeed, every lock to
remain stable, an authorised edit scope, and complete hard validation. A failed
hard check yields failure; otherwise any pending target yields unknown.
Each target includes its full request and canonical request checksum so a
registered target cannot be replaced with an easier predicate under the same ID.

An accepted-revision claim also requires `revision_provenance` from the
operator: its accepted/no-op status, source identity, exact mask/locks and
output identity must match the request. Without it the difference can still
be inspected, but overall success remains unknown and `accepted_revision`
is false. G-D admission additionally requires that record and its source/output
artifacts to belong to the actual D story and match an accepted refinement
iteration and its attributable calls. An invented before-plan cannot establish
gate controllability merely by differing from the final D plan.

All outcomes are **plan-only**. Structural compliance is separate from final
prose realisation, literary improvement, originality and scientific gate
decisions. Synthetic fixtures establish software behaviour only.

The public `replay_report(path)` rechecks every source checksum, recomputes
the predicates and boundary evidence from the original request, and requires
an exact match to the saved report. Gate admission uses replay rather than
trusting a saved success flag.

## Structural diversity

`nde diversity-report INPUT.json --out REPORT` reads a manifest containing
`plans`, each with `path`, a distinct integer `seed`, and `functions`.
The latter maps stable scene IDs to existing uncertainty-aware readings,
for example `{"s1": {"value": "setup"}}`. These are explicit caller
annotations; neither scene titles nor summaries are classified automatically.
Missing, unknown and ambiguous function labels remain unknown; a pair's
function distance is null if either plan has an uncertain label.

The versioned metric `ordinal-position-multiset-jaccard-v1` aligns scenes by
zero-based chronology, characters and facts by their declared list positions,
and setup references by the setup scene position. IDs and their spelling do
not enter the features. Function features contain scene position and exact
registered function label. Edge features contain the edge type and aligned
endpoints: causal, setup/payoff, reveal-to-holder and knowledge-use-by-holder.
Reveals with multiple holders produce one feature per holder. Duplicate
features retain their multiplicity. This metric measures changes at fixed
ordinal positions, without attempting semantic graph isomorphism or inferring
equivalence after scene insertion/reordering.

For each pair of distinct seeds, distance is one minus the sum of minimum
feature counts divided by the sum of maximum counts (multiset Jaccard).
Both empty edge sets have distance zero. The range is 0 (identical features)
to 1 (disjoint features). Renaming IDs alone yields zero. Invalid plans retain
their identities and a failed validator flag; edge distance is null rather
than a misleading number. All annotated readings and source checksums remain
in the evidence so missing labels and input changes are inspectable.
