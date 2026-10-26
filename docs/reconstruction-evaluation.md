# Reconstruction evaluation

`nde reconstruction-score DATASET CANDIDATE COMPARATOR --out OUTPUT` reads a
protected validation/test dataset and the exact saved denoising and AR prediction
artifacts. It writes immutable `scores.json`, readable `report.md`, and a
source-linked `meaning-review.json` packet. The candidate direction is denoising
minus AR. Training identities, initialization, vocabulary, seed and declared
training/inference budgets are compared explicitly; actual work is retained.

Only hidden eligible fields contribute to accuracy. Hard targets and diagnostic
soft targets have separate denominators. Unknown/ambiguous readings are retained
as context and are ineligible labels. Missing predictions and invalid decodes
remain in the complete expected task and validator denominators. Copying visible
fields cannot increase hidden-field accuracy; attempts to predict an unmasked
position are validation failures.

Typed causal, setup/payoff and knowledge edge precision/recall/F1 are computed on
edges touched by hidden hard fields. Reference identities are normalized by
canonical collection position. Duplicate predictions add false positives;
repeated gold edges represent one relation. Empty gold denominators are
unavailable, rather than automatic perfect recovery. Long-range recovery is
reported separately: causal or setup/payoff endpoints have at least one
intervening scene; knowledge edges use the earliest represented prior reveal or
explicit initial-holder anchor, and an unknown origin is unavailable. Knowledge
uses follow the using character's acquisition history; another character's
earlier knowledge cannot turn a local use into long-range recovery.

The established narrative validators inspect reconstructed retained views with
empty excluded channels. This is structural validity of the registered view,
not evidence that the full source-linked tier-L annotation has been recovered.
Rows retain actual/requested masking rates, schedules and parent author/work
provenance so related scenes/windows/families can be aggregated together.

The meaning-review packet is evaluator-only. Its default status is `missing`;
LLM diagnostics and synthetic known-answer fixtures cannot stand in for actual
owner assessment or establish a scientific model advantage.

`nde reconstruction-overlap TRAINING QUERY --out OUTPUT` saves an immutable
`overlap.json` and readable report. The training index is never modified. Exact
source/work and protected-author matches are flagged independently of annotation
family or edition names. The inspectable metric is multiset Jaccard distance over
canonical typed retained records on shared channels. ID-only renaming is
normalized; missing channels are unavailable and uncertainty is preserved.
Distance is a candidate similarity measure, not proof of memorisation or its
absence. Ties are ordered by record identity.

## Registered G3 workflow

`nde reconstruction-g3 readiness CONFIG` performs a no-training check.
`nde reconstruction-g3 register CONFIG --out PROTOCOL.json` freezes thresholds
and evidence before held-out predictions, paired scores or owner meaning labels
exist. `nde reconstruction-g3 report PROTOCOL.json --out OUTPUT` writes an
immutable decision and readable report; it never starts training or Phase 7.

The version-1 configuration identifies `training`, protected test `evaluation`,
local `base_model` (`model.json` or its directory), frozen `experiment.json`,
exact future `score_reports`, `overlap_report`, and the retained C/D `reference`
artifact. Real readiness also requires `g1_batch` (plus any `g1_realisation`
packets) and `g2_gate`: outcomes are recomputed through the established gate
validators. G-D passage is not required. The model weights, datasets, export,
G2 identity, experiment job pairs, seeds, train subsets, target vocabulary and
resource budgets are bound to the protocol. Related examples remain aggregated
by registered `aggregation: work` or `author`.

Each G3 protocol covers one audited training population and its registered
author-grouped subsets. The overlap audit covers that population; every subset
retains its approved export/G2 lineage. A broader sweep may compare several
training populations or granularities, but each requires its own G3 registration
and audit. Unrelated, unaudited populations cannot enter the same gate decision.

Defaults are `validator_target: 0.95`, `positive_margin: 0.01`, final registered
checkpoint selection, and the three separately reported long-range edge
families. Every family must meet the positive candidate-minus-AR margin.
Requested and actual masking rates must both be within 30–50% for primary
results; other tasks remain identified outside the primary analysis. Missing
results, stale checkpoints, incomplete training, changed inputs, unequal budgets
or unresolved overlap remain pending evidence and retain their denominators.
Completed metric failures retain the C/D reference and record the stop action.

`test_only: true` permits explicitly synthetic software fixtures without real
G1/G2 passage; a fixture criteria pass always remains scientifically pending.
Synthetic scores and owner judgements cannot establish a literary advantage.

Owner meaning evidence remains separate. If `require_meaning_review` is true,
freeze `owner_review`, `minimum_owner_samples` (default 1), and
`meaning_preservation_minimum` (default 0.8) before collecting labels. The review
is a version-1 sealed `owner-meaning-review` record with nonempty `reviewer`,
`test_only`, exact `score_ids`, input snapshots and `samples`. Each sample names
`score_id`, `task_id`, and the actual owner's `judgement`: `preserved`, `changed`
or `uncertain`. Missing, stale, uncertain, synthetic or unverified owner evidence
cannot satisfy the registered requirement. The saved decision binds the review
checksum while preserving its status separately from deterministic metrics.
