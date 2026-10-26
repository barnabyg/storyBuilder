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
explicit initial-holder anchor, and an unknown origin is unavailable.

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
