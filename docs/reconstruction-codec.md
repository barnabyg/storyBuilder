# Structural reconstruction artifacts

**Status:** G1 and its one retest did not pass, so [the diffusion track has stopped](g1-four-pairs.md). Real training exports need a passed G2, which needs a passed G1, so only test-only exports can reach this workflow. The tooling remains, tested on fixtures only.

The additive Phase 6 codec preserves the released narrative-plan and corpus-export
formats. `reconstruction-codec encode EXPORT --out DIRECTORY` validates the Phase 5
export before reading its training annotations. `inspect DIRECTORY` repeats those
checks. `decode DIRECTORY --record 0 --out retained.json` writes an explicitly
filtered retained view, not a complete tier-L annotation.

The versioned specification, schema identity, fixed vocabulary identity and export
identity are recorded in `encoding.json`. Lists retain chronology and explicit
length slots; object keys have canonical lexical order. IDs become collection and
position references (for example `scenes.0`). Source text, spans, checksums, original
IDs, plan titles and excluded channels never enter the view or tokens. Full original
annotations remain at their unchanged checksummed paths in evaluator provenance.
Only registered channel fields are retained. Character identities are anonymous
reference scaffolding; excluded plot retains scene identity scaffolding only.

Tokens are BOS=2, EOS=3 and UTF-8 bytes plus four; PAD=0 and MASK=1 are reserved.
There is no learned vocabulary and no fitting on evaluation data. Scalar slots
carry path, primitive/reference type, channel and hard/soft policy separately from
eligibility. Unknown and ambiguous readings retain confidence/alternatives but
have no prediction loss. ID and length scaffolding never has prediction loss.
Window preparation may additionally disable loss for channels whose context falls
outside the view. Hard and soft policies remain distinct in the saved artifact.

Decoding exactly reverses retained canonical content, including reading uncertainty
and internal references. Evidence/source data and excluded labels are deliberately
not reversible from tensors. Source/export/schema/codec changes are rejected;
scene/token capacity violations produce errors without truncation. Retained-view
validator success cannot establish that omitted annotation channels are complete.

`reconstruction-tasks create ENCODING --schedule CONFIG --out DIRECTORY` freezes
`tasks.json` and a readable report. The configuration contains `seed`,
`repetitions` and a `schedules` list of `{ "kind": "fields", "rate": 0.3 }`.
Kinds are random eligible scalar fields, whole scenes, whole channels and a
contiguous canonical field span. Rates must be 10–70%; unit counts round upward.
Scene masks include eligible fields in annotations/links touching the scene,
including both endpoints of causal edges and the setup scene of payoff links.
Immutable ID/list-length scaffolding and unknown readings remain visible.
Optional `channel` restricts eligible channels. Span `start` and `count` allow
registered first/middle/last/full boundary experiments; explicit count overrides
the requested rate and the report records actual field coverage separately.
Empty eligible channels produce zero coverage rather than inventing targets.

The task saves the exact visible slots, their identity, allowed positions and a
separate evaluator-only target list. Seeds bind source record, schedule and
repetition. `model_input` exposes compact path/value tuples only, with distinct
mask objects for hidden values; it never serializes targets, gold views, source
text or auxiliary provenance. Both objectives consume this same saved input.
Reconstruction requires exactly the allowed prediction positions and preserves
all observations; missing/extra predictions or invalid decoded references produce
diagnostics. Hard targets have loss weight 1; soft targets have registered weight
0.25; scaffolding, excluded and uncertain fields have zero loss.
