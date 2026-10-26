# Structural reconstruction artifacts

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
