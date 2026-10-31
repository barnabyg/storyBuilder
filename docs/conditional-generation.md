# Conditional full generation

**Status:** G1 and its one retest did not pass, so [the diffusion track has stopped](g1-four-pairs.md). Real generation needs a passed G3, so the code refuses it. The tooling remains, tested on fixtures only.

`nde conditional-generate generate request.json --out proposal-run` generates a
structural proposal from an identified conditional checkpoint, permitted
conditions and explicit caller authority. It never accepts a target annotation,
scene list or inferred target layout. This is a separate full-generation mode;
Phase 6 reconstruction keeps its original codec and 10–70% corruption contract.

The request has `checkpoint`, `conditions`, `seed`, `guidance_scale`, `locks`,
`constraints`, `max_attempts`, `max_tokens`, `max_attention_cells` and
`max_seconds`. Conditions follow [the preparation semantics](conditional-data.md).
An omitted conditions object is entirely unknown. The checkpoint registers
scene/entity/record/token capacity before inference. Scene ranges are inclusive
caller requests; an unsupported upper boundary is an error, never a truncation.

The model generates collection counts, identities, order, references and
structural values from a completely masked output. Each attempt starts without
observed structural target content and uses a deterministic seed derived from
the request seed and attempt index. There is no C/D or LLM structural-planning
fallback. An invalid learned sample remains a rejected sample.

Locks map existing ID-based scalar selectors to caller-supplied JSON values,
for example `characters/characters.0/name`. Locks cannot supply a whole graph,
evidence, sources, envelope identities or new records. Code applies the locks
and caller constraint records, then validates the entire plan. Unknown lock
identities reject the candidate. Contradictory hard constraints fail before
sampling. A natural-language ending remains a literary intention; it does not
create a `required_ending` fact automatically.

The recovery policy is bounded rejection. Every retry re-masks all editable
structural content; validation does not expand caller authority. Schema,
references, chronology, character knowledge, state transitions, constraints and
locks must all pass before selection. The workflow stops at the first accepted
proposal or the attempt/time/token/attention limit. Guidance passes count toward
actual work. A durable reservation precedes each model attempt so interrupted
work remains attributable.

The output preserves rejected candidates, generated tokens, seeds, diagnostics
and work in numbered attempt records and readable `report.md`. A selected
`proposal.json` includes a content identity, original conditions, locks, source
input checksums and an explicit text-completion manifest. Inspect with
`nde conditional-generate inspect proposal-run`. Completed output is immutable;
use a new directory for another generation.

`structural-proposal` means the structural checks passed. It does not mean a
completed narrative plan, literary adherence, owner approval or a passed G-E.
Descriptions, names and subtext still require the separately authorised
[text-completion workflow](conditional-completion.md). Fixture checkpoints and
proposals remain test-only. Real conditional training and experiments require
genuine passed G3 evidence; implementation and synthetic fixtures do not advance
that scientific gate.
