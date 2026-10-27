# Conditional layouts

Phase 7 adds `nde-conditional-layout` version 1 alongside the released Phase 6
byte codec. Reconstruction still observes collection lengths and canonical IDs,
and retains its 10–70% corruption policy. Full generation predicts an entire
canonical JSON structural view, including collection lengths, anonymous IDs,
order and reference topology. Its output block is sized by registered capacity,
never by a target annotation, and is fully masked before a forward pass. Learned
EOS terminates the proposal. Missing EOS, malformed JSON, unsupported collections,
duplicate identities, invalid chronology/references and capacity excess reject a
proposal; no sequence is silently truncated.

`conditional-layout encode DATASET CAPACITY.json --out DIRECTORY` converts an
approved conditional dataset into inspectable training targets;
`conditional-layout inspect DIRECTORY/layouts.json` checks the saved layout
identity, token/view consistency and narrative validation. Capacity specifies
`max_scenes`, `max_characters`, `max_records` (all identified records including
beats), and `max_tokens` (including BOS/EOS). These are upper bounds, not an
observed scene count or target layout. Scene order is chronology; all references
use collection/position identities inherited from the filtered training codec.

Names, titles, summaries, descriptions and interpretive readings become the
explicit placeholder `Unfilled generated text`. Source prose, evidence and
owner labels never enter a layout. State and relationship keys and reading values
remain learned symbolic content; replacing those with a common placeholder would
erase transitions. Unknown readings retain their null values and uncertainty.
The schema-required text skeleton is a structural proposal, pending the separate
restricted text-completion workflow and complete narrative validation.

`conditional-train preflight CONFIG.json`, `train CONFIG.json --out RUN`,
`resume CONFIG.json --out RUN`, `inspect CHECKPOINT`, and
`sample CHECKPOINT --out DIRECTORY` expose the learning path. The native local
attention architecture is fully fine-tuned from identified local base weights.
The config binds the approved conditional dataset, capacity, adaptation, seed,
optimizer policy and token/attention/time allowances. Checkpoints atomically save
parameters, Adam state, RNG, sampler position and accumulated resource use.
Resume rejects changed inputs and retains interrupted work reservations and time.

Tiny random CPU weights and synthetic datasets require `test_only: true`.
Real Phase 7 training requires a current, genuine passed G3 protocol; synthetic
criteria and completed implementation tickets cannot authorize advancement.
Fixture parameter updates and layout sampling provide software evidence only,
not literary adherence or a G-E decision.
