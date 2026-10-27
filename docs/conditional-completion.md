# Restricted completion of generated proposals

`nde conditional-complete run CONFIG --out DIRECTORY` completes the free-text
manifest on a saved conditional structural proposal. It is a hybrid step:
learned structure stays fixed and the language model supplies only authorised
text. A successful completion is a mechanically validated generated narrative
plan, not source evidence, an owner-approved reading, or evidence of literary
adherence.

The configuration names a `proposal`, `fields` containing exact ID-keyed leaf
selectors, a gateway `provider`, a `model` binding, and an explicit token `price`.
The selected fields must come from the proposal's `completion_fields` manifest.
Ambiguous readings can retain their ambiguity while their permitted text is
completed. Scalar alternatives use selectors such as
`subtext/subtext.0/reading/alternatives/0`; completion preserves the alternatives'
count, order and unselected text, as well as reading status and confidence.
Completing a subset saves a partial result; publication of a final plan requires
the whole manifest. Caller locks take precedence over the manifest.

Names, titles, summaries, setup/fact descriptions and known interpretive reading
text may be completed. IDs, list order and count, references, symbolic states,
constraints, uncertainty statuses, evidence and provenance cannot be selected.
Unknown interpretive values stay unknown. The response contract is an exact
dictionary of requested selectors to nonempty strings, rather than permission
to return or replace a plan. Every resulting candidate is checked against both
the narrative validator and an independent comparison with the learned source.

The workflow retains the original proposal, the original plan, raw responses,
exact validation failures, selected text, prompt/configuration/model identities,
and call/token/cost/cache accounting. The gateway supplies response caching and
cost reservations; `max_cost_usd`, `max_tokens`, and `max_repairs` bound calls.
Each call has a durable intent written before the request. An interruption keeps
that pending intent and any cached response; it never causes an invalid final
plan to be published. A rerun must use a new output directory and cannot rewrite
the original proposal or a completed run.

After complete validation, `plan.json`, `validation.json`, `cards.md` and
`cards.html` expose the completed plan. Partial, rejected, provider-failed and
budget-stopped runs keep their diagnostics and last valid artifact without
publishing a final plan. All filled text is identified as generated fiction.
No completion advances G3, G-E, or a real E experiment.
