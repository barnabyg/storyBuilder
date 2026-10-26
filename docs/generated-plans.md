# Generated narrative plans

`uv run nde plan generate configs/plans-c.yaml` generates a tier-L plan for
one dev brief. Configure the planner role, explicit prices and hard run/day
caps before a paid run; this command makes provider calls. It rejects test
briefs, which require the protocol-bound gate workflow.

The `plans` section accepts `briefs_dir`, `briefs`, `repeat` and `params`.
Planning parameters are `skeleton_prompt`, `channels_prompt` and an optional
list of explicitly represented `constraints` using the released narrative
constraint schema. The brief's natural-language ending is carried as a
literary requirement, not converted into invented symbolic canon.

The whole-story skeleton establishes stable character/scene/beat identities,
scene functions, character arcs and ending intentions. Each subsequent call
returns that same envelope with the current scene's channels added. Accepted
identities and channels cannot be rewritten by later calls. Schema and hard
semantic failures share the configured bounded repair loop; generated plans
cannot invent source text or evidence spans.

The printed run directory contains the usual manifest, call ledger and report,
plus `plans.jsonl`. Each plan's folder contains `plan.json`, `validation.json`,
`cards.md`, `cards.html`, `skeleton.json`, every original/repair response with
its diagnostics under `planning/`, and `planning-provenance.json`. Provenance
records the brief, schema, code, model, prompt checksums, parameters, seeds,
repair limit, selected plan checksum and every attributable planning call.
Cached calls retain their experimental tokens and cost zero new spend.

Exhausted repairs exit 1 and retain failed artifacts with `status: failed`;
provider failures and budget aborts retain their run status and any completed
attempts. Malformed configuration exits 2 before calling a provider. No
plan-only result is recorded as a successfully generated story. Existing
`nde plan inspect`, `validate`, `project` and `schema` commands operate on the
saved `plan.json` as before.

For an offline manual check, run the fake-provider fixture
`uv run pytest tests/test_plan_generation.py -q`. It needs no API key, paid
service or external download; its sample arcs and intentions are synthetic.
Actual literary quality and owner approval remain unmeasured by these tests.
