# Candidate D coarse planning

Use `pipeline: baseline_d` in an ordinary `nde run CONFIG.yaml` story experiment.
D generates `intentions.json` (stable ordered scene IDs, characters, story arc,
and explicit reader-affect trajectories), then `skeleton.json` conditioned on
those records, then completes each scene's channels. Higher-level intentions,
identities and caller constraints are checked on every candidate, with the same
bounded schema/semantic repair loop and gateway as C. Each rejected output and
its diagnostics are kept under `planning/`.

The brief, including target length, scene bounds and ending, enters every stage.
The ending is a literary constraint; generated plans cannot invent source spans
or establish facts from that literary statement. Explicit locked constraints use
the released narrative schema. D's global stage requires at least three scenes
and a 3–5 point affect trajectory, matching the schema's coarse trajectory units.

D uses C's existing ledger and scene-note assembly and B/C's unchanged
`scene_prose@1` prose role/template. Shared story selection, length retries and
equal-budget prose revisions retain their established behavior.

`planning-provenance.json` binds the pipeline, brief, schema, planner binding,
parameters, prompt checksums, code commit, seed, repairs and attributable calls.
The run's config/manifest supply the complete configuration identity. Accepted
plans include validation diagnostics and Markdown/HTML cards. Failed stages keep
partial artifacts and never create a successful final story.

Fake-provider tests in `tests/test_baseline_d.py` exercise conditioned requests,
locked/global attacks, bounded failure, budget aborts and zero-spend replay.
Scientific Phase 4 runs and literary judgements require separate G1/G-D evidence.

## Bounded structural refinement

Set D's `critic_iterations` (0–10, default 0) and bind a `critic` role to enable
refinement. Preparation requires explicit family metadata on planner, prose and
critic. The critic must belong to a different family from both generators.
`allow_critic_generator_family: true` is a separate explicit critic waiver; judge
waivers do not apply. The waiver is retained in refinement records.

`plan_critic@1` receives the selected plan checksum and deterministic diagnostics.
Every issue supplies that checksum, a `RevisionMask`, a concrete proposed change
and rationale. Stale, unknown, conflicting, locked or unsupported selections are
recorded and rejected. Overlapping issue targets reject that proposed batch.
Disjoint issues form one explicit candidate mask; constraints are always locked,
alongside `revision_locks` supplied by the caller. Related edge edits require
explicit mask authority. Scene and channel units use the same operator contract
as `nde plan revise`.

Each candidate passes schema, semantic, lock and boundary checks. Bounded repairs
use the run's `structured.max_repairs`. No actionable issues, unchanged outputs,
two consecutive failed candidates, or iteration exhaustion stop the loop. Provider
and hard-budget interruption retain the last accepted plan and failed story
artifacts. All critic/regeneration/repair calls are charged to the story's
experimental budget. Cached calls keep those tokens with zero new provider spend.
`refinement.json` records suggestions, masks, rejections, stopping reason and exact
selected identity; `planning-provenance.json` includes the complete attributable
initial-plus-refinement history. Critic preferences cannot override hard checks
and provide no scientific or literary-quality score.

## Selected C versus D development comparisons

`uv run nde compare-cd run CONFIG.yaml --selection SELECTION.json --out NEW_FOLDER`
validates the historical owner tier selection and its development evidence before
any new provider call. The config must contain exactly C (`baseline_c`) and D
(`baseline_d`), distinct dev briefs and shared C/D equal-budget matching. C's
selected tier, normalized planning parameters, schema, planner/prose bindings and
planning/shared-prose prompt identities must match the reviewed baseline. Adding
D's critic role and prompts does not invalidate C's reviewed identity. Synthetic
selections are visibly labelled test-only and cannot establish a scientific gate.

This command uses the existing story run, bounded matching revisions, final prose
selection and realisation tooling. Reports expose expected pairs even after an
abort, all attributable generation tokens, list-price costs, new spend, final
artifact identities, length validity and matching outcomes. `compare-cd report
FOLDER` rechecks saved artifacts and selected-C evidence; changed or missing
inputs make the report incomplete. The command never hides failed pairs by
reporting only an intersection of successes. Output folders are immutable run
registries; choose a new folder for another experiment.

D can also import a completed saved plan/revision with `params.saved_plan` and
`params.provenance` pointing at its final `plan.json` and complete
`planning-provenance.json`. Brief, schema, prompt/model/parameter identities,
seed, repair bounds, critic iteration history, final checksum and complete call
sequences are validated before prose. Each historical planning, critic, rejected
regeneration and repair call is attributed exactly once with its original token
budget and list-price cost, zero new spend and `imported_from` source identity.
Partial or standalone revision provenance lacking its upstream D planning calls
cannot enter the comparison with an artificially free planner. Imports preserve
readability of earlier D initial-planning provenance without critic histories.
