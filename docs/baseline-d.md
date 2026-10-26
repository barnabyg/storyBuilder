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
