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
