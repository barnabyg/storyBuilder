# G1 on four pairs

**Status (30 October 2026):** registered, with stories generated and no reading yet.
G1 has no result.

G1 asks whether multi-channel plans produce stories the owner finds better than
beat-only plans at equal budget. It compares the selected M tier (C) with
Baseline B.

## Why four pairs

The plan first asked for 30 pairs, about ten hours of reading. The owner declined
that on 27 October 2026. On 30 October, before reading anything, the owner decided:
"Four pairs of stories is more than enough for me to determine if there is a
consistent difference."

This makes G1 the owner's judgement of a consistent difference, not a statistical
test. With no real difference and no ties, chance alone gives 3 or 4 wins of 4 about
31% of the time, so a pass is a decision to continue rather than proof of an effect.

## Rule

G1 passes when both of these hold:

- C scores strictly higher than B on coherence in **at least 3 of 4 pairs**. Ties
  count as non-wins.
- C's mean originality is at least B's, the owner's no-average-loss decision.

Overall preference and strength are recorded but do not decide the gate. If G1 does
not pass, the plan's fail action applies: at most one follow-up on fresh briefs, or
stop the diffusion track.

The machine-readable protocol is
[`benchmark/prereg/g1-20261030.json`](../benchmark/prereg/g1-20261030.json).

## What the pairs are

The four pairs use dev briefs brief-03 to brief-06, frozen in
`benchmark/pilots/mb-20261027/`, and were not used to choose the tier. They were
generated on 30 October by run
`20261030T150215Z-dev-M-B-pilot-20261027-v1-78ddef` for US$0.48, and all four passed
strict readiness. They were first registered as a
[development pilot](../benchmark/prereg/dev-mb-pilot-20261027.txt) that could not
pass G1. Those approved records stay unchanged as history, and the G1 protocol
supersedes that limit. The 30 frozen test briefs stay untouched.

G1 binds its stories through `nde eval g1-run`, which replays the same config from
the response cache with no provider spend. The story checksums must match the
original run.

## Reading

The agent guides the owner through one story and one question at a time, and enters
the answers. The owner keeps no notes. For each pair, the owner gives:

1. coherence for each story, from 1 to 5;
2. originality for each story, from 1 to 5;
3. which story they preferred overall;
4. how strongly, from 1 to 3.

Reading takes about 80 minutes, with breaks whenever the owner likes. The owner
does not work on the code, so no 48-hour wait applies. The report may still show its
standard timing caveats. Conditions stay hidden until all four pairs are scored and
the batch is closed. The agent never opens `sealed/` before then.

## Commands

Run each command on one line. The agent runs these for the owner:

```powershell
uv run nde eval g1-check benchmark/prereg/g1-20261030.json configs/dev-mb-pilot-20261027.yaml --selection docs/evidence/dev-tier-selection-2026-10-27/selection.json --schema-evidence .nde/g-schema-packet/evidence-indexed --out runs/g1-check
uv run nde eval g1-run benchmark/prereg/g1-20261030.json configs/dev-mb-pilot-20261027.yaml --selection docs/evidence/dev-tier-selection-2026-10-27/selection.json --schema-evidence .nde/g-schema-packet/evidence-indexed --runs-dir runs
uv run nde eval g1-batch benchmark/prereg/g1-20261030.json configs/dev-mb-pilot-20261027.yaml --selection docs/evidence/dev-tier-selection-2026-10-27/selection.json --schema-evidence .nde/g-schema-packet/evidence-indexed --name g1-20261030 --from RUN C --from RUN B
uv run nde eval serve --batch g1-20261030
uv run nde g1-report batches/g1-20261030 --out runs/g1-decision
```

Extract `.nde/g-schema-packet/` from
`docs/evidence/g-schema-2026-10-27/evidence.zip` first.
