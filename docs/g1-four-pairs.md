# G1 on four pairs

**Status (30 October 2026):** read and closed. **G1 and its one permitted retest did
not pass, so the diffusion track stops.** See the
[first evidence](evidence/g1-2026-10-30/README.txt) and
[retest evidence](evidence/g1-retest-2026-10-30/README.txt).

G1 asks whether multi-channel plans produce stories the owner finds better than
beat-only plans at equal budget. It compares the selected M tier (C) with
Baseline B.

## Result

| Brief | C coherence | B coherence | C originality | B originality | Preferred |
| --- | --- | --- | --- | --- | --- |
| brief-03 | 1 | 1 | 2 | 1 | C, strength 3 |
| brief-04 | 1 | 4 | 2 | 4 | B, strength 3 |
| brief-05 | 1 | 2 | 2 | 3 | B, strength 2 |
| brief-06 | 1 | 3 | 2 | 3 | B, strength 2 |

C won 0 pairs on coherence, tied 1 and lost 3, against the 3 wins required. Mean
originality, C minus B, was -0.75, against a minimum of 0.

Two caveats affect how far this reaches. The owner preferred the story shown first in
every pair, and C was first only for brief-03. To match budgets, B also received 13
to 17 whole-story revision calls per brief, while C spent its budget on planning and
had one prose pass. So this result may reflect revision rather than planning. It does
not change the outcome.

## Retest

**Result: did not pass.** C won 2 of 4 on coherence and needed 3.

| Brief | C coherence | B coherence | Preferred |
| --- | --- | --- | --- |
| brief-11 | 1 | 3 | B, strength 2 |
| brief-12 | 4 | 3 | C, strength 3 |
| brief-13 | 3 | 2 | C, strength 1 |
| brief-14 | 1 | 3 | B, strength 2 |

The story read second won in all four pairs, and each method was second in exactly
two. So this result cannot separate method from reading order. It shows no consistent
advantage for the richer plans, and C used about four times B's tokens.

This is the one follow-up the plan allows. The owner chose it on 30 October 2026,
after seeing the first result, with these changes: "I think a fairer retest is a good
idea, but I would drop originality. Also, I don't mind spending a bit more to get a
better model involved." The owner also said "coherence is critical", and that
"comparing two terrible stories is much harder than comparing two reasonable stories."

| Setting | First G1 | Retest |
| --- | --- | --- |
| Briefs | dev brief-03 to brief-06 | held-out test brief-11 to brief-14, the first four by number |
| Model, both roles | `deepseek-flash` | `gpt-6.1-sol` at medium reasoning effort |
| Editing | B revised until budgets matched; C not revised | Both conditions get exactly 2 passes |
| Budgets | Matched within 10% | Not matched: C also pays for planning, and the report shows both |
| Decided by | Coherence and originality | Coherence only: at least 3 of 4 wins, ties count as non-wins |
| Shown order | Random | Random, restricted so each condition is first in 2 of 4 pairs |

The owner chose the model: "I've got a ton of credit on the OpenAI API, can we use
their models please. Maybe gpt-6.1-sol set to medium thinking." Sampling temperature is
not set, because this reasoning model does not take it.

The owner also explained the first result's one preference for C: both pharmacist
stories scored very low, and "comparing two terrible stories is much harder than
comparing two reasonable stories." That preference carries little weight.

The retest did not require originality, and the owner dropped it from later reviews.
The reader still offers it as an optional score. The G1 default and the unused G-D and
exemplar workflows still require it. Preference and strength were still recorded, and
did not decide the gate. The first result stays preserved and is not pooled with this
one. The retest did not pass, so the diffusion track stopped: no further follow-up is
allowed.

Because budgets were not matched, a pass could not have claimed that planning wins at
equal cost, only that the richer plans gave this reader more coherent stories at the
extra cost of planning.

The protocol is
[`benchmark/prereg/g1-retest-20261030.json`](../benchmark/prereg/g1-retest-20261030.json)
and the config is [`configs/g1-retest-20261030.yaml`](../configs/g1-retest-20261030.yaml).
A rehearsal of one pair on dev brief-01 cost US$0.41, and the run cost US$1.76 against
a US$3 cap. The reading batch was `g1-retest-20261030`, built by running
the commands below with these files. Its key records seed 3, which makes each
method come first twice. Those commands do not choose such a seed, and no code or
protocol field requires a balanced order, so the balance rests on that one recorded
choice. The agent did not see the order until the batch closed.

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

Overall preference and strength were recorded but did not decide the gate. The plan's
fail action for a G1 that did not pass was at most one follow-up on fresh briefs, or
stopping the diffusion track. Both have now happened.

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
supersedes that limit. The first G1 left the 30 frozen test briefs untouched; the
retest later used four of them. The owner read other stories from these four briefs
during G-instr on 19 and 20 October, so the premises may have felt familiar.

G1 binds its stories through `nde eval g1-run`. Run
`20261030T153611Z-dev-M-B-pilot-20261027-v1-c27ddd` replayed the same config entirely
from the response cache: 145 calls, no provider spend, and the same eight story
checksums as the original run.

## Reading

The agent guided the owner through one story and one question at a time, and entered
the answers. The owner kept no notes. For each pair of the first G1, the owner gave:

1. coherence for each story, from 1 to 5;
2. originality for each story, from 1 to 5;
3. which story they preferred overall;
4. how strongly, from 1 to 3.

The owner confirmed the 3-of-4 rule before reading ("Yes, 3 of 4"). The reading batch
is `g1-20261030-b`. An earlier unread batch, `g1-20261030`, was replaced before reading
because the agent saw that its shown order put the same condition first in every
pair. The replacement's order was not inspected.

Reading was planned at about 80 minutes, with breaks whenever the owner liked. The
owner does not work on the code, so no 48-hour wait applied. The report may still show
its standard timing caveats. Conditions stayed hidden until all four pairs were scored
and the batch closed, and the agent did not open `sealed/` before then.

## Commands

These are the commands the agent ran for the first G1, each on one line. They would
now refuse, because the batch and decision folders already exist:

```powershell
uv run nde eval g1-check benchmark/prereg/g1-20261030.json configs/dev-mb-pilot-20261027.yaml --selection docs/evidence/dev-tier-selection-2026-10-27/selection.json --schema-evidence .nde/g-schema-packet/evidence-indexed --out runs/g1-check
uv run nde eval g1-run benchmark/prereg/g1-20261030.json configs/dev-mb-pilot-20261027.yaml --selection docs/evidence/dev-tier-selection-2026-10-27/selection.json --schema-evidence .nde/g-schema-packet/evidence-indexed --runs-dir runs
uv run nde eval g1-batch benchmark/prereg/g1-20261030.json configs/dev-mb-pilot-20261027.yaml --selection docs/evidence/dev-tier-selection-2026-10-27/selection.json --schema-evidence .nde/g-schema-packet/evidence-indexed --name g1-20261030-b --from runs/RUN C --from runs/RUN B
uv run nde eval serve --batch g1-20261030-b
uv run nde g1-report batches/g1-20261030-b --out runs/g1-decision
```

Extract `.nde/g-schema-packet/` from
`docs/evidence/g-schema-2026-10-27/evidence.zip` first.
