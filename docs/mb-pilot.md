# M/B development pilot and the G1 decision

**Status (30 October 2026):** generated once and admitted. Run
`20261030T150215Z-dev-M-B-pilot-20261027-v1-78ddef` cost US$0.48 at commit
`f578753`, and all four pairs passed strict readiness. The blind batch
`dev-mb-pilot-20261027` is built and sealed, and reading has not started. Formal
G1 has no protocol and no result.

The owner declined the 30-pair G1 read on 27 October 2026 ("No, that's not going
to happen. I can spare a few hours at most."). The registered replacement compares
the selected M tier with Baseline B on four development briefs. It is preliminary
development evidence from one reader. **It cannot pass G1** and does not unlock D,
corpus, reconstruction or E work.

The approved records are immutable, because the approval binds their exact bytes:

| Record | Path |
| --- | --- |
| Plain account | [`benchmark/prereg/dev-mb-pilot-20261027.txt`](../benchmark/prereg/dev-mb-pilot-20261027.txt) |
| Protocol | [`benchmark/prereg/dev-mb-pilot-20261027.json`](../benchmark/prereg/dev-mb-pilot-20261027.json) |
| Owner approval | [`benchmark/prereg/dev-mb-pilot-20261027-approval.yaml`](../benchmark/prereg/dev-mb-pilot-20261027-approval.yaml) |
| Cohort | [`benchmark/prereg/dev-mb-pilot-20261027-cohort.json`](../benchmark/prereg/dev-mb-pilot-20261027-cohort.json) |
| Config | [`configs/dev-mb-pilot-20261027.yaml`](../configs/dev-mb-pilot-20261027.yaml) |
| Briefs | `benchmark/pilots/mb-20261027/dev/`, brief-03 to brief-06 |

The config's first line still says "DRAFT". That comment predates approval and
stays because editing it would break the approved checksum.

## Fixed terms

- Four pairs, one repetition each, with no dropping, top-up, new seeds or
  quality-based retries.
- One generation invocation under the US$1.50 total, per-run and $3 per-day caps.
- Every pair must pass strict readiness before any reading. Otherwise the pilot
  stops as incomplete, with failed attempts retained.
- Coherence and originality for both stories, then overall preference and strength.
- At most two hours of owner reading and answers, excluding breaks and agent work.
  Stop at the cap. An unfinished batch stays open and sealed, and is recorded as
  incomplete.
- Reading starts at least 48 hours after the owner's last generating-code work.
  The owner confirmed on 30 October 2026 that they do not work on the code, so no
  wait applies. If `eval serve` warns about recent agent commits, the report records
  that as a caveat.
- Conditions stay hidden until all scores are complete and the batch is closed.

## Commands

Run each command on one line, from the repository root:

```powershell
uv run nde run configs/dev-mb-pilot-20261027.yaml
$run = (Get-ChildItem runs -Directory -Filter '*-dev-M-B-pilot-20261027-v1-*' | Sort-Object Name | Select-Object -Last 1).FullName
uv run nde eval readiness benchmark/prereg/dev-mb-pilot-20261027-cohort.json --benchmark benchmark/pilots/mb-20261027 --from $run C --from $run B --out runs/dev-mb-pilot-readiness
uv run nde eval batch dev-mb-pilot-20261027 --from $run C --from $run B --cohort benchmark/prereg/dev-mb-pilot-20261027-cohort.json --benchmark benchmark/pilots/mb-20261027 --optional-scores --primary-score coherence --require-score coherence --require-score originality
uv run nde eval serve --batch dev-mb-pilot-20261027
uv run nde eval report dev-mb-pilot-20261027
```

Run the first command only once. `eval serve` warns if the generating code changed
less than 48 hours ago, and the report records that warning. Under the
[guided owner review](../AGENTS.md#guided-owner-review), the agent asks one question
at a time and never opens `sealed/` while reading is in progress.

The report lists coherence wins, ties and losses, each pair's scores and the mean
paired originality difference. It also prints binomial p-values, which mean little
at four pairs: no result at this size reaches p < 0.05. The owner's
no-average-originality-loss preference applies descriptively.

## The decision G1 still needs

The plan's G1 needs 30 pairs, about ten hours of reading. The owner has declined
that, so G1 cannot complete as written. Only the owner can choose what replaces it:

| Option | Owner reading | Consequence |
| --- | --- | --- |
| Stop the diffusion track | None further | Ship the schema, benchmark, harness and planner as the result (proposal §14). |
| Register a smaller formal G1 | 3–4 hours | Needs a new committed protocol, a plan amendment and a change to the 30-brief check in `src/nde/eval/g1.py`. Later gates inherit a weaker test. |
| Read the original 30 pairs after all | About 10 hours | No criteria change. `G1Protocol` already supports it. |

The pilot's four pairs cannot be pooled into any of these, because they use dev
briefs. A smaller G1 would use a pre-registered fixed subset of the untouched test
briefs. These exact binomial figures show what a smaller size gives up, with ties
ignored (real ties lower every pass chance):

| Pairs | Wins to pass | Two-sided p | Hours | Pass chance if C truly wins 60% / 70% / 85% of pairs |
| --- | --- | --- | --- | --- |
| 9 | 8 | 0.039 | 3 | 7% / 20% / 60% |
| 12 | 10 | 0.039 | 4 | 8% / 25% / 74% |
| 30 | 21 | 0.043 | 10 | 18% / 59% / 99% |

A smaller gate is only likely to detect a large effect. Failing one would say little
about a moderate effect, and the plan's fail action would still stop the track.

Record the chosen option and its rule before reading the pilot, so that its result
cannot shape the rule afterwards.
