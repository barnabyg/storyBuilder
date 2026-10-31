# Extraction pilot

**Status:** G1 and its one retest did not pass, so [the diffusion track has stopped](g1-four-pairs.md). A real pilot needs a passed G1, so the code refuses it. The tooling remains, tested on fixtures only.

The pilot reuses the independent-family engine; it does not introduce another
annotation format. Configure a frozen cohort and each expected work:

```json
{
  "cohort": "data/cohort/cohort.json",
  "extraction_config": "configs/pilot-families.json",
  "works": [{
    "work_id": "story-id",
    "work_dir": "data/story-id",
    "segmentation": "data/story-segments/segmentation.json"
  }],
  "g1_batch": "batches/closed-g1",
  "g1_decision": "data/g1-result/decision.json",
  "per_run_usd": 10,
  "per_day_usd": 20,
  "max_stories": 50
}
```

Run `nde corpus-pilot run pilot.json --out data/pilot-v1`. Use absolute paths
for portable invocation from different working directories. The family config
uses [the extraction settings](extraction.md) with `annotators` replacing
`annotator`: at least two distinct declared families, explicit prices, bounded
repairs, an optional per-story `budget.per_run_usd` (without it no story cap
applies; the pilot `per_run_usd` still caps the run), and a positive
`budget.per_day_usd` no greater than the pilot daily cap. Every family shares any
story cap; every story shares the durable pilot cap and daily ledger.

Before calls, `pilot.json` freezes source/cohort/configuration identities,
schema, prompt and planned pass inventory. Features are computed locally.
The engine saves individual family/pass responses and cards. Repeating the
same command resumes this version: it preserves completed work, attempts,
ledger totals and caps. An interrupted request permits cache-only recovery;
missing cache evidence leaves recovery required. Changed inputs require a
new output directory. Exhausted, provider-failed and budget-aborted work
does not receive fresh retry limits on resume.

`nde corpus-pilot report data/pilot-v1` reports every planned
story/family/pass, including failed or not-yet-attempted rows. Experimental
tokens and list-price costs include cache hits; actual new spend is separate.
Durable spend/reservations not attributed to checkpointed responses remain
explicit unresolved accounting. Scaling estimates divide all attempted costs
by completed stories, expose incomplete coverage, and assume unchanged story
length, models, passes and retry distribution. They are not automatic scaling
authorisation.

Real runs replay the recorded G1 evidence before extraction and refuse unless it
passed. G-D is not a
prerequisite. Deterministic development fixtures set `test_only: true`, omit
G1 evidence and use local fixture-provider endpoints. Their output cannot
establish scientific success. Normal pilot runs are bounded to 50 works;
larger execution is a separate G2-controlled command.
