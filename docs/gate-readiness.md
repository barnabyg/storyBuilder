# Strict gate readiness

`uv run nde eval readiness cohort.json --benchmark benchmark --from runs/RUN C
--from runs/RUN B --out readiness/G1` checks an explicitly registered cohort
before owner reading. Run the command on one line. It writes `readiness.json`
and `readiness.md` even when registration or run evidence is malformed.

Exit codes are **0** (ready), **1** (incomplete or invalid evidence) and **2**
(malformed registration or run files). Every expected pair remains in the
report. A failed generation never reduces the denominator.

The JSON registration has these fields:

```json
{
  "schema_version": "1",
  "id": "registered-experiment-id",
  "brief_checksums": {"brief-11": "COPY_64_HEX_SHA256_FROM_FROZEN_JSON"},
  "split_checksums": {"test": "COPY_REGISTERED_FROZEN_SPLIT_SHA256"},
  "conditions": ["C", "B"],
  "repeat": 1,
  "length_tolerance": 0.1,
  "budget_measure": "generation_tokens",
  "budget_tolerance": 0.1
}
```

Include the complete planned brief cohort, not just successful briefs. Repetition
identities use the run records' zero-based numbering. `brief_checksums` binds
each brief's LF-normalized bytes; `split_checksums` uses `nde.briefs.split_checksum`.
Readiness requires matching frozen splits and unchanged brief content. The two
`--from` conditions must follow registration order.

Admission checks actual final `story.md`, its selected raw output, word count,
target, frozen inputs and saved generation status. New runs also save the final
story checksum in `stories.jsonl`; older records remain readable and their raw
selected output supplies the consistency check. All attributable call tokens
count, including repairs, length regeneration and revisions. Cache hits keep
experimental tokens and add zero new provider spend. Matching uses the registered
fraction of the larger budget, inclusive at the boundary.

For strict batch creation add `--cohort cohort.json --benchmark benchmark` to
`nde eval batch`, or call `nde.eval.readiness.build_gate_batch` with `GateCohort`,
benchmark path, two `Source` objects and `GateBatchSettings`. Both recheck
admission immediately before creating the sealed batch and store its readiness
evidence under `sealed/readiness/`. Without `--cohort`, `nde eval batch` keeps its
ordinary intersection behaviour. Readiness alone does not validate a
scientific protocol, establish a passed prerequisite gate or certify blinding.
