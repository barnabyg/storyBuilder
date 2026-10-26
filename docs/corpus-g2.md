# G2 readiness and training exports

Complete a frozen [pilot](corpus-pilot.md), then freeze its
[review packet](corpus-review.md). Register G2 **before owner labels**:

```json
{
  "pilot": "data/pilot-v1",
  "review": "data/review-v1",
  "min_stories": 50,
  "min_owner_per_channel": 3,
  "min_coverage": 1,
  "owner_threshold": 0.8,
  "scaling_size": 300,
  "scaling_budget_usd": 100
}
```

`nde corpus-g2 register g2-config.json --out data/g2-protocol.json` freezes
criteria and population identities. The review population must cover the
pilot population and every core channel. `owner_threshold` cannot be below
0.8. Paths are interpreted from the working directory; absolute paths are
recommended for recorded runs.

Collect actual source-linked owner labels, then run
`nde corpus-g2 report data/g2-protocol.json --out data/g2-v1`.
The immutable `decision.json` and readable report bind the protocol, pilot,
classification results, reviews, gold revisions and costs. A report never
starts extraction. Missing/stale evidence, unknown classifications,
inadequate coverage or unresolved costs remain pending. Core plot, causality,
information, goal and belief channels must be reliable independently.
Core failures retain an explicit stop-before-training action. A synthetic
criteria pass still has scientific outcome pending.

`nde corpus-export create data/g2-v1 --out data/training-v1` creates an
immutable versioned training view only from an approved genuine G2 result.
For deterministic tests only, `--test-only` permits a synthetic criteria-pass
record while keeping every artifact marked synthetic. Inspect with
`nde corpus-export inspect data/training-v1`.

Exports retain separate original annotations per model family and exact
source/schema/classification identities. Only eligible training works and
authors are included; protected gold/validation/test material stays excluded.
The channel target policy marks reliable channels hard, weak channels soft,
and unmeasurable or pending channels excluded. Original complete annotations
remain readable; the filtered training view does not claim to be a complete
tier-L narrative plan. Span bases and references stay attached to originals.
Consumers must apply the target policy, as the exemplar workflow does.

Larger extraction is separately invoked with
`nde corpus-g2 scale expansion.json --gate data/g2-v1 --out data/expansion-v1`.
It replays a genuine passed G2, requires unchanged model/extraction settings,
preserves earlier frozen assignments, checks registered size and spending
caps, and uses the same pilot orchestration. Protected authors cannot enter
training through expansion. Pending, stale, failed or synthetic G2 evidence
cannot authorise real scaling. No Phase 6 training is implemented here.
