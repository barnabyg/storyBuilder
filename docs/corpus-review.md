# Corpus channel review

`nde corpus-review freeze review.json --out data/review-v1` freezes the population,
criteria and deterministic sample before any owner labels. Paths in configurations
are resolved from the working directory; absolute paths make saved experiments
unambiguous. A minimal registration is:

```json
{
  "cohort": "data/cohort-v1/cohort.json",
  "inputs": [{
    "work_id": "story-id",
    "work_dir": "data/story-id",
    "segmentation": "data/story-segments/segmentation.json",
    "features": "data/story-features/features.json",
    "families": "data/story-families",
    "gold_session": "data/owner-corrected-session"
  }],
  "per_stratum": 1,
  "min_owner_per_channel": 3,
  "min_gold_per_channel": 1,
  "reliable_agreement": 0.8,
  "reliable_owner": 0.8,
  "reliable_gold": 0.8,
  "weak_owner": 0.5
}
```

The default channels are plot, causality, information, goal, belief,
relationships and interpretation. Sampling strata combine channel, period,
genre, split, disagreement and explicit uncertainty. Each stratum contributes
up to `per_stratum` works, ordered by the registered seed and content identity.
`max_samples` rejects an oversized packet rather than silently dropping strata.
`packet.json` includes source-linked claims, annotation paths, deterministic
features, inclusion counts and current/missing/stale gold evidence. Exact
normalized claims use the independent-family alignment rules and multiset
union denominators; agreement is reproducibility evidence, not literary truth.

Record each actual judgement in a JSON file:

```json
{
  "sample_id": "id from packet.json samples",
  "owner": "owner name",
  "judgement": "defensible",
  "notes": "Source-linked assessment and relevant uncertainty."
}
```

Run `nde corpus-review label data/review-v1 judgement.json`, then
`nde corpus-review report data/review-v1`. Labels may be accurate, defensible,
not_defensible or unknown. Later labels append history; the latest judgement
for a sample is used. Unknown is reported separately and cannot satisfy the
required known-owner denominator. Reports show all reviewed and missing counts.

Reliable channels become hard targets; weak channels are soft-only;
unmeasurable and pending channels are excluded. Core failures stay explicit.
Current approved gold revisions and sufficient known owner labels are required
for real reliability. Changed source, segmentation, features, annotations,
configuration or gold revisions make the packet stale and classifications
pending without deleting labels. Freeze a new packet to review changed inputs.

Synthetic fixtures explicitly set `test_only: true` in registration and labels.
Their classifications cannot establish scientific reliability or pass G2.
