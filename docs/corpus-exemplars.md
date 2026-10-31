# Human-plan exemplars in development

**Status:** G1 and its one retest did not pass, so [the diffusion track has stopped](g1-four-pairs.md). Real training exports need a passed G2, which needs a passed G1, so only test-only exports can reach this workflow. The tooling remains, tested on fixtures only.

`nde exemplars` compares an unchanged baseline with an exemplar-conditioned
version of the same C or D pipeline. It accepts development briefs only. This
optional workflow does not run or select a held-out gate, and does not block G2.

Use a versioned training export from the corpus export workflow. Every use
revalidates its frozen cohort, protected authors, source and annotation identities,
schema, classification and target policy. A selected work must be eligible for
training. Validation/test/gold works and their authors cannot become exemplars.

The experiment config must contain exactly these two conditions, with identical
parameters and pipelines (both `baseline_c`, or both `baseline_d`):

```yaml
stories:
  briefs_dir: ../benchmark
  briefs: {ids: [brief-01]}
  conditions:
    - {name: baseline, pipeline: baseline_c, params: {tier: L}}
    - {name: exemplars, pipeline: baseline_c, params: {tier: L}}
  equal_budget:
    conditions: [baseline, exemplars]
    tolerance: 0.10
    max_revision_passes: 4
```

Retain the ordinary provider, role, price and spending-cap settings. Both arms
share prose settings, briefs, repetition seeds and all other planning settings.
The command adds the selected export checksum and work identities only to the
`exemplars` arm:

```powershell
uv run nde exemplars run configs/exemplar-dev.yaml --export data/training-export --works work-one work-two --out runs/exemplar-dev
uv run nde exemplars report runs/exemplar-dev
```

`--family <declared-family>` selects that family's annotation for each work.
Without it, the first family in sorted order is used, and its exact identity is
recorded. Separate family annotations are never combined into a consensus plan.

The configured exemplar export can also be specified directly in C/D parameters
as `exemplars: {dataset: /absolute/path/dataset.json, dataset_sha256: ..., work_ids:
[work-one]}`. Imported saved plans cannot acquire new conditioning. Omitting
`exemplars`, or setting it to null, preserves the existing prompt behavior.

Only approved target fields enter the structural prompts. Hard targets and
uncertain, soft-only targets are labeled separately. Excluded channels and whole
source texts are omitted. Structural IDs and source span bases remain available
for interpreting references. A reduced example is explicitly a set of targets,
never a purported complete tier-L plan. Source evidence must not be copied into
a generated plan. C's skeleton/channel requests and D's intentions, skeleton,
channel, critic and structural revision requests retain the exact supplied
examples in `corpus-exemplars.json` and their request artifacts. Prose uses the
existing shared contract.

All calls pass through the ordinary gateway and accounting pipeline. Extra
exemplar input tokens, repairs, critics and revisions count toward the comparison;
cache hits preserve experimental tokens, while new provider spend is separate.
Unmatched budgets and failed/incomplete stories remain visible. The reports show
deterministic plan validity, selected final-prose identities, costs, and the usual
source-bound realisation review packets. Re-reporting checks source/configuration,
export and run identities before accepting evidence.

`nde exemplars review <directory> <review.json>` records owner coherence and
originality scores linked to the comparison checksum and both selected final
prose checksums. Reviews are immutable history. Synthetic exports and their
reviews remain explicitly test-only; local fake providers are required by the
comparison command. Without actual owner evidence, literary improvement remains
pending. LLM literary estimates are diagnostic only, and owner development
scores do not themselves establish a scientific gate result.
