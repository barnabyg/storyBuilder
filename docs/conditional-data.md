# Conditional training data

**Status:** G1 and its one retest did not pass, so [the diffusion track has stopped](g1-four-pairs.md). Real training exports need a passed G2, which needs a passed G1, so only test-only exports can reach this workflow. The tooling remains, tested on fixtures only.

`nde conditional-data prepare config.json --out prepared` creates a separate
version-1 `conditional-dataset`; `nde conditional-data inspect prepared` checks
its complete source lineage. Phase 6 codecs, reconstruction tasks and datasets
keep their existing identities and corruption semantics.

Preparation accepts exactly one approved `export` or existing explicit
reconstruction `evaluation` authorization. Exports retain their approved channel
policy. Evaluation examples and their conditions remain evaluator-only; optimizer
loaders reject them. Families of one work share an author/work `group_id`, and
duplicate work/family pairs are rejected. No source export or annotation is edited.

Conditions have these meanings:

| Field | Meaning |
| --- | --- |
| `genre` | A supplied or derived genre label, not a symbolic constraint. |
| `themes` | Short literary theme labels, not annotated theme records. |
| `ending_intent` | Natural-language intended ending; never converted to a fact, lock or canon. |
| `target_words` | Inclusive requested word-count range, not measured source length. |
| `scene_range` | Inclusive caller-supplied allowed scene-count range, not a target layout. |

Missing values stay unknown. Each provided label carries `origin` (`metadata` or
`derived`), `status` (`known`, `uncertain` or `unknown`) and numeric `uncertainty`
between zero and one. Unknown labels carry null values. Derived labels require a
`method`; model-based derivation also identifies both `model` and `prompt`.
Optional `input_files` bind derivation source/configuration bytes. A condition
record is a training label with recorded provenance, never owner-validated truth.
Range conditions require supplied metadata; deriving an exact scene count or
source length would expose hidden target scaffolding and is rejected.

The preparation JSON has a `conditions` list. Each entry identifies `work_id`,
optionally `family`, and a `labels` object. Omitting `family` applies the same
conditions to every family of that work. For example:

```json
{
  "export": "approved/dataset.json",
  "conditions": [{
    "work_id": "lamp",
    "labels": {
      "genre": {"value": "fantasy", "origin": "metadata", "status": "known", "uncertainty": 0},
      "ending_intent": {"value": "The light returns, at a cost.", "origin": "derived", "status": "uncertain", "uncertainty": 0.4, "method": "annotator reading"}
    }
  }]
}
```

Inference consumes only the strict `Conditions` value object. Target codec views,
layout counts, original identities, source prose, derivation files and evaluator
metadata live outside that boundary. Arbitrary plan objects or unregistered
condition fields are rejected. Literary labels cannot establish their own truth
or grant edit authority.
Literary strings cannot contain JSON object/array delimiters (`{}`, `[]`),
including embedded serialized targets disguised by a natural-language prefix.

The dataset binds preparation bytes, derivation inputs, approved export, frozen
cohort, schema and codec policy. Inspection reconstructs the expected examples;
resealing modified examples does not authorize them. Changed sources or settings
require a new output directory; completed datasets are immutable.
