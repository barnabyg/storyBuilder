# Synthetic augmentation is a separate ablation

`nde reconstruction-augmentation create CONFIG --out DIRECTORY` freezes a new
optimizer dataset without changing its parent. `inspect DIRECTORY` validates
the entire source chain and regenerates expected content before accepting it.
The configuration is JSON:

```json
{
  "dataset": "path/to/training/dataset.json",
  "recipe": "training-label-template-v1",
  "seed": 7,
  "synthetic_per_human": 1,
  "max_generated_examples": 10000,
  "max_generation_tokens": 1000000
}
```

Only approved optimizer training examples can supply inputs. An optional
`source_example_ids` list restricts generation further; any unknown/protected
example ID is rejected. The registered local generator transforms eligible
retained textual labels with an explicit synthetic variant tag. Its exact
instruction, permitted label values, generator/version, seed, parent IDs and
output IDs are recorded. It consumes no source prose, excluded channels, full
annotations, gold labels, evaluation labels or benchmark outputs. It makes no
provider calls. This inexpensive recipe exercises textual augmentation mechanics;
it does not claim new literary structure or independent human meaning.

Original examples remain byte-for-byte equivalent in the derived manifest.
Synthetic examples carry explicit `synthetic` and generation provenance, retain
their parent author/work group, graph references, uncertainty and channel target
policies, and never increase independent human story/author counts. Mixture
accounting records both exact counts and measured synthetic fraction. Zero
synthetic examples preserves the human-only population. Generation has explicit
example and byte-token resource limits; reports record actual prompt/output
byte tokens, string transformations, provider calls and provider cost.

Recipe selection defaults to pre-registration. Validation-based selection must
declare `selection_basis: "validation"` and a separate protected validation
dataset in `selection_validation`; training/test evidence cannot occupy that
role. Validation content is checked as provenance and is never passed to the
generator. Frozen input checksums reject later recipe/source changes.

Pass the human parent and augmented dataset separately as `train_datasets` in a
registered reconstruction sweep. Both objectives must use the same augmentation
artifact, author-group subsets, initialization, tokenizer and resource policy;
fixed held-out tasks are shared across the two ablations. Each retains its own
training/checkpoint/prediction/score identities and actual work counters.

`nde reconstruction-augmentation report CONFIG --out DIRECTORY` accepts
`augmentation`, `human_scores` and `augmented_scores` paths. It checks fixed
evaluation identities, both objective pairs, source group equality, model/resource
matching and checkpoint ancestry, then recomputes paired scores from their bound
predictions. The report compares human-only and augmented scores, records all four
checkpoints and training/inference work, and charges generation separately. It
cannot authorize G3 or Phase 7; software fixtures and synthetic agreement leave
scientific improvement unestablished.
