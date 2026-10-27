# Brief-conditioned denoising and guidance

The conditional checkpoint representation is `nde-conditional-denoising-v1`.
This uses the registered native attention architecture with a fully masked,
fixed-capacity output block. Layout/count/identity/reference and structural-value
tokens are learned jointly; no observed target layout is passed to inference.
The entire allowed `Conditions` object is encoded in a fixed-width context
block, alongside registered capacity. Provenance, evaluator labels, source text,
target plans and target lengths remain outside that block. Oversized context or
output fails without truncation.

Training uses classifier-free conditioning: a Bernoulli draw drops the entire
literary-condition block with `condition_drop_probability` strictly between 0 and 1 (default
0.5). A dropped block uses an explicit null condition. Capacity remains visible
in both modes. Every selected mode is saved in the checkpoint history. Both
conditional and unconditional examples update the same attention backbone and
denoising cross-entropy objective. The endpoints are rejected because a wholly
conditioned or wholly unconditioned run cannot train both guidance modes. Small
finite runs still record exactly which modes actually occurred.

Approved hard channels have loss weight 1 and soft channels retain Phase 6's
weight 0.25. Uncertain asserted literary values, alternatives and confidence have
zero value loss; explicit unknown status, null absence and JSON syntax remain
learnable. New count and
identity targets inherit the approved collection policy. Plot-excluded context
scaffolding cannot authorize layout learning.

Sampling uses `unconditional + scale * (conditional - unconditional)` logits.
`--guidance-scale` is finite and in [0, 8]. Scale 0 uses only the unconditional
pass, scale 1 only the conditional pass, and other scales use both. Every actual
forward pass is charged to token, attention-cell and wall-time allowances.
`sampling_temperature: 0` selects argmax deterministically; a value in (0, 2]
enables seeded categorical sampling. Missing EOS, invalid JSON, invalid structure,
and caller scene-range violations remain rejected candidates. Guidance cannot
change edit authority or override explicit locks in the generation workflow.

Configuration adds `max_condition_tokens` (default 1024),
`condition_drop_probability`, and `sampling_temperature` to the layout-training
configuration. The base model must have room for the full registered condition
and output blocks. For example, a capacity of 4096 output tokens needs at least
5120 positions with the default condition block. A `sample` command accepts
`--conditions CONDITIONS.json` and `--guidance-scale SCALE`; unknown conditions
default to the documented null/empty values.

The checkpoint binds the condition schema, representation, guidance policy,
training configuration, approved dataset and local base initialization.
Initialization is an explicit native full fine-tune of identified local weights;
renaming an unconditioned reconstruction checkpoint cannot make it conditional.
Unsupported checkpoint kinds or representation identities fail clearly.
Checkpoints retain model, optimizer, RNG, sampler position, selected conditioning
modes, accumulated work and interrupted-step reservations. Resume revalidates all
transitive inputs and continues the original allowances. Inspecting/sampling a
saved checkpoint is self-contained: persisted base dimensions and weights suffice
after training files are removed; provenance paths are not inference inputs.

Tiny CPU fixtures establish actual parameter updates, conditioning mechanics and
recovery behavior. They are explicitly test-only and do not establish literary
condition adherence, owner approval or scientific advancement. Real training
requires a current genuine passed G3 protocol.
