# Baseline C story generation

Register a condition as `pipeline: baseline_c` in an ordinary story experiment.
It uses the same `prose` role, `scene_prose@1` prompt, normalization, assembly,
length retries and budget revisions as B. Its difference is the validated
narrative plan and the current scene's planned-state notes.

```yaml
stories:
  briefs_dir: ../benchmark
  briefs: {ids: [brief-01]}
  conditions:
    - name: C
      pipeline: baseline_c
      params: {tier: L}
```

Use the provider, planner/prose role, prices, seed and hard caps from your
experiment config. `uv run nde run CONFIG.yaml` generates a whole-story
skeleton and scene channels before writing any prose. Tier `S`, `M` or `L`
projects that full plan before creating scene notes and replaying the ledger;
all original planning costs remain attributable to the condition.

To write from an already generated plan, select its one matching brief and
set both `params.saved_plan` and `params.provenance` to the saved `plan.json`
and `planning-provenance.json` files. Use absolute paths (relative paths resolve
from the command's working directory). Keep the original seed, planner binding,
prompts, repair limit and planning parameters. Imports check the brief,
schema, prompt/configuration identities, source checksum and complete planning
call sequence before writing prose. A hand-written plan without attributable
planning provenance cannot silently enter a budget comparison.

Imported planning calls retain their experimental tokens and list-price costs,
with zero new spend and `imported_from` identifying the source plan checksum.
The original call records remain in saved provenance. The current run's complete
configuration is retained in `config.json`.

Each story folder includes the full `source-plan.json`, selected/projected
`plan.json`, inspection cards and diagnostics, `plan-selection.json`, and the
planned-only `ledger.json`. Each length attempt retains raw scene outputs and
`scene-NN.inputs.json` with the stable scene ID, selected plan/ledger checksums,
relevant scene intentions and correctly timed before-state summary. Replay is
rebuilt from the initial plan on every attempt. It is not evidence that the
prose realised those intentions.

Provider or budget failures retain a failed story record and completed partial
artifacts. Length exhaustion retains the closest attempt with `in_range: false`,
which strict gate admission must reject. No live calls or owner assessments are
needed to verify the implementation: `uv run pytest tests/test_baseline_c.py
 tests/test_baseline_b.py tests/test_instrument.py -q` exercises the public
fake-provider workflow. Inspect `attempts/1/scene-01.inputs.json` in its synthetic
runs: the first scene must exclude later acquisitions; later summaries must
include completed transitions; a second attempt's first-scene input must reset.
Real literary quality and scientific gate readiness require separate evidence.
