# Pre-registration: gate G-instr (instrument check)

**Phase:** 1, P1.8 and the Phase 1 exit gate of
[the implementation plan](../../docs/IMPLEMENTATION_PLAN.md).
**Drafted:** 19 October 2026, copied from the plan's defaults.

## Status

Drafted for the owner to confirm. Adjust the thresholds or fail actions if
needed, then commit this file **before** reading the `p1-instrument` batch.
The commit that confirms it is the pre-registration.

- [ ] Owner confirmed on: (date of the confirming commit)

## Hypothesis

The evaluation instrument (blind reading by the owner, plus the LLM judge
panel) detects a known, deliberately caused structural degradation. Stories
from Baseline B (a beat outline, then scene-by-scene prose) are preferred over
**B-scrambled** stories, whose scenes follow the same outline in a shuffled
order (a seeded permutation that is never the original) with freshly written
prose, from the same planner, prose model and templates.

## Conditions

- **B:** `baseline_b` in `configs/p1-instrument.yaml`.
- **B-scrambled:** `baseline_b_scrambled` in the same run: B's outline for the
  same brief, scenes shuffled, prose regenerated.
- Both: `planner` `openai/gpt-6-luna`, `prose` `deepseek/deepseek-flash`, the
  shared prose contract (`scene_prose@1`), normalisation, and length control
  (within 10% of 2,000 words, up to 3 attempts).
- **Judge:** one LLM judge, OpenAI `gpt-6.1-sol` at medium reasoning effort
  (`configs/p1-judges.yaml`), in pairwise mode (both orders) and rubric mode.
  It shares the OpenAI family with the planner, so the plan's family rules are
  waived by the owner's choice: only OpenAI and DeepSeek APIs are available.
  Both conditions use the same planner outline and prose model, so
  self-preference cannot favour one condition over the other in this check.

## Primary metric

The owner's forced-choice preference per pair in the blind batch
`p1-instrument`: the number of pairs in which the intact B story is preferred.

Secondary, reported but not deciding the gate: the judge's Cohen's kappa
against the owner, position consistency, rubric means, failure
tags, and the owner's condition-guess accuracy.

## Sample size

n = 10 pairs: one B story and one B-scrambled story for each of the 10 dev
briefs (`benchmark/dev/`), repetition 1. All 10 pairs are read; none are
dropped. A story flagged out of the length range is still read, and the flag is
noted in the result.

## Pass thresholds

The gate passes only if all three hold:

1. The owner prefers intact B in **at least 9 of 10** pairs.
2. The judge (`gpt-6.1-sol`) agrees with the owner in **at least 80%** of the
   pairs (position-inconsistent and incomplete pairs count as disagreement).
3. The A and B pipelines ran end to end within their cost caps
   (`configs/stories-ab.yaml` and `configs/p1-instrument.yaml`).

For reference, 9 of 10 has a two-sided exact binomial p of about 0.021
against no preference.

## Procedure

1. Calibrate on the calibration batch (P1.9) and close it.
2. Confirm and commit this pre-registration.
3. Generate (`configs/p1-instrument.yaml`), build the batch, and run the judge
   panel (`configs/p1-judges.yaml`), as in `docs/instrument-check.md`.
4. Wait at least 48 hours after last working on the generating code, then read
   all 10 pairs blind in the reader and close the batch.
5. Run `uv run nde eval report p1-instrument` and record the result in the
   plan.

If the report flags the batch **possibly unblinded** (condition guesses right
in more than 70% of pairs), record that with the result.

## Fail actions

- **If the owner cannot reliably detect the damage** (fewer than 9 of 10): the
  stories are too short or the prose dominates. Increase the target length, or
  restrict reading to targeted excerpts, then rerun the check before going
  further.
- **If the judge fails where the owner succeeds** (below 80% agreement): drop
  the judge from screening and budget more owner reading time for later
  gates.
- **If a pipeline exceeded its cost cap:** fix the cost estimate or the cap and
  rerun the generation; the cache makes completed calls free.
