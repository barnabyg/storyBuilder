# Rubric and failure taxonomy

The rubric is what the blind reader and the LLM judges score stories against.
It is defined once, in `benchmark/rubric/v<n>.yaml`, and versioned. See P1.2
in [the implementation plan](IMPLEMENTATION_PLAN.md).

## What it contains

- **Scale:** every dimension is scored from `scale.min` to `scale.max`
  (1 to 5 in v1).
- **Dimensions:** engagement, coherence, emotional impact, character
  believability, pacing and originality. Each has a short definition and
  anchors for the low end, the middle and the high end of the scale. The
  points in between are interpolated.
- **Failure tags:** a fixed vocabulary of named failures (unearned change,
  dropped thread, continuity error, melodrama, generic, rushed ending,
  repetition), each with a one-line definition and an example. The reader also
  accepts free-text tags; this list is the vocabulary the judges use and that
  reports count.

The file is validated by `nde.eval.rubric.Rubric`: IDs must be unique, every
dimension needs all three anchors, and the file's `version` must match its
file name. `load_rubric(directory, version)` returns the validated rubric.

## Versioning

A released rubric version's text is fixed, exactly like a prompt template:
`benchmark/rubric/checksums.json` records its SHA-256, and loading a version
whose text no longer matches is rejected. To change the wording:

1. Copy `benchmark/rubric/v1.yaml` to `v2.yaml`, set `version: 2` and edit it.
2. Record it:

   ```powershell
   uv run nde rubric lock
   ```

3. Commit the new file and the updated checksums together.

`nde rubric lock` never re-records a changed version; it reports it and exits
2. `uv run nde rubric lock --check` reports unrecorded or changed versions
without writing anything, and the test suite (so `nde-verify`) checks the
committed rubric the same way.

Every batch and judge run records the rubric version it used, so old results
stay interpretable after the wording changes.

## Status

Rubric v1 was drafted by an agent and has not changed since. G-instr used it. No
separate owner sign-off of the wording is recorded; confirm or version it before
a new gate registers it.
