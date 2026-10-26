# Reproducible G-schema evidence

This closes the reporting implementation, not the owner's G-schema review.
The bundled validator controls are synthetic. They provide mechanical evidence;
they never establish literary accuracy or owner approval.

```powershell
uv run nde schema-evidence --manifest data/manifest.calibration.yaml --annotations .nde/annotations --out .nde/g-schema
uv run nde schema-evidence --replay .nde/g-schema/inputs.json --out .nde/g-schema-replay
```

The first command reads local saved inputs only. It never downloads a source,
calls a provider, starts a timer or creates an owner review. Annotation sessions
are read from `<annotations>/<story-id>/session.json` using the story IDs in the
provenance manifest. Use `nde annotate draft --out .nde/annotations/<story-id>`
when preparing each story. The existing manifest inventories three actual
public-domain works: *An Occurrence at Owl Creek Bridge*, *The Gift of the Magi*
and *The Yellow Wallpaper*. The report measures their extracted word counts
when local sources exist; it does not claim all three satisfy the roughly
2,000–5,000-word selection target. Missing sources and out-of-range lengths are
shown as pending, so replace unsuitable selections before the owner gate.
The source licence basis is recorded provenance, not a new legal determination.

Outputs are `evidence.md`, `evidence.html`, machine-readable `evidence.json`,
`mutation-results.json`, and a portable `inputs.json` snapshot. The snapshot
contains the source bytes, manifest records, schema, hydrated controls and
mutations, sessions, original model drafts, submitted corrections and timer logs.
It carries source/text, schema, prompt, configuration, correction and code/run
identifiers. The replay command recomputes from this saved snapshot without
relying on the original directories. The report ID is a checksum of the full
snapshot. Replaying identical inputs with the same code/schema produces the
same packet bytes. Use the recorded code revision if the schema has changed.
Snapshots are local research evidence and include source texts and review notes.

## Automated evidence and gate status

The package bundles **44 distinct semantic violation categories** in
`src/nde/evidence/mutations.json`, with four valid synthetic controls. Categories
cover global identity/presence, causal references/self-links/order/cycles,
setup/payoff references/order, fact holders/acquisition/presence, character and
relationship transitions, locked values/ending facts, interpretive references,
ordinal trajectory order, motif endpoints, and exact source/span versions/bounds.
The registry is the documented, machine-readable category list: each entry names
its invariant, injection, intended diagnostic and severity. All categories and
actual diagnostics appear in each report. Additional schema-level tests, such
as ordinal numeric bounds, remain in their focused channel test files.

Each mutation is independently applied to its saved valid control. A catch
requires its intended diagnostic and severity. Unrelated parsing rejection
counts as a miss. Mechanical coverage requires at least 20 distinct categories,
all eight named families, all controls valid, and at least 95% caught. Use
`--mutations <registry.json>` to inspect a different registry; referenced control
paths are relative to that registry. Saved snapshots contain the control data.

Mechanical evidence also checks raw source identity, extracted span basis,
annotation provenance/config/schema, preserved original draft, numbered
correction checksums, current correction validity and timer consistency. Owner
rejection and excessive measured correction time are reported as owner-gate
failures separately from mechanical validator results.

Exit codes are **0 passed**, **1 failed**, **2 malformed/unreadable command
input**, and **3 pending**. Passing fixture tests, validators or generated drafts
cannot pass G-schema. Missing annotation, current-revision owner review,
literary notes, completed correction intervals, or trajectory comparisons keeps
it pending. In-flight/recovery-required generations cannot pass. Known invalid
sources/corrections, missed mechanical thresholds, explicit owner rejection,
non-discriminating trajectories, or correction time at/above 7,200 seconds fail
the corresponding criterion. The report retains every unresolved diagnostic.

Trajectory comparisons use the coarse shape and ordinal values, ignoring IDs
so renamed scenes cannot create spurious distinctions. Unknown values remain
unknown. At least one pair must differ mechanically; the owner must separately
confirm whether the differences are meaningful. Automatic difference is not
literary confirmation.

## Small remaining owner procedure

1. Select three to five provenance-checked stories in range, draft tier L and
   inspect the existing scene cards. Preserve the original draft.
2. Start/resume the annotation timer only during active correction; pause for
   idle time and automated generation. Recover an interrupted interval explicitly.
   Submit separately versioned corrections with `nde annotate revise`.
3. Resolve hard diagnostics. Resolve or explicitly remove human-assessment
   constraints after assessing them, preserving the literary judgement in review
   notes. A generic accepted review does not automatically clear unresolved
   human constraints or missing/ambiguous locked facts.
4. Pause the timer and explicitly review the **latest** correction using
   `nde annotate review ... --owner ... --notes ... --accept` and, only after
   comparison, `--meaningful-trajectories`. Earlier-revision reviews do not
   establish approval of a newer correction. Regenerate the report.

The target is under about two hours of measured correction per story. Fail
action: **cut or merge slow or non-discriminating channels, then repeat on
another story**. No routine manual regression testing is required to accept
the reporting implementation. Tests use isolated synthetic sources and explicitly
labelled test-only reviews; none are evidence that the actual owner reviewed
the real inventory.
