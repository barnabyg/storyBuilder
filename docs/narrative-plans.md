# Narrative plan inspection

Narrative schema 1 is separate from `StoryOutline`, briefs and run records.
No existing format is migrated or reinterpreted. JSON input must explicitly
name `schema_version: "1"`; unsupported versions fail with exit 2. The migration
entry point currently dispatches only version 1 to itself, because no older
narrative plan format has been released.

```powershell
uv run nde plan validate examples/harbour.plan.json
uv run nde plan inspect examples/harbour.plan.json --out .nde/harbour
uv run nde plan schema --out .nde/narrative.schema.json
```

Inspection writes `plan.json`, `validation.json`, `cards.md` and `cards.html`.
Exit 0 means mechanically valid, 1 means hard validation findings, and 2 means
unreadable, malformed or unsupported input. Invalid but parseable plans still
produce cards and diagnostics for correction. Advisory and human-assessment
findings do not imply mechanical failure or literary approval.

IDs are stable strings and globally unique across story, characters, sources,
scenes and beats. Scene and beat list position defines order; IDs never imply
chronology. Beats and character presence are optional. Character references
must resolve; duplicate presence is invalid.

Passages use evidence spans, never copied text masquerading as offsets. A span
names a source ID and its exact SHA-256, with half-open `[start, end)` Unicode
codepoint offsets into the embedded LF-only UTF-8 source text. The text checksum,
version binding and bounds are validated. Raw acquisition provenance can be
recorded independently in `provenance`. JSON Schema export is deterministic;
cross-record invariants additionally require the plan validator.

The example is deterministic and requires no provider, keys or downloads.
Existing outline and experiment commands keep their original contracts.
