# Planned scene state ledger

```powershell
uv run nde plan replay examples/harbour-state.plan.json --out .nde/harbour-ledger
```

The command writes `ledger.json`, `ledger.md` and `validation.json`. Exit 0
means the represented plan can be replayed, 1 means semantic validation failed
(diagnostics are saved, with no new ledger), and 2 means an unreadable or
unsupported input. Inspect `validation.json` even after success: advisory and
human-assessment items remain unresolved by replay.

Every ledger is explicitly **planned-only**. It records the source plan's ID,
tier and SHA-256 of its canonical represented JSON, and ordered before/after
snapshots keyed by stable scene IDs. This does not establish what the prose
realised. Replay leaves the source plan unchanged and produces identical output
for identical represented input.

Chronology follows the scene list. Initial knowledge includes only explicit
initial holders. Reveals add each holder after their scene, so a fact revealed and used in the same
scene is missing from that scene's `planned_state_before`, although the
[validator](narrative-structure.md) accepts it as held from the scene's start.
Setups open after
their scene and close after their payoff scene, and motifs accumulate only at
completed occurrences. Knowledge uses are attached to their scene separately.
State and relationship keys enter the ledger from their first explicit
`before` reading when replay reaches that scene; later transitions never supply
earlier state. Unknown and ambiguous readings retain status, confidence and
alternatives. A multi-scene motif's record accumulates the IDs of its completed scenes without
exposing later ones.

The compact summary uses a deterministic relevance policy, rather than a
character cutoff that might discard hard facts. It includes every recorded
constraint as a **requirement**, all current canon/value readings, goals/beliefs
for characters present in the current scene, relationships involving a present
character, knowledge held by at least one present character, every open setup,
and completed motif occurrences. A required ending is a target, not an assertion
that it already occurred. No future reveal or acquisition appears in the
before summary. No source text is copied into the summary. S/M omissions stay
absent: replay does not reconstruct channels removed by projection.

The Python API is `nde.narrative.ledger.replay(plan)`, returning a `Ledger` whose
ordered `scenes` hold `before`, `after`, `summary` (before-state prose context),
`after_summary` and `knowledge_uses`. Invalid plans raise `PlanError` containing
named validation diagnostics. `render_ledger(plan, out)` writes the same public
artifacts as the command.

Manual check: run the command above, open `.nde/harbour-ledger/ledger.md`, and
confirm `s1` begins with “protect the lamp”, `s2` begins with “save the boat”,
and only the final after-state records “guide the boat home”. The relationship
should move from suspicious through cooperating to allies, while the final
belief remains ambiguous with duty/friendship alternatives. These are planned
fixture transitions, not owner assessments or generated-story evidence.
