# Character state, relationships and constraints

```powershell
uv run nde plan inspect examples/harbour-state.plan.json --out .nde/state-cards
uv run nde plan validate .nde/state-cards/plan.json
```

The example uses two characters in two scenes. It retains changing goals and
beliefs, an unchanged locked value, a relationship transition and an explicit
ending fact. Cards show IDs, scenes, before/after readings, alternatives,
confidence, mechanical constraint status and pending literary assessment.

`states` contains transitions keyed by `(character_id, kind, key)`. Kinds are
`goal`, `belief`, `value` and `canon`. `character_id: null` identifies story-wide
canon; other kinds require a character. Characters must exist and be present
in the referenced scene. There can be only one transition for a key in a scene.
Validators follow **scene order**, independently of transition storage order.
A known `before` reading contradicting the preceding known `after` reading is
an error identifying the transition, scene and character. A change within a
transition is allowed: the point is to describe change, not prohibit it.

`relationships` contains transitions keyed by the unordered, distinct set of
`participant_ids` and `key`. All participants must exist and be present in the
scene. Before/after continuity and one-transition-per-scene rules also apply.
Participant list order does not change relationship identity.

Shared `Reading` records preserve `value`, `status`, optional `confidence`
between 0 and 1, and `alternatives`. A `known` reading requires a value;
`unknown` requires null, and `ambiguous` can carry a tentative value or null
with alternatives. Confidence defaults to unrecorded, never certainty. These
are annotations, not owner review or verified literary truth. Missing/uncertain
readings cannot establish a mechanical constraint as satisfied.

`constraints` names the same `(character_id, state_kind, key)` facts. A `locked`
constraint compares every explicit before/after reading with `expected`, while
`required_ending` compares the latest explicit after reading in scene order.
The represented state carries forward when no subsequent transition changes
it. This verifies an explicit ending fact, not whether prose earns the ending.
The validator does not infer facts from summaries or external canon.

Known mismatches are hard errors. Absent facts, unknown/ambiguous values and
`literary` constraints produce `human` findings, remain unresolved in cards and
cannot by themselves establish G-schema readiness. Mechanical constraints need
a nonempty expected value. Literary questions use `note` and always require
human assessment. Reference and continuity errors remain hard even if a
separate constraint's represented value matches.

## Mutation evidence

`examples/narrative-state.mutations.json` registers the valid control and
independent injected categories for the aggregate gate report. Each case must
produce its intended **error code** through saved-plan inspection; rejecting
unrelated syntax does not count. The integration tests run these mutations
through the public CLI, saved diagnostics and both card formats.

| Category | Expected diagnostic |
| --- | --- |
| Unknown state character | `state_character_reference` |
| Unknown state scene | `state_scene_reference` |
| Contradictory state before value | `state_transition` |
| Unknown relationship participant | `relationship_participant` |
| Duplicate relationship participant | `relationship_participant_duplicate` |
| Unknown relationship scene | `relationship_scene_reference` |
| Contradictory relationship before value | `relationship_transition` |
| Changed locked value | `locked_value` |
| Contradicted ending fact | `required_ending` |
| Unknown constraint character | `constraint_character_reference` |
| Missing mechanical expected value | `constraint_expected` |
| Unscoped goal/belief/value | `state_character_required` |
| Duplicate state transition | `state_duplicate` |
| Duplicate relationship transition | `relationship_duplicate` |
| Absent state character | `state_presence` |
| Absent relationship participant | `relationship_presence` |
| Unscoped non-canon constraint | `constraint_character_required` |

Additional named integration cases cover uncertainty bounds and valid controls. No API keys,
provider calls, downloads or owner availability are needed for implementation
acceptance. A manual inspection of the command above should show the final
goal “guide the boat home,” relationships changing from “suspicious” to “allies,”
and `hc3` awaiting human assessment; this is an optional content check, not a
routine regression requirement.
