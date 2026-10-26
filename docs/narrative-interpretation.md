# Interpretive channels and trajectories

Use the existing [plan inspection workflow](narrative-plans.md). The saved
`examples/trajectory-rising.plan.json` and `examples/trajectory-falling.plan.json`
are deterministic presentation fixtures, not gold annotations or owner evidence.

```powershell
uv run nde plan inspect examples/trajectory-rising.plan.json --out .nde/rising
uv run nde plan inspect examples/trajectory-falling.plan.json --out .nde/falling
```

Schema 1 adds list-valued `arcs`, `reader_affect`, `tension`, `themes`, `motifs`,
`discourse` and `subtext` channels. Every record has a globally unique `id`, a
`reading`, optional `scene_ids` and optional `evidence` spans. An empty
`scene_ids` list identifies a story-level record. `kind` explicitly separates
`observation` from `interpretation`; the default is interpretation. Observations
describe explicit textual properties, while interpretations retain a reading's
`status`, optional `confidence` and optional `alternatives` without treating them
as established facts. Unknown values render as missing values with unknown
status, not as confident labels. Confidence is optional in [0, 1]; absence is
shown as unrecorded, never silently converted to certainty.

An `Arc` may name a `character_id`. `Arc`, `ReaderAffect` and `Tension` may include
a trajectory, respectively agency, mood and tension. A trajectory contains
**three to five** ordered `{scene_id, value}` points on the coarse ordinal scale
**0–4**. Zero denotes low intensity/agency/mood and four high; the numbers are
ordered categories, not interval measurements. A null value means unknown and
remains unknown in cards. A shape must be one of `rising`, `falling`, `flat`,
`peak`, `valley`, `oscillating`, `unknown` or `ambiguous`. Shapes are recorded
readings, not curve fits or literary judgements inferred from numbers. Points
must refer to existing scenes in chronological order; several coarse points
may concern the same scene. Missing optional trajectories remain absent.

`motif_echoes` link distinct motif occurrence IDs through typed `repetition`,
`transformation` or `contrast` edges. Endpoints must exist. They may carry their
own evidence. Every annotation or echo evidence span uses the shared exact source
ID, source checksum and half-open Unicode codepoint offsets. It is checked
against the embedded LF source text, including exact version and bounds.

Markdown and HTML show the values, record/scene/character IDs, interpretation or
observation status, confidence, alternatives, evidence and ordinal sequences.
Fixture comparisons establish that saved and rendered trajectories differ;
only a separately recorded owner assessment can establish that those differences
are meaningful literary distinctions.

## Mutation categories

The public CLI/file tests retain valid controls and expect the intended validator
diagnostic, rather than accepting any arbitrary parse failure as a catch.

| Category | Expected diagnostic |
| --- | --- |
| Annotation refers to missing scene | `interpretive_scene_reference` |
| Arc refers to missing character | `arc_character_reference` |
| Trajectory refers to missing scene | `trajectory_scene_reference` |
| Trajectory reverses chronology | `trajectory_order` |
| Motif echo has a missing endpoint | `motif_endpoint` |
| Motif echo links an occurrence to itself | `motif_self_link` |
| Cross-channel ID duplicates a scene ID | `duplicate_id` |
| Evidence names a missing source | `span_source` |
| Evidence names a different source version | `span_version` |
| Evidence exceeds source offsets | `span_bounds` |
| Ordinal exceeds upper bound | Pydantic `less_than_equal` at trajectory point value |
| Trajectory has fewer than three points | Pydantic `too_short` at trajectory points |
| Shape is outside the explicit enum | Pydantic `enum` at trajectory shape |

All listed semantic diagnostics are hard errors. These checks establish
structural consistency, not the accuracy of interpretations.
