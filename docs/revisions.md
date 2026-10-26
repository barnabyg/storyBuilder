# Structural revisions

`uv run nde plan revise config.yaml` uses the normal experiment gateway, response
cache, run/day caps and call registry. It is dev-only, and never overwrites its
source. A minimal `plans` section is:

```yaml
plans:
  briefs_dir: ../benchmark
  briefs: {ids: [brief-01]}
  params:
    saved_plan: ../examples/narrative-plan.json
    revision:
      version: '1'
      mask:
        channels: [arcs, reader_affect]
        units: []
      locks: [arcs/arc-keeper/reading, scenes/scene-0, constraints]
      instruction: Give the unlocked arc and affect records a new trajectory.
```

Supply a priced `planner` role, versioned prompt directory, seed, structured
repair limit and hard budget caps in the surrounding experiment config as for
`nde plan generate`. `saved_plan` resolves relative to the config. A source must
pass deterministic validation and every selector must resolve before any call.

The version-1 operator supports whole channel lists and stable slash selectors
such as `arcs/arc-keeper/reading`, `scenes/scene-0/summary` and
`scenes/scene-0/beats/beat-0/summary`. List offsets and wildcards are unsupported.
Only existing selectors are accepted. A whole channel authorises adding/removing
channel records; selecting a record authorises its content, with its identity
and surrounding list order preserved. Scene and beat IDs/order/count, title,
characters, sources, schema/tier, plan identity and all constraint records are
immutable. These restrictions fail validation even if a provider requests broader
permission. Existing brief constraints are retained as written, not converted
from literary prose into invented symbolic facts.

Locks may cover any existing subtree, including a whole channel, scene, edge,
constraint or nested field. Broad selections exclude nested locks. A selection
that is entirely inside a locked subtree is a preflight mask/lock conflict.
Unselected content and locked content are checked independently of semantic
validity. A provider returns the complete candidate; the operator rejects scope
or lock violations rather than silently splicing an unrelated replacement.

Canonical bytes are UTF-8 JSON with sorted object keys, `ensure_ascii=False`,
default JSON separators, and no trailing newline. Plan identities use the
released `serialized_plan` envelope, matching the planned-state ledger's
`plan_identity`. Subtree identities serialize their complete JSON values. Object
key order and file formatting are immaterial; list order is significant.

Each unique run writes `plans.jsonl` and `calls.jsonl`. Per-plan artifacts include
`plan.json`, `validation.json`, Markdown/HTML cards, raw repair attempts and
specific diagnostics under `planning/revision-<name>/`, and
`revisions/<name>/provenance.json` with the mask/locks, input/output canonical
checksums, lock identities, changed units, schema/code/model/prompt/brief/seed
identities and every attributable call. The saved run configuration binds all
run settings. `change-summary.md` lists actual changes. No-op results explicitly
record `noop` and retain the same identity. Schema, semantic, scope and lock
repairs share the configured finite repair limit. Failed/budget-aborted/provider
runs retain diagnostics and `last-accepted` cards/plan, and do not expose a failed
candidate as `plan.json`. A complete artifact directory is published by atomic
rename; no existing destination is replaced. Cached replay retains experimental
tokens and list-price cost while adding zero new provider spend.
