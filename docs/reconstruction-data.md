# Protected reconstruction datasets

Phase 6 adds separate version 1 artifacts. It does not change the released
training-only corpus export or narrative-plan schema. All commands below read
bound source files again and reject changed identities.

Create the training encoding, then prepare optimizer examples:

```powershell
nde reconstruction-codec encode path/to/export --out encoded
nde reconstruction-data prepare prepare.json --out training
nde reconstruction-data inspect training
```

`prepare.json` accepts `encoding`, `granularity` (`whole-story`, `window`, or
`scene`), `window_scenes`, `stride`, `seed`, `repetitions`, `max_scenes`,
`max_tokens`, and `schedules`. For example:

```json
{
  "encoding": "encoded",
  "granularity": "window",
  "window_scenes": 3,
  "stride": 1,
  "schedules": [{"kind": "fields", "rate": 0.3}]
}
```

Each example retains its parent author/work/family and source checksum, original
annotation path, half-open scene offsets, and exact source evidence offsets.
These provenance fields are evaluator metadata, outside model input. Family
variants and overlapping windows do not increase independent story counts.
Optimizer loading rejects evaluation roles. The vocabulary is fixed and requires
no fitting on either training or evaluation labels. Capacity limits reject the
example; there is no truncation or implicit padding of real content.

For a partial window, links requiring scenes outside the window are recorded as
boundary omissions, with context explicitly unknown. Local causal and information
content can remain visible, but both graph channels have zero target eligibility
in partial views. This conservative policy avoids teaching missing cross-window
edges as negative labels. Original complete annotations remain available for
inspection; the model receives only the filtered representation.

Evaluation uses a separately supplied authorization, never the training export's
record list. Create an authorization config with `export`, `cohort`, `split`
(`validation` or `test`), `authorized_by`, `test_only`, and `annotations` entries
containing `work_id`, `family`, and `path`. Paths must identify valid annotations
of works in that exact frozen held-out split, with matching source checksums.
Gold and excluded assignments cannot enter this workflow. The exported channel
policy applies to evaluation as well.

```powershell
nde reconstruction-data authorize-evaluation authorize.json --out authorization
nde reconstruction-data prepare prepare-held.json --out evaluation
```

`prepare-held.json` uses `"evaluation": "authorization"` in place of `encoding`.
The resulting manifest is marked `evaluator-only`; it never adds held-out records
to the approved training export. Synthetic fixtures require matching test-only
status throughout and cannot establish scientific approval.
