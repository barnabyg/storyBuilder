# Selected C/D development experiment

Use `nde compare-cd run CONFIG --selection SELECTION --out NEW_FOLDER` with:

```yaml
stories:
  briefs_dir: ../benchmark
  briefs: {ids: [brief-01]}
  conditions:
    - name: C
      pipeline: baseline_c
      params: {tier: L}  # Must match the selected C record.
    - name: D
      pipeline: baseline_d
      params:
        critic_iterations: 2
        revision_locks: []  # Optional stable selectors from the accepted plan.
  length: {tolerance: 0.1, max_attempts: 3}
  equal_budget:
    conditions: [C, D]
    tolerance: 0.1
    max_revision_passes: 4
```

The complete config must also supply the original selected C planner/prose
bindings, a separate-family critic binding, prices, hard per-run/day caps,
provider settings, seed and versioned prompt directory. No concrete model or
scientific defaults are selected by this example. Remove the illustrative lock
unless the generated plan has that actual scene ID. Generation/failure tests use
fake providers exclusively. G-D admission is a separate protocol-bound workflow.
