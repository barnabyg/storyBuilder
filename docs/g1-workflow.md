# Registered G1 workflow

G1 generation and reading are separate from software acceptance. Actual G1 needs
a passed **owner** G-schema packet, an actual owner-selected tier backed by dev
reviews, and a committed pre-registration. Phase 2 software completion alone
does not establish those prerequisites. Synthetic fixtures are always marked
`test_only: true`; that mode accepts small frozen cohorts and restricts providers
to localhost. It cannot establish the scientific gate.

Prepare an experiment configuration using exactly `C: baseline_c` and
`B: baseline_b`, the selected C tier, the shared prose role/template, the exact
frozen test brief cohort, equal generation-token budgets and explicit positive
per-run and per-day dollar caps. Arbitrary extra calls and judge tuning inputs
are rejected. Keep dev comparison files, owner review records and the immutable
tier selection available for validation.

Create a JSON `G1Protocol` (schema exported by
`nde.eval.g1.G1Protocol.model_json_schema()`) and commit it. It includes:

- `id`, `schema_version: "1"`, `test_only: false`, `tier`;
- `cohort`: the [strict readiness registration](gate-readiness.md), with C first,
  B second, exactly 30 distinct frozen **test** brief IDs/checksums, one repetition,
  registered length and generation-token budget tolerances;
- `candidate: "C"`, `comparator: "B"`, `primary: "coherence"`,
  `ties: "non_wins"`, `win_threshold: 21` (or the explicit registered threshold),
  `required_dimensions: ["coherence", "originality"]`;
- `originality: {"analysis": "mean_paired_difference", "margin": NUMBER}`.
  The margin must be specified before reading. This supported descriptive
  safeguard requires mean paired C-minus-B originality **at least minus the
  margin**. It makes no inferential noninferiority claim. Other analysis names
  are rejected;
- `reading_mode: "full"` (all rubric marks and condition guesses required) or
  `"required_only"` (coherence/originality for both stories, preference and strength
  required; other marks/guesses optional), and `rubric_version: 1`;
- `config_sha256` from `load_config(path).hash`, `selection_sha256` and
  `schema_evidence_sha256` (LF-normalized file hashes, the latter for
  `evidence.json`);
- `generation_limits`: `max_repairs`, `max_length_attempts`,
  `max_revision_passes`, `per_run_usd`, `per_day_usd`, exactly matching config;
- `stopping_rule: "fixed_sample_no_top_up"` and
  `fail_action: "stop_or_one_fresh_preregistered_followup"`.

The supported command sequence (run each command on one line):

```text
uv run nde eval g1-check protocol.json configs/g1.yaml --selection selection.json --schema-evidence schema-packet --out g1-check
uv run nde eval g1-run protocol.json configs/g1.yaml --selection selection.json --schema-evidence schema-packet --runs-dir runs
uv run nde eval g1-batch protocol.json configs/g1.yaml --selection selection.json --schema-evidence schema-packet --name g1 --from runs/RUN C --from runs/RUN B --seed 123
uv run nde eval serve --batch g1
```

`g1-check` makes no provider calls, writes JSON/Markdown and exits 0 when ready,
2 when blocked. `g1-run` rechecks prerequisites before calling a provider, records
the run's protocol/config/evidence identities in `g1-generation.json`, and exits
1 on generation failure. `g1-batch` requires those generation identities and
strict complete-cohort readiness. It rejects incomplete or altered runs rather
than pairing their successful intersection. Existing generation/batch commands
keep their ordinary behavior.

The G-schema packet is replayed from `inputs.json` and must reproduce
`evidence.json` with passed mechanical and owner gates. A changed tier/config,
schema, prompt, model parameter, cap, frozen brief or protocol blocks admission.
The complete protocol, source-run manifests/calls/story indices, readiness and
all kept story artifacts are bound under `sealed/g1.json`. A later report can
detect changed or missing sources. Use a new batch name for another attempt;
the builder never overwrites an existing batch.

Optional `g1-batch --owner-code-work-at ISO8601_WITH_TIMEZONE` records the owner's
reported latest code-work time. Omission remains unknown, never confirmed
blinding. Keep at least the registered reading delay and record condition guesses
where possible. The reader serves only the blind view and scores; sealed files,
plans and ledger metadata have no reader HTTP route. Missing required scores
prevent submission and closure, including incomplete scores already on disk.

## Browser acceptance

Default tests use synthetic offline fixtures and download no browser. The browser
test uses an already installed Node.js, Playwright package and Chromium/Chrome:
set `NDE_BROWSER_NODE` to the Node executable, `NODE_PATH` to its installed
`node_modules`, and optionally `NDE_BROWSER_CHANNEL=chrome`, then run
`uv run pytest tests/test_g1_browser.py`. It exercises rendered required controls,
HTTP validation, persisted scores, closure and sealed condition-map exclusion.
Without `NDE_BROWSER_NODE`, this one test reports its prerequisite as skipped.

Manual regression: with a synthetic batch, submit only preference/strength and
expect an error naming both required dimensions; complete coherence/originality
for both stories and expect the score to persist after reload and closure to
become available. Requests for `/sealed/key.json` and `/sealed/g1.json` must return
404. Use synthetic data for this check, so the real held-out batch stays blind.
