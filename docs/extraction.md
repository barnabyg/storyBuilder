# Evidence-linked corpus extraction

`nde extract run WORK --segments SEGMENTS --config CONFIG --out OUTPUT` runs
three versioned channel passes: structural/causal/information, character state,
then interpretation. Each response is a tier-L plan. Earlier accepted channels
are locked; conflicting rewrites, dangling references and invalid evidence cause
the same bounded repair loop. Exact segment identities and full canonical source
spans are required. Structural/state claims inherit their linked scene evidence;
known interpretive readings additionally require their own source evidence.
These are mechanical support checks, not proof of literary correctness.

Configuration contains `providers`, `annotator`, explicit `prices`, `budget`,
`seed` (default 7) and `max_repairs` (default 1 per pass). It uses the existing
gateway provider/model/price/budget schemas. Instructions and narrative schema
are versioned by checkpoint checksums. Changed source, segmentation, prompts,
schema, configuration, cache or accounting paths require a new output version.

`extraction.json` checkpoints every expected pass, original response, diagnostic,
request attribution, token count, retries and list-price/new-spend cost. Original
responses are also retained separately. `annotation/` contains the assembled
plan, validation and source-linked cards. `report.md` and `report.html` explicitly
describe an unreviewed model annotation. Existing Phase 2 sessions and corrections
are unchanged. `nde extract report OUTPUT` verifies current inputs and originals.

Resumption retains repair counts and durable ledger totals. An interrupted paid
request can only be replayed from its exact cached response. Without that evidence
it remains `recovery_required`; rerunning never silently repeats the paid call.
Cache hits retain experimental tokens and list-price costs. Nested ledger scopes
enforce a shared cohort/day cap and a story cap without double-counting daily
spend. Unresolved in-flight calls and durable accounting totals remain visible.
