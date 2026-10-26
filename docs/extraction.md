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
Checkpoints also retain the complete acquired work provenance and the resolved
provider endpoints, including environment overrides. Same-text author/licence
metadata changes or endpoint changes cannot silently reuse old evidence. Earlier
checkpoint JSON remains readable; checkpoints lacking these bindings cannot be
resumed or admitted as current evidence and require a new output version.

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

## Independent model families

`nde extract families WORK --segments SEGMENTS --config CONFIG --out OUTPUT`
uses the same settings with `annotators` replacing `annotator`. Supply at least
two different provider/model bindings with distinct, explicit `family` metadata.
Aliases pointing at the same endpoint/model, missing families and case-insensitive
duplicate families are rejected before any call. A declaration records the user's
family provenance; it does not independently authenticate a provider's model.

Each family receives only the verified source, segmentation and its own accepted
passes/repairs. Separate directories retain each original and failure state.
`families.json` records expected coverage; one failed or unstarted family leaves
the story incomplete. Story caps cover all independent families together and
remain enforced when a larger cohort cap is supplied by pilot orchestration.

`nde extract families-report OUTPUT` revalidates source/config/schema identities
and original responses, then recomputes agreement. Scenes align on exact source
spans, entities on unique normalized names and reference claims on their normalized
content. Generated record IDs are excluded. Duplicate names remain explicitly
ambiguous. Exact normalized claims use multiset intersection/union: `agreed`
is intersection size, `total` is union size, `compared` is the smaller inventory,
and `unmatched` retains both inventories' missing or contradictory claims.
Empty channels have no measured agreement, rather than a perfect score.

Reports include core plot, causality, information, goal and belief channels,
state/interpretive channels and a grouped interpretation row. They retain
disagreements; they never produce merged consensus labels or claim owner approval.
Three or more families produce all pairwise comparisons. Aggregate durable story
spend and any new spend not attributable to a saved response remain visible after
crash recovery. Actual literary correctness and gold review remain pending.
