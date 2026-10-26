# Compare development tiers and freeze the owner selection

Use a B/C equal-budget story config on an explicit set of dev briefs. Both
conditions use the shared prose role and `scene_prose@1`. The command generates
three ordinary experiments: B versus projected C-S, B versus C-M, and B versus
C-L. Full tier-L planning runs before projection in each experiment. Identical
planning calls share the response cache across tiers, retain their attributable
experimental tokens and add no new provider spend on a cache hit. Prose and
budget revisions use the existing shared generation path.

```powershell
uv run nde tiers compare configs/stories-bc-equal-budget.yaml --out .nde/dev-tiers-1
uv run nde tiers report .nde/dev-tiers-1
```

`compare` requires exactly `B: baseline_b` and `C: baseline_c`, equal-budget
matching for B/C, and distinct dev-only brief IDs. Arbitrary calls, plan runs,
judge runs, frozen held-out identities and test selections are rejected before
generation. Use generation and daily budget caps in the config. This is a
generation command and may make paid provider calls; offline automated tests
use the local fake provider.

The folder contains the source config, each tier's config and ordinary run
artifacts, C realisation packets, an immutable `comparison.json` and reproducible
JSON/Markdown reports. Reports expose all stories, failed/incomplete cohorts,
length checks against final prose, per-condition attributable planning/prose/
repair/regeneration/revision calls and tokens, new provider spend, budget
matching, validity and realisation evidence. Unreviewed realisation and owner
judgements remain pending; a software pass cannot establish literary quality.
Inspect each `realisation/<tier>/<brief>-r<repetition>` packet with the existing
[realisation workflow](realisation.md). Reviews against superseded final prose
remain stale.

Exit 0 means the generation comparison is complete, 1 means incomplete saved
evidence, and 2 means malformed inputs or an incompatible configuration. Owner
reading may still be pending at exit 0. Existing output folders are refused so
an attempt cannot silently replace historical outputs.

Record the owner's actual dev assessment separately. A review JSON file has
`comparison_sha256` (the comparison's `id`), `reviewer`, `test_only: false`, and
`judgements`. Each judgement identifies a `brief`, `tier`, zero-based
`repetition`, `coherence` and `originality` scores from 1 to 5, and a nonempty
`note`. The sample is explicit; unreviewed stories stay unreviewed. Appending a
review preserves earlier files.

```powershell
uv run nde tiers review .nde/dev-tiers-1 owner-dev-review.json
uv run nde tiers select .nde/dev-tiers-1 --tier M --owner "Owner name" --rationale "Reason based on the recorded dev sample" --out .nde/selection-1.json
uv run nde tiers check .nde/selection-1.json configs/stories-bc-equal-budget.yaml
```

Selection requires complete generation/budget comparisons and explicit owner
judgements for at least one story in the chosen tier. It records that reviewed
sample rather than pretending the full dev set was assessed. The selected tier,
owner/rationale, exact dev checksums, configuration, schema, actual prompt
checksums, model parameters, immutable comparison and owner-review file
identities are bound into the content ID. Selection files use exclusive
creation: changes require a new historical selection. Synthetic tests use
explicitly marked `test_only` reviews and `select --test-only`; they cannot
satisfy real G1 prerequisites. No LLM judge can submit a tier selection.

`nde.tiers.load_selection(path)` verifies the selection content identity.
`selection_problems(selection, config_path)` checks the original comparison and
review artifacts, current dev brief checksums, chosen tier, schema, actual used
prompt text and model/generation/repair/budget settings. The name, brief selector
and directory may change from dev to test; condition ordering is canonicalised
and the chosen tier is checked separately. Other relevant settings must match.
Model IDs alone do not establish model-parameter compatibility. `tiers check`
returns 0 for matching current evidence/settings, 1 for stale or mismatched
evidence/settings, and 2 for an unreadable or malformed selection/config. Set
the config's C `params.tier` to the selected tier before checking it.

Manual test with configured credentials: generate a small dev comparison,
inspect `report.md`, and confirm that S scene input notes contain no beliefs or
motifs, M includes goals/beliefs without motifs, and L retains both. All runs
should preserve full planning tokens despite projection. Before owner review,
selection must refuse with an explicit missing-owner-judgement error. Changing
the prose model, prompt text, limits or dev briefs after selection must make
`selection_problems` nonempty. Actual owner sampling and rationale remain
scientific prerequisites for the held-out G1 workflow.
