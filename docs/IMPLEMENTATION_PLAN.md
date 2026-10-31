# Narrative Diffusion Engine — Implementation Plan

**Status:** Working plan v3; tooling through Phase 7 implemented; G-schema passed; G1 and its retest did not pass; diffusion track stopped; separate continuity approach accepted by the owner
**Created:** 18 October 2026
**Updated:** 31 October 2026, recording the owner's decision on the continuity approach
**Source:** [Narrative_Diffusion_Writing_Engine_Proposal.md](../Narrative_Diffusion_Writing_Engine_Proposal.md)

## Current position

- **Phase 0 complete:** G0 passed on 18 October; the gateway, cache, budgets,
  experiment registry and canonical verification are implemented.
- **Phase 1 complete:** G-instr closed on 20 October through its pre-registered
  fail action. The owner's coherence scoring is the validated structural
  instrument; the LLM judge is diagnostic only. The 10 dev briefs and 30 frozen
  test briefs are in place.
- **G-schema passed on 27 October:** three real, source-linked annotations were
  corrected through guided owner review and explicitly approved. The owner
  confirmed meaningful differences in the qualitative tension and ending-response
  patterns. All 44 mutation categories were caught, each story took under two
  hours of measured owner review, and the final evidence replay is byte-identical.
  See the [frozen decision and evidence](evidence/g-schema-2026-10-27/decision.json).
- **Phases 2–5 tooling implemented:** the schema/annotation workflow, C and D
  planners, registered evaluations, extraction pilot and G2 export/scaling
  controls are on `main`. Their implementation issues are closed.
- **Phase 6 tooling is also on `main`:** structural encoding, protected
  datasets, local denoising/matched AR training, scoring, audits, sweeps, G3
  reporting and the optional augmentation workflow. Issues #66–75 are closed.
- **Implementation is not a scientific pass.** Beyond the dated G-schema result,
  G1 did not pass, and no genuine G-D, G2 or G3 outcome is recorded in this plan. Synthetic fixture criteria can
  pass while the scientific outcome remains pending. Closed implementation
  tickets and green verification do not authorise later research phases.
- **G1 is now four pairs:** on 30 October, before any reading, the owner decided
  that four pairs are enough to judge whether there is a consistent difference. The
  [G1 protocol](g1-four-pairs.md) was registered on the four generated M/B pairs.
- **G1 did not pass on 30 October:** C won 0 coherence pairs of 4, tying 1, and fell
  short on originality. See the [frozen decision and evidence](evidence/g1-2026-10-30/README.txt).
- **The retest did not pass on 30 October:** on four held-out pairs with `gpt-6.1-sol`
  and equal editing, decided on coherence alone, C won 2 of 4 against the 3 required.
  See the [retest result](g1-four-pairs.md#retest) and
  [evidence](evidence/g1-retest-2026-10-30/README.txt).
- **The diffusion track has stopped under the registered fail action.** The schema,
  benchmark, harness and planner are the result (proposal §14). Real D/corpus,
  reconstruction and E experiments are not authorised: they still require a passed G1.
  Phase 6–7 tooling remains, tested on fixtures only, and Phase 8 is not implemented.
- **The owner accepted the separate [continuity approach](continuity.md) on 31 October:**
  character sheets, a scene clock, knowledge checks, targeted repair and plain-style
  rules. After reading one story on a development brief, the owner scored coherence 4
  of 5, reported none of the four defects, and decided no blind comparison or further
  proof runs are needed: "it clearly works right now". This is the owner's judgement,
  not a blind result or a gate. It does not reopen G1 or restart the diffusion track.
  See the [evidence](evidence/continuity-2026-10-31/README.txt).

| Phase | Implementation status on `main` | Implementation issues | Operating guide |
| --- | --- | --- | --- |
| P2 | Schema, validators, tiers, cards, annotation/review and G-schema evidence | [#32–38](https://github.com/barnabyg/storyBuilder/issues?q=is%3Aissue+%22%5BP2%5D%22), closed | [Plans](narrative-plans.md), [annotations](annotations.md), [G-schema](schema-evidence.md) |
| P3 | State ledger, C, equal budgets, realisation, tier selection, G1 admission/decision | [#39–47](https://github.com/barnabyg/storyBuilder/issues?q=is%3Aissue+%22%5BP3%5D%22), closed | [C](baseline-c.md), [G1 workflow](g1-workflow.md), [G1 decision](g1-decision.md) |
| P4 | Masked revision, coarse-to-fine D, bounded critic, controls/diversity, G-D | [#48–55](https://github.com/barnabyg/storyBuilder/issues?q=is%3Aissue+%22%5BP4%5D%22), closed | [Revisions](revisions.md), [D](baseline-d.md), [G-D](gd.md) |
| P5 | Provenance, splits, segmentation/features, independent extraction, pilot/review, exports and exemplars | [#56–65](https://github.com/barnabyg/storyBuilder/issues?q=is%3Aissue+%22%5BP5%5D%22), closed | [Corpus](corpus.md), [extraction](extraction.md), [G2](corpus-g2.md) |
| P6 | Codec/tasks, datasets, CPU ML models, paired evaluation, audits/sweeps, G3 and augmentation | [#66–75](https://github.com/barnabyg/storyBuilder/issues?q=is%3Aissue+%22%5BP6%5D%22), closed | [Training](reconstruction-training.md), [experiments](reconstruction-experiments.md), [G3](reconstruction-evaluation.md) |
| P7 | Conditional examples, variable layouts, brief-conditioned training with guidance, bounded generation, text completion, D/E inpainting, the Candidate E run pipeline, adherence/diversity/similarity reporting, registered G-E preparation and the G-E decision report; no G-E result | [#76–85](https://github.com/barnabyg/storyBuilder/issues?q=is%3Aissue+%22%5BP7%5D%22): closed | [Data](conditional-data.md), [generation](conditional-generation.md), [inpainting](conditional-inpainting.md), [E](candidate-e.md), [evidence](conditional-report.md), [G-E](ge.md) |

Status above was checked against `main` at `de0beaf` and GitHub Issues on
30 October 2026. The dated G0/G-instr results below remain historical records;
the task tables retain their original plan IDs, which differ from GitHub issue
numbers. Operating guides describe the implemented interfaces. Future changes
to scientific criteria require a new committed protocol, not reinterpretation
of a completed result.

## 1. Decisions taken for this plan

| Topic | Decision | Consequence |
| --- | --- | --- |
| Primary goal | Research first: answer the gates in proposal §11 | Minimal UI. CLI plus generated HTML/Markdown reports. The one exception is a small local reader app for blind evaluation. No plan-editor UI until the core thesis passes. |
| Stack | Python throughout | One schema language (Pydantic) from extraction through to diffusion training. No cross-language contract. |
| LLMs | Mix of low-cost APIs (e.g. DeepSeek Flash (`deepseek-flash`), GPT-6 Luna (`gpt-6-luna`)), plus local models if they prove good enough | A provider-agnostic gateway behind an OpenAI-compatible interface, with model choice set per role in config. Structured-output reliability on cheap models becomes an early, measured risk. |
| Evaluators | The project owner is the only human evaluator for now | Owner reading time is the scarcest resource and drives the evaluation design (§4). Results describe one reader's judgement. Any external claim later needs a panel. |

### Working assumptions (change any of these if they are wrong)

- One developer, part-time. Effort is given in relative sizes, not dates.
- Gate evaluations use stories of about **2,000 words with 5–8 scenes**. That is long enough for setup/payoff and arc shape to matter, and short enough to read a pair in about 20 minutes.
- Implementation verification uses tiny local CPU models and needs no GPU or model downloads. Before real Phase 6 training, select compatible local pretrained weights and measure resource requirements; neither the current tests nor a nominal GPU memory size establishes feasibility. Other model architectures need an explicit adapter.
- The corpus is public-domain only at first (Standard Ebooks, Project Gutenberg). Licensed modern fiction is out of scope until Phase 6 shows that data volume is the bottleneck.

## 2. Strategy: how this plan de-risks the build

The proposal's stages are sound, but run in their listed order they would spend most of the effort before the central bet is tested. This plan makes four structural changes.

1. **Test the core thesis before building the corpus pipeline.** The proposal's recommended first experiment compares beat-only plans with multi-channel plans. It needs no extracted corpus, because both plans can be generated by prompting. If richer plans do not improve stories, the large annotated dataset is only useful for training, and training would no longer be justified. So the cheap thesis test (Phase 3) runs **before** extraction at scale (Phase 5).
2. **Validate the measuring instrument before trusting it.** With a single human evaluator, the evaluation harness is the riskiest component. Phase 1 proves that it can detect a known, deliberately caused degradation before any real comparison is run.
3. **Budget owner reading for development and gates.** Following G-instr, deterministic checks and small owner-reviewed dev samples screen ablations and variants. LLM judges provide diagnosis only until validated for the intended task against the owner's verdicts. Allow for that additional owner reading before scheduling a gate.
4. **Make the strongest non-diffusion baseline the bar diffusion must clear.** Iterative refinement (Candidate D) is built and tuned before any denoiser. A trained model is only worth pursuing if it beats the reference baseline at equal budget: D if G-D passes, otherwise C (proposal §10.3).

Every phase ends in a **gate** with a pre-registered pass criterion and an explicit **fail action**. Stopping after any gate still leaves usable outputs: a benchmark, a schema, a planner, a dataset.

```text
P0 Foundations ─► P1 Benchmark + instrument check ─► P2 Schema v0 ─► P3 THESIS TEST (B vs C)
                                                                          │ pass
                                          ┌───────────────────────────────┴──────────────┐
                                          ▼                                              ▼
                                P4 Iterative refinement (D)               P5 Extraction pilot + corpus
                                          └───────────────────────────────┬──────────────┘
                                                                          ▼
                                                       P6 Denoising feasibility (reconstruction)
                                                                          │ pass
                                                                          ▼
                                                       P7 Conditional diffusion planner (E)
                                                                          ▼
                                                       P8 End-to-end evaluation + decision
```

Phases 4 and 5 are independent and can run in parallel, for example on separate branches or worktrees.

This diagram describes scientific prerequisites. Fixture-based software has
been implemented through Phase 7 ahead of those decisions; real workflows
retain evidence checks and do not infer permission from code availability.

## 3. Cross-cutting foundations

These pieces are built once in Phase 0 and used by every later phase.

### 3.1 LLM gateway (`nde.llm`)

- One client for **OpenAI-compatible endpoints**. That covers DeepSeek, OpenAI and local servers (llama.cpp, Ollama and vLLM all expose this API). Add an adapter only for a provider that is not compatible.
- **Roles, not models, in code.** The roles are `annotator`, `planner`, `prose`, `critic` and `judge`, and each maps to a provider/model/params in config. Switching a role to a local model is then a config change.
- **Response cache.** Calls are content-addressed by (provider, model, params, prompt-template version, rendered prompt, seed) and stored in SQLite. This makes re-runs free and reproducible, and lets an analysis be re-run without new spend.
- **Cost ledger and hard budget caps** per run and per day. A run aborts cleanly when it exceeds its cap.
- **Structured output.** Requests use JSON-schema mode where the provider supports it. Each response is validated against Pydantic, followed by at most N repair attempts. Schema-validity rate and repair rate are logged **per model**, because this is the most likely failure point for low-cost models.
- **Family separation rule.** The judge and critic must come from a different model family than the planner and prose generator being judged, to reduce self-preference bias. Any waiver must be explicit in config and reported with the result; the G-instr waiver was specific to that experiment and is not automatic permission for later comparisons. Critic suggestions are bounded by validators and owner review, not treated as validated evaluation scores.

### 3.2 Experiment registry (`nde.runs`)

- Every experiment is a YAML config. Running it produces a run directory with a manifest: git SHA, config hash, seeds, model IDs, prompt versions, benchmark split IDs, costs and output paths.
- Prompts live in `prompts/` as versioned templates. Changing a template's text requires a new version.
- Reports (Markdown and HTML) are generated from run directories, never edited by hand.

### 3.3 Blind evaluation reader (`nde.eval.reader`)

- A small local web app on localhost (FastAPI plus server-rendered HTML). It shows one story pair at a time, in randomised order, with all condition markers stripped and formatting normalised.
- It records a forced-choice preference with strength (1–3), rubric scores, free-text failure tags, and a **"which condition do you think A was?"** guess. The guess measures whether blinding has leaked.
- The pair-to-condition key stays sealed until the batch is closed.

### 3.4 Engineering baseline

- Python 3.12+, managed with `uv`. Dependencies are pinned in `uv.lock`. `torch` and other ML dependencies sit in an optional dependency group so that Phases 0–5 stay light.
- One canonical, non-mutating verification command, `uv run nde-verify`, runs the gates in order:
  1. format check (`ruff format --check`)
  2. lint (`ruff check`, warnings as errors)
  3. type check (`pyright --strict`)
  4. static bug rules (the bugbear and security rule sets in ruff; anything not covered is documented)
  5. tests (`pytest`)
  6. dependency audit (`pip-audit`) and a secret scan
  7. build check (`uv build`)
- An optional localhost live verification dashboard is implemented; headless verification uses the same gates.
- GitHub hosts `barnabyg/storyBuilder`, with `main` as the default branch. GitHub Actions runs the same command headless on pushes and pull requests; remote and CI setup are complete.
- CI has a second CPU ML job: `uv sync --locked --group ml`, then
  `NDE_REQUIRE_ML=1 uv run --locked --group ml nde-verify` (set the environment
  variable separately in PowerShell). It runs actual parameter updates,
  checkpoint recovery and paired inference without downloading weights. The
  ordinary dependency set stays lightweight; optional ML tests may skip there.
- LLM calls are never made in unit tests. Tests use recorded cache fixtures. A separate opt-in `live` test marker exercises the real providers.
- Automated command/file/report tests are the primary implementation acceptance
  evidence. Browser tests cover critical reader flows when the documented
  installed-browser prerequisite is enabled. Remaining owner work is literary
  evidence and scientific interpretation, not routine regression testing.
- Each phase is developed on its own feature branch, with one commit per ticket.

### 3.5 Repository layout

```text
src/nde/
  schema/     released brief, benchmark and response models
  narrative/  versioned plans, validators, tiers, ledger and scene cards
  llm/        gateway, cache, cost ledger, structured-output repair
  runs/       experiment registry, manifests, report generation
  plan/       B/C/D planning, imports, masked revision and critic refinement
  prose/      scene-level prose realisation
  eval/       judges, blind reader, strict readiness, G1/G-D protocols
  annotations/ source-linked draft/correction sessions and owner review
  evidence/   reproducible G-schema evidence and mutation registry
  corpus/     acquisition, splits, segmentation/features, extraction, G2/export
  reconstruction/ codec, datasets, native ML, scoring, audits, sweeps and G3
prompts/      versioned prompt templates
configs/      experiment configs
benchmark/    briefs (dev/test), rubric, pre-registrations, gold annotations
data/         gitignored; manifest with sources and checksums committed
docs/
tests/
```

## 4. Evaluation protocol with a single evaluator

This protocol applies to future gates and incorporates the Phase 1 decisions.

- **Splits.** About 40 briefs are split into 10 **dev** briefs, used freely during development, and 30 **test** briefs. Outputs for test briefs are generated only for gate runs and read only blind.
  - **The test split is frozen (20 October 2026):** `brief-11` to `brief-40`, checksums in `benchmark/frozen.json`. The briefs were LLM-drafted through `configs/draft-briefs.yaml` (run `20261020T195417Z-draft-briefs-7b48ae`, 40 drafts), given a first curation pass by Claude (10 rejected, light edits) and reviewed and accepted by the owner. Drafts 2, 7, 8, 11, 20, 22, 23, 26, 28 and 37 were rejected; the rest became `brief-11` to `brief-40` in draft order. The run directory and drafts are local only (`runs/` and `benchmark/drafts/` are gitignored).
- **Sample unit and size.** A story-comparison gate uses one pair per distinct
  brief at a fixed, pre-registered sample size. G1 uses four pairs, by owner
  decision. Its result describes a consistent difference for this reader and does
  not rule out a moderate effect when it fails.
  About 80 distinct briefs is a planning target for a larger confirmation or
  final comparison; calculate power for its primary metric, expected ties and
  chosen threshold before committing that protocol. Additional seeds on one
  brief do not create independent brief-level observations.
- **No automatic extension.** Reading a gate's pairs does not authorise topping
  up the same test to a larger size. A larger follow-up requires a separate,
  pre-registered fixed-size experiment on fresh held-out briefs, reported
  separately. The existing frozen test split must remain unchanged; a new
  benchmark version, checksums and tooling support are prerequisites. Any
  sequential or pooled analysis needs its stopping rule and error control
  specified before the first relevant batch is read.
- **Reading cost.** At about 2,000 words per story, budget about 20 minutes per pair, so about 80 minutes for the four-pair G1.
- **Leakage controls.**
  - All conditions use the same prose model and prose prompt template.
  - Story length is held within ±10% of target. Regenerate outside that range
    only within pre-registered attempt and cost caps. If either story remains
    out of range or generation fails, the batch is not ready for gate reading:
    resolve the problem before reading or record the gate as incomplete. Do
    not silently drop the pair or reduce n. Current generation retains the
    closest draft and flags it, and batch creation can omit failed stories;
    the implemented strict readiness and registered gate workflows enforce
    this policy; ordinary non-gate batch creation retains its old behaviour.
    This does not retroactively alter G-instr's explicit allowance for short
    stories.
  - Headers and markup are stripped.
  - Reading happens at least 48 hours after the owner last worked on the generating code.
  - If the owner's condition guesses are right in more than about 70% of pairs, the gate is flagged as possibly unblinded.
- **LLM judge panel.** Judges are diagnostic only following G-instr. When run,
  use at least two model families, with pairwise comparisons in both orders,
  or explicitly record a family-rule waiver. Report agreement with the owner
  (percentage agreement and Cohen's κ), ties and position consistency.
  Screening can resume only after validation for the intended task against
  owner scores under a pre-registered protocol on separate dev data.
  Until then, small owner-reviewed dev samples and deterministic checks
  select variants. Judges never pass a gate on their own, and revalidating
  a judge is not a prerequisite for Phase 2 or G1.
- **Structure as its own measure** (decided after G-instr attempt 1). The owner's
  natural "which is better" judgement weighs the quality of the writing, so it can
  miss structural differences when the prose is equally good. Gates that test
  structure therefore make a required structural score (the rubric's coherence)
  the primary metric, with the overall preference as a secondary signal.
  For G1 and other registered win-count comparisons, count a win only when
  the candidate scores strictly higher; ties count as non-wins and stay in
  the denominator. Report the two-sided exact binomial
  test against a win probability of 0.5, as the reader report does. Put the
  candidate first in the batch's `--from` arguments, because the report tests
  that first condition. A pass requires the pre-registered win threshold in
  the candidate's favour; a small two-sided p-value alone is insufficient.
  Report overall preference and originality separately, without substituting
  them for coherence. G-D instead applies its separately registered mean
  paired difference and margin; it does not reuse the G1 win threshold.
- **Pre-registration.** Before each gate run, commit its supported
  machine-readable protocol and a human-readable account of the hypothesis,
  candidate and comparator, primary metric,
  distinct brief IDs and checksums, n, tie treatment, pass threshold, required
  rubric scores, reading mode, originality safeguard, length and budget
  admission checks, stopping rule and fail action. Specify the margin and
  analysis for any claim of "not worse" or "at least as good"; a
  non-significant difference alone does not establish it. Thresholds in this
  plan are defaults to confirm at that point. Preserve completed protocols and
  results; amendments govern new runs, not results already read.
  The historical G-instr protocol remains in `benchmark/prereg/g-instr.md`.
  G1, G-D, G2 and G3 now have versioned JSON protocols and evidence bindings;
  follow their operating guides rather than treating a prose checklist as
  sufficient admission. G2 criteria are frozen before owner labels; G3 is
  registered before held-out predictions, paired scores and owner meaning
  labels. The supported G1 originality and G-D coherence/originality analysis
  is a registered mean paired difference with a margin: a **descriptive**
  safeguard, not inferential noninferiority. Unsupported analyses are rejected.
- **Scope of claims.** Gate results describe the owner's taste. That is acceptable for research-first internal decisions. A small external panel is a prerequisite for any public claim.

## 5. Phases

Sizes: **S** is about 1–2 weeks part-time, **M** about 3–5 weeks, **L** about 6 weeks or more.

### Phase 0 — Foundations (M)

**Goal:** the infrastructure every experiment depends on, so that later phases only write experiment logic.

| Ticket | Work |
| --- | --- |
| P0.1 | Project scaffolding: `uv`, pinned tools, `nde-verify` canonical command with all gates, CI workflow, `.gitignore`, `data/` manifest convention |
| P0.2 | Live verification dashboard for `nde-verify` |
| P0.3 | LLM gateway: OpenAI-compatible client, role→model config, retries and backoff, provider error normalisation |
| P0.4 | Response cache (SQLite, content-addressed) and cost ledger with budget caps |
| P0.5 | Structured-output layer: schema mode, Pydantic validation, bounded repair, per-model validity metrics |
| P0.6 | Experiment registry: config → run dir → manifest; `nde run <config>`; report generation skeleton |
| P0.7 | Model smoke benchmark: for each candidate model (the two APIs plus any local model), measure JSON-schema validity on a moderately nested schema, latency, and cost per 1,000 output tokens |

**Exit gate:**
- `nde-verify` passes cleanly.
- Re-running a run from cache costs nothing and reproduces its outputs byte-for-byte.
- P0.7 shows at least one affordable model reaching **≥95% schema validity after repair**.

**Fail action:** if no low-cost model reaches that validity, simplify the schema shape (flatter, smaller per-call sub-schemas) or add a single mid-tier model only for the `planner` role. Record the decision.

**G0 result (18 October 2026): passed.** Run `20261018T213124Z-benchmark-g0-80c381` of
`configs/benchmark-g0.yaml`: 20 `StoryOutline` samples per model (two briefs × 10),
at most 2 repairs, reasoning or thinking disabled for both models.

| Model | First-attempt validity | Validity after repair | Median / p95 latency | Cost per outline |
| --- | --- | --- | --- | --- |
| `gpt-6-luna` | 95% | **100%** | 16.2 s / 19.2 s | ≈ $0.0010 |
| `deepseek-flash` | 100% | **100%** | 8.6 s / 10.3 s | ≈ $0.0026 at peak rates (about half off-peak) |

**Decision:** `gpt-6-luna` is the `planner` model. `deepseek-flash` is the fallback. No
schema simplification and no mid-tier model are needed.

- Luna costs less per outline and supports JSON-schema response mode natively.
  DeepSeek has no schema mode, so the schema rides in every prompt, nearly
  doubling its input tokens. DeepSeek was twice as fast and valid on every
  first attempt, so it stays benchmarked and ready.
- The fallback is a config change: point the `planner` role at
  `deepseek/deepseek-flash` with `thinking: {type: disabled}`, as in
  `configs/benchmark-g0.yaml`. Use it if Luna becomes unavailable, its validity
  drops, or its price rises. The gateway does not fail over automatically.
- Family separation: with an OpenAI planner, the judge and critic roles must
  come from another family, such as DeepSeek. If the planner falls back to
  DeepSeek, the judge and critic assignments must change too.
- Caveat: 20 samples per model. With no failures, the 95% lower confidence
  bound on validity is about 86%, not 95%. Every run report shows per-model
  validity; revisit this decision if it falls below the threshold in later
  phases. Raising `repeat` to 30 per brief (60 samples) would make the pass
  statistically confident for about $0.15 more.

**Manual test:** `uv run nde run configs/smoke.yaml` produces a run directory and a cost report. Running it a second time reports zero new spend.

### Phase 1 — Benchmark and instrument validation (M) — proposal Stage 0

**Goal:** a credible way to compare approaches, shown to detect real differences.

| Ticket | Work |
| --- | --- |
| P1.1 | Brief format (Pydantic) and about 40 briefs: genre, premise, protagonist, central conflict, themes, ending constraint, target length. Split into dev and test, then freeze the test split. |
| P1.2 | Rubric and failure taxonomy: engagement, coherence, emotional impact, character believability, pacing, originality; failure tags such as unearned change, dropped thread, continuity error, melodrama, generic |
| P1.3 | **Baseline A:** single prompt from premise to story |
| P1.4 | **Baseline B:** beat outline, then scene-by-scene prose. Uses the same prose model and template that every later condition will use. |
| P1.5 | Budget accounting: generation-token and call counts per condition, and an equal-budget mechanism. A cheaper condition gets extra revision passes until its budget matches the more expensive one (±10%). |
| P1.6 | Blind reader app (§3.3) and statistics module (exact binomial test, κ, bootstrap confidence intervals for rubric scores) |
| P1.7 | LLM judge panel: pairwise and rubric modes, both orders |
| P1.8 | **Instrument check:** B against **B-scrambled**, where scene order is shuffled and the prose regenerated, on 10 dev briefs. Read blind by the owner and scored by the judges. |
| P1.9 | Calibration: the owner rates 3–5 public-domain human short stories and some Baseline A/B outputs, to anchor the rubric scale |

**Original exit gate (historical; amended for attempt 2 below):**
- The owner prefers intact B over scrambled in **≥9 of 10 pairs**.
- Both judge families agree with the owner in ≥80% of those pairs (for
  G-instr: the single judge; see the judge decision below).
- The A and B pipelines run end to end within the cost caps.

**G-instr judge decision (19 October 2026):** the judge panel is one model,
OpenAI `gpt-6.1-sol` at medium reasoning effort (`configs/p1-judges.yaml`).
Only OpenAI and DeepSeek APIs are available and the stories come from both
families, so no judge can satisfy the family rules of §3.1 and §4. They are
waived explicitly in the judge config (`judges.min_families: 1`,
`judges.allow_generator_families: true`), and every run manifest and batch
report flags the waiver. In G-instr both conditions share the planner and
prose models, so self-preference cannot favour either one; revisit the
panel before Phase 3, where conditions differ in their planners.

**G-instr attempt 1 (19 October 2026): not passed.** Run
`20261019T160211Z-p1-instrument-5222de` (10 B and 10 B-scrambled stories on the
dev briefs), batch `p1-instrument`, read from a skim with optional rubric marks
(pre-registered amendment).

| Criterion | Threshold | Result |
| --- | --- | --- |
| Owner prefers intact B | ≥ 9 of 10 | 7 of 10 (two-sided exact binomial p = 0.34) |
| Judge agrees with the owner | ≥ 80% | 70%; the judge (`gpt-6.1-sol`) preferred B in 10 of 10, 100% position-consistent |
| Pipelines within cost caps | — | yes: $0.57 in total, every run within its cap |

- Not flagged possibly unblinded: no condition guesses were given. The reader
  warned that the generating code changed 7 hours before reading; the change
  was the agent's, not the owner's.
- 7 of the 20 stories were short of the ±10% length range (1,560–1,786 words),
  split evenly between the conditions.
- **Diagnosis:** the owner saw the strange sequences while skimming but judged
  "which is better" by the quality of the writing, as they normally would: the
  prose dominated, as the fail action anticipates. The judge's perfect detection
  shows the damage is in the text. Because the prose model rewrites every scene
  for the shuffled plan, a scrambled story reads as an odd but fluent story
  rather than an obviously broken one.
- **Flaw found:** every brief used the same scene permutation, because the
  shuffle seed did not include the brief. Fixed before attempt 2.

**Decision (owner, 19 October 2026):** rerun G-instr (attempt 2) with structure
as its own measure. The owner scores **coherence** for both stories in every
pair (required); the primary metric is the share of pairs in which intact B
scores higher. The overall preference stays as a secondary signal. Stories are
skimmed in full, and the pre-registration is amended before reading. The same
approach applies to later structure-focused gates (see §4).

**G-instr attempt 2 (20 October 2026): not passed (judge); gate closed by the
pre-registered fail action.** Run `20261019T200559Z-p1-instrument-2-a640ad`
(fresh stories, seed 2410, per-brief scramble), batch `p1-instrument-2`
(coherence as the required primary score, skim reading), judge run
`20261019T202110Z-p1-judges-2-5ce6c5`.

| Criterion | Threshold | Result |
| --- | --- | --- |
| Intact B scores higher on coherence (owner) | ≥ 9 of 10 | **9 of 10**, no ties (two-sided exact binomial p = 0.021): passed |
| Judge's coherence picks agree with the owner's | ≥ 80% | **70%**: failed. Rubric-mode picks: 8 B, 0 scrambled, 2 ties |
| Pipelines within cost caps | — | yes: $0.15 for the stories, $0.40 for the judge |

- Owner mean coherence: 3.6 for B and 1.8 for B-scrambled. The owner's overall
  preference (secondary) went to B in 9 of 10. The judge's pairwise preference
  (secondary) went to B in 10 of 10, 90% agreement with the owner, 100%
  position-consistent.
- The judge never scored a scrambled story higher on coherence. Its failure
  came from two ties on the coarse 1–5 scale, plus the one pair (brief 03) in
  which the owner preferred the scrambled story.
- 4 of the 20 stories were short of the length range (1,599–1,713 words), split
  evenly between the conditions.
- Caveats: the reader warned that the generating code changed 19 hours before
  reading (the agent's scramble fix, not the owner's work). The pre-registration's
  "owner confirmed attempt 2" box was not ticked, but its terms were committed
  before reading and were the owner's own decisions.

**Decision (owner, 20 October 2026):** apply the pre-registered fail action.
The owner's blind reading, judging structure explicitly through the coherence
score, is the validated instrument. The LLM judge is **dropped from screening**:
it is not validated for structural judgement, and later gates report it for
diagnosis only. G-instr is closed, and so is Phase 1.

**Fail action:** if the owner cannot reliably detect gross structural damage, the stories are too short or the prose dominates. Increase the target length or restrict reading to targeted excerpts before going further. If the judges fail where the owner succeeds, drop judges from screening and budget more owner reading time.

**Manual test:** `uv run nde eval serve --batch p1-instrument` opens the reader on `localhost`. Complete the 10 pairs. `uv run nde eval report p1-instrument` reveals the key and prints the result.

### Phase 2 — Narrative schema v0 (M) — proposal Stage 1

**Implementation:** complete on `main`; issues #32–38 closed. The released
plan schema is version `1` (the phase's "v0" is its original design label).
G-schema reporting includes 44 registered semantic mutation categories and
four valid controls. Actual gold corrections, timing and meaningful-trajectory
review are preserved in the [frozen G-schema packet](evidence/g-schema-2026-10-27/decision.json).
See [the evidence workflow](schema-evidence.md).

**G-schema result (27 October 2026): passed.** The owner approved the final
annotations for *The Gift of the Magi* (2,057 words, 48.8 minutes),
*The Lady, or the Tiger?* (2,708 words, 44.3 minutes), and
*An Occurrence at Owl Creek Bridge* (3,728 words, 52.1 minutes).
The shared qualitative comparison took a further 2.5 minutes, tracked separately.
All 44 mutation categories were caught. Frozen-input replay through the public
CLI reproduces the evidence bytes; the portable bundle includes source texts,
original drafts, correction history, timers, owner judgements and presentation
hashes. The selected works are [protected gold sources](../data/manifest.gschema-20261027.yaml).

This used a guided, assisted correction workflow: the owner reviewed one small
question at a time, then explicitly approved each final version. Agent preparation
was excluded from owner timers. No unprovided numeric confidence/intensity ratings
or whole-story agency curves were invented. The result establishes representation
and practical owner review for these three stories and this owner; it does not
establish story-quality gains, external-reader agreement, or later gates.

**Goal:** the smallest inspectable representation that can express meaningful differences between scenes and stories.

**Design decisions made here:**
- The **scene** is the primary unit. Scene beats are an optional ordered child list, and passages exist only as text-offset spans. This resolves proposal §13 decision 2 for now; finer resolution is added only if the gates show it is needed.
- **Observation and interpretation are separate types.** Interpretive fields carry `confidence`, allow the values `unknown` and `ambiguous`, and hold optional alternative readings.
- Trajectories (mood, tension, agency) are **coarse ordinal sequences** over three to five points with a shape enum, not floats. This avoids false precision.
- Graph structure is a **typed edge list** keyed by stable IDs: `causes`, `sets_up`/`pays_off`, `reveals`, `motif_echo`. The ordered scene list sits alongside the edge list, so neither replaces the other.

| Ticket | Work |
| --- | --- |
| P2.1 | Pydantic models: Story, Arc, Scene, Beat, CharacterState, Relationship, KnowledgeFact (with holders and reveal point), ReaderAffect, Tension, Theme, Motif, Discourse (observables), Subtext, Constraint, Edge. Export to JSON Schema. Schema version field and migration stub; preserve readable existing brief, outline and run formats. Include one complete example plan. |
| P2.2 | Deterministic validators: referential integrity; causes point backwards in time; setups precede payoffs; unpaid setups are flagged; no character uses knowledge before its reveal; character presence is consistent; hard constraints hold |
| P2.3 | Validator **mutation tests**: inject at least 20 known violation types into valid plans and require that **≥95%** are caught |
| P2.4 | **Representation tiers** for later ablation. **S:** beats only. **M:** beats plus goals/beliefs, causality and information state. **L:** all channels. Generated as projections of one schema. |
| P2.5 | Gold annotations: 3–5 public-domain short stories annotated to tier L. An LLM drafts each one and the owner corrects it; the owner's correction time per story is recorded. |
| P2.6 | Plan renderer: schema to readable Markdown/HTML "scene cards", needed for reviewing gold annotations and generated plans |

**Implementation order:** P2.1 first; then P2.2–P2.3 and P2.4. Build P2.6
before owner correction in P2.5, so the schema's usability is tested through
the renderer. Keep the ticket IDs stable even though P2.6 precedes P2.5.

**Exit gate:**
- The gold annotations show visibly different state trajectories between stories, so the schema is not flattening them.
- The validators pass P2.3.
- Correcting one 2,000–5,000-word story takes the owner **under about 2 hours**. A longer time signals excessive schema complexity.

**Fail action:** cut or merge the channels that are slowest to annotate or least discriminating. Repeat on one more story.

### Phase 3 — Core thesis test: structured planning (M–L) — proposal Stage 3, moved earlier

**Implementation:** complete on `main`; issues #39–47 closed, including
P3.0 strict readiness. Generated/imported C plans retain planning provenance
and costs; all tiers use B's shared prose contract. Real G1 additionally
requires a passed owner G-schema packet, current dev selection and committed
protocol. No real G1 result is added by this status update.

**Development decision (27 October 2026):** the owner selected **M** after
reviewing three versions of one dev brief with tier labels concealed. M scored
4/5 for coherence and 5/5 for originality; S and L each scored 3/5 and 4/5.
The owner found M meaningfully more coherent and less contrived. Both generated
dev briefs passed the final validity, length and equal-budget checks across all
tiers. Six fixed sampled intentions were accepted (M: 2 of 4 represented items;
L: 4 of 15); the remaining intentions and the reserve brief remain unreviewed.
The [immutable selection](evidence/dev-tier-selection-2026-10-27/selection.json)
binds the actual reviews and generation settings. All tiers incurred full-L
planning before projection. G1 registration and held-out evaluation remain open.

**Reading decision (30 October 2026):** the owner declined a 30-pair read on
27 October and on 30 October set G1 at four pairs, before any reading: "Four pairs
of stories is more than enough for me to determine if there is a consistent
difference." The [registered protocol](g1-four-pairs.md) uses the four already
generated dev-brief pairs. Their briefs were not used to choose the tier. The
originality decision, no average loss (margin 0), is preserved.

**Goal:** answer the first real question. **Do multi-channel plans produce better stories than beat-only plans at equal budget?**

| Ticket | Work |
| --- | --- |
| P3.0 | Gate readiness: check the exact expected brief pairs, successful generation, final story lengths and equal-budget tolerance before opening a gate batch. Reject incomplete batches for gate use and report every unresolved pair. Preserve existing non-gate run and batch formats. |
| P3.1 | **State ledger:** canon, knowledge holders, open setups, motif registry and character states, updated after each scene and summarised compactly for the prose step |
| P3.2 | **Baseline C planner:** prompted generation of a tier-L plan. A whole-plan skeleton (arcs, scene functions, ending) comes first, then channels per scene. The plan is validated against the schema, and validator failures are fed back for bounded repair. |
| P3.3 | Plan-conditioned prose: each scene is realised from its scene record plus the ledger summary, using the same prose model and template as Baseline B |
| P3.4 | **Realisation check:** review whether prose realises planned deltas (belief change, trust change, information reveal, tension shape). Report owner-reviewed realisation on a specified dev sample; any LLM estimate is diagnostic and reported separately. This separates planner failure from prose failure. |
| P3.5 | Tier ablation (S/M/L) on dev briefs, screened with validators and a budgeted owner-reviewed sample. LLM judgements are diagnostic only unless separately validated under §4. Freeze the chosen tier before the test run. This answers proposal §13 decision 1, the smallest useful representation. |
| P3.6 | **Gate run:** C (the chosen tier, first in the batch sources) against B on four frozen briefs at equal budget, admitted through P3.0. Owner blind reading requires coherence and originality scores for both stories, plus overall preference. Judges, if run, are diagnostic only. |

**Exit gate (Gate 1, the thesis):**
- C scores strictly higher than B on **coherence in at least 3 of 4 pairs**;
  ties count as non-wins. The owner's retest decides on this alone. This is a judgement of consistency by the owner, not a
  statistical test: with no real difference and no ties, chance alone gives 3 or 4
  wins about 31% of the time. Overall preference is secondary.
- For the first G1 only: C meets the pre-registered "not worse" safeguard on
  **originality**, with required scores for both stories and an explicit margin and
  analysis. On 30 October 2026 the owner dropped originality from later reviews.

**Promising but inconclusive** (for example, 2 coherence wins of 4 with no
losses): G1 has not passed. At most one follow-up is allowed: either a comparison
on fresh briefs under §4, or the diagnostic retry below. Do not add pairs to the
completed test or tune against its revealed stories.

**Fail action:**
- First diagnose using dev briefs. If owner-reviewed realisation is low, retry
  once with a stronger prose model in **both** conditions. Pre-register the
  changed comparison and use fresh held-out briefs; preserve and report the
  first result separately. If realisation is fine but coherence does not
  improve, the richer plan itself is not helping.
- If the gate still fails after the single allowed follow-up, or no follow-up
  is justified or feasible, **stop the diffusion track**. Ship the schema,
  benchmark, harness and planner as the result (proposal §14).

**Implemented workflows:** `uv run nde plan generate configs/plans-c.yaml`
writes a dev plan and cards. To realise a saved plan, configure
`baseline_c` with the paired saved-plan/provenance inputs and run
`uv run nde run <config>`, as documented in [Baseline C](baseline-c.md).
Use the [realisation packet](realisation.md) to assess actual prose adherence;
a valid plan or planned-state ledger alone does not establish it. These paid
dev commands are not required for routine implementation acceptance.

### Phase 4 — Iterative refinement planner, Candidate D (M) — real experiments require Gate 1

**Implementation:** complete on `main`; issues #48–55 closed. Channel/span
operators, global-intention planning, bounded critic refinement, control and
diversity reports, equal-budget C/D comparisons and G-D admission/decisions
are implemented. See [D](baseline-d.md) and [G-D](gd.md). A real G1 pass is
recomputed before registered G-D generation; implementation fixtures cannot
promote D as the scientific reference baseline.

**Goal:** the strongest non-trained planner. It is diffusion-inspired, uses existing LLMs, and doubles as the first version of structural editing.

| Ticket | Work |
| --- | --- |
| P4.1 | Coarse-to-fine generation: story-level arc and affect curve → scene skeleton → full scene channels, each level conditioned on the one above |
| P4.2 | **Mask-and-regenerate operators:** regenerate a span of scenes or selected channels while locked scenes, edges and constraints stay fixed. This gives structural inpainting and constraint locking (proposal §8). |
| P4.3 | Critic loop: a critic from a different model family plus the deterministic validators propose targeted issues, and only the affected units are regenerated. The number of iterations is a budget parameter. |
| P4.4 | **Controllability metrics:** locked elements stay byte-stable across revisions; requested changes such as "two episodes of renewed hope in the middle third" are checked by validators where expressible and owner review on a specified dev sample; judges are diagnostic only. Diversity across seeds is measured structurally, as edit distance over scene functions and edge sets, not over prose |
| P4.5 | Gate run: C against D at equal budget on the test briefs, owner blind |

**Exit gate:** D meets the pre-registered "at least as good" criterion against
C on owner coherence scores, with overall preference secondary and the
originality safeguard from §4, and controllability metrics meet their
pre-registered targets. D becomes the **reference baseline** for any trained
planner. Define the comparison margin and analysis before the run.

**Fail action:** D is not required for Phase 6. If it does not meet the exit
criteria, C becomes the reference baseline instead. Either way, record why
refinement did not help, since that informs the denoiser design. Later phases
use "reference baseline" to mean D if G-D passes, otherwise C.

### Phase 5 — Extraction pilot and corpus (L) — proposal Stage 2; can run in parallel with Phase 4

**Implementation:** complete on `main`; issues #56–65 closed, including
optional exemplar comparisons. The pipeline supports source-linked corpus
acquisition, author-disjoint frozen cohorts, segmentation/features, independent
model-family extraction, a resumable pilot and bounded stratified review.
The [G2 workflow](corpus-g2.md) gates real training exports and larger runs on
replayed scientific evidence. Available software is not evidence that the
50-story pilot, owner review or 300–1,000-story expansion has been completed.

**Goal:** find out which channels can be extracted reliably from human fiction, and build the training dataset needed if diffusion is pursued.

| Ticket | Work |
| --- | --- |
| P5.1 | Corpus acquisition: public-domain short fiction (prefer Standard Ebooks for clean CC0 texts, then Gutenberg). Each work has a provenance record (source, edition, licence basis, checksum, author, period, genre). Splits are made **by author**. |
| P5.2 | Segmentation: chapter and scene boundaries with text offsets, confidence and parent links. The scene splitter is evaluated against the scene boundaries in the P2.5 gold stories. |
| P5.3 | Deterministic features: dialogue proportion, sentence-length distribution, entity mentions and co-occurrence, viewpoint heuristics |
| P5.4 | LLM extraction to the schema in passes per channel group, with **at least two model families** per story on the pilot sample. Evidence spans are required, and disagreements are stored, not merged away. |
| P5.5 | Reliability analysis per channel: inter-model agreement, agreement with the owner on a stratified review sample, and agreement with the P2.5 gold stories. Each channel is classified as **reliable**, **weak** (soft target only) or **unmeasurable** (dropped from training targets). |
| P5.6 | Pilot on about 50 stories, then a cost-per-story estimate. Scale to about 300–1,000 stories only after P5.5 passes. |
| P5.7 | Optional: use extracted human plans as few-shot exemplars for the C and D planners, and screen on dev briefs with deterministic checks and owner review whether grounding in human structure reduces generic plans; judges are diagnostic only |

**Exit gate (Gate 2):**
- Core channels (plot events, causality, information state, character goals and beliefs) are rated **reliable**.
- The owner judges at least about 80% of reviewed annotations "accurate or defensible".
- Scaling cost fits the budget.

**Fail action:** restrict the training representation to the reliable channels. If the core causal and knowledge channels are unreliable, a trained planner cannot learn them from this data, so stop before Phase 6 and keep the reference baseline as the result.

### Phase 6 — Denoising feasibility (L) — proposal Stage 4; requires Gates 1 and 2

**Implementation:** tooling present on `main`; issues #66–75 are still open
and need acceptance/closure reconciliation. The codec, corruption tasks,
author-safe datasets, CPU models, paired scoring, overlap audit, bounded
sweeps, registered G3 and optional augmentation are documented below. No
scientific G3 pass or real pretrained-model result is recorded here.

**Goal:** show that plans survive corruption and recovery better than a matched autoregressive model, before any conditional generation is attempted.

**Design decision (proposal §13 decision 4):** start with **discrete masked denoising over serialised schema tokens**, not continuous latent diffusion. The plan is already structured and editable, discrete masking maps directly onto inpainting and constraint locking, and it avoids the latent-decodability risk named in §5.4. Continuous latents become a later comparison, not the first bet.

**Main risk: data scarcity.** A thousand short stories with 5–8 scenes each is a very small training set. Mitigations, which are tested and not assumed:
- Initialise from a small pretrained language model fine-tuned with a masked-denoising objective, rather than training from scratch.
- Use window- and scene-level training examples in addition to whole stories.
- Treat augmentation with LLM-generated plans as a separate ablation, because it risks learning the generator's clichés (proposal §5.3).

| Ticket | Work |
| --- | --- |
| P6.1 | Plan serialisation and tokenisation: a canonical field order, typed slots, edges as ID references, and variable scene counts represented with length tokens or empty slots |
| P6.2 | Masking schedules: random fields, whole scenes, whole channels, contiguous spans (the middle act) |
| P6.3 | Denoiser prototype, a fine-tuned small pretrained model with a masked-denoising objective |
| P6.4 | **Matched AR comparison model:** same base model, same data, same parameter count, trained left to right with infilling. This is required to attribute any gain to denoising (proposal §9 item 8). |
| P6.5 | Reconstruction metrics per masking rate (10–70%): field accuracy per channel, edge recovery F1, validator pass rate, and an owner-reviewed "meaning preserved" assessment on a specified sample; any judge score is diagnostic only |
| P6.6 | Memorisation check: nearest-neighbour similarity of reconstructions and samples against the training plans |

**Implemented scope and limits:**

- [Codec/tasks](reconstruction-codec.md): a fixed 260-token byte vocabulary,
  canonical typed slots and references, explicit list lengths, and hard/soft/
  excluded policies. Retained views are not complete tier-L annotations.
  Masking supports fields, scenes, channels and contiguous canonical field
  spans; requested and actual rates are recorded separately.
- [Datasets](reconstruction-data.md): whole-story/window/scene examples and
  separate evaluator-only author-disjoint views. Partial windows disable
  graph-channel targets when external context is unavailable. They do not
  teach omitted cross-window edges as negative facts.
- [Local training](reconstruction-training.md): the supported native backbone
  is a residual output-query attention/MLP block with byte/position embeddings.
  Local pretrained weights can be registered with provenance; unrelated model
  architectures need an adapter. Random initialization is test-only. Denoising
  and causal AR use matched base identities, parameter counts and declared
  resource caps; unequal actual inference work is reported, not called equal.
- [Scoring/G3](reconstruction-evaluation.md) and
  [sweeps](reconstruction-experiments.md): masked-field and long-range edge
  recovery, full failure denominators, overlap audits, 25/50/100% author-grouped
  subsets and durable training/inference token/time limits. One G3 protocol
  covers one audited training population; owner meaning review is separate and
  enforced when required by the registered protocol.
- [Augmentation](reconstruction-augmentation.md): a separately reported local
  training-label transformation ablation. It makes no provider calls and does
  not claim new literary structure or independent human stories. Broader
  LLM-generated-plan augmentation remains future experimental work.

**Exit gate (Gate 3):**
- At 30–50% masking, the denoiser's reconstructions pass the validators at a pre-registered rate.
- On the long-range links (setup/payoff, causal and knowledge edges) it beats the matched AR model by a pre-registered margin.

**Fail action:** if denoising does not beat AR on long-range structure, the specific advantage of diffusion is not shown. Stop and report. D or C stays the planner.

### Phase 7 — Conditional diffusion planner, Candidate E (L) — proposal Stage 5

**Implementation:** P7.1–P7.4 tooling and P7.5 reporting (#76–83) are
implemented and verified on CPU fixture checkpoints only: the Candidate E run
pipeline, D/E inpainting and the [condition-adherence, diversity and
training-similarity report](conditional-report.md) are available. The
[registered G-E preparation](ge.md) (#84: readiness, protocol, bound dev run and
owner review packet) and the [G-E decision](ge.md#g-e-decision) (#85: criteria,
iteration history, reference retention and Phase 8 eligibility) are implemented,
so Phase 7 tooling is complete. No G-E protocol is registered and no G-E outcome
is recorded: the scientific gate remains pending on a genuine G3 pass.
Real E experiments still require a genuine G3 pass; reconstruction software or
synthetic criteria are insufficient, and E refuses non-fixture checkpoints
without passed G3 evidence.

**Goal:** generate new whole-story plans from briefs.

| Ticket | Work |
| --- | --- |
| P7.1 | Conditioning on brief attributes (genre, themes, ending constraint, length); classifier-free guidance or an equivalent, with a condition-adherence test set |
| P7.2 | Generation from a fully masked plan with locked constraints; inference-time validator checks, with rejection or targeted re-masking of invalid units |
| P7.3 | Hybrid decoding: E proposes the structure, and an LLM fills in the free-text fields (event descriptions, subtext prose) where the tokenised form is too coarse |
| P7.4 | Inpainting and alternatives using the same operator interface as P4.2, so that D and E are interchangeable behind one API |
| P7.5 | Diversity and conventionality: structural diversity across seeds; similarity to training plans and to the reference baseline's outputs |

**Exit gate:** E meets the pre-registered validator pass-rate comparison
against the reference baseline, and a budgeted owner-reviewed dev sample
shows E competitive on structure under a defined criterion. Judges are
diagnostic only unless separately validated under §4. Only then commit to
the larger owner reading workload in Phase 8. Allow at most one development
iteration if this gate fails, then stop.

### Phase 8 — End-to-end evaluation and decision (M) — proposal Stage 6

**Implementation:** not started. The existing reader and gate infrastructure
are reusable foundations; the E comparison and final decision are future work.

| Ticket | Work |
| --- | --- |
| P8.1 | Final comparison at equal budget with the same prose model. The owner reads E against the reference baseline (D if G-D passed, otherwise C) as the primary comparison, and that baseline against B for context. Coherence is primary; overall preference is secondary and originality is a required safeguard. Plan a separately pre-registered, fixed-size comparison on about 80 fresh held-out briefs under §4; prepare and freeze that benchmark version before generation. |
| P8.2 | Error analysis by failure tag; cost per story for each condition |
| P8.3 | Decision report against proposal §10's core criterion: does E repeatably and meaningfully beat the strongest equal-budget structured baseline? |

Proposal Stage 7 (novel length, graph diffusion, tooling for revision) is **out of scope** for this plan. It is planned separately only if Phase 8 justifies it.

## 6. Gate summary

The dated G0, G-instr and G-schema outcomes are recorded scientific decisions in
this plan. For the remaining rows, implemented gate tooling does not establish
a result: retain pending status until genuine bound evidence is replayed and
the outcome is recorded.

| Gate | Phase | Question | Default pass (confirm in pre-registration) | If it fails |
| --- | --- | --- | --- | --- |
| G0 (passed, 18 Oct 2026) | P0 | Can low-cost models produce our structures? | ≥95% schema validity after repair | Simplify the schema, or use a mid-tier planner model |
| G-instr (closed, 20 Oct 2026) | P1 | Does the instrument detect known damage? | Historical: owner ≥9/10 coherence wins, judge ≥80% agreement | Owner 9/10; judge 70%. Closed through the fail action: owner coherence is the instrument, judges diagnostic only |
| G-schema (passed, 27 Oct 2026) | P2 | Is the schema expressive and practical? | 44/44 categories caught; all three owner reviews under two hours; meaningful qualitative differences confirmed | Cut channels |
| **G1** (failed/inconclusive, 30 Oct 2026; 0 of 4 coherence wins, [evidence](evidence/g1-2026-10-30/README.txt); retest 2 of 4, [evidence](evidence/g1-retest-2026-10-30/README.txt)) | P3 | **Do multi-channel plans help?** | C higher on owner coherence in at least 3 of 4 pairs; ties are non-wins; originality safeguard met (first G1 only) | At most one fresh-held-out follow-up (larger confirmation or diagnostic retry), then stop the diffusion track |
| G-D (not recorded) | P4 | Does refinement help? | D meets pre-registered coherence comparison against C, originality safeguard and controllability targets | C becomes the reference baseline |
| **G2** (not recorded) | P5 | Can core channels be extracted? | Core channels reliable; ≥80% defensible | Reduce the representation, or stop before training |
| **G3** (not recorded) | P6 | Does denoising beat AR on long-range structure? | Pre-registered validator target and positive long-range margin over matched AR | Stop; D or C is the result |
| G-E (not recorded) | P7 | Is E competitive enough to evaluate? | Pre-registered validity comparison and owner-reviewed dev structure criterion against the reference baseline | Iterate once, then stop if still below criteria |
| **Final** (future) | P8 | Does diffusion deliver a real gain? | E beats the reference baseline on owner coherence at the fixed planned n; originality safeguard met | A valid negative result; publish the internal report |

## 7. Risk register

Mapped to the uncertainties in proposal §9.

| Risk | § | Where it is tested | Mitigation |
| --- | --- | --- | --- |
| Low-cost models are unreliable on nested structured output | — | P0.7 | Per-model validity metrics; smaller sub-schemas; a mid-tier fallback for one role only |
| Single-evaluator bias or broken blinding | §9.10 | P1.8, every gate | Condition-guess tracking, 48-hour delay, sealed keys, pre-registration |
| The prose model dilutes the plans | §9.7 | P3.4 | Owner-reviewed realisation reported separately from coherence and overall preference |
| Unvalidated judges steer variant selection | §9.10 | P3.5, P7 | Deterministic checks and owner-reviewed dev samples; judges diagnostic until separately validated |
| Gate inputs or sample size change after results are known | §9.10 | P3.0, every gate | Exact pair, length and budget checks before reading; fixed n, explicit tie rules and fresh held-out follow-ups |
| Annotator clichés and bias | §5.3 | P5.4–P5.5 | Two or more model families, retained disagreements, owner review, gold stories |
| Labels that cannot be measured | §9.1 | P5.5 | Reliable/weak/unmeasurable classification; weak channels used as soft targets only |
| Too little data for diffusion | §9.9 | P6 | Pretrained initialisation, window-level samples, augmentation tested as an ablation |
| Gains come from compute, not method | §9.8 | P1.5, P6.4, P8.1 | Equal-budget accounting; a matched AR model |
| Conventional or generic plans | §9.5 | P4.4, P7.5 | Structural diversity and similarity-to-training metrics; originality rubric |
| Better global shape costs distinctive scenes | §9.6 | P3.6, P8.1 | Rubric dimensions reported separately, not only as an overall preference |
| Scope creep into product UI | — | — | The research-first decision; the plan editor is deferred until after G1 |

## 8. Where each open design decision in proposal §13 is resolved

| # | Decision | Resolved in |
| --- | --- | --- |
| 1 | Smallest useful state | P3.5 tier ablation, refined by P5.5 reliability |
| 2 | Primary unit | P2: the scene (beats optional); revisited only if gates fail on local pacing |
| 3 | Encoder without expensive labels | P5 (LLM extraction with validation) and P6 (pretrained initialisation) |
| 4 | Continuous or discrete first | P6: discrete masked denoising first |
| 5 | Preserving causal and knowledge graphs | P2.2 validators as hard constraints at inference (P7.2); edge-recovery metrics (P6.5) |
| 6 | Variable scene counts | P6.1 length tokens or empty slots; tested in P7 |
| 7 | Minimum corpus | P5.6 and the P6 learning curves (train on 25%, 50% and 100% of the data) |
| 8 | Hard versus soft constraints | P2: knowledge, causality, locked canon and ending are hard; affect and style are soft |
| 9 | Fair revision budget | P1.5 equal-budget mechanism |
| 10 | What justifies trained diffusion | G1 + G2 + G3 together |

## 9. Still open

Implementation through Phase 7 does not resolve the following scientific and
operational work. Keep missing evidence pending rather than repeating fixture
checks as a substitute for owner judgement.

1. **Research gates:** G-schema and the sampled dev review/M selection are complete.
   [G1 and its one permitted retest did not pass](g1-four-pairs.md), so the diffusion
   track stops. Any return to it needs a new owner decision, and would then follow with independently registered G-D/G2 work and
   genuine G1/G2 prerequisites for G3. Record each actual outcome; existing gate
   commands already enforce admission.
2. **Issue close-out:** Phase 6 and 7 implementation issues (#66–85) are closed.
   Tooling on `main` is not a reason to manufacture a scientific result.
3. **Fresh held-out briefs, before a G1 follow-up or Phase 8:** plan authorship,
   owner curation, benchmark versioning and independent freezing without
   changing the current test split. Decide the fixed sample size and stopping
   rule before generating or reading those outputs.
4. **Real model and compute selection:** choose compatible local pretrained
   weights (or scope a new adapter), establish measured memory/time capacity
   and register matched budgets before substantive training. CPU fixture
   success does not determine a useful model size or GPU requirement.
5. **External panel, before public claims:** decide whether and when to add
   readers. Current results describe the owner's judgement only.

**Resolved:** GitHub remote and Actions CI are established. Brief authorship
is LLM drafting followed by owner curation (see [docs/briefs.md](briefs.md));
the initial 30 test briefs are frozen. LLM judge screening is suspended under
the G-instr fail action; separate validation is needed only if screening is
to resume.
