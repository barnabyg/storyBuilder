# Model smoke benchmark (gate G0)

Gate G0 asks whether low-cost models can reliably produce the project's nested
structures. A model passes when it reaches **at least 95% schema validity
after repair** on a moderately nested schema. See Phase 0 in
[the implementation plan](IMPLEMENTATION_PLAN.md).

## What it measures

`configs/benchmark-g0.yaml` asks every candidate model to plan story outlines
that must validate against `StoryOutline` (`src/nde/schema/benchmark.py`). The
schema is shaped like the future narrative schema: nested objects, bounded
lists, enums, optional fields, and ID references that must resolve across the
document (scenes point at defined characters, locations and earlier scenes).
Invalid responses get up to `structured.max_repairs` repair requests carrying
the validation errors.

The run report's **Benchmark (gate G0)** table has one row per candidate:

| Column | Meaning |
| --- | --- |
| Calls | outlines requested (a call and its repairs count once) |
| First-attempt validity | share valid with no repair |
| Validity after repair | share valid after at most `max_repairs` repairs; **the G0 metric** |
| Repair rate | share that needed at least one repair |
| Latency median / p95 | per call, all attempts included (p95 is nearest-rank) |
| Cost per 1k output tokens | list-price token cost divided by output tokens |
| Total cost | list-price token cost of every attempt, cached or not |
| Meets G0 | validity after repair at or above `benchmark.threshold` |

Costs in the table are at list price even for cache hits, so a cached re-run
reports the same table; the **Cost** section shows what the run actually spent.

## Running it (project owner)

1. Check each candidate's model ID and each price in
   `configs/benchmark-g0.yaml` against the provider's current lists. The
   `gpt-6-luna` ID and price were checked against OpenAI's model page on
   18 October 2026.
2. Put the API keys in `.env` (see `.env.example`): `DEEPSEEK_API_KEY`,
   `OPENAI_API_KEY`.
3. Set `budget.per_run_usd` to what you are willing to spend.
4. Run:

   ```powershell
   uv run nde run configs/benchmark-g0.yaml
   ```

5. Open `report.md` in the printed run directory. A second run makes no
   calls and costs nothing; it reads every response from the cache.
6. Record the G0 decision, and any fallback (a simpler schema shape, or a
   mid-tier model for the `planner` role only), in
   `docs/IMPLEMENTATION_PLAN.md`.

The number of samples per model is the sum of each call's `repeat` (20 by
default: two briefs, ten repetitions each). Repetition *i* uses seed + *i*
for every candidate, so candidates see the same seeds. Increase `repeat` for a
tighter estimate: at 20 samples, one invalid outline is 5 percentage points.

## How the per-run cap protects you

Before every request, including repairs, the gateway reserves an estimate of
its cost: input tokens from the prompt length and output tokens from the
candidate's `max_tokens` (or `max_completion_tokens`). If the reservation would take the run past
`budget.per_run_usd` (or the day past `budget.per_day_usd`), the request is
not made and the run stops with `aborted: budget`. Every request is checked,
however many repairs the responses need.

The estimate is conservative but not a guarantee. Output is bounded, because
providers stop at `max_tokens`. Input is estimated at 3 characters per token,
which is generous for English, but unusually token-dense text can exceed it.
So the cap can be overshot by at most the underestimated input cost of the
last request. That is small next to the cap when `max_tokens` is set; leave
some headroom if the cap is a hard limit for you. Candidates take turns
for each repetition, so a run cut short by its cap still has comparable
samples for every candidate, and its report still shows the table for what
was collected. Re-running after raising the cap reuses every cached response
and only pays for the rest.

Set `max_tokens` on every candidate, or `max_completion_tokens` for OpenAI
reasoning models such as `gpt-6-luna`, which reject `max_tokens` in Chat
Completions. Without either, the estimate assumes
`budget.assumed_output_tokens` (1,024 by default).

## Adding a candidate model

1. If the provider is new, add it under `providers` with its `base_url`,
   `api_key_env` and `json_schema` (true only if it accepts
   `response_format: {type: json_schema}`), and add its key to `.env` and
   `.env.example`.
2. Add the model under `benchmark.candidates` with the same `params` as the
   other candidates, so the comparison stays fair.
3. Add its price under `prices` as `provider/model`. A local model gets an
   explicit zero price.
4. Run the benchmark again: existing candidates come from the cache, so only
   the new model costs anything.

For a local model, uncomment the `local` provider and candidate (see
[Local models](experiments.md#local-models)).
