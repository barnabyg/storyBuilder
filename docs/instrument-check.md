# Instrument check (gate G-instr)

**Status: closed 20 October 2026 through the pre-registered fail action.** In
attempt 2, the owner's coherence scoring passed at 9/10 and judge agreement failed
at 70%. The owner's coherence scoring is the instrument, and the LLM judge is
diagnostic only. The steps below are the historical procedure.

Gate G-instr asked whether the evaluation instrument (the owner reading blind,
plus the LLM judge panel) detects a known, deliberately caused degradation:
Baseline B against **B-scrambled**, the same outline told in a shuffled scene
order with freshly written prose. See P1.8 and the Phase 1 exit gate in
[the implementation plan](IMPLEMENTATION_PLAN.md), and the pre-registration in
[`benchmark/prereg/g-instr.md`](../benchmark/prereg/g-instr.md).

## What is set up

| Piece | Where |
| --- | --- |
| B-scrambled condition | `baseline_b_scrambled` (`src/nde/stories/scrambled.py`): B's outline for the brief (the same cached call), scenes shuffled with a seeded permutation that is never the identity, prose regenerated through the shared prose contract. The permutation is kept as `permutation.json` beside the story. |
| Generation config | `configs/p1-instrument.yaml`: B and B-scrambled on the 10 dev briefs, planner `gpt-6-luna`, prose `deepseek-flash`, per-run cap $2 |
| Judge config | `configs/p1-judges.yaml`: pairwise and rubric modes on batch `p1-instrument`, with one judge, `gpt-6.1-sol` at medium reasoning effort (your choice). It shares the planner's OpenAI family, so the family rules are waived explicitly and the reports say so. Check its model ID and price. |
| Pre-registration | `benchmark/prereg/g-instr.md`: hypothesis, primary metric, n = 10, thresholds and fail actions, copied from the plan for you to confirm |

## From a clean checkout to the reader

In PowerShell 7, from the folder that should hold the checkout:

```powershell
git clone https://github.com/barnabyg/storyBuilder.git
Set-Location storyBuilder
uv sync
Copy-Item .env.example .env
# Edit .env: DEEPSEEK_API_KEY and OPENAI_API_KEY (planner and judge).

uv run nde run configs/p1-instrument.yaml
$run = (Get-ChildItem runs -Directory -Filter '*-p1-instrument-*' | Sort-Object Name | Select-Object -Last 1).FullName
uv run nde eval batch p1-instrument --from $run B --from $run B-scrambled --optional-scores
uv run nde run configs/p1-judges.yaml
uv run nde eval serve --batch p1-instrument
```

Expected:

- `nde run configs/p1-instrument.yaml` prints `run succeeded: runs\...`. Its
  `report.md` lists 10 B and 10 B-scrambled stories, each with its outline,
  and the cost per condition. Stories still outside the length range after 3
  attempts are listed under **Out of length range**.
- `nde eval batch` prints `built batch 'p1-instrument': 10 pairs`.
- `nde run configs/p1-judges.yaml` prints `run succeeded`. Its report shows 40
  judge calls (10 pairs, 2 orders plus 2 stories) and a family-separation
  warning, and the verdicts
  are in `batches\p1-instrument\judges\`.
- `nde eval serve` prints the reader's address,
  `http://127.0.0.1:8770/`, and warns if the generating code changed less than
  48 hours ago.

Every step is cached: re-running any of them costs nothing for the calls
already made.

**Do not read the pairs yet.** Reading is the gate itself:
calibrate first, confirm and commit the pre-registration, and wait at least
48 hours after you last worked on the generating code. Stop the reader with
Ctrl+C; it resumes where it left off.

## Attempt 2

Attempt 1 did not pass; the plan records why. Attempt 2 uses fresh stories and
coherence as the primary measure (the pre-registration's Attempt 2 section):

```powershell
uv run nde run configs/p1-instrument-2.yaml
$run = (Get-ChildItem runs -Directory -Filter '*-p1-instrument-2-*' | Sort-Object Name | Select-Object -Last 1).FullName
uv run nde eval batch p1-instrument-2 --from $run B --from $run B-scrambled --optional-scores --primary-score coherence
uv run nde run configs/p1-judges-2.yaml
uv run nde eval serve --batch p1-instrument-2
```

In each pair, score coherence for both stories and give your preference; the
rest is optional. Then close the batch and run
`uv run nde eval report p1-instrument-2`. The report leads with the coherence
picks.

## After reading

When all 10 pairs are judged, close the batch on the reader's overview page,
then:

```powershell
uv run nde eval report p1-instrument
```

The report reveals the key, with the owner's win counts and binomial p-value,
rubric means, failure tags, guess accuracy (and the possibly-unblinded flag),
and the judge's agreement, kappa and position consistency, flagged as sharing
the planner's family. Record the
result in the Phase 1 section of the plan.
