# Instrument check (gate G-instr)

Gate G-instr asks whether the evaluation instrument (the owner reading blind,
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
| Judge config | `configs/p1-judges.yaml`: pairwise and rubric modes on batch `p1-instrument`. **You choose the panel**: two families other than openai and deepseek. The committed proposal (Anthropic Haiku 5.5 and Mistral Small) needs its model IDs, endpoints and prices checked. |
| Pre-registration | `benchmark/prereg/g-instr.md`: hypothesis, primary metric, n = 10, thresholds and fail actions, copied from the plan for you to confirm |

## From a clean checkout to the reader

In PowerShell 7, from the repository root:

```powershell
git clone https://github.com/barnabyg/storyBuilder.git
Set-Location storyBuilder
uv sync
Copy-Item .env.example .env
# Edit .env: DEEPSEEK_API_KEY, OPENAI_API_KEY, and the judge panel's keys
# (ANTHROPIC_API_KEY and MISTRAL_API_KEY for the committed proposal).

uv run nde run configs/p1-instrument.yaml
$run = (Get-ChildItem runs -Directory -Filter '*-p1-instrument-*' | Sort-Object Name | Select-Object -Last 1).FullName
uv run nde eval batch p1-instrument --from $run B --from $run B-scrambled
uv run nde run configs/p1-judges.yaml
uv run nde eval serve --batch p1-instrument
```

Expected:

- `nde run configs/p1-instrument.yaml` prints `run succeeded: runs\...`. Its
  `report.md` lists 10 B and 10 B-scrambled stories, each with its outline,
  and the cost per condition. Stories still outside the length range after 3
  attempts are listed under **Out of length range**.
- `nde eval batch` prints `built batch 'p1-instrument': 10 pairs`.
- `nde run configs/p1-judges.yaml` prints `run succeeded`. Its report shows 80
  judge calls (10 pairs, 2 judges, 2 orders plus 2 stories), and the verdicts
  are in `batches\p1-instrument\judges\`.
- `nde eval serve` prints the reader's address,
  `http://127.0.0.1:8770/`, and warns if the generating code changed less than
  48 hours ago.

Every step is cached: re-running any of them costs nothing for the calls
already made.

**Do not read the pairs yet.** Reading is the gate itself (the next ticket):
calibrate first, confirm and commit the pre-registration, and wait at least
48 hours after you last worked on the generating code. Stop the reader with
Ctrl+C; it resumes where it left off.

## After reading

When all 10 pairs are judged, close the batch on the reader's overview page,
then:

```powershell
uv run nde eval report p1-instrument
```

The report reveals the key, with the owner's win counts and binomial p-value,
rubric means, failure tags, guess accuracy (and the possibly-unblinded flag),
and each judge family's agreement, kappa and position consistency. Record the
result in the Phase 1 section of the plan.
