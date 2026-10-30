# Continuity stories

G1 and its one retest did not pass, so the diffusion track has stopped. This is a
separate, exploratory attempt to fix the defects the owner found while reading
generated stories: flowery language, characters appearing without introduction,
characters knowing things they should not, and unrealistic timing. It does not
reopen G1.

```powershell
uv run nde continuity brief-01
```

The command takes one development brief, using the models and budget in
[`configs/continuity-dev.yaml`](../configs/continuity-dev.yaml) (`--config` chooses
another). It refuses a held-out brief before any call, and so does any config that
names one for this pipeline. Run it with `OPENAI_API_KEY` set. It stops at the
config's `budget.per_run_usd`.

## What it does

1. **Plan.** Before any prose, the planner makes a sheet for each character:
   name, appearance, back-story, and when and how they enter. It then plans the
   scenes.
2. **Write.** Each scene is written from the sheets and short, concrete plain-style
   rules. Length control works as it does for every other story.
3. **Check.** A model reads the chosen story with numbered paragraphs and reports
   what it finds, citing paragraph numbers. Plain code then compares that reading
   with the plan and decides what to flag.

Automated checks guide repair and never establish story quality. Only the owner's
reading does that.

## What it saves

The story's folder, `stories/continuity/<brief>/r1/` in the run directory, holds:

| File | Contents |
| --- | --- |
| `story.md` | the story |
| `summary.md` | a readable summary: length, cost, characters and problems |
| `continuity.json` | the same record for tools, with the full flagged-problem list and cost |
| `character-sheets.json`, `plan.json` | the sheets and the whole plan |
| `style-rules.md` | the plain-style rules the writer received |
| `audits/round-<n>/<check>.json` | each check's reading of the story |

Every flagged problem has the same three fields: `problem_type`, `passage` (the
story's exact words) and `reason`.

## Checks

| Problem type | What it flags |
| --- | --- |
| `character_introduction` | a named person with no sheet, no introduction, a different entry scene from the plan, or a planned person who never appears |
