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
   scenes, recording for each how much story time passes and how long its key events
   should realistically take.
2. **Write.** Each scene is written from the sheets and short, concrete plain-style
   rules. Length control works as it does for every other story.
3. **Check.** A model reads the chosen story with numbered paragraphs and reports
   what it finds, citing paragraph numbers. Plain code then compares that reading
   with the plan and decides what to flag.
4. **Repair.** For each flagged passage, the writer gets the sheets, the plan, the
   whole story and the problems, and supplies new text for that passage alone. Code
   splices it in, so every other word stays as it was. The checks then run again. This
   repeats up to `repair_tries` times (a condition parameter, 2 by default) and stops
   early if nothing is flagged or no text changes. The writer may give a passage back
   unchanged if it is fine or any fix would make the story worse. Every try goes
   through the same budget. Problems still found by the last check are reported as
   unresolved.

Automated checks guide repair and never establish story quality. Only the owner's
reading does that.

## What it saves

The story's folder, `stories/continuity/<brief>/r1/` in the run directory, holds:

| File | Contents |
| --- | --- |
| `story.md` | the story, after any repairs |
| `story.unrepaired.md` | the story as first written |
| `summary.md` | a readable summary: length, cost, characters, problems and repairs |
| `continuity.json` | the same record for tools: the first check's `flagged_problems`, `flags_by_round`, `unresolved_problems`, cost, and `finished`, which stays false if a budget stop ended the run early |
| `repairs.json` | each change, with the problems it addressed and the paragraph before and after |
| `character-sheets.json`, `plan.json` | the sheets and the whole plan |
| `style-rules.md` | the plain-style rules the writer received |
| `knowledge.json` | scene by scene, what each character acted on and how they learned it |
| `audits/round-<n>/<check>.json` | each check's reading of the story |

Every flagged problem has the same three fields: `problem_type`, `passage` (the
story's exact words) and `reason`.

## Checks

| Problem type | What it flags |
| --- | --- |
| `character_introduction` | a named person with no sheet, no introduction, a different entry scene from the plan, or a planned person who never appears |
| `impossible_timing` | an event the prose makes too quick or too slow, or a scene whose stated time strays from the plan's clock |
| `unexplained_knowledge` | a character acting on a fact the story gives them no way to know, or learns only later |

The clock check uses the plan's range for a planned key event, and the checker's own
realistic range for any other event. A scene's stated time may differ from the plan
by a factor of two either way, plus five minutes.

The knowledge check accepts a fact only if the reading cites where it came from: a
paragraph no later than the one where the character acts on it, or a numbered item
on their sheet's list of what they know at the start.
