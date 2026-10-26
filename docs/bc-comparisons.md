# Equal-budget B-versus-C development comparisons

`configs/stories-bc-equal-budget.yaml` compares tier-L C and B on two dev
briefs, with the same prose role and `scene_prose@1`. It reuses the established
token-budget revision mechanism. Review the model/pricing settings and provide
the keys named in the config before any live call. To run:

```powershell
uv run nde run configs/stories-bc-equal-budget.yaml
uv run nde report runs/RUN
```

Every run writes `comparison.json` and a "Final selected comparison" section
in `report.md`. They enumerate the expected pairs, including absent or failed
stories. Each selected story has a path, content checksum, final word count,
revision count, attributable calls/tokens, list-price token cost and actual new
provider spend. Cache hits keep experimental budgets and add no new spend.
Matching records show before/after budgets, tolerance, pass limits, overshoot
and interrupted revisions. The previous final story and partial revision
artifacts remain available if a provider or hard cap interrupts revision.

A pair is comparable only when both final artifacts exist, generation and
length checks succeed, recorded budgets match attributable calls and matching
is complete. A successful run exit alone is insufficient evidence of a valid
comparison. Ordinary non-gate batch generation keeps its existing behavior;
gate admission must use strict readiness. The report uses final `story.md`,
including any selected budget revision, as do evaluation and realisation.

Manual check: run the development config, inspect `comparison.json`, and open
each reported final path. The content hash and word count must match that
file. If a pair was revised, compare it with `story.original.md`; evaluation
must display the current `story.md`. Rerunning identical cached inputs should
show the same token budgets and final prose with zero new spend.
