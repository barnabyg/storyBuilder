# AGENTS.md

## Agent skills

### Issue tracker

Issues live in GitHub Issues for `barnabyg/storyBuilder`, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Project hygiene

For hygiene cycle status, retries, setup or disabling its startup trigger, see `docs/agents/project-hygiene.md`.

## Guided owner review

For manual testing or qualitative judgement that needs the owner, use the
one-question-at-a-time approach that worked in the G-schema review:

1. Prepare the inputs, drafts and automated checks first. Present clean,
   readable material; keep technical files and bookkeeping agent-side.
2. Review one item at a time. Let the owner read or try it, then ask one short,
   concrete question per turn and wait for the answer. Reveal the next step
   when needed, rather than handing over the whole test plan.
3. Use brief summaries for factual checks and open questions for interpretation
   or feelings. Accept yes, no, uncertainty or a plain-English correction.
   When something is unclear, show the relevant source context before continuing.
4. Preserve the owner's words and the scope of each answer. Distinguish facts
   from interpretations, character emotions from reader reactions, and action
   from a sense of control. Keep unprovided ratings and uncertain details unknown.
5. Apply feedback as versioned corrections and run the relevant checks. At an
   item checkpoint, show a short corrected summary and ask what is wrong or
   missing. Present remaining material corrections before requesting approval
   of the exact final version; approval of one claim covers only that claim.
6. For comparisons, summarise the agreed patterns in plain language and ask
   whether the differences are meaningful to the owner.
7. When measuring review effort, begin on explicit readiness, pause for breaks
   and agent work, and record actual intervals. Preserve the inputs, feedback,
   approved versions and required replay evidence. Finish with the result and
   any remaining owner work clearly stated.
