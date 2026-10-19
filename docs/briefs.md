# Briefs

A **brief** is the benchmark's unit of input: what story to write. Every
condition writes one story per brief, and blind comparisons pair stories by
brief. See P1.1 and §4 (Splits) in [the implementation plan](IMPLEMENTATION_PLAN.md).

## Format

One YAML file per brief, at `benchmark/<split>/<id>.yaml`:

```yaml
id: brief-01                # stable; must match the file name
split: dev                  # dev | test; must match the folder
genre: literary fiction
premise: >-                 # the situation the story starts from
  On the night her late husband's fishing boat is finally sold, ...
protagonist:
  name: Ada Venn
  description: a widowed former keeper in her sixties ...
central_conflict: >-
  Ada must choose between exposing the buyer ... and letting ...
themes: [grief, loyalty, letting go]   # one to four
ending_constraint:
  kind: bittersweet         # happy | tragic | bittersweet | ambiguous | twist | open
  detail: The boat is lost, but Ada keeps her neighbour's trust ...
target_words: 2000          # optional; 1,000 to 8,000 (default 2,000)
min_scenes: 5               # optional; 2 to 12 (default 5)
max_scenes: 8               # optional; 2 to 12 (default 8)
```

The model is `nde.schema.brief.Brief`. Missing fields, unknown fields, an
out-of-range target length and a scene range with `min_scenes` above
`max_scenes` are all rejected, with the field named.

Brief IDs are unique across both splits.

## Splits

- **dev** (`benchmark/dev/`): ten briefs, used freely during development.
  Dev briefs may be edited at any time; the dev split is not frozen.
- **test** (`benchmark/test/`): the held-out briefs. Outputs for test briefs are
  generated only for gate runs and read only blind. The test split is frozen
  once curated.

## Checking briefs

```powershell
uv run nde briefs check
```

It validates every brief and every frozen split and names each problem. It
exits 0 when everything passes and 1 otherwise. `nde-verify` runs the same
check on the committed briefs (in the tests gate).

## Freezing a split

```powershell
uv run nde briefs freeze test
```

Freezing records the SHA-256 of every brief in the split in
`benchmark/frozen.json`, the same way `prompts/checksums.json` records prompt
templates. After that, editing, adding or removing a brief in that split fails
the check, naming the brief. To see this, change one character in the frozen
fixture split and check it:

```powershell
uv run nde briefs check --dir tests/fixtures/frozen-split
```

It fails with `test/brief-91 changed after the test split was frozen`; restore
the file with `git checkout tests/fixtures/frozen-split`. A frozen split cannot
be refrozen over changes:
restore the briefs instead. Runs record the split's checksum in their
manifest, so a gate result names exactly the briefs it used.

## Drafting and curating briefs

Briefs are drafted by an LLM and curated by the owner (plan §9, open item 3).
Drafting is an ordinary run, so it is cached, costed and capped:

```powershell
uv run nde run configs/draft-briefs.yaml
uv run nde briefs drafts <run dir> --out benchmark/drafts
```

`configs/draft-briefs.yaml` asks the `planner` model for four batches of ten
briefs with the versioned prompt `brief_draft@1`, as structured output
validated against `BriefDrafts`. `nde briefs drafts` writes each drafted brief
to its own file under `benchmark/drafts/` (gitignored). Drafts never land in a
split by themselves.

To curate:

1. Read each draft. Edit it, or delete it.
2. Aim for variety across genre, ending kind and conflict type, and avoid
   near-duplicates of existing briefs. The config's `guidance` variable
   steers a drafting run, for example by listing premises to avoid.
3. For each draft you keep, add `id: brief-NN` (the next free number) and
   `split: dev` or `split: test`, and move it to `benchmark/<split>/<id>.yaml`.
4. Run `uv run nde briefs check`.
5. For the test split, run `uv run nde briefs freeze test` once all 30 briefs
   are in, and commit the briefs and `benchmark/frozen.json` together.

The ten committed dev briefs were drafted by an LLM (Claude, while
implementing this ticket) rather than through the drafting config, and are
committed for the owner to review.
