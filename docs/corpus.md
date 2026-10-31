# Corpus acquisition and deterministic evidence

The `nde corpus` commands add version-1 corpus artifacts alongside the existing
calibration and owner-annotation formats. They do not modify the frozen brief
benchmark, approve a licence claim, or establish a scientific gate result.

## Acquire one work

Save a JSON or YAML work specification:

```json
{
  "work_id": "lamp",
  "source_id": "lamp-edition-1",
  "title": "The Lamp",
  "author_id": "ann-author",
  "author": "Ann Author",
  "source": "https://example.org/collection.txt",
  "edition": "Publisher edition 1",
  "licence_basis": "Supplied public-domain basis and jurisdiction",
  "licence_evidence": "Source statement, URL and date of the owner's check",
  "format": "text",
  "start": "START OF THE LAMP",
  "end": "END OF THE LAMP",
  "period": "unknown",
  "genre": "unknown"
}
```

```text
uv run nde corpus import work.yaml collection.txt --out data/corpus/lamp
uv run nde corpus fetch work.yaml --out data/corpus/lamp
uv run nde corpus inspect data/corpus/lamp
```

`work.json`, `source.raw`, `text.txt` and `report.html` publish together. A failed
download, extraction or publication never replaces a verified directory. Fetch
reuses a verified local copy without contacting the source again. Supply
`raw_sha256` to enforce an expected download checksum; otherwise acquisition
records the first bytes. Source IDs identify editions/files and can be shared
by works selected from the same collection. Work IDs identify the literary work
across editions. Record author aliases before freezing cohorts.

Plain-text markers must each occur exactly once, in order, on boundary lines;
only text between their lines is retained. Gutenberg defaults are `*** START OF`
and `*** END OF`. Paragraph hard wrapping is joined, matching title/byline and
leading production credits are removed, and blank paragraphs preserve explicit
chapter and break lines. For UTF-8 XHTML, use `format: xhtml`, `section_id` naming
exactly one selected work, and null `start`/`end`. Paragraphs/headings and `hr`
breaks are retained in document order. Entity/DTD declarations, invalid XML,
missing IDs and non-UTF-8 input are rejected. XHTML acquisition intentionally
requires an explicit work selection rather than guessing which front matter
belongs to a work. Licence and provenance statements are supplied evidence;
metadata checks do not establish legal eligibility.

## Freeze protected author-disjoint cohorts

```json
{
  "id": "pilot-v1",
  "works": ["corpus/lamp", "corpus/another-work"],
  "seed": 7,
  "author_aliases": {"ann-pen-name": "ann-author"},
  "protected_work_ids": ["lamp"],
  "protected_author_ids": [],
  "excluded_work_ids": [],
  "author_splits": {},
  "train_percent": 80,
  "validation_percent": 10
}
```

Work paths are relative to the configuration file. `author_splits` may explicitly
assign `train`, `validation`, `test`, `gold` or `excluded`. Otherwise the stable
author ID and seed determine a SHA-256 bucket. Gold reservations protect every
work by the canonical author. Only `train` records are eligible for training or
exemplars. Ambiguous/unknown identities, unregistered aliases sharing the same
normalized author name, duplicate work/content identities, source-ID conflicts
and incomplete edition/licence metadata prevent freezing.

```text
uv run nde corpus split freeze cohort.yaml --out data/cohorts/pilot-v1
uv run nde corpus split inspect data/cohorts/pilot-v1
```

The frozen manifest stores configuration and work-record checksums, canonical
authors, assignments, eligibility and reasons. Inspection revalidates current
inputs. Existing assignments cannot be overwritten. Expansion uses a new config
and output version with `previous` pointing at the old cohort; old works and
author assignments must remain intact. Keep original configuration/source paths
available: immutable artifacts refer to their exact inputs, rather than silently
accepting moved or edited files.

## Segment and compare source boundaries

```text
uv run nde corpus segment data/corpus/lamp --out data/segments/lamp
uv run nde corpus segment-check data/corpus/lamp data/segments/lamp
uv run nde corpus segment-compare data/corpus/lamp data/segments/lamp --gold gold.json --out data/boundary-check/lamp
```

Offsets are half-open Unicode-codepoint intervals on the exact canonical LF
text, never UTF-8 byte or UTF-16 offsets. Any line starting with the word `chapter`,
`book` or `part`, in any case, defines a chapter, so a prose line such as "Part of
the crowd turned away." also counts; lines of three or more stars, dashes or underscores
define scene breaks. A story without such markers remains one uncertain scene;
unmarked temporal or viewpoint transitions are not silently labelled as known
breaks. Each scene has a parent chapter, confidence and reason. Bounds, ordering,
non-overlap, containment and coverage are checked; only whitespace and explicit
break lines may be gaps. Consecutive breaks do not invent empty scenes.

An owner boundary reference has `version`, `work_id`, `text_sha256`,
`segmentation_sha256` (the reviewed reference revision), `approved`, `owner`,
`recorded_at`, `synthetic` (default false), `tolerance_codepoints` (default zero)
and ordered unique `scene_starts` including zero. Register the tolerance before
review. Comparisons exclude the trivial work-start boundary and match remaining
starts one-to-one to the nearest available reference within tolerance, breaking
ties toward the earlier offset. Reports retain predicted/reference denominators,
matched counts and precision/recall; an empty denominator is null. Missing,
stale, unapproved or synthetic references leave scientific evidence pending.

## Extract textual features

```text
uv run nde corpus features data/corpus/lamp data/segments/lamp --aliases aliases.json --out data/features/lamp
uv run nde corpus features-check data/corpus/lamp data/segments/lamp data/features/lamp
```

Aliases are an optional JSON map such as `{"eva": ["Éva", "Eva Smith"]}`. The
feature record and HTML report bind source, segmentation, method and alias
checksums, with per-scene and per-work observations and global source spans.
The same inputs reproduce the same outputs without services or model downloads.

- Tokens are Unicode letter words with internal apostrophes, or digit runs.
- Sentence lengths count those tokens between `.`, `!` and `?` runs, preserving
  common title abbreviations (`Dr`, `Mr`, `Mrs`, `Ms`, `Prof`, `St`, `Sr`, `Jr`,
  `etc`) and decimal points. These are measurements under a documented heuristic,
  not a universal sentence parser.
- Dialogue proportion counts tokens inside balanced straight/curly quotations
  divided by all tokens. Apostrophes inside words do not start speech. Quoted
  titles may still count; quotation is observable but speech is an interpretation.
  Unbalanced quotation or a missing token denominator yields null.
- Entity aliases use case-insensitive longest whole-name matching. Shared aliases
  retain every candidate and no resolved identity. Cooccurrence counts scenes
  containing each resolved entity pair. Without aliases the entity measurement
  is unsupported, rather than a measured zero.
- Viewpoint uses first/third-person pronoun cues with retained spans, including
  quoted speech; mixed or absent cues remain explicit. This cannot establish
  literary viewpoint or entity coreference.

Feature inspection rejects changed source, segmentation, aliases or saved values.
Old annotation spans retain their original basis and are never renormalized.
