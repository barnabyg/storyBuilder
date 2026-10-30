# Realisation review

Create a packet for a successful story with its validated `plan.json` saved
beside the selected `story.md`:

```powershell
uv run nde realisation create runs/RUN --condition C --brief brief-01 --out runs/review-01
uv run nde realisation assess runs/review-01 owner-review.json
uv run nde realisation report runs/review-01
```

`--repetition` is zero-based and defaults to 0. The packet includes the final
prose, its checksum, the plan checksum, stable intention/scene IDs and an
explicit assessment denominator. Review belief changes, relationships, reveals
and tension intentions. This is planned-state evidence, not automatic proof of
literary accuracy. Each packet names its sampled brief; an unsampled story has
no owner realisation evidence.

The review JSON contains `packet_sha256`, a nonempty `reviewer`, `test_only`
(true for synthetic software fixtures), and `assessments`. Each assessment has
`item_id`, `rating` (`realised`, `not_realised`, or `unknown`) and a nonempty
`note`. Optional `start`, `end` and `quote` must exactly identify Unicode
offsets in the final prose. Optional `diagnostic_provenance` records a model,
prompt and other identities for diagnostic estimates; nonempty `model` and
`prompt` identifiers are required. These never count as
owner assessments. Synthetic reviews likewise cannot complete actual review.
Legacy estimates without those identifiers stay readable as unresolved diagnostics.

Whole-story revisions make scene alignment unresolved. Assess the final prose
directly; the workflow never transfers spans from superseded scene drafts.
Changed or missing source plans, story records or final prose make the packet
stale. Create a new packet for the new story; immutable assessment history is
retained in the old packet. Reports explicitly distinguish missing and unknown
assessments. Report exit 1 means stale source evidence, exit 2 malformed input,
and exit 0 a generated report, including a report pending actual owner review.

Manual check: generate a C story, create a packet, and open `packet.md`. Its
prose must equal the selected story after all length and budget revisions.
Submit a partial review and confirm the denominator stays fixed and missing
items remain pending. Replacing `story.md` must make `report` exit 1 and retain
the saved assessments in `history/`.

Baseline B packets use the saved `outline.json` and final prose. Their
denominator is explicitly zero for channels the beat outline does not
represent; the workflow never invents belief/relationship/reveal/tension
intentions to fill those gaps.
