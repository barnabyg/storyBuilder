# Source-linked annotations

This is a small Phase 2 workflow for three to five stories, using the existing
gateway, cache, ledger, versioned prompts and call provenance. It does not run
the later bulk extraction pipeline. Tooling acceptance is automated; the
owner's literary evidence remains a separate, pending G-schema requirement.

Import a local public-domain source through the provenance workflow first.
The manifest must record its checksum, edition and licence basis. Drafting
rechecks the raw bytes and derives an exact normalized LF text checksum for
Unicode codepoint evidence spans. No source is downloaded by annotation commands.

```powershell
uv run nde calibration import owl-creek-bridge path/to/source.txt --manifest data/manifest.calibration.yaml
uv run nde annotate draft owl-creek-bridge --manifest data/manifest.calibration.yaml --config configs/annotation.yaml --out .nde/annotations/owl-creek-bridge
uv run nde annotate status .nde/annotations/owl-creek-bridge
```

Copy and configure `configs/annotation.yaml`: set the annotator provider/model,
current prices and hard run/day caps before generation. Keys stay in environment
variables. The prompt is checksum-versioned as `annotate_l@1`. Annotation requests
use the complete tier-L schema. A single `max_repairs + 1` attempt limit covers
both syntax and semantic failures. Every completion, error, token count, cost,
cache flag and reported model is retained in `session.json`. `original.txt`
retains the first raw model response even when repairs follow. `draft/` contains
the accepted plan, diagnostics and Markdown/HTML cards. These are **unreviewed**.

Repeat the exact draft command to resume; changed source, config, prompt or schema
inputs are rejected. Saved attempts consume the original repair limit and saved
spend consumes the original run cap. An accepted saved response can finish
rendering after an interruption without another provider call. Identical calls
in another session use the shared cache with zero new spend. Exhausted sessions
cannot gain extra attempts merely by restarting. Use a new directory and an
explicitly revised configuration for a new experiment.

Run totals are also persisted transactionally in the shared ledger, using the
session ID, before/after each provider call. Resume requires the same ledger and
cache paths. An in-flight call is replayed only from cache: if its response was
not safely cached, the session becomes `recovery_required` and makes no further
provider calls. Preserve that session and reconcile its provider receipt before
starting another experiment. A crash before settlement can retain a conservative
reservation in the ledger; it is unresolved accounting, not measured token cost.

Record only active owner correction time. Start immediately before reading or
editing, pause for idle time, generation and breaks, then resume explicitly:

```powershell
uv run nde annotate timer .nde/annotations/owl-creek-bridge start
uv run nde annotate timer .nde/annotations/owl-creek-bridge pause
uv run nde annotate timer .nde/annotations/owl-creek-bridge resume
uv run nde annotate timer .nde/annotations/owl-creek-bridge pause
uv run nde annotate revise .nde/annotations/owl-creek-bridge path/to/correction.json
```

Only completed start/pause intervals count. If work was interrupted with the timer
active, `timer <directory> recover` discards the unfinished interval. It never
counts downtime as correction effort. Status reports timing as unavailable
while a timer is active or no completed intervals exist.

Revisions are numbered and preserve the submitted JSON, validation results and
cards. Invalid submissions are retained but cannot be approved. If interrupted
after saving a submission, repeat `revise` with the saved `submitted.json` to
complete that version. Original drafts and prior correction versions remain.

Only an explicit owner action records literary review, bound to the checksum of
a valid correction version and accompanied by actual completed correction time:

```powershell
uv run nde annotate review .nde/annotations/owl-creek-bridge 1 --owner Bob --notes "Record specific literary judgements and remaining uncertainty here." --accept --meaningful-trajectories
```

Omit `--accept` to record rejection. Use `--meaningful-trajectories` only after
checking the differences against other annotated stories; differing fixture
curves cannot establish this. Reviews are append-only records; a later revision
requires its own review. Test-only review actions exercise storage in isolated
temporary sessions and supply no evidence about real stories.

No routine manual regression testing is needed. The remaining owner work is to
correct and review the selected real stories, record active effort, and judge
whether retained channels distinguish them. If correction is slow or channels
do not discriminate, cut or merge those channels and repeat on another story.
