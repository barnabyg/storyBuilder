G-schema passed on 27 October 2026.

Use a repository checkout at the code revision recorded in g-schema-result.json and install its locked environment. From that checkout, replay the extracted packet with explicit paths:
  uv run nde schema-evidence --replay PATH_TO_EXTRACTED_PACKET/evidence-indexed/inputs.json --out PATH_TO_REPLAY_OUTPUT

The command makes no provider calls. Compare evidence.json, evidence.md, evidence.html, mutation-results.json and inputs.json with evidence-indexed/.
evidence-indexed/ is the final native evidence. annotations-indexed/ contains original model attempts, all correction versions, timers and final native owner reviews.
annotations/ contains earlier drafts superseded after evidence-offset defects; their costs remain included.
Core and final review pages are frozen as presented. Hashes bind the exact versions accepted by the human owner.
guided-feedback.jsonl and the comparison records preserve the human judgements and the scope of each approval.
Numeric confidence/intensity values are unrecorded rather than fabricated. One unresolved driftwood setup is intentional because the intervening event is not shown.
This is one owner reviewing three stories in an assisted workflow. It establishes G-schema only, not improved story quality or later gates.
No credentials, API caches, database files, model weights or local runtime are included.

Protected gold: exclude these works and their authors from this project's training/tuning cohorts. The project manifest is data/manifest.gschema-20261027.yaml; the same source records are included here in manifest.yaml.
