# Registered G-D evaluation

`nde gd check PROTOCOL CONFIG --selection SELECTION --g1-batch CLOSED_G1
--g1-decision DECISION_JSON --out CHECK` is a no-spend prerequisite check.
It recomputes the closed G1 result and its bound artifacts. A passed issue,
a hand-written result, or a synthetic G1 decision cannot authorise real gate
generation. Test-only protocols require local fake providers and produce
test-only evidence; they never promote a scientific reference baseline.

The protocol JSON follows `nde.eval.gd.GDProtocol`. Commit it before generation.
It names D as candidate and C as comparator, a fixed distinct frozen cohort and
explicit `n`, the established C tier and selection checksum, model/schema/prompt
and configuration checksums, hard spending limits, repair/length/budget-revision
limits, critic iterations, reading mode, and prior benchmark exposure.
`control_targets` maps each target ID to the canonical SHA-256 of its complete
`ControlTarget` request. Do not choose these targets after reading the results.

Both `coherence` and `originality` require
`{"analysis": "mean_paired_difference", "margin": NUMBER}`. The implemented
comparison is **descriptive**: mean paired D-minus-C ordinal score must be at
least minus the registered margin. Ties contribute zero and remain in the
denominator. It provides no confidence interval or inferential noninferiority
claim. A nonsignificant difference is never treated as equality. Unsupported
analyses and omitted margins are rejected; G1's win-count rule is not reused.

After the check passes, use `nde gd run` with the same inputs and optional
`--runs-dir`, `--cache`, and `--ledger` paths. This uses the shared experiment,
prose, equal-budget, length-selection and accounting machinery. The run saves
`gd-generation.json` binding its final manifest, calls and story artifacts to
the pre-registration.
Ordinary `nde run` development output cannot masquerade as registered generation.

Prepare deterministic plan-only control reports using `nde control-report`.
Then run `nde gd batch` with the same inputs, `--name NAME`,
`--from RUN D --from RUN C`, and `--control BRIEF REPORT_JSON` once per brief.
Admission recomputes controls, checks registered predicate definitions and exact
accepted D plan identities, and uses strict cohort, final-length and budget
readiness. Known failed predicates remain available for a failed gate outcome;
unknown predicates and invalid lock/scope/validator evidence block admission.
Control reports must name the accepted operator's provenance and match its
saved source, mask, output and calls in D's refinement history. Merely
constructing a different before-plan cannot establish a gate edit. Keep the
complete revision artifacts; a plan/provenance pair alone is insufficient for
this gate evidence.
The batch seals its protocol, controls, prerequisite history and artifact hashes.

Use the existing `nde eval serve --batch NAME` reader. Coherence and originality
are required for both stories, as are preference and strength. The reader serves
only prose and scoring controls; it cannot serve the sealed condition mapping,
plans, critics or ledgers. G-D condition-guess options use opaque labels.
The 48-hour delay, prior benchmark exposure and blinding limitations must still
be interpreted in the scientific report. Implementation fixtures are not owner
evaluation and do not establish G1 or G-D scientific outcomes.

## Closed decision

`nde gd-report BATCH --out NEW_FOLDER [--realisation PACKET]` recomputes the
registered prerequisites, sealed cohort, required scores and control reports.
It writes deterministic `decision.json`, Markdown and HTML. The output folder
must be new: previous scientific records are never overwritten. Open batches
and missing or inconsistent evidence produce a pending result; the report
does not open a sealed key before closure.

Coherence, originality and every registered control predicate must pass.
Overall preference is secondary. The report retains the exact final story and
plan identities, experimental tokens, list-price cost, new spend, planning and
revision provenance, owner realisation evidence where supplied, and diagnostic
critic/judge separation. It records prior benchmark exposure, reading delay
limitations, missing guesses and condition-guess accuracy as blinding caveats.

Only a complete actual scientific pass creates `reference-baseline.json` naming
D. A complete actual failure retains C. That record binds configuration, tier,
schema, prompts, models and planning artifact provenance. Synthetic fixture
outcomes can show that registered criteria passed or failed, but their
scientific outcome remains pending and they never create a reference record.
G-D failure alone does not block Phase 6 when its independent G1/G2 prerequisites
hold; reporting never launches another experiment.
