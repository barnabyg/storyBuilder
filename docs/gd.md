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
analyses and omitted margins are rejected; G1's 21/30 rule is not reused.

After the check passes, use `nde gd run` with the same inputs and optional
`--runs-dir`, `--cache`, and `--ledger` paths. This uses the shared experiment,
prose, equal-budget, length-selection and accounting machinery. The run saves
`gd-generation.json` binding its final manifest to the pre-registration.
Ordinary `nde run` development output cannot masquerade as registered generation.

Prepare deterministic plan-only control reports using `nde control-report`.
Then run `nde gd batch` with the same inputs, `--name NAME`,
`--from RUN D --from RUN C`, and `--control BRIEF REPORT_JSON` once per brief.
Admission recomputes controls, checks registered predicate definitions and exact
accepted D plan identities, and uses strict cohort, final-length and budget
readiness. Known failed predicates remain available for a failed gate outcome;
unknown predicates and invalid lock/scope/validator evidence block admission.
The batch seals its protocol, controls, prerequisite history and artifact hashes.

Use the existing `nde eval serve --batch NAME` reader. Coherence and originality
are required for both stories, as are preference and strength. The reader serves
only prose and scoring controls; it cannot serve the sealed condition mapping,
plans, critics or ledgers. G-D condition-guess options use opaque labels.
The 48-hour delay, prior benchmark exposure and blinding limitations must still
be interpreted in the scientific report. Implementation fixtures are not owner
evaluation and do not establish G1 or G-D scientific outcomes.
