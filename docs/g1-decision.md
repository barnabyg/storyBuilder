# G1 decision reports

After completing and closing a protocol-bound G1 batch:

```powershell
uv run nde g1-report batches/G1 --out runs/g1-decision
```

Optional repeatable `--realisation PACKET_DIR` arguments attach current owner
realisation reports. Packet story hashes must belong to the admitted cohort;
stale and absent evidence remains explicit. Optional judge agreement is
available in the ordinary `nde eval report` output and remains diagnostic.

The immutable output folder contains the complete `decision.json` (protocol,
run/admission identities, result checksum, costs, reading caveats, failure tags
and secondary overall preferences) and a summary `decision.md`. Choose a new
folder to regenerate a report; completed outputs cannot be overwritten.

C wins coherence only when its score is strictly higher than B's. Ties are
non-wins in the full registered denominator. The exact two-sided binomial
result is reported with C's direction explicit; significance in B's direction
cannot pass C's threshold. Originality uses the supported registered analysis,
mean paired C-minus-B score, and passes when that mean is at least the
negative registered margin. This is a descriptive registered safeguard,
not an inferential noninferiority test. Preference alone cannot pass G1.

Missing scores, inconsistent cohorts or changed bound artifacts make the
criteria pending. Complete actual evidence yields passed or failed/inconclusive
according to the registered criteria. A synthetic protocol reports its fixture
criteria result separately and keeps the scientific outcome pending. Owner
code-work timing and absent condition guesses are explicit caveats; absent
information cannot confirm blinding.

The report describes at most one separately pre-registered follow-up on fresh
held-out briefs. It never generates one, extends n, pools results or changes the
completed protocol. Real owner reading, actual tier choice, passed G-schema
evidence and the scientific decision remain separate from software fixtures.

Exit 0 means a report was written (inspect its outcome, which may be pending),
and exit 2 means malformed/unclosed input or an existing output folder. Manual
check: after an actual closed batch exists, verify a tied pair contributes no C
win, compare the final identities with the admitted stories, and confirm the
report reproduces into a second folder byte for byte. Replacing a bound final
story must produce a pending result in a new report while preserving the old
decision.
