The G1 retest did not pass on 30 October 2026.

The owner read four blind pairs of M (C) against Baseline B under
benchmark/prereg/g1-retest-20261030.json: held-out briefs, gpt-6.1-sol at medium
reasoning effort in both conditions, and exactly 2 editing passes for both. C won 2
pairs on coherence and lost 2, against the 3 wins required. The owner preferred C in
2 pairs and B in 2.

Owner coherence scores, with the preference and its strength, by brief:
  brief-11: C 1, B 3; preferred B, strength 2
  brief-12: C 4, B 3; preferred C, strength 3
  brief-13: C 3, B 2; preferred C, strength 1
  brief-14: C 1, B 3; preferred B, strength 2

Mean coherence: C 2.25, B 2.75.

Caveats:
- The story read second won coherence and preference in all four pairs. Each condition
  was second in exactly two, so this result cannot separate method from reading order.
- Budgets were deliberately not matched: C used 333,797 generation tokens and B 85,401.
- The report's 48-hour warning reflects agent commits. The owner does not work on the
  code. Condition guesses were not collected.
- This describes one reader and four pairs.

The plan allows one follow-up, and this was it, so the diffusion track stops. The run
cost US$1.76. evidence.zip holds the closed batch, the run and the reading log.
decision.json and decision.md are the report from `nde g1-report`, which reproduced byte
for byte in a second folder.
