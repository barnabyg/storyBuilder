# Representation tiers

```powershell
uv run nde plan project examples/trajectory-rising.plan.json --tier S --out .nde/tier-s
uv run nde plan project examples/harbour-state.plan.json --tier M --out .nde/tier-m
uv run nde plan project examples/trajectory-rising.plan.json --tier L --out .nde/tier-l
```

Each command validates the source, projects an independent plan, validates the
projection and writes the ordinary `plan.json`, `validation.json`, `cards.md`
and `cards.html`. Invalid source plans cannot be made valid by hiding offending
channels. Normal inspect/validate commands read every projected plan.

| Field group | S | M | L |
| --- | --- | --- | --- |
| Structural envelope: version, plan ID/title/tier, character identities, sources, ordered scenes and beats | retained | retained | retained |
| Scene metadata: ID/title/summary, character presence, source spans; beat ID/summary/spans | retained | retained | retained |
| Causal links, setups/payoffs, explicit knowledge facts/reveals/uses | omitted | retained | retained |
| Goal and belief transitions, including their uncertainty/alternatives | omitted | retained | retained |
| Value/canon transitions, relationships, hard/literary constraints | omitted | omitted | retained |
| Arcs/agency, reader affect/mood, tension, themes, motifs/echoes, discourse, subtext | omitted | omitted | retained |

S is the ordered beat representation with its structural identity and provenance
envelope. Scene summary is a structural synopsis, not an interpretive channel.
Source text, checksums and evidence spans belong to that envelope so spans remain
verifiable. M adds goals/beliefs, causality and information state. L retains all
supported channels, uncertainty and evidence. Relationship interpretations belong
only to L, including when they describe beliefs about other characters.

Excluded channel keys are absent from exported JSON. Reloading supplies empty
defaults internally; excluded nonempty channels are validation errors rather
than silently ignored. M filters state records to `goal` and `belief`; every
retained reference targets a retained scene, character, source, setup or fact.
The source plan is unchanged. Retained IDs and list ordering never change, and
repeating the same projection is deterministic and idempotent. Promoting an
already reduced plan cannot reconstruct omitted channels.

No benchmark brief, outline, evaluation or run format changes. Automated file
integration tests cover all tiers and both renderers; routine manual regression
testing is unnecessary.
