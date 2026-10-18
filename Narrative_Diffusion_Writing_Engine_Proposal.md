# Narrative Diffusion: A Proposal for a Multi-Layer AI Story Planning Engine

**Status:** Concept proposal — for technical planning and validation  
**Date:** 18 October 2026  
**Working name:** Narrative Diffusion Engine (NDE)

## 1. Executive summary

This proposal describes an AI-assisted creative writing system that models **the structure of a story, not merely its sequence of words or plot events**. The starting hypothesis is that engaging fiction exhibits multiple interacting patterns at different timescales: coherent prose, local mood, tension, character and relationship development, information release, thematic development, motifs, and whole-story emotional trajectories.

Conventional language models learn some of these patterns implicitly through token prediction. Story-beat planners expose plot structure but often leave tone, subtext, pacing and thematic continuity underspecified. The proposed system would instead **extract representations of narrative states from human-written fiction**, learn the patterns that govern them, and generate a coherent story plan before a separate language model writes the prose.

The principal research hypothesis is that a **conditional diffusion or diffusion-inspired generative model** could learn to construct and revise *whole-story narrative-state structures*: begin from noise, condition on instructions such as genre, themes and intended ending, and iteratively denoise toward a structured plan. This might be advantageous over strictly left-to-right story planning because a story's ending, motifs, emotional trajectory and causal structure constrain its earlier scenes.

**The proposal is exploratory, not a claim that narrative diffusion is already proven superior.** The first task is to demonstrate that narrative structure can be represented, extracted and evaluated reliably. Only then should a diffusion planner be trained and compared against simpler baselines.

## 2. Problem and motivation

### 2.1 What story beats miss

A beat-based plan says *what happens*: a character enters a room, confronts an ally, discovers a betrayal. It may not specify:

- How the scene should **feel** to the reader, including the emotional trajectory within the scene.
- Whether tension should climb, plateau, briefly release, or sharply reverse.
- What the characters openly say versus privately intend.
- Which relationships, beliefs or levels of agency change.
- Which themes are advanced, complicated or subverted.
- Whether a detail foreshadows a future event or pays off an earlier one.
- How the scene serves a longer arc of dread, hope, tragedy, irony or transformation.

The result can be prose that is locally plausible but globally undirected: scenes happen in a sensible order without a compelling cumulative reader experience.

### 2.2 Working thesis

**Narrative quality depends in part on coordinated state changes across multiple timescales.** A system that explicitly plans those changes may produce stories that are more coherent and engaging than a system guided only by plot beats or style prompts.

An important refinement is to model a **narrative-state transition**, not just a label for “the next story element.” The same external event can produce different narrative effects depending on the character's state, reader knowledge, tone and larger arc.

## 3. Objectives and non-objectives

### Objectives

1. Define a machine-readable representation for narrative development across sentences/paragraphs, scene beats, scenes, chapters and complete stories.
2. Extract that representation from a legally usable corpus of human-written stories, with measurable annotation quality.
3. Learn statistical structure within and between narrative states.
4. Generate editable whole-story plans conditioned on prompts and constraints.
5. Convert these plans into prose while preserving continuity, causality, character development and intended emotional movement.
6. Evaluate against strong autoregressive and prompt-based planning baselines, including blind human judgments.

### Non-objectives for the initial project

- Training a general-purpose foundation LLM from scratch.
- Replacing the prose-writing LLM with diffusion in the first implementation.
- Assuming that a numerical “tension” or “quality” score fully captures literary merit.
- Perfectly formalising creativity, taste, theme or subtext.
- Producing publishable novels without human editing or oversight.

## 4. Proposed narrative representation

Represent each scene as a **structured, multi-channel narrative state**, with explicit transitions to adjacent states and links across the wider story.

| Channel | Examples of tracked information | Why it matters |
| --- | --- | --- |
| Plot events | Actions, revelations, decisions, reversals | External narrative progression |
| Causality | Preconditions, consequences, dependencies | Prevents arbitrary sequences of events |
| Character goals and beliefs | Wants, fears, mistakes, secrets, intentions | Makes action psychologically motivated |
| Character trajectory | Confidence, agency, guilt, shame, resolve | Enables change over time |
| Relationships | Trust, attachment, resentment, dependence, power | Supports developing interpersonal drama |
| Reader affect | Unease, relief, dread, wonder, grief, humour | Models intended reader experience |
| Tension and pacing | Escalation, release, delay, surprise | Controls intensity and rhythm |
| Information state | Who knows what, dramatic irony, withheld facts | Supports suspense, mystery and revelation |
| Theme | Competing values, moral dilemmas, recurring questions | Creates deeper structural meaning |
| Motifs and payoffs | Objects, imagery, promises, echoes and callbacks | Connects distant scenes |
| Discourse and style | Viewpoint, narrative distance, dialogue density, rhythm | Connects structural plans to actual prose |
| Subtext | Expressed desire vs hidden intent; implied power dynamics | Prevents scenes from becoming mechanically literal |

**Representation principle:** Distinguish *observable textual properties* (e.g., dialogue proportion) from *interpretations* (e.g., dread or thematic pressure). Interpretive labels should allow uncertainty, disagreement and multiple readings.

### 4.1 Hierarchical timescales

A practical hierarchy is:

`story → act/arc → chapter → scene → scene beat → passage`

Higher levels hold slow-moving constraints and long-range dependencies; lower levels hold local emotional movement, rhythm and information release. The model should not require each chapter to have the same number of scenes or each scene to have the same number of beats.

### 4.2 State, transition and graph links

Each unit may carry:

- **State:** what is true at its beginning and end.
- **Delta:** what changes over the unit.
- **Trajectory:** how mood, tension or agency evolve *within* the unit.
- **Links:** what earlier events caused it, and what later events it sets up.
- **Constraints:** invariants that must remain true (e.g., a character cannot know a secret yet).

Some aspects of a story are sequential, but many are relational or causal. The representation will likely need both an ordered scene list **and** graph-like links for knowledge, causality, motifs and payoff.

### 4.3 Illustrative scene record

The following is a **design sketch**, not a settled schema or scientifically calibrated measurement:

```yaml
scene_id: S12
position: 12
plot:
  event: Mara questions Elias about an old photograph
  function: confirms a lie without open confrontation
characters:
  Mara:
    visible_goal: casual conversation
    hidden_goal: test Elias with false information
    belief_delta: uncertainty -> conviction
  Elias:
    visible_goal: reassure Mara
    hidden_goal: conceal earlier involvement
relationships:
  Mara_to_Elias:
    trust_delta: decreases
reader_experience:
  mood_curve: politeness -> discomfort -> quiet menace
  tension_curve: gradual rise; no release
information:
  reader_knows: Mara has discovered the contradiction
  Elias_knows: he has not yet been exposed
subtext: a conversation about a photograph is a contest over truth
motifs:
  - photograph
  - reflection
long_range_links:
  caused_by: [S04]
  sets_up: [S18]
constraints:
  - no overt accusation
  - no resolution of the betrayal
```

## 5. Data acquisition and extraction

### 5.1 Corpus

Use a substantial, diverse corpus of **human-written fiction with clear rights to analyse and train on**. Candidate sources include public-domain works and suitably licensed contemporary writing. Record genre, language, period, author, work, edition and provenance to support auditing and meaningful train/test splits.

A public-domain-only pilot is plausible, but risks historical and stylistic bias. Modern genre coverage would require careful licensing.

### 5.2 Segmentation

Split each work at multiple resolutions: chapters and scenes first, then scene beats and passages. Scene-boundary detection will be imperfect; retain original text offsets, confidence and hierarchical parent links so segmentations can be corrected or revised.

### 5.3 Annotation

Use LLM-assisted extraction to produce structured labels; combine it with deterministic measurements where appropriate (dialogue density, sentence-length distribution, entity occurrences). Include evidence spans from the source prose wherever possible.

Key safeguards:

- Multiple independent annotation passes on a sample, with disagreements retained rather than hidden.
- Human review on a stratified sample of genres and periods.
- Confidence estimates and an `unknown`/`ambiguous` option.
- Clear separation of observations from interpretive judgments.
- Extraction of before/after states and causal evidence, not merely mood tags.
- Versioned schema, prompts and annotation-model identifiers.

The annotation model's literary assumptions are a serious source of bias. A model trained solely on its own unverified interpretations could learn **annotation clichés** rather than human storytelling patterns.

### 5.4 Representation learning

Compare three ways of representing narrative states:

1. **Structured symbols:** inspectable fields, labels, links and trajectories. Best for editing and validation; potentially too rigid.
2. **Continuous embeddings:** compact vectors learned from scenes and structured annotations. Well suited to continuous latent diffusion, but harder to interpret and constrain.
3. **Hybrid representation:** structured state and graph constraints plus learned latent vectors for nuanced, hard-to-label aspects of narrative craft.

**Recommended starting point:** hybrid. Preserve a human-readable plan as the source of truth while experimenting with embeddings as learned supporting representations.

Critically, an ordinary text embedding is not automatically an appropriate diffusion latent. A useful latent space must preserve narratively important distinctions and support decoding into valid, coherent states. That requires explicit testing and possibly a trained encoder/decoder.

## 6. Generative model hypothesis

### 6.1 Why a diffusion-style planner?

A left-to-right narrative predictor learns `next_state | previous_states`. A diffusion-style model would instead learn to recover a plausible **whole-plan representation** from corrupted or partially specified versions of it, conditioned on creative requirements.

Potential advantages to test:

- Global consistency between opening, midpoint and ending.
- Whole-arc design of mood, tension and character change.
- Foreshadowing and payoff placement across distant scenes.
- Structural editing: preserve key scenes and regenerate a weak middle act.
- Generation of several distinct plans from the same constraints.
- Revision without committing irreversibly to a previously generated sequence.

These are **hypotheses**, not guaranteed properties of diffusion. A diffusion model can still produce conventional, incoherent or causally impossible plans.

### 6.2 Candidate approaches

| Approach | Mechanism | Advantage | Principal difficulty |
| --- | --- | --- | --- |
| Continuous latent diffusion | Noise and denoise scene/arc vectors | Natural whole-sequence generation and interpolation | Learned latent may be uninterpretable or undecodable |
| Discrete diffusion | Mask/corrupt and restore narrative tokens | Editable, inspectable symbols | Requires expressive tokens; may flatten nuance |
| Graph diffusion | Generate/edit scene, character, causal and motif relationships | Matches nonlocal story dependencies | Complex representation, training and constraint handling |
| Diffusion-inspired iterative refinement | Start with a rough plan, repeatedly revise under constraints | Fastest route to testing the workflow with existing LLMs | Not evidence that a trained diffusion model is beneficial |

**Recommended sequence:** establish iterative refinement and autoregressive baselines first; then test continuous latent or discrete diffusion; defer graph diffusion until evidence justifies the complexity.

### 6.3 Training and inference, conceptually

**Training**

1. Encode a human-authored story as a hierarchical narrative plan.
2. Represent that plan as suitable continuous latents, discrete tokens or a hybrid.
3. Corrupt parts or all of the plan according to a chosen noise/masking schedule.
4. Provide optional conditioning attributes and any preserved constraints.
5. Train a denoiser to predict cleaner states (or the noise/mask targets, depending on formulation).
6. Decode and validate reconstructed plans; measure whether narrative information survives corruption and recovery.

**Generation**

1. Accept user inputs: genre, premise, desired style, themes, protagonist, central conflict, ending constraints, target length, etc.
2. Initialise a noisy or heavily masked story-plan representation.
3. Iteratively generate a candidate plan while holding explicit constraints fixed.
4. Decode into editable scene and arc cards.
5. Run deterministic consistency checks and model-based literary critiques.
6. Present the plan for human revision or generate prose scene by scene.

Genre and style keywords are useful starting conditions but often too underspecified by themselves. The interface should allow richer constraints without requiring them.

### 6.4 Planning should not be mistaken for storytelling

The denoiser's job is to propose *narrative structure*. A separate prose model should realise individual scenes, informed by the relevant local plan and a compact persistent global state. The prose model may suggest justified changes to the plan, but those changes should be explicit and tracked rather than silently rewriting canon.

## 7. End-to-end system architecture

```text
Rights-cleared human-written stories
             |
             v
   Segmentation and provenance
             |
             v
  Narrative annotation extractor
             |
             v
  Structured states + linked graph
             |
             v
   Narrative encoder / dataset
             |
             +----------------------+
             |                      |
             v                      v
   Baseline planners       Diffusion/denoising
 (prompted and AR)           narrative planner
             |                      |
             +-----------+----------+
                         |
                         v
          Editable whole-story plan
                         |
                 Validation / critique
                         |
                         v
             Scene-level prose LLM
                         |
                         v
        Continuity + affect evaluation
                         |
                         v
                Draft and revisions
```

Four functions remain distinct:

- **Planner:** proposes a coherent narrative trajectory.
- **Prose generator:** realises each scene in language.
- **State ledger:** preserves canon, character knowledge, event consequences, motifs and other cross-scene dependencies.
- **Evaluator/editor:** identifies failures and proposes revisions, with human authority over subjective creative decisions.

## 8. Planned capabilities

The target product should eventually support:

- **From brief to outline:** produce a multi-layer plan from a premise and constraints.
- **Affect shaping:** increase slow-burning dread, reduce melodrama, introduce a deliberate tonal rupture.
- **Structural inpainting:** keep the beginning and ending but regenerate the middle.
- **Character-arc repair:** repair a change of heart that currently feels unearned.
- **Payoff management:** strengthen setup, callbacks, dramatic irony and reveals.
- **Constraint locking:** preserve canon, events, critical character choices, or an intended ending while revising the rest.
- **Alternatives:** offer meaningfully different trajectories rather than minor paraphrases.
- **Traceability:** show why each proposed scene exists and how it contributes to larger arcs.

Example:

> Keep the betrayal and tragic ending, but redesign the middle third so that the relationship deteriorates gradually, with two believable episodes of renewed hope before the final reversal.

## 9. Critical scientific and engineering uncertainties

The project should explicitly test rather than assume the following:

1. **Extraction validity:** Can annotators reliably agree on states such as subtext, agency, reader affect and theme? Which labels are only weakly measurable?
2. **Latent adequacy:** Can a compressed representation preserve causality, knowledge states and distant setup/payoff relationships?
3. **Structural well-formedness:** Can denoising produce variable-length, causally coherent plans with character consistency?
4. **Condition adherence:** Do genre, theme, ending and style requirements actually influence the plan in controlled ways?
5. **Quality versus conventionality:** Does training encourage stock arcs and genre clichés rather than interesting departures?
6. **Global versus local trade-offs:** Does better whole-story shape come at the expense of distinctive scenes and prose?
7. **Planner/prose alignment:** Does the language model realise the intended changes, or dilute them during drafting?
8. **Value of diffusion specifically:** Are any gains attributable to diffusion, rather than richer annotations, better prompts or extra revision compute?
9. **Data limitations:** How much legally usable fiction and annotation fidelity are needed to learn meaningful long-range patterns?
10. **Evaluation reliability:** Can an improvement be demonstrated under blinded comparisons instead of relying on agreeable model critics?

## 10. Evaluation strategy

Measure the complete system, while also separately testing its components.

### 10.1 Reconstruction and representation tests

- Can the encoder/decoder recover meaningful character, event and relationship changes?
- Do causal and knowledge relationships survive compression and corruption?
- Are mood/tension trajectories reproducible across independent annotations?
- How often do reconstructed plans violate schema and continuity constraints?

### 10.2 Generation and planning tests

- Causal coherence: do events follow from established preconditions and intentions?
- State consistency: do characters possess the right knowledge and capabilities at the right times?
- Affect adherence: does the intended emotional trajectory appear in the plan and ultimately the prose?
- Setup/payoff quality: are callbacks earned and relevant, not merely repeated tokens?
- Controllability: can a specified trait or arc be changed while locked elements remain stable?
- Diversity: do multiple generations differ structurally rather than cosmetically?
- Originality/conventionality: do plans avoid near-duplicates and generic formulae?

### 10.3 Comparative tests

Hold constant the premise, reference constraints, target length, prose model and approximate generation/revision budget. Compare:

- **Baseline A:** one LLM prompted to write from a premise.
- **Baseline B:** story-beat outline followed by scene generation.
- **Baseline C:** multi-layer structured plan generated autoregressively or through prompting.
- **Candidate D:** multi-layer plan generated using iterative refinement.
- **Candidate E:** the trained diffusion planner, if earlier stages succeed.

Use blinded human evaluation on complete short stories and targeted excerpts. Ask readers about engagement, coherence, emotional impact, character believability, pacing and originality; also collect preference rankings and specific failures. Model-based critics can assist diagnosis but should not be the sole success measure. Split data by **entire work and author** where feasible to reduce leakage.

**Core decision criterion:** Does the trained diffusion planner deliver a repeatable, meaningful improvement over a strong, equal-budget structured-planning baseline?

## 11. Staged feasibility programme

This is an **evidence-driven sequence**, not yet an implementation schedule.

| Stage | Goal | Deliverable | Gate to proceed |
| --- | --- | --- | --- |
| **0. Define the benchmark** | Establish what improvement would mean | Evaluation set, rubrics and baseline outputs | Credible way to compare approaches |
| **1. Narrative schema** | Make states useful and inspectable | Versioned schema plus hand-reviewed example stories | Schema expresses meaningful differences without excessive complexity |
| **2. Extraction pilot** | Test automated annotations | Small rights-cleared corpus with source-linked labels | Human review finds annotations sufficiently accurate/useful |
| **3. Structured planner baseline** | Test whether richer story states help at all | Prompt-based/AR planner plus prose generator and critic | Clear gains over beat-only planning |
| **4. Denoising feasibility** | Show that narrative plans can survive corruption/recovery | Prototype latent or discrete denoiser | Reconstructions preserve meaning, constraints and key links |
| **5. Conditional generation** | Generate new whole-story plans | Conditional diffusion planner, editable plans and inpainting | Competitive or better outcomes than structured baselines |
| **6. End-to-end evaluation** | Check real reader impact | Blind comparison study, error analysis and cost assessment | Improvement is material enough to justify extension |
| **7. Scale and refine** | Support novels and richer graph structure | Hierarchical/graph planning and robust revision tools | Long-range quality scales without prohibitive complexity |

### Recommended first experiment

**Do not begin by training diffusion.** Start with a small set of carefully selected, rights-cleared short stories. Hand-check the extracted structure. Generate several new stories from a shared brief using (a) a beat-only outline and (b) a multi-channel scene-state plan, keeping the prose model and revision budget identical. Blindly compare the results.

If the richer plan demonstrably improves reader experience, the next question is whether training a model to *generate* those plans improves on a prompted or autoregressive planner. If the answer is no, diffusion training may be unnecessary.

## 12. Implementation considerations for later planning

The following are planning topics rather than settled choices:

- **Data format:** versioned JSON/YAML records with text offsets, provenance, schema validation and graph edges.
- **Storage:** source texts, extracted records and experimental runs must remain reproducible and linked.
- **Model components:** initially reuse established language models for annotation, planning and prose; train specialist components only when a clear bottleneck is identified.
- **Training objective:** select noise process and loss functions appropriate to continuous, discrete or graph states. Consider auxiliary losses for causal/knowledge consistency if reconstruction loss alone is insufficient.
- **Variable story length:** investigate hierarchical generation, masked scene slots, length conditioning and expansion/collapse operators rather than fixed scene counts.
- **Constraints:** enforce hard invariants with validators, not just model instructions; treat affect and style as soft targets.
- **Compute:** benchmark dataset sizes, model footprints and end-to-end costs before selecting hardware or training scale.
- **Experiment tracking:** maintain seeds, prompts, versions, data splits, model checkpoints and evaluation judgments.
- **User experience:** the system's most valuable output may be an *editable story plan* rather than fully automatic prose.
- **Legal/ethical practice:** license verification, attribution/provenance, corpus governance, memorisation checks and respect for authors' rights.

## 13. Open design decisions

These should be resolved through experiments or during the subsequent technical plan:

1. What is the smallest useful narrative state representation?
2. Is the primary unit a scene, a scene beat, or a multiresolution hierarchy?
3. Can an acceptable narrative encoder be built without an expensive labelled dataset?
4. Should the first denoiser operate on continuous vectors or masked discrete narrative tokens?
5. How should causal graphs and character knowledge be preserved during denoising?
6. How will the model generate variable numbers of scenes and subplots?
7. What is the minimum rights-cleared corpus needed for a meaningful test?
8. Which changes must the system treat as hard constraints rather than stylistic preferences?
9. How much iterative editing is allowed before comparing costs and quality fairly?
10. What result would justify the jump from diffusion-inspired refinement to a trained diffusion model?

## 14. Expected outcome and success definition

A successful research prototype would accept a creative brief, generate an editable multi-layer narrative plan, realise it as a short story using an existing LLM, and show **measurably better reader-rated narrative coherence and/or engagement** than a strong beat-based and structured non-diffusion baseline at a comparable budget.

Even if diffusion offers no benefit, the project could still deliver useful outcomes: a narrative-state schema, an annotated fiction dataset, an evaluation benchmark, and an improved structured-planning tool. A failure to demonstrate benefit from diffusion would be a valid result rather than a failure of the broader investigation.

## 15. Central proposition

> **Move from predicting what event happens next to designing what changes next in the characters, relationships, reader's understanding, emotional pressure and thematic trajectory — and use a whole-story generative model to coordinate those changes before writing prose.**

The key architectural bet is that **narrative planning and prose realisation are distinct, cooperating problems**. The key experimental bet is that **whole-structure denoising may provide stronger global narrative control than left-to-right planning**. Neither proposition should be treated as proven until evaluated.

---

*Origin: Synthesised from the discussion of story beats, multi-layer narrative states, automated extraction from human-written fiction, next-state modelling, diffusion-based planning, and prose generation. Research examples mentioned in the discussion inform the hypothesis but are not presented here as independent validation. This document is intended to precede a technical build plan.*
