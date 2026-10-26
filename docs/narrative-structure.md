# Causality, setups and explicit knowledge

Use the existing `nde plan validate` and `nde plan inspect` commands with
`tests/fixtures/narrative-structure.json` for a complete offline example.
The same schema and card workflow handles minimal plans with all these lists empty.

## Relationships and order

All structural records have globally unique stable `id` fields. Scene list order
defines chronology, independently of the order of records in any link list.

| Plan channel | Record fields | Meaning |
| --- | --- | --- |
| `causal_links` | `id`, `kind: causal`, `cause_scene_id`, `effect_scene_id` | An earlier cause points forward to a later effect. Both scene endpoints must resolve. Self-links, reversed links and cycles are errors. |
| `setups` | `id`, `scene_id`, `description` | A setup introduced in a particular scene. |
| `payoff_links` | `id`, `kind: setup_payoff`, `setup_id`, `payoff_scene_id` | An existing setup points forward to a strictly later payoff scene. One setup may have several payoffs. A setup with no valid payoff emits advisory `unpaid_setup`. |
| `knowledge_facts` | `id`, `description`, `initial_holder_ids` | An explicit fact and its holders before the first scene. Initial holders need not appear in the first scene; they must be known characters. |
| `reveal_links` | `id`, `kind: reveal`, `fact_id`, `scene_id`, `holder_ids` | At the **start** of this scene the listed, present characters first acquire this fact. Each holder is unique and may acquire a fact only once; a reminder is not another acquisition. |
| `knowledge_uses` | `id`, `fact_id`, `scene_id`, `holder_id` | A present character explicitly uses a fact already held initially or acquired at or before this scene's start. Duplicate use records for the same fact, holder and scene are errors. |

The `kind` literals are exported in JSON Schema and are written explicitly when
a plan is saved, even if omitted in input. All fact, scene and character
references must resolve. Reveal holders must be nonempty. Multiple holders can
acquire a fact together. A same-scene reveal and use is valid because acquisition
is defined at scene start; finer within-scene ordering is not represented.

These checks validate recorded facts and links. They do not infer knowledge,
causality, or the quality of a payoff from prose. `unpaid_setup` is advisory
because an unresolved setup can be intentional. Errors exit 1; advisory
findings retain exit 0. Both severities appear in Markdown, HTML and
`validation.json` with the record ID and relevant scene/fact/holder identifiers.

## Registered mutation categories

`tests/test_narrative_structure.py` injects each category into the saved valid
control and exercises file loading, validation, saved diagnostics and card
rendering through the public CLI. A catch means the **intended** code appears,
not merely that input parsing fails. The aggregate G-schema report may group
multiple injection examples under the distinct invariant categories below.

| Category | Expected diagnostic | Injection |
| --- | --- | --- |
| structural ID collision | `duplicate_id` | Give a causal link an existing scene ID. |
| dangling causal endpoint | `causal_endpoint` | Name a nonexistent cause scene. |
| causal self-link | `causal_self_link` | Make a scene cause itself. |
| causal chronology | `causal_order` | Reverse cause/effect, or reorder scenes. |
| causal cycle | `causal_cycle` | Add a return edge from the effect to its cause. |
| dangling setup scene | `setup_scene_reference` | Introduce a setup in a nonexistent scene. |
| dangling payoff setup | `payoff_setup_reference` | Pay off an unknown setup. |
| dangling payoff scene | `payoff_scene_reference` | Pay off in a nonexistent scene. |
| setup/payoff chronology | `payoff_order` | Pay off in the setup's scene or before it. |
| unpaid setup (advisory) | `unpaid_setup` | Remove all payoff links. |
| duplicate fact identity | `duplicate_fact_id` | Give two knowledge facts the same ID. |
| invalid knowledge holder | `knowledge_holder` | Use unknown initial, reveal or use holders. |
| duplicate holder/fact reference | `duplicate_fact_reference` | Repeat an initial or reveal holder. |
| dangling knowledge scene | `knowledge_scene` | Reveal or use in an unknown scene. |
| dangling knowledge fact | `knowledge_fact` | Reveal or use a nonexistent fact. |
| duplicate acquisition | `duplicate_acquisition` | Acquire a fact again in another scene. |
| duplicate explicit use | `duplicate_knowledge_use` | Duplicate a fact/holder/scene use under another ID. |
| absent knowledge holder | `knowledge_presence` | Reveal a fact to a character absent from its scene. |
| premature knowledge use | `knowledge_before_acquisition` | Use a fact in the first scene when revealed later. |

Valid controls include initial knowledge used in the first scene, first-scene
and last-scene reveals, same-scene use, a payoff in the final scene and a reveal
with multiple holders. All tests require no keys, network downloads or live
providers. No owner literary review is needed to verify this implementation.
