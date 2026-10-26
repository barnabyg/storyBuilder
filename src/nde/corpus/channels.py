"""Logical channel membership shared by agreement, review and exemplar targets."""

CHANNEL_FIELDS = {
    "plot": ("scenes",),
    "causality": ("causal_links", "setups", "payoff_links"),
    "information": ("knowledge_facts", "reveal_links", "knowledge_uses"),
    "relationships": ("relationships",),
    "interpretation": (
        "arcs",
        "reader_affect",
        "tension",
        "themes",
        "motifs",
        "motif_echoes",
        "discourse",
        "subtext",
    ),
}
