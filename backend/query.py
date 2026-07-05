from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from .graph import Fact, KnowledgeGraph


@dataclass
class RetrievedContext:
    matched_entities: list[str]
    facts: list[Fact]


_STOPWORDS = {
    "i", "me", "my", "you", "your", "the", "a", "an", "is", "are", "was", "were",
    "do", "does", "did", "have", "has", "had", "be", "been", "being", "of", "to",
    "in", "on", "at", "by", "for", "with", "about", "as", "can", "could", "would",
    "should", "will", "what", "where", "when", "who", "why", "how", "and", "or",
    "but", "if", "this", "that", "these", "those", "tell", "say", "remind", "me",
    "again", "please", "help",
}

# Half-life for the recency boost. After this many seconds the recency
# contribution to the score has decayed to 1/e of its initial value.
_RECENCY_HALF_LIFE_S = 3 * 24 * 3600  # ~3 days

# Signals that the user is asking about the past ("where did I live before?").
# Only then do closed facts become retrievable — tagged, so the responder can
# speak of them in the past tense.
_PAST_MARKERS = re.compile(
    r"\b(used to|previously|before|no longer|anymore|in the past|back then|"
    r"formerly|did i|was i|used-to)\b",
    re.IGNORECASE,
)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
            if t not in _STOPWORDS and len(t) > 2]


def retrieve(kg: KnowledgeGraph, text: str, k: int = 8) -> RetrievedContext:
    """Pull the most relevant facts for a user query.

    Score = ``confidence + 0.5 * token_overlap + 0.4 * recency`` where
    ``recency = exp(-Δt / half_life)``. Retracted / superseded facts are
    normally filtered out; when the query asks about the past ("used to",
    "before", ...) they are included with a score penalty so current facts
    still rank first.
    """
    toks = set(_tokens(text))
    include_past = bool(_PAST_MARKERS.search(text))
    matched_entity_keys: list[str] = []
    for key, ent in kg.entities.items():
        ent_toks = set(_tokens(ent.name))
        if ent_toks & toks:
            matched_entity_keys.append(key)
    matched_entity_keys.append("user")

    now = time.time()
    scored: list[tuple[float, Fact]] = []
    seen_ids: set[str] = set()
    for key in matched_entity_keys:
        for f in kg.facts_about(key, depth=2, include_inactive=include_past):
            if f.id in seen_ids:
                continue
            if f.valid_until is not None and not include_past:
                continue
            seen_ids.add(f.id)
            fact_text = f"{f.subject} {f.predicate} {f.object}"
            overlap = len(set(_tokens(fact_text)) & toks)
            age = max(0.0, now - f.last_active_at)
            recency = math.exp(-age / _RECENCY_HALF_LIFE_S)
            score = f.confidence + 0.5 * overlap + 0.4 * recency
            if f.valid_until is not None:
                score -= 0.3  # past facts rank below current ones
            scored.append((score, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    facts = [f for _, f in scored[:k]]
    return RetrievedContext(matched_entities=matched_entity_keys, facts=facts)


def format_context_for_llm(ctx: RetrievedContext) -> str:
    if not ctx.facts:
        return "(no prior knowledge)"
    lines = []
    for f in ctx.facts:
        status = "" if f.valid_until is None else " [no longer true]"
        lines.append(f"- {f.subject} {f.predicate} {f.object} (conf={f.confidence:.2f}){status}")
    return "\n".join(lines)
