from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from .embeddings import EMBEDDER
from .graph import Fact, KnowledgeGraph


@dataclass
class RetrievedContext:
    matched_entities: list[str]
    facts: list[Fact]


# Poids de la similarité sémantique dans le score (0 = lexical pur).
_SEM_WEIGHT = 1.5


def _fact_text(f: Fact) -> str:
    """Texte lisible d'un fait, pour le lexical ET l'embedding.

    "user struggles_with derivatives" → "I struggles with derivatives" :
    on remplace user→I et les underscores par des espaces pour se rapprocher
    d'une phrase naturelle."""
    subj = "I" if f.subject.strip().lower() == "user" else f.subject
    return f"{subj} {f.predicate.replace('_', ' ')} {f.object}"


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

    Score = ``confidence + 0.5·token_overlap + 0.4·recency + 1.5·cosine`` where
    ``recency = exp(-Δt / half_life)`` and ``cosine`` is the semantic similarity
    between the query and the fact (0 when no embedding backend is installed —
    the layer then falls back to pure lexical matching). Retracted / superseded
    facts are normally filtered out; when the query asks about the past
    ("used to", "before", ...) they are included with a score penalty so current
    facts still rank first.
    """
    toks = set(_tokens(text))
    include_past = bool(_PAST_MARKERS.search(text))

    matched_entity_keys: list[str] = []
    for key, ent in kg.entities.items():
        ent_toks = set(_tokens(ent.name))
        if ent_toks & toks:
            matched_entity_keys.append(key)
    matched_entity_keys.append("user")

    # Candidats — les faits reliés aux entités repérées lexicalement...
    candidates: dict[str, Fact] = {}
    for key in matched_entity_keys:
        for f in kg.facts_about(key, depth=2, include_inactive=include_past):
            if f.valid_until is not None and not include_past:
                continue
            candidates[f.id] = f
    # ...et, quand la recherche sémantique est dispo, TOUS les faits en jeu :
    # ainsi une requête sans mot commun ("what am I bad at?" vs struggles_with)
    # retrouve quand même sa réponse par le sens.
    if EMBEDDER.available:
        for f in kg.facts.values():
            if f.valid_until is not None and not include_past:
                continue
            candidates[f.id] = f

    if not candidates:
        return RetrievedContext(matched_entities=matched_entity_keys, facts=[])

    cand = list(candidates.values())

    # Similarité sémantique, calculée en lot (requête vs chaque fait).
    sims: dict[str, float] = {}
    q_vec = EMBEDDER.encode_query(text)
    if q_vec is not None:
        doc_vecs = EMBEDDER.encode_docs([_fact_text(f) for f in cand])
        if doc_vecs is not None:
            sims = {f.id: EMBEDDER.cosine(q_vec, dv) for f, dv in zip(cand, doc_vecs)}

    now = time.time()
    scored: list[tuple[float, Fact]] = []
    for f in cand:
        overlap = len(set(_tokens(_fact_text(f))) & toks)
        age = max(0.0, now - f.last_active_at)
        recency = math.exp(-age / _RECENCY_HALF_LIFE_S)
        score = (
            f.confidence
            + 0.5 * overlap
            + 0.4 * recency
            + _SEM_WEIGHT * sims.get(f.id, 0.0)
        )
        if f.valid_until is not None:
            score -= 0.3  # les faits passés passent sous les faits actuels
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
