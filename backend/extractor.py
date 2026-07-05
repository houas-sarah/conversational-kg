from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .llm import GroqClient

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None


@dataclass
class ExtractedTriple:
    subject: str
    predicate: str
    object: str
    confidence: float
    op: str = "assert"             # "assert" or "retract"
    valid_until: str | None = None  # "now" sentinel or None


@dataclass
class Extraction:
    entities: list[dict]
    triples: list[ExtractedTriple]
    method: str
    # The user's message rewritten with pronouns/references resolved to real
    # names. Equal to the raw text when there was nothing to resolve.
    resolved_text: str = ""


SPACY_LABEL_TO_KIND = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "DATE": "Time",
    "TIME": "Time",
    "PRODUCT": "Product",
    "EVENT": "Event",
    "WORK_OF_ART": "Work",
    "LANGUAGE": "Language",
    "NORP": "Group",
    "FAC": "Facility",
    "MONEY": "Quantity",
    "QUANTITY": "Quantity",
}


# ── Predicate canonicalisation ───────────────────────────────────────────
#
# The LLM (and the rules) can emit many synonyms for the same relation. Without
# canonicalisation the graph fragments — `studies` / `is_studying` / `learns`
# become three different edges. We map every emitted predicate through a
# controlled vocabulary; unknown predicates pass through unchanged but are
# normalised to snake_case so at least casing/spacing don't fragment further.

PREDICATE_VOCAB: dict[str, list[str]] = {
    "named":            ["named", "is_called", "goes_by", "known_as"],
    "is":               ["is", "is_a"],
    "lives_in":         ["lives_in", "resides_in", "based_in", "is_from", "from"],
    "works_at":         ["works_at", "employed_by", "works_for"],
    "current_employer": ["current_employer"],
    "studies":          ["studies", "is_studying", "learns", "learning", "majoring_in", "taking"],
    "studies_at":       ["studies_at", "attends", "enrolled_in", "student_at", "goes_to"],
    "has_friend":       ["has_friend", "is_friend_of", "befriended", "friend_with"],
    "has_supervisor":   ["has_supervisor", "is_supervised_by", "advisor_is", "supervised_by"],
    "likes":            ["likes", "enjoys", "fond_of"],
    "loves":            ["loves", "adores"],
    "dislikes":         ["dislikes", "doesnt_like", "doesn_t_like"],
    "hates":            ["hates", "detests"],
    "prefers":          ["prefers", "would_rather"],
    "knows":            ["knows", "is_familiar_with", "familiar_with"],
    "understands":      ["understands", "gets", "grasps"],
    "struggles_with":   ["struggles_with", "stuck_on", "having_trouble_with",
                         "finds_difficult", "needs_help_with", "needs_help"],
    "working_on":       ["working_on", "currently_doing"],
    "completed":        ["completed", "finished", "done_with"],
    "feels":            ["feels", "is_feeling"],
    "wants":            ["wants", "wishes", "would_like"],
    "speaks":           ["speaks", "talks_in"],
    "mentioned":        ["mentioned"],
}

_SYN_TO_CANONICAL: dict[str, str] = {
    syn: canon for canon, syns in PREDICATE_VOCAB.items() for syn in syns
}


def _canonicalize_predicate(pred: str) -> str:
    """Map a predicate to its canonical form via the controlled vocabulary.

    Unknown predicates are returned snake_cased; the graph still stores them
    but they will sit in their own bucket until the vocab is extended.
    """
    p = pred.strip().lower().replace(" ", "_").replace("-", "_")
    return _SYN_TO_CANONICAL.get(p, p)


# ── Rule-based fallback patterns ─────────────────────────────────────────

_OBJ = r"([A-Za-z][\w\s\-]{0,40}?)(?=\s+(?:and|but|because|so)\b|\s*[.,!?;]|\s*$)"

_PATTERNS = [
    (re.compile(rf"\bi(?:'m| am)\s+(?:really\s+|kind of\s+|so\s+)?struggling with\s+{_OBJ}", re.I), "user", "struggles_with"),
    (re.compile(rf"\bi(?:'m| am)\s+stuck on\s+{_OBJ}", re.I), "user", "struggles_with"),
    (re.compile(rf"\bi need help with\s+{_OBJ}", re.I), "user", "needs_help_with"),
    (re.compile(rf"\bi(?:'m| am)\s+working on\s+{_OBJ}", re.I), "user", "working_on"),
    (re.compile(rf"\bi(?:'m| am)\s+studying\s+{_OBJ}", re.I), "user", "studies"),
    (re.compile(rf"\bi study\s+{_OBJ}", re.I), "user", "studies"),
    (re.compile(rf"\bi like\s+{_OBJ}", re.I), "user", "likes"),
    (re.compile(rf"\bi love\s+{_OBJ}", re.I), "user", "loves"),
    (re.compile(rf"\bi prefer\s+{_OBJ}", re.I), "user", "prefers"),
    (re.compile(rf"\bi (?:hate|dislike)\s+{_OBJ}", re.I), "user", "dislikes"),
    (re.compile(rf"\bi work (?:at|for)\s+{_OBJ}", re.I), "user", "works_at"),
    (re.compile(rf"\bi live in\s+{_OBJ}", re.I), "user", "lives_in"),
    (re.compile(r"\bmy name is\s+([A-Z][a-z]+)\b", re.I), "user", "named"),
    (re.compile(rf"\bi understand\s+{_OBJ}", re.I), "user", "understands"),
    (re.compile(rf"\bi know\s+{_OBJ}", re.I), "user", "knows"),
    (re.compile(rf"\bi(?:'ve| have)\s+finished\s+{_OBJ}", re.I), "user", "completed"),
    (re.compile(rf"\bi finished\s+{_OBJ}", re.I), "user", "completed"),
    (re.compile(rf"\bi feel\s+{_OBJ}", re.I), "user", "feels"),
    (re.compile(rf"\bi want\s+(?:to\s+)?{_OBJ}", re.I), "user", "wants"),
]

# Pronouns must never end up as graph nodes.
_SELF_PRONOUNS = {"i", "me", "myself"}
_BARE_PRONOUNS = {
    "she", "he", "it", "they", "them", "her", "him", "hers", "his",
    "their", "theirs", "this", "that", "these", "those", "we", "us",
    "someone", "something", "anyone", "anything", "everyone",
}
# Vague placeholders the LLM sometimes emits instead of a real entity.
_JUNK_NODES = {"unknown", "unclear", "n/a", "na", "none", "nothing", "stuff", "things", "thing"}

_STOPWORDS_TAIL = {
    "it", "this", "that", "them", "those", "these", "things", "stuff", "some",
    "her", "him", "he", "she", "they", "we", "us", "you",
}

_TRAILING_ADVERBS = re.compile(
    r"\b(?:lately|recently|today|yesterday|tomorrow|now|sometimes|usually|often|always|"
    r"still|currently|actually|eventually|already|too|very|really|quite|kind of|sort of|"
    r"this (?:semester|year|month|week|morning|evening|afternoon)|next (?:week|month|year))\b\s*$",
    re.IGNORECASE,
)


def _clean_object(s: str) -> str:
    s = s.strip().rstrip(".,!?;:").lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(?:the|a|an|some)\s+", "", s)
    prev = None
    while prev != s:
        prev = s
        s = _TRAILING_ADVERBS.sub("", s).strip()
    return s


def _sanitize_triples(triples: list[ExtractedTriple]) -> list[ExtractedTriple]:
    """Map first-person pronouns to 'user'; drop triples with unresolved pronoun
    or junk-placeholder slots."""
    out: list[ExtractedTriple] = []
    for t in triples:
        subj = "user" if t.subject in _SELF_PRONOUNS else t.subject
        obj = "user" if t.object in _SELF_PRONOUNS else t.object
        if not subj or not obj or not t.predicate:
            continue
        if subj in _BARE_PRONOUNS or obj in _BARE_PRONOUNS:
            continue
        if subj in _JUNK_NODES or obj in _JUNK_NODES:
            continue
        op = "retract" if t.op == "retract" else "assert"
        out.append(ExtractedTriple(
            subject=subj,
            predicate=_canonicalize_predicate(t.predicate),
            object=obj,
            confidence=t.confidence,
            op=op,
            valid_until=t.valid_until,
        ))
    return out


def _spacy_entities(text: str) -> list[dict]:
    if _NLP is None:
        return []
    doc = _NLP(text)
    out = []
    for ent in doc.ents:
        kind = SPACY_LABEL_TO_KIND.get(ent.label_, "Concept")
        out.append({"text": ent.text, "label": ent.label_, "kind": kind})
    return out


def _spacy_noun_chunks(text: str) -> list[str]:
    if _NLP is None:
        return []
    doc = _NLP(text)
    return [chunk.text.lower().strip() for chunk in doc.noun_chunks if len(chunk.text) > 2]


def _rule_extract(text: str) -> list[ExtractedTriple]:
    triples: list[ExtractedTriple] = []
    seen: set[tuple[str, str, str]] = set()
    for pat, subj, pred in _PATTERNS:
        for m in pat.finditer(text):
            obj = _clean_object(m.group(1))
            if not obj or obj in _STOPWORDS_TAIL or len(obj) > 60:
                continue
            key = (subj, pred, obj)
            if key in seen:
                continue
            seen.add(key)
            triples.append(ExtractedTriple(
                subject=subj, predicate=pred, object=obj, confidence=0.75,
            ))
    return triples


# ── LLM prompt ───────────────────────────────────────────────────────────

LLM_SYSTEM = """You extract knowledge-graph triples from one turn of an ongoing conversation, for a long-running conversational memory.

You are given the recent CONVERSATION SO FAR and the NEW USER MESSAGE. Always read the new message in the context of that conversation.

Return ONLY a JSON object with this exact shape:
{
  "resolved_text": "...",
  "entities": [{"name": "...", "kind": "Person|Location|Organization|Subject|Topic|Activity|Object|Concept|Time|Feeling"}],
  "triples": [
    {
      "op": "assert" | "retract",
      "subject": "...",
      "predicate": "<snake_case_verb>",
      "object": "...",
      "valid_until": "now" | null,
      "confidence": 0.0-1.0
    }
  ]
}

WHAT TO EXTRACT:
- A triple for EVERY relationship stated -- including those that do not involve the speaker.
- Use "user" (lowercase) only for the speaker; use real names for everyone else.
- Predicates are snake_case present-tense verbs. Prefer these canonical forms when applicable: named, lives_in, works_at, studies, studies_at, has_friend, has_supervisor, likes, loves, dislikes, hates, prefers, wants, feels, knows, understands, struggles_with, working_on, completed.
- Keep objects short (1-4 words). Extract only what is stated -- never invent.
- QUESTIONS: a question states no new facts -- return an empty "triples" array (still fill "resolved_text").

OPERATIONS & TIME:
- "op" defaults to "assert". Use "retract" when the user says a previously-true fact is NO LONGER true:
  - "I don't like coffee anymore" -> retract on (user, likes, coffee).
  - "I no longer study biology"  -> retract on (user, studies, biology).
  - "I quit my job at X"          -> retract on (user, works_at, X).
- For statements about something the user USED TO do (already in the past, no longer true): emit op="assert" with valid_until="now". This records the historical truth.
  - "I used to live in Paris" -> {"op":"assert", ..., "predicate":"lives_in", "object":"Paris", "valid_until":"now"}.
- A plain present-tense negation is a NEW fact, not a retraction: "I don't like broccoli" -> assert (user, dislikes, broccoli). Use "retract" ONLY with change signals like "anymore", "no longer", "not ... any more", "quit", "stopped".
- Otherwise valid_until is null.

COREFERENCE:
- Replace every pronoun/reference (she, he, it, they, them, her, him, this, that, ...) with the real name using the conversation.
- "resolved_text" is the new user message rewritten as a standalone sentence with those replacements. If nothing to resolve, copy unchanged.
- NEVER output a pronoun as subject or object. If unresolved, skip that triple.

EXAMPLE 1 -- assertion with multiple relations
CONVERSATION SO FAR: (no earlier messages -- this is the first turn)
NEW USER MESSAGE: My friend Lina studies physics at Riverside University.
OUTPUT:
{"resolved_text": "My friend Lina studies physics at Riverside University.",
 "entities": [{"name": "Lina", "kind": "Person"}, {"name": "physics", "kind": "Subject"}, {"name": "Riverside University", "kind": "Organization"}],
 "triples": [{"op": "assert", "subject": "user", "predicate": "has_friend", "object": "Lina", "valid_until": null, "confidence": 0.95},
             {"op": "assert", "subject": "Lina", "predicate": "studies", "object": "physics", "valid_until": null, "confidence": 0.95},
             {"op": "assert", "subject": "Lina", "predicate": "studies_at", "object": "Riverside University", "valid_until": null, "confidence": 0.9}]}

EXAMPLE 2 -- coreference + assertion
CONVERSATION SO FAR:
User: My friend Lina studies physics at Riverside University.
Assistant: Got it -- noted Lina.
NEW USER MESSAGE: She's been struggling with quantum mechanics.
OUTPUT:
{"resolved_text": "Lina has been struggling with quantum mechanics.",
 "entities": [{"name": "quantum mechanics", "kind": "Subject"}],
 "triples": [{"op": "assert", "subject": "Lina", "predicate": "struggles_with", "object": "quantum mechanics", "valid_until": null, "confidence": 0.9}]}

EXAMPLE 3 -- a question (no new facts)
CONVERSATION SO FAR:
User: My friend Lina studies physics at Riverside University.
Assistant: Got it.
NEW USER MESSAGE: What is she working on?
OUTPUT:
{"resolved_text": "What is Lina working on?", "entities": [], "triples": []}

EXAMPLE 4 -- explicit retraction
CONVERSATION SO FAR:
User: I love coffee.
Assistant: Good to know.
NEW USER MESSAGE: Actually I don't like coffee anymore.
OUTPUT:
{"resolved_text": "I don't like coffee anymore.",
 "entities": [{"name": "coffee", "kind": "Object"}],
 "triples": [{"op": "retract", "subject": "user", "predicate": "loves", "object": "coffee", "valid_until": null, "confidence": 0.9}]}

EXAMPLE 5 -- historical fact (used to)
CONVERSATION SO FAR: (no earlier messages)
NEW USER MESSAGE: I used to live in Paris.
OUTPUT:
{"resolved_text": "I used to live in Paris.",
 "entities": [{"name": "Paris", "kind": "Location"}],
 "triples": [{"op": "assert", "subject": "user", "predicate": "lives_in", "object": "Paris", "valid_until": "now", "confidence": 0.9}]}

If nothing is extractable, return empty "entities" and "triples" but still fill "resolved_text". Output ONLY the JSON object, no prose."""


# ── Extractor ────────────────────────────────────────────────────────────


class HybridExtractor:
    def __init__(self, llm: Optional[GroqClient] = None):
        self.llm = llm or GroqClient()

    async def extract(self, text: str, history: str = "") -> Extraction:
        """Extract entities and triples from one turn.

        `history` is the recent conversation transcript; it lets the LLM
        resolve pronouns ("she") and detect retractions ("anymore") before
        any triple touches the graph.

        Method reporting policy: if the LLM call succeeds (no exception), its
        output is authoritative and we report ``hybrid:llm+spacy`` — even
        when it returns zero triples (e.g. for a question). The rule path
        runs only when the LLM is unavailable or threw.
        """
        entities = _spacy_entities(text)
        resolved_text = text

        if self.llm.available:
            try:
                user_prompt = (
                    f"CONVERSATION SO FAR:\n{history or '(no earlier messages -- this is the first turn)'}\n\n"
                    f"NEW USER MESSAGE:\n{text}"
                )
                data = await self.llm.chat_json(LLM_SYSTEM, user_prompt, temperature=0.1, max_tokens=700)

                resolved = str(data.get("resolved_text", "") or "").strip()
                if resolved:
                    resolved_text = resolved

                for e in data.get("entities", []) or []:
                    name = str(e.get("name", "")).strip()
                    if name and not any(name.lower() == x["text"].lower() for x in entities):
                        entities.append({"text": name, "label": "LLM", "kind": e.get("kind", "Concept")})

                triples = _sanitize_triples([
                    ExtractedTriple(
                        subject=str(t.get("subject", "user")).strip().lower(),
                        predicate=str(t.get("predicate", "")).strip().lower().replace(" ", "_"),
                        object=str(t.get("object", "")).strip().lower(),
                        confidence=float(t.get("confidence", 0.85)),
                        op=str(t.get("op", "assert")).strip().lower(),
                        valid_until=(t.get("valid_until") if t.get("valid_until") in ("now", None) else None),
                    )
                    for t in (data.get("triples", []) or [])
                    if t.get("predicate") and t.get("object")
                ])
                # LLM is authoritative when it responds successfully — even
                # with an empty triples list (e.g. a question).
                return Extraction(entities, triples, "hybrid:llm+spacy", resolved_text)
            except Exception:
                pass  # fall through to rules

        # Rule-based fallback — only when the LLM is unavailable or threw.
        triples = _rule_extract(text)
        if not triples:
            for c in _spacy_noun_chunks(text)[:3]:
                triples.append(ExtractedTriple(
                    subject="user", predicate="mentioned", object=c, confidence=0.4,
                ))
        triples = _sanitize_triples(triples)
        return Extraction(entities, triples, "rules+spacy", resolved_text)
