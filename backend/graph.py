from __future__ import annotations

import dataclasses
import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import networkx as nx


@dataclass
class Fact:
    """A timestamped knowledge-graph edge with a validity interval.

    - asserted_at: when the user *told* us this fact.
    - valid_from:  when the fact *became true* in the world (default = asserted_at).
    - valid_until: when it stopped being true. None = still true now.

    A fact is "active" iff valid_until is None. Supersession (a conflicting
    new fact replacing this one) closes valid_until and sets superseded_by.
    Explicit retraction ("I don't X anymore") also closes valid_until but
    leaves superseded_by unset.
    """

    id: str
    subject: str
    predicate: str
    object: str
    asserted_at: float
    valid_from: float
    valid_until: float | None = None
    confidence: float = 0.8
    source_turn: int = 0
    superseded_by: str | None = None
    reinforcements: int = 0
    last_asserted_at: float | None = None

    @property
    def timestamp(self) -> float:
        """Backward-compat alias for asserted_at."""
        return self.asserted_at

    @property
    def last_active_at(self) -> float:
        """Most recent time the user asserted this fact (for recency scoring)."""
        return self.last_asserted_at or self.asserted_at

    @property
    def is_active(self) -> bool:
        return self.valid_until is None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Entity:
    name: str
    kind: str = "Concept"
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    mentions: int = 1


def _norm(s: str) -> str:
    return s.strip().lower()


_FACT_FIELDS = {f.name for f in dataclasses.fields(Fact)}


class KnowledgeGraph:
    """Property graph with time-aware facts and recency-based conflict handling.

    Storage is a NetworkX MultiDiGraph in memory plus a JSON file on disk.
    Each edge is a Fact (subject -predicate-> object) carrying validity
    intervals. Conflict resolution closes the old fact's valid_until; the
    explicit ``retract`` API closes it without designating a replacement.
    """

    CONFLICTING_PREDICATES = {
        "named", "is", "lives_in", "works_at", "studies", "studies_at",
        "has_supervisor", "has_status", "current_subject", "current_focus",
        "current_employer", "is_at", "located_in",
    }

    # Predicates that are mutually exclusive *about the same object*:
    # asserting one closes any active sibling on the same (subject, object).
    # "I understand derivatives now" supersedes "I struggle with derivatives";
    # "I hate coffee" supersedes "I like coffee".
    OBJECT_EXCLUSIVE_FAMILIES: list[frozenset[str]] = [
        frozenset({"likes", "loves", "prefers", "dislikes", "hates"}),
        frozenset({"understands", "struggles_with"}),
        frozenset({"working_on", "completed"}),
    ]

    _FAMILY_OF: dict[str, frozenset[str]] = {
        p: fam for fam in OBJECT_EXCLUSIVE_FAMILIES for p in fam
    }

    def __init__(self, storage_path: Path):
        self.path = Path(storage_path)
        self.g: nx.MultiDiGraph = nx.MultiDiGraph()
        self.entities: dict[str, Entity] = {}
        self.facts: dict[str, Fact] = {}
        self._lock = threading.RLock()
        self._load()

    # ── persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        # Entities
        for e in data.get("entities", []):
            self.entities[_norm(e["name"])] = Entity(**e)
            self.g.add_node(_norm(e["name"]), **e)

        # Facts — migrate old schema (single "timestamp" field) into the
        # asserted_at / valid_from / valid_until model.
        raw_facts: list[dict] = data.get("facts", [])
        for f in raw_facts:
            if "timestamp" in f and "asserted_at" not in f:
                f["asserted_at"] = f.pop("timestamp")
            f.setdefault("valid_from", f.get("asserted_at", time.time()))
            f.setdefault("valid_until", None)
            clean = {k: v for k, v in f.items() if k in _FACT_FIELDS}
            fact = Fact(**clean)
            self.facts[fact.id] = fact

        # For legacy facts that were marked superseded_by without an explicit
        # valid_until, derive validity from the superseding fact.
        for f in self.facts.values():
            if f.superseded_by and f.valid_until is None:
                succ = self.facts.get(f.superseded_by)
                if succ:
                    f.valid_until = succ.asserted_at

        # Register edges last so they carry the migrated data.
        for f in self.facts.values():
            self.g.add_edge(_norm(f.subject), _norm(f.object), key=f.id, **f.as_dict())

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "entities": [asdict(e) for e in self.entities.values()],
                "facts": [f.as_dict() for f in self.facts.values()],
            }
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    # ── entities ───────────────────────────────────────────────────────

    def upsert_entity(self, name: str, kind: str = "Concept") -> Entity:
        key = _norm(name)
        now = time.time()
        with self._lock:
            if key in self.entities:
                ent = self.entities[key]
                ent.last_seen = now
                ent.mentions += 1
                if kind != "Concept":
                    ent.kind = kind
            else:
                ent = Entity(name=name.strip(), kind=kind, first_seen=now, last_seen=now)
                self.entities[key] = ent
                self.g.add_node(key, **asdict(ent))
            self.g.nodes[key].update(asdict(ent))
            return ent

    # ── facts ──────────────────────────────────────────────────────────

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.8,
        source_turn: int = 0,
        valid_until: float | None = None,
    ) -> tuple[Fact, list[Fact]]:
        """Add a fact and return (new_fact, superseded_facts).

        If ``valid_until`` is provided (e.g. "I used to live in Paris"), the
        new fact is recorded as historical — already inactive at insertion —
        and never supersedes anything: a past fact cannot displace a current
        one. Otherwise it is active until something else closes it.

        Re-asserting an identical active fact reinforces it (confidence bump,
        recency refresh) instead of duplicating the edge.
        """
        with self._lock:
            self.upsert_entity(subject)
            self.upsert_entity(obj)
            predicate = _norm(predicate).replace(" ", "_")
            now = time.time()
            subj_n, obj_n = _norm(subject), _norm(obj)
            is_active_assertion = valid_until is None

            # Renforcement : si l'utilisateur répète exactement le même fait
            # actif, on le renforce (confiance + récence) au lieu de dupliquer
            # l'arête dans le graphe.
            if is_active_assertion:
                for f in self.facts.values():
                    if (
                        f.valid_until is None
                        and _norm(f.subject) == subj_n
                        and f.predicate == predicate
                        and _norm(f.object) == obj_n
                    ):
                        f.confidence = min(0.99, max(f.confidence, confidence) + 0.05)
                        f.reinforcements += 1
                        f.last_asserted_at = now
                        self._update_edge(f)
                        return f, []

            # Résolution de conflit — seule une assertion *active* peut fermer
            # des faits existants. Deux déclencheurs :
            #   1. prédicats un-à-un : même (sujet, prédicat), objet différent
            #   2. familles d'exclusivité : même (sujet, objet), prédicat frère
            #      (ex. "comprend" ferme "a du mal avec" sur le même objet)
            superseded: list[Fact] = []
            if is_active_assertion:
                family = self._FAMILY_OF.get(predicate)
                for f in self.facts.values():
                    if f.valid_until is not None or _norm(f.subject) != subj_n:
                        continue
                    one_to_one_clash = (
                        predicate in self.CONFLICTING_PREDICATES
                        and f.predicate == predicate
                        and _norm(f.object) != obj_n
                    )
                    family_clash = (
                        family is not None
                        and f.predicate != predicate
                        and f.predicate in family
                        and _norm(f.object) == obj_n
                    )
                    if one_to_one_clash or family_clash:
                        f.valid_until = now
                        f.superseded_by = "PENDING"
                        self._update_edge(f)
                        superseded.append(f)

            fact = Fact(
                id=uuid.uuid4().hex[:12],
                subject=subject.strip(),
                predicate=predicate,
                object=obj.strip(),
                asserted_at=now,
                valid_from=now,
                valid_until=valid_until,
                confidence=confidence,
                source_turn=source_turn,
            )
            for f in superseded:
                f.superseded_by = fact.id
                self._update_edge(f)
            self.facts[fact.id] = fact
            self.g.add_edge(_norm(subject), _norm(obj), key=fact.id, **fact.as_dict())
            return fact, superseded

    def retract(self, subject: str, predicate: str, obj: str) -> Fact | None:
        """Close an active fact matching (subject, predicate, object).

        Used when the user explicitly says a fact is no longer true
        ("I don't like coffee anymore"). Matches the exact predicate first;
        if none, falls back to siblings in the same exclusivity family so
        "don't like coffee anymore" also closes a stored "loves coffee".
        Returns the retracted fact, or None if no active match was found.
        """
        with self._lock:
            now = time.time()
            pred_n = _norm(predicate).replace(" ", "_")
            subj_n, obj_n = _norm(subject), _norm(obj)
            family = self._FAMILY_OF.get(pred_n, frozenset())
            exact: Fact | None = None
            sibling: Fact | None = None
            for f in self.facts.values():
                if (
                    f.valid_until is not None
                    or _norm(f.subject) != subj_n
                    or _norm(f.object) != obj_n
                ):
                    continue
                if f.predicate == pred_n:
                    exact = f
                    break
                if sibling is None and f.predicate in family:
                    sibling = f
            target = exact or sibling
            if target is not None:
                target.valid_until = now
                self._update_edge(target)
            return target

    def _update_edge(self, f: Fact) -> None:
        """Sync a fact's mutated state into its NetworkX edge data."""
        s, t = _norm(f.subject), _norm(f.object)
        if self.g.has_edge(s, t, key=f.id):
            self.g[s][t][f.id].update(f.as_dict())

    # ── queries ────────────────────────────────────────────────────────

    def active_facts(self) -> list[Fact]:
        return [f for f in self.facts.values() if f.valid_until is None]

    def facts_about(self, entity: str, depth: int = 2, include_inactive: bool = False) -> list[Fact]:
        key = _norm(entity)
        if key not in self.g:
            return []
        seen: set[str] = set()
        frontier = {key}
        out: list[Fact] = []
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                if node in seen:
                    continue
                seen.add(node)
                for _, tgt, k, _data in self.g.out_edges(node, keys=True, data=True):
                    f = self.facts.get(k)
                    if f and (include_inactive or f.valid_until is None):
                        out.append(f)
                        next_frontier.add(tgt)
                for src, _, k, _data in self.g.in_edges(node, keys=True, data=True):
                    f = self.facts.get(k)
                    if f and (include_inactive or f.valid_until is None):
                        out.append(f)
                        next_frontier.add(src)
            frontier = next_frontier - seen
        seen_ids: set[str] = set()
        deduped: list[Fact] = []
        for f in sorted(out, key=lambda x: x.asserted_at, reverse=True):
            if f.id in seen_ids:
                continue
            seen_ids.add(f.id)
            deduped.append(f)
        return deduped

    # ── views ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """JSON snapshot for the frontend graph viz."""
        nodes = [
            {
                "id": key,
                "label": ent.name,
                "kind": ent.kind,
                "mentions": ent.mentions,
                "last_seen": ent.last_seen,
            }
            for key, ent in self.entities.items()
        ]
        edges = [
            {
                "id": f.id,
                "source": _norm(f.subject),
                "target": _norm(f.object),
                "predicate": f.predicate,
                "timestamp": f.asserted_at,        # legacy alias
                "asserted_at": f.asserted_at,
                "valid_from": f.valid_from,
                "valid_until": f.valid_until,
                "confidence": f.confidence,
                "superseded": f.valid_until is not None,
            }
            for f in self.facts.values()
        ]
        return {"nodes": nodes, "edges": edges}

    def stats(self) -> dict:
        active = self.active_facts()
        return {
            "nodes": len(self.entities),
            "edges_total": len(self.facts),
            "edges_active": len(active),
            "conflicts": len(self.facts) - len(active),
        }
