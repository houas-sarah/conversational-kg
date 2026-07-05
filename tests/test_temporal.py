from __future__ import annotations

import time

import pytest

from backend.graph import KnowledgeGraph
from backend.extractor import _canonicalize_predicate


class TestRetract:
    def test_retract_closes_active_fact(self, kg: KnowledgeGraph):
        kg.add_fact("user", "likes", "coffee")
        r = kg.retract("user", "likes", "coffee")
        assert r is not None
        assert r.valid_until is not None
        assert r.is_active is False
        actives = [f for f in kg.active_facts() if f.predicate == "likes"]
        assert len(actives) == 0

    def test_retract_missing_returns_none(self, kg: KnowledgeGraph):
        assert kg.retract("user", "likes", "absinthe") is None

    def test_retract_does_not_affect_other_facts(self, kg: KnowledgeGraph):
        kg.add_fact("user", "likes", "coffee")
        kg.add_fact("user", "likes", "tea")
        kg.retract("user", "likes", "coffee")
        active_likes = sorted(f.object for f in kg.active_facts() if f.predicate == "likes")
        assert active_likes == ["tea"]

    def test_retract_persists_through_save_reload(self, kg: KnowledgeGraph):
        kg.add_fact("user", "likes", "coffee")
        kg.retract("user", "likes", "coffee")
        kg.save()
        reloaded = KnowledgeGraph(kg.path)
        likes = [f for f in reloaded.facts.values() if f.predicate == "likes"]
        assert len(likes) == 1                  # historical row preserved
        assert likes[0].valid_until is not None  # but inactive
        assert likes[0] not in reloaded.active_facts()


class TestValidity:
    def test_add_with_valid_until_marks_inactive(self, kg: KnowledgeGraph):
        past = time.time() - 1
        fact, _ = kg.add_fact("user", "lives_in", "Paris", valid_until=past)
        assert fact.valid_until == past
        assert fact not in kg.active_facts()

    def test_supersession_sets_valid_until_and_superseded_by(self, kg: KnowledgeGraph):
        first, _ = kg.add_fact("user", "lives_in", "Algiers")
        assert first.valid_until is None
        second, sup = kg.add_fact("user", "lives_in", "Paris")
        assert [f.id for f in sup] == [first.id]
        assert kg.facts[first.id].valid_until is not None
        assert kg.facts[first.id].superseded_by == second.id

    def test_legacy_kg_json_migrates_to_validity_fields(self, tmp_path):
        # Simulate the pre-temporal schema on disk.
        import json
        legacy = {
            "entities": [
                {"name": "user", "kind": "Person", "first_seen": 1.0, "last_seen": 1.0, "mentions": 1},
                {"name": "paris", "kind": "Location", "first_seen": 1.0, "last_seen": 1.0, "mentions": 1},
                {"name": "tokyo", "kind": "Location", "first_seen": 2.0, "last_seen": 2.0, "mentions": 1},
            ],
            "facts": [
                {"id": "a", "subject": "user", "predicate": "lives_in", "object": "paris",
                 "timestamp": 1.0, "confidence": 0.8, "source_turn": 1, "superseded_by": "b"},
                {"id": "b", "subject": "user", "predicate": "lives_in", "object": "tokyo",
                 "timestamp": 2.0, "confidence": 0.8, "source_turn": 2, "superseded_by": None},
            ],
        }
        path = tmp_path / "kg.json"
        path.write_text(json.dumps(legacy), encoding="utf-8")
        kg = KnowledgeGraph(path)
        # superseded fact got a derived valid_until from the successor's asserted_at
        assert kg.facts["a"].valid_until == 2.0
        assert kg.facts["a"].asserted_at == 1.0
        assert kg.facts["b"].valid_until is None
        active = [f.object for f in kg.active_facts() if f.predicate == "lives_in"]
        assert active == ["tokyo"]


class TestCanonicalization:
    @pytest.mark.parametrize("synonym, canonical", [
        ("studies", "studies"),
        ("is_studying", "studies"),
        ("learns", "studies"),
        ("attends", "studies_at"),
        ("enrolled_in", "studies_at"),
        ("student_at", "studies_at"),
        ("doesnt_like", "dislikes"),
        ("Studies", "studies"),         # case-insensitive
        ("studies at", "studies_at"),    # whitespace → underscore
        ("is-friend-of", "has_friend"),  # hyphens too
    ])
    def test_canonicalize_maps_synonyms(self, synonym, canonical):
        assert _canonicalize_predicate(synonym) == canonical

    def test_canonicalize_passes_through_unknown(self):
        assert _canonicalize_predicate("totally_made_up_predicate") == "totally_made_up_predicate"
