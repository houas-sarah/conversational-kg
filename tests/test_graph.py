from __future__ import annotations

import threading
from pathlib import Path

import pytest

from backend.graph import KnowledgeGraph


class TestBasicOperations:
    def test_add_returns_fact_with_id(self, kg: KnowledgeGraph):
        fact, sup = kg.add_fact("Alice", "likes", "tea")
        assert fact.id
        assert fact.timestamp > 0
        assert sup == []

    def test_entity_normalized_lowercase(self, kg: KnowledgeGraph):
        kg.add_fact("Alice", "likes", "tea")
        assert "alice" in kg.entities

    def test_active_facts_count(self, kg: KnowledgeGraph):
        kg.add_fact("a", "rel", "b")
        kg.add_fact("c", "rel", "d")
        assert len(kg.active_facts()) == 2


class TestConflictResolution:
    def test_one_to_one_predicate_supersedes(self, kg: KnowledgeGraph):
        kg.add_fact("user", "lives_in", "Algiers")
        new, sup = kg.add_fact("user", "lives_in", "Paris")
        assert len(sup) == 1
        assert sup[0].object == "Algiers"

    def test_multi_value_predicate_keeps_all(self, kg: KnowledgeGraph):
        for obj in ["tea", "coffee", "chocolate"]:
            kg.add_fact("user", "likes", obj)
        likes = [f for f in kg.active_facts() if f.predicate == "likes"]
        assert len(likes) == 3

    def test_chain_supersession_four_deep(self, kg: KnowledgeGraph):
        a, _ = kg.add_fact("user", "studies", "biology")
        b, _ = kg.add_fact("user", "studies", "chemistry")
        c, _ = kg.add_fact("user", "studies", "physics")
        d, _ = kg.add_fact("user", "studies", "math")
        assert kg.facts[a.id].superseded_by == b.id
        assert kg.facts[b.id].superseded_by == c.id
        assert kg.facts[c.id].superseded_by == d.id
        assert kg.facts[d.id].superseded_by is None
        actives = [f for f in kg.active_facts() if f.predicate == "studies"]
        assert len(actives) == 1
        assert actives[0].object == "math"


class TestPersistence:
    def test_save_and_reload_preserves_state(self, kg: KnowledgeGraph, tmp_path: Path):
        kg.add_fact("user", "likes", "pizza", confidence=0.95)
        kg.add_fact("user", "lives_in", "Rome")
        kg.add_fact("user", "lives_in", "Tokyo")
        kg.save()
        reloaded = KnowledgeGraph(kg.path)
        assert len(reloaded.entities) == len(kg.entities)
        assert len(reloaded.facts) == len(kg.facts)
        assert reloaded.stats()["conflicts"] == kg.stats()["conflicts"]
        active_lives = [f for f in reloaded.active_facts() if f.predicate == "lives_in"]
        assert len(active_lives) == 1
        assert active_lives[0].object == "Tokyo"

    def test_corrupted_json_does_not_crash(self, tmp_path: Path):
        bad = tmp_path / "kg.json"
        bad.write_text("{this is not valid json", encoding="utf-8")
        kg = KnowledgeGraph(bad)
        assert len(kg.entities) == 0
        assert len(kg.facts) == 0

    def test_missing_data_directory_autocreates(self, tmp_path: Path):
        path = tmp_path / "a" / "b" / "c" / "kg.json"
        kg = KnowledgeGraph(path)
        kg.add_fact("x", "y", "z")
        kg.save()
        assert path.exists()


class TestGraphSemantics:
    def test_case_insensitive_entity_identity(self, kg: KnowledgeGraph):
        kg.add_fact("Alice", "likes", "tea")
        kg.add_fact("alice", "likes", "coffee")
        kg.add_fact("ALICE", "likes", "water")
        alice_keys = [k for k in kg.entities if k == "alice"]
        assert len(alice_keys) == 1

    def test_self_loop_allowed(self, kg: KnowledgeGraph):
        f, _ = kg.add_fact("user", "knows", "user")
        assert f.id in kg.facts
        assert len(kg.entities) == 1

    def test_predicate_whitespace_normalizes(self, kg: KnowledgeGraph):
        f, _ = kg.add_fact("a", "lives in", "b")
        assert f.predicate == "lives_in"

    def test_two_hop_traversal_from_non_user_node(self, kg: KnowledgeGraph):
        kg.add_fact("calculus", "has_topic", "derivatives")
        kg.add_fact("derivatives", "type", "math")
        kg.add_fact("user", "studies", "calculus")
        facts = kg.facts_about("derivatives", depth=2)
        pairs = {(f.subject, f.object) for f in facts}
        assert ("calculus", "derivatives") in pairs
        assert ("user", "calculus") in pairs


class TestSnapshot:
    def test_no_dangling_edges(self, kg: KnowledgeGraph):
        kg.add_fact("a", "to", "b")
        kg.add_fact("b", "to", "c")
        kg.add_fact("c", "to", "a")
        snap = kg.snapshot()
        node_ids = {n["id"] for n in snap["nodes"]}
        for e in snap["edges"]:
            assert e["source"] in node_ids
            assert e["target"] in node_ids

    def test_stats_invariant(self, kg: KnowledgeGraph):
        kg.add_fact("u", "lives_in", "x")
        kg.add_fact("u", "lives_in", "y")
        kg.add_fact("u", "lives_in", "z")
        s = kg.stats()
        assert s["edges_total"] == s["edges_active"] + s["conflicts"]
        assert s["edges_active"] == 1
        assert s["conflicts"] == 2


class TestConcurrency:
    def test_threadsafe_writes_no_loss(self, kg: KnowledgeGraph):
        def worker(i: int):
            for j in range(20):
                kg.add_fact(f"u{i}", "knows", f"item{j}")
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(kg.facts) == 100
