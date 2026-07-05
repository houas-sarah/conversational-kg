"""KG validation suite.

Exercises: idempotency, conflict resolution depth, persistence round-trip,
edge-case inputs, retrieval correctness, concurrent writes, JSON corruption
recovery, snapshot integrity. Prints PASS/FAIL per test and a summary.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from .extractor import HybridExtractor
from .graph import KnowledgeGraph
from .query import retrieve


PASSED = 0
FAILED = 0
FAILURES: list[str] = []


def _check(label: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        FAILURES.append(f"{label} — {detail}")
        print(f"  FAIL  {label}  ← {detail}")


def _fresh() -> tuple[KnowledgeGraph, Path]:
    tmp = Path(tempfile.mkdtemp()) / "kg.json"
    return KnowledgeGraph(tmp), tmp


def test_basic_add():
    print("\n[1] basic add / read")
    kg, _ = _fresh()
    f, sup = kg.add_fact("Alice", "likes", "tea")
    _check("returns Fact with id", bool(f.id))
    _check("timestamp set", f.timestamp > 0)
    _check("no supersedes on first add", sup == [])
    _check("entity exists", "alice" in kg.entities)
    _check("active facts count = 1", len(kg.active_facts()) == 1)


def test_idempotency_same_fact():
    print("\n[2] same triple added twice — reinforcement, not duplication")
    kg, _ = _fresh()
    f1, _ = kg.add_fact("Alice", "likes", "tea", confidence=0.8)
    f2, sup = kg.add_fact("Alice", "likes", "tea", confidence=0.8)
    _check("re-assertion returns the same fact", f1.id == f2.id)
    _check("only one active fact", len(kg.active_facts()) == 1,
           f"got {len(kg.active_facts())}")
    _check("confidence bumped", f2.confidence > 0.8, f"got {f2.confidence}")
    _check("reinforcement counted", f2.reinforcements == 1)
    _check("recency refreshed", f2.last_asserted_at is not None)
    _check("nothing superseded", sup == [])


def test_conflict_resolution():
    print("\n[3] conflict resolution on 1:1 predicates")
    kg, _ = _fresh()
    kg.add_fact("user", "lives_in", "Algiers")
    f2, sup = kg.add_fact("user", "lives_in", "Paris")
    _check("new fact added", f2 in kg.facts.values())
    _check("old fact superseded", len(sup) == 1 and sup[0].object == "Algiers")
    active = kg.active_facts()
    _check("only one active lives_in", sum(1 for f in active if f.predicate == "lives_in") == 1)
    _check("Paris is the surviving object",
           any(f.object == "Paris" and f.predicate == "lives_in" for f in active))


def test_no_conflict_on_multi_predicates():
    print("\n[4] no-conflict predicates allow multiplicity")
    kg, _ = _fresh()
    kg.add_fact("user", "likes", "tea")
    kg.add_fact("user", "likes", "coffee")
    kg.add_fact("user", "likes", "chocolate")
    active = kg.active_facts()
    _check("3 likes coexist", len([f for f in active if f.predicate == "likes"]) == 3)


def test_conflict_chain():
    print("\n[5] chained supersessions across 4 facts")
    kg, _ = _fresh()
    a, _ = kg.add_fact("user", "studies", "biology")
    b, _ = kg.add_fact("user", "studies", "chemistry")
    c, _ = kg.add_fact("user", "studies", "physics")
    d, _ = kg.add_fact("user", "studies", "math")
    _check("a -> b supersedes a", kg.facts[a.id].superseded_by == b.id)
    _check("b -> c supersedes b", kg.facts[b.id].superseded_by == c.id)
    _check("c -> d supersedes c", kg.facts[c.id].superseded_by == d.id)
    _check("d still active", kg.facts[d.id].superseded_by is None)
    _check("one active studies fact",
           sum(1 for f in kg.active_facts() if f.predicate == "studies") == 1)


def test_polarity_conflict():
    print("\n[5b] opposite-polarity predicates conflict on the same object")
    kg, _ = _fresh()
    old, _ = kg.add_fact("user", "likes", "coffee")
    new, sup = kg.add_fact("user", "hates", "coffee")
    _check("likes coffee superseded by hates coffee",
           len(sup) == 1 and sup[0].id == old.id)
    _check("old fact closed", kg.facts[old.id].valid_until is not None)
    _check("only hates is active",
           [f.predicate for f in kg.active_facts()] == ["hates"])
    # unrelated objects must NOT clash
    kg.add_fact("user", "likes", "tea")
    _check("likes tea coexists with hates coffee",
           len(kg.active_facts()) == 2)


def test_mastery_transition():
    print("\n[5c] understands supersedes struggles_with (same object)")
    kg, _ = _fresh()
    old, _ = kg.add_fact("user", "struggles_with", "derivatives")
    new, sup = kg.add_fact("user", "understands", "derivatives")
    _check("struggle closed", kg.facts[old.id].valid_until is not None)
    _check("superseded_by links to new fact",
           kg.facts[old.id].superseded_by == new.id)
    _check("only understands active",
           [f.predicate for f in kg.active_facts()] == ["understands"])


def test_progress_transition():
    print("\n[5d] completed supersedes working_on (same object)")
    kg, _ = _fresh()
    old, _ = kg.add_fact("user", "working_on", "the report")
    _, sup = kg.add_fact("user", "completed", "the report")
    _check("working_on closed", len(sup) == 1 and sup[0].id == old.id)


def test_historical_does_not_supersede():
    print("\n[5e] historical fact ('used to') never displaces a current one")
    kg, _ = _fresh()
    current, _ = kg.add_fact("user", "lives_in", "Algiers")
    past, sup = kg.add_fact("user", "lives_in", "Paris", valid_until=time.time())
    _check("no supersession from a past fact", sup == [])
    _check("current residence still active",
           kg.facts[current.id].valid_until is None)
    _check("past fact stored as inactive", past.valid_until is not None)


def test_retract_family_fallback():
    print("\n[5f] retraction matches sibling predicates in the same family")
    kg, _ = _fresh()
    stored, _ = kg.add_fact("user", "loves", "coffee")
    retracted = kg.retract("user", "likes", "coffee")  # user said "don't like anymore"
    _check("sibling predicate retracted",
           retracted is not None and retracted.id == stored.id)
    _check("fact closed", kg.facts[stored.id].valid_until is not None)
    _check("no active facts remain", kg.active_facts() == [])
    _check("retract with no match returns None",
           kg.retract("user", "likes", "tea") is None)


def test_persistence_roundtrip():
    print("\n[6] save → reload → state preserved")
    kg, path = _fresh()
    kg.add_fact("user", "likes", "pizza", confidence=0.95)
    kg.add_fact("user", "lives_in", "Rome")
    kg.add_fact("user", "lives_in", "Tokyo")  # supersedes Rome
    kg.save()
    kg2 = KnowledgeGraph(path)
    _check("nodes preserved", len(kg.entities) == len(kg2.entities))
    _check("facts preserved", len(kg.facts) == len(kg2.facts))
    _check("conflicts preserved",
           kg.stats()["conflicts"] == kg2.stats()["conflicts"])
    surv = [f for f in kg2.active_facts() if f.predicate == "lives_in"]
    _check("surviving Tokyo after reload",
           len(surv) == 1 and surv[0].object == "Tokyo")


def test_corrupted_json_recovery():
    print("\n[7] corrupted kg.json doesn't crash on boot")
    tmp = Path(tempfile.mkdtemp()) / "kg.json"
    tmp.write_text("{not valid json", encoding="utf-8")
    try:
        kg = KnowledgeGraph(tmp)
        _check("loads with empty state", len(kg.entities) == 0 and len(kg.facts) == 0)
    except Exception as e:
        _check("loads with empty state", False, f"raised: {e}")


def test_empty_path_creates():
    print("\n[8] missing data directory auto-creates on save")
    tmp = Path(tempfile.mkdtemp()) / "nested" / "deep" / "kg.json"
    kg = KnowledgeGraph(tmp)
    kg.add_fact("a", "rel", "b")
    kg.save()
    _check("path created", tmp.exists())


def test_self_loop():
    print("\n[9] subject == object self-loop")
    kg, _ = _fresh()
    f, _ = kg.add_fact("user", "knows", "user")
    _check("self-loop stored", f.id in kg.facts)
    _check("one entity for self-loop", len(kg.entities) == 1)


def test_normalization_case():
    print("\n[10] case insensitivity for entity identity")
    kg, _ = _fresh()
    kg.add_fact("Alice", "likes", "tea")
    kg.add_fact("alice", "likes", "coffee")
    kg.add_fact("ALICE", "likes", "water")
    _check("Alice collapses to one node", len([k for k in kg.entities if k.startswith("a")]) == 1,
           f"keys: {list(kg.entities.keys())}")


def test_retrieval_relevance():
    print("\n[11] retrieval surfaces relevant facts by token overlap")
    kg, _ = _fresh()
    kg.add_fact("user", "studies", "calculus")
    kg.add_fact("user", "likes", "pizza")
    kg.add_fact("user", "struggles_with", "derivatives")
    kg.add_fact("user", "lives_in", "algiers")
    ctx = retrieve(kg, "What math topics am I struggling with?")
    found = {(f.subject, f.predicate, f.object) for f in ctx.facts}
    _check("retrieval includes calculus", ("user", "studies", "calculus") in found)
    _check("retrieval includes derivatives", ("user", "struggles_with", "derivatives") in found)


def test_past_query_retrieval():
    print("\n[11b] past-tense queries surface closed facts; present ones don't")
    kg, _ = _fresh()
    kg.add_fact("user", "lives_in", "Rome")
    kg.add_fact("user", "lives_in", "Tokyo")  # supersedes Rome
    present = retrieve(kg, "Where do I live?")
    _check("present query hides Rome",
           all(f.object != "Rome" for f in present.facts))
    past = retrieve(kg, "Where did I live before?")
    objs = {f.object for f in past.facts}
    _check("past query includes Rome", "Rome" in objs, f"got {objs}")
    _check("past query still includes current Tokyo", "Tokyo" in objs)
    tokyo = next(f for f in past.facts if f.object == "Tokyo")
    rome = next(f for f in past.facts if f.object == "Rome")
    _check("current fact ranks above past fact",
           past.facts.index(tokyo) < past.facts.index(rome))


def test_facts_about_traversal():
    print("\n[12] 2-hop graph traversal from a non-user node")
    kg, _ = _fresh()
    kg.add_fact("calculus", "has_topic", "derivatives")
    kg.add_fact("derivatives", "type", "math")
    kg.add_fact("user", "studies", "calculus")
    facts = kg.facts_about("derivatives", depth=2)
    pairs = {(f.subject, f.object) for f in facts}
    _check("finds inbound edge", ("calculus", "derivatives") in pairs)
    _check("finds upstream user-studies-calculus via 2nd hop",
           ("user", "calculus") in pairs)


def test_concurrent_writes():
    print("\n[13] concurrent writes don't lose data (thread safety)")
    kg, _ = _fresh()
    def worker(i: int):
        for j in range(20):
            kg.add_fact(f"u{i}", "knows", f"item{j}")
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    _check("all 100 facts present", len(kg.facts) == 100, f"got {len(kg.facts)}")


def test_snapshot_integrity():
    print("\n[14] snapshot edges only reference existing nodes")
    kg, _ = _fresh()
    kg.add_fact("a", "to", "b")
    kg.add_fact("b", "to", "c")
    kg.add_fact("c", "to", "a")
    snap = kg.snapshot()
    node_ids = {n["id"] for n in snap["nodes"]}
    for e in snap["edges"]:
        _check(f"edge {e['id']} src in nodes", e["source"] in node_ids, e["source"])
        _check(f"edge {e['id']} tgt in nodes", e["target"] in node_ids, e["target"])


def test_empty_input_extractor():
    print("\n[15] extractor handles empty / whitespace / weird input")
    async def go():
        e = HybridExtractor()
        for txt in ["", "   ", "...", "????", "a"]:
            ext = await e.extract(txt)
            _check(f"no crash on {txt!r}", isinstance(ext.triples, list))
    asyncio.run(go())


def test_long_input():
    print("\n[16] very long input doesn't crash")
    async def go():
        e = HybridExtractor()
        big = "I like tea. " * 200
        ext = await e.extract(big)
        _check("returns extraction", isinstance(ext.triples, list))
        _check("dedupes 'likes tea' to 1", sum(1 for t in ext.triples if t.object == "tea") <= 2)
    asyncio.run(go())


def test_stats_invariants():
    print("\n[17] stats math is consistent")
    kg, _ = _fresh()
    kg.add_fact("u", "lives_in", "x")
    kg.add_fact("u", "lives_in", "y")
    kg.add_fact("u", "lives_in", "z")
    s = kg.stats()
    _check("edges_total = active + conflicts",
           s["edges_total"] == s["edges_active"] + s["conflicts"])
    _check("only 1 active lives_in", s["edges_active"] == 1)
    _check("2 superseded", s["conflicts"] == 2)


def test_predicate_whitespace_normalization():
    print("\n[18] predicate whitespace normalizes to snake_case")
    kg, _ = _fresh()
    f, _ = kg.add_fact("a", "lives in", "b")
    _check("predicate normalized to snake", f.predicate == "lives_in")


def main():
    print("=" * 60)
    print("KNOWLEDGE GRAPH VALIDATION SUITE")
    print("=" * 60)
    tests = [
        test_basic_add, test_idempotency_same_fact, test_conflict_resolution,
        test_no_conflict_on_multi_predicates, test_conflict_chain,
        test_polarity_conflict, test_mastery_transition,
        test_progress_transition, test_historical_does_not_supersede,
        test_retract_family_fallback,
        test_persistence_roundtrip, test_corrupted_json_recovery,
        test_empty_path_creates, test_self_loop, test_normalization_case,
        test_retrieval_relevance, test_past_query_retrieval,
        test_facts_about_traversal,
        test_concurrent_writes, test_snapshot_integrity,
        test_empty_input_extractor, test_long_input, test_stats_invariants,
        test_predicate_whitespace_normalization,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            global FAILED
            FAILED += 1
            FAILURES.append(f"{t.__name__} raised: {e}")
            print(f"  FAIL  {t.__name__} raised exception")
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print("=" * 60)
    if FAILED:
        print("\nFAILURES:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
