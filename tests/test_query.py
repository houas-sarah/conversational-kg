from __future__ import annotations

from backend.graph import KnowledgeGraph
from backend.query import retrieve, format_context_for_llm


class TestRetrieval:
    def test_surfaces_subject_overlap(self, kg_with_facts: KnowledgeGraph):
        ctx = retrieve(kg_with_facts, "Tell me about my calculus work.")
        objs = {f.object for f in ctx.facts}
        assert "calculus" in objs

    def test_surfaces_derivative_query(self, kg_with_facts: KnowledgeGraph):
        ctx = retrieve(kg_with_facts, "What am I struggling with?")
        objs = {f.object for f in ctx.facts}
        assert "derivatives" in objs

    def test_user_facts_always_included(self, kg_with_facts: KnowledgeGraph):
        ctx = retrieve(kg_with_facts, "random query with no entity match")
        assert "user" in ctx.matched_entities

    def test_ignores_superseded_facts(self, kg: KnowledgeGraph):
        kg.add_fact("user", "lives_in", "Algiers")
        kg.add_fact("user", "lives_in", "Paris")
        ctx = retrieve(kg, "Where do I live?")
        cities = [f.object for f in ctx.facts if f.predicate == "lives_in"]
        assert "Paris" in cities
        assert "Algiers" not in cities


class TestFormatting:
    def test_empty_context_renders_safely(self, kg: KnowledgeGraph):
        ctx = retrieve(kg, "anything")
        formatted = format_context_for_llm(ctx)
        assert isinstance(formatted, str)
        assert formatted

    def test_nonempty_context_lists_facts(self, kg_with_facts: KnowledgeGraph):
        ctx = retrieve(kg_with_facts, "What do I like?")
        formatted = format_context_for_llm(ctx)
        assert "likes" in formatted
