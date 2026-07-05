from __future__ import annotations

import pytest

from backend.extractor import HybridExtractor


pytestmark = pytest.mark.asyncio


@pytest.fixture
def extractor() -> HybridExtractor:
    return HybridExtractor()


class TestEdgeCases:
    @pytest.mark.parametrize("text", ["", "   ", "...", "????", "a", "!!!!", "\n\n\t"])
    async def test_weird_input_does_not_crash(self, extractor: HybridExtractor, text: str):
        result = await extractor.extract(text)
        assert isinstance(result.triples, list)

    async def test_long_input_handled(self, extractor: HybridExtractor):
        big = "I like tea. " * 200
        result = await extractor.extract(big)
        assert isinstance(result.triples, list)
        tea_count = sum(1 for t in result.triples if t.object == "tea")
        assert tea_count <= 2


class TestRuleBasedExtraction:
    """Tests that pass regardless of whether the LLM is available."""

    async def test_struggling_extracts(self, extractor: HybridExtractor):
        result = await extractor.extract("I'm really struggling with derivatives.")
        triples = [(t.subject, t.predicate) for t in result.triples]
        assert any(s == "user" and p == "struggles_with" for s, p in triples)

    async def test_name_extracts(self, extractor: HybridExtractor):
        result = await extractor.extract("My name is Sarah.")
        triples = [(t.subject, t.predicate, t.object) for t in result.triples]
        assert any(p == "named" and o == "sarah" for _, p, o in triples)

    async def test_location_extracts(self, extractor: HybridExtractor):
        result = await extractor.extract("I live in Algiers.")
        triples = [(t.subject, t.predicate, t.object) for t in result.triples]
        assert any(p == "lives_in" and o == "algiers" for _, p, o in triples)
