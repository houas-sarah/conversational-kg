from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from backend.graph import KnowledgeGraph


@pytest.fixture
def kg(tmp_path: Path) -> KnowledgeGraph:
    """Fresh, isolated KnowledgeGraph backed by a temp file."""
    return KnowledgeGraph(tmp_path / "kg.json")


@pytest.fixture
def kg_with_facts(kg: KnowledgeGraph) -> KnowledgeGraph:
    """KG pre-loaded with a small canonical fact set, useful for retrieval tests."""
    kg.add_fact("user", "studies", "calculus")
    kg.add_fact("user", "struggles_with", "derivatives")
    kg.add_fact("user", "lives_in", "algiers")
    kg.add_fact("user", "likes", "pizza")
    kg.add_fact("user", "likes", "coffee")
    return kg
