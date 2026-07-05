"""Seed the knowledge graph with realistic demo conversations.

Runs the same extractor + graph pipeline as the live chat, so the demo state is
produced by the real system (not hand-written triples). Each "session" is a
short multi-turn conversation; turns are spaced out in time so the temporal
ordering in the graph looks like a real session history.

Usage:
    .venv\\Scripts\\python.exe -m backend.seed_data
    .venv\\Scripts\\python.exe -m backend.seed_data --reset
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from .extractor import HybridExtractor
from .graph import KnowledgeGraph
from .llm import GroqClient


SESSIONS: list[dict] = [
    {
        "name": "Day 1 — getting to know each other",
        "turns": [
            "My name is Sarah and I'm a computer science student at Riverside University.",
            "I'm currently building a project on knowledge graphs.",
            "My supervisor is Dr. Karim Haddad.",
            "I live in Algiers.",
            "I love couscous and I really hate broccoli.",
        ],
    },
    {
        "name": "Day 3 — coursework comes up",
        "turns": [
            "I'm studying calculus this semester.",
            "I'm really struggling with derivatives lately.",
            "I prefer studying in the evening.",
            "I want to understand integration too eventually.",
        ],
    },
    {
        "name": "Day 5 — side interests",
        "turns": [
            "I love pop music, especially while I code.",
            "I work on Python projects in my free time.",
            "I'm working on a chatbot for my project.",
            "I feel motivated when the code finally runs.",
        ],
    },
    {
        "name": "Day 12 — progress update (will conflict with earlier facts)",
        "turns": [
            "I finished the calculus chapter on derivatives.",
            "I understand derivatives now actually.",
            "I'm studying linear algebra now.",
        ],
    },
]


async def seed(reset: bool = False) -> None:
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "kg.json"
    if reset and data_path.exists():
        data_path.unlink()
        print(f"[reset] removed {data_path}")

    kg = KnowledgeGraph(data_path)
    llm = GroqClient()
    extractor = HybridExtractor(llm)

    print(f"[info] llm available: {llm.available}")
    if llm.available:
        print(f"[info] model: {llm.model}")
    print()

    turn_no = 0
    for session in SESSIONS:
        print(f"━━━ {session['name']} ━━━")
        for msg in session["turns"]:
            turn_no += 1
            print(f"  [t{turn_no:02d}] » {msg}")
            ext = await extractor.extract(msg)
            for t in ext.triples:
                if not (t.subject and t.predicate and t.object):
                    continue
                fact, superseded = kg.add_fact(t.subject, t.predicate, t.object, t.confidence, turn_no)
                marker = "⊕"
                print(f"         {marker} {fact.subject} ─{fact.predicate}→ {fact.object}  (conf={fact.confidence:.2f})")
                for s in superseded:
                    print(f"         ⊖ supersedes: {s.subject} ─{s.predicate}→ {s.object}")
            if llm.available:
                await asyncio.sleep(0.4)
        print()

    kg.save()
    stats = kg.stats()
    print("━" * 60)
    print(f"[done] saved to {data_path}")
    print(f"       nodes={stats['nodes']}  edges_active={stats['edges_active']}  superseded={stats['conflicts']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="wipe existing kg.json first")
    args = parser.parse_args()
    asyncio.run(seed(reset=args.reset))


if __name__ == "__main__":
    main()
