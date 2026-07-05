from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class Turn:
    """One exchange: what the user said and how the assistant replied."""

    user: str
    assistant: str


class ConversationMemory:
    """Short-term working memory — the last few turns of dialogue.

    The knowledge graph is *long-term* memory. This class is the rolling window
    the extractor and responder need to make sense of the *current* message:
    resolving pronouns ("she", "it"), follow-up questions ("and her?"), and
    anything that only means something in context.

    It is deliberately small and in-memory: working memory, not an archive.
    The graph keeps the facts that matter for good.
    """

    def __init__(self, max_turns: int = 8):
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def record(self, user: str, assistant: str) -> None:
        self._turns.append(Turn(user=user.strip(), assistant=assistant.strip()))

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def is_empty(self) -> bool:
        return len(self._turns) == 0

    def transcript(self, last: int | None = None) -> str:
        """Render recent dialogue as plain text for an LLM prompt."""
        turns = self.turns
        if last is not None:
            turns = turns[-last:]
        if not turns:
            return "(no earlier messages — this is the first turn)"
        lines: list[str] = []
        for t in turns:
            lines.append(f"User: {t.user}")
            lines.append(f"Assistant: {t.assistant}")
        return "\n".join(lines)
