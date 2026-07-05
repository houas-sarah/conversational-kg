from __future__ import annotations

import hashlib
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .conversation import ConversationMemory
from .graph import KnowledgeGraph


# A session id comes from the client (localStorage), so it is untrusted input.
# We strip it to a safe charset before use and *hash* it for any filename, so a
# crafted id can never escape the sessions directory (path traversal).
_SID_STRIP = re.compile(r"[^A-Za-z0-9_-]")


def clean_sid(raw: str | None) -> str | None:
    """Return a safe session id, or None if nothing usable was supplied."""
    if not raw:
        return None
    s = _SID_STRIP.sub("", raw)[:64]
    return s or None


def new_sid() -> str:
    return uuid.uuid4().hex


@dataclass
class Session:
    """One visitor's isolated world: their graph, dialogue and turn counter."""

    sid: str
    kg: KnowledgeGraph
    convo: ConversationMemory
    turns: int = 0
    last_access: float = field(default_factory=time.time)


class SessionManager:
    """Per-visitor isolation for a multi-user deployment.

    Each session id owns a private KnowledgeGraph + ConversationMemory, so two
    people using the live demo never see or overwrite each other's memory.

    Idle sessions are evicted (TTL) and the total is capped (LRU) so a public
    demo cannot grow without bound. Eviction only drops the in-memory copy; the
    session's JSON file stays on disk, so a returning visitor (same browser →
    same id) recovers their graph.
    """

    def __init__(
        self,
        storage_dir: Path,
        ttl_s: float = 2 * 3600,      # evict a session after 2h idle
        max_sessions: int = 500,      # then cap total live sessions
    ):
        self.dir = Path(storage_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_s
        self.max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def _path_for(self, sid: str) -> Path:
        # Hash the (already-cleaned) id → a filename that cannot traverse the fs.
        h = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:16]
        return self.dir / f"{h}.json"

    def get(self, sid: str) -> Session:
        """Fetch (or lazily create + load) the session for ``sid``."""
        with self._lock:
            self._evict_idle()
            s = self._sessions.get(sid)
            if s is None:
                s = Session(
                    sid=sid,
                    kg=KnowledgeGraph(self._path_for(sid)),
                    convo=ConversationMemory(),
                )
                self._sessions[sid] = s
                self._enforce_cap()
            s.last_access = time.time()
            return s

    def reset(self, sid: str) -> None:
        """Clear one session's graph + dialogue, leaving the session alive."""
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                return
            s.kg.entities.clear()
            s.kg.facts.clear()
            s.kg.g.clear()
            s.convo.clear()
            s.turns = 0
            s.kg.save()

    def _evict_idle(self) -> None:
        now = time.time()
        dead = [k for k, v in self._sessions.items() if now - v.last_access > self.ttl]
        for k in dead:
            self._sessions.pop(k, None)

    def _enforce_cap(self) -> None:
        overflow = len(self._sessions) - self.max_sessions
        if overflow <= 0:
            return
        # Drop the least-recently-used sessions first.
        lru = sorted(self._sessions.items(), key=lambda kv: kv[1].last_access)
        for k, _ in lru[:overflow]:
            self._sessions.pop(k, None)

    @property
    def active_count(self) -> int:
        return len(self._sessions)
