from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .embeddings import EMBEDDER
from .extractor import HybridExtractor
from .llm import GroqClient
from .query import retrieve
from .responder import Responder
from .session import SessionManager, clean_sid, new_sid


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SESSIONS_DIR = ROOT / "data" / "sessions"
FRONTEND = ROOT / "frontend"


class Hub:
    """WebSocket fan-out, scoped per session so a turn only reaches the
    sockets that belong to the same visitor."""

    def __init__(self):
        self.rooms: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def add(self, sid: str, ws: WebSocket):
        async with self._lock:
            self.rooms.setdefault(sid, set()).add(ws)

    async def remove(self, sid: str, ws: WebSocket):
        async with self._lock:
            room = self.rooms.get(sid)
            if room:
                room.discard(ws)
                if not room:
                    self.rooms.pop(sid, None)

    async def broadcast(self, sid: str, msg: dict[str, Any]):
        dead: list[WebSocket] = []
        for ws in list(self.rooms.get(sid, ())):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.remove(sid, ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Précharge le modèle d'embeddings en tâche de fond : le premier message de
    # l'utilisateur n'a pas à attendre le chargement (~1-2 s à froid).
    threading.Thread(target=lambda: EMBEDDER.available, daemon=True).start()
    yield


app = FastAPI(title="the commonplace — Conversational KG", lifespan=lifespan)

llm = GroqClient()
extractor = HybridExtractor(llm)
responder = Responder(llm)
sessions = SessionManager(SESSIONS_DIR)
hub = Hub()


class ChatIn(BaseModel):
    text: str


def _resolve_sid(header: str | None, query: str | None) -> str:
    """Pick a session id from the request, minting a fresh one if absent.

    The client sends its id via the ``X-Session-Id`` header (REST) or the
    ``sid`` query param (WebSocket). When neither is present we generate one
    and echo it back so the client can adopt and persist it.
    """
    return clean_sid(header) or clean_sid(query) or new_sid()


@app.get("/api/status")
def status(
    x_session_id: str | None = Header(default=None),
    sid: str | None = Query(default=None),
):
    resolved = _resolve_sid(x_session_id, sid)
    s = sessions.get(resolved)
    return {
        "sid": resolved,
        "llm_available": llm.available,
        "llm_model": llm.model if llm.available else None,
        "extraction_method": "hybrid (spaCy + Groq LLM)" if llm.available else "rules + spaCy",
        "retrieval": f"semantic ({EMBEDDER.backend})" if EMBEDDER.available else "lexical",
        "stats": s.kg.stats(),
        "active_sessions": sessions.active_count,
    }


@app.get("/api/graph")
def graph(
    x_session_id: str | None = Header(default=None),
    sid: str | None = Query(default=None),
):
    resolved = _resolve_sid(x_session_id, sid)
    return sessions.get(resolved).kg.snapshot()


@app.get("/api/facts")
def facts(
    limit: int = 50,
    x_session_id: str | None = Header(default=None),
    sid: str | None = Query(default=None),
):
    resolved = _resolve_sid(x_session_id, sid)
    s = sessions.get(resolved)
    items = sorted(s.kg.facts.values(), key=lambda f: f.timestamp, reverse=True)[:limit]
    return [f.as_dict() for f in items]


@app.post("/api/reset")
def reset(
    x_session_id: str | None = Header(default=None),
    sid: str | None = Query(default=None),
):
    resolved = _resolve_sid(x_session_id, sid)
    sessions.reset(resolved)
    return {"ok": True, "sid": resolved}


@app.post("/api/chat")
async def chat(
    payload: ChatIn,
    x_session_id: str | None = Header(default=None),
    sid: str | None = Query(default=None),
):
    resolved = _resolve_sid(x_session_id, sid)
    return await _process_turn(resolved, payload.text)


async def _process_turn(sid: str, text: str) -> dict:
    s = sessions.get(sid)
    kg = s.kg
    s.turns += 1
    turn = s.turns
    t0 = time.time()

    # Mémoire courte : les derniers tours de dialogue. L'extracteur s'en sert
    # pour résoudre les pronoms ("she" -> "Lina") avant que le moindre triplet
    # ne touche le graphe.
    history = s.convo.transcript()

    extraction = await extractor.extract(text, history)
    resolved = (extraction.resolved_text or text).strip() or text

    new_facts = []
    superseded_all = []
    retracted_all = []
    for t in extraction.triples:
        if not t.subject or not t.object or not t.predicate:
            continue
        if t.op == "retract":
            retracted = kg.retract(t.subject, t.predicate, t.object)
            if retracted is not None:
                retracted_all.append(retracted)
            continue
        # Historical fact ("I used to live in Paris") → already closed at insert.
        valid_until = time.time() if t.valid_until == "now" else None
        fact, superseded = kg.add_fact(
            t.subject, t.predicate, t.object,
            confidence=t.confidence,
            source_turn=turn,
            valid_until=valid_until,
        )
        new_facts.append(fact)
        superseded_all.extend(superseded)

    # Colour-code the graph: apply the entity kinds the extractor found
    # (person / topic / place ...) instead of leaving every node as "Concept".
    kinds = {
        (e.get("text") or "").strip().lower(): (e.get("kind") or "Concept")
        for e in extraction.entities
        if (e.get("text") or "").strip()
    }
    kinds.setdefault("user", "Person")
    for key, kind in kinds.items():
        ent = kg.entities.get(key)
        if ent and kind and kind.lower() != "concept":
            ent.kind = kind
            if key in kg.g:
                kg.g.nodes[key]["kind"] = kind

    # On interroge le graphe avec le texte résolu : ainsi "what is she studying?"
    # retrouve bien l'entité à laquelle "she" fait référence.
    ctx = retrieve(kg, resolved)
    reply = await responder.reply(ctx, new_facts, text, history)

    s.convo.record(text, reply)
    kg.save()
    elapsed = time.time() - t0

    result = {
        "sid": sid,
        "turn": turn,
        "user": text,
        "resolved_text": resolved,
        "reply": reply,
        "extraction_method": extraction.method,
        "entities": extraction.entities,
        "new_facts": [f.as_dict() for f in new_facts],
        "superseded": [f.as_dict() for f in superseded_all],
        "retracted": [f.as_dict() for f in retracted_all],
        "recalled_facts": [f.as_dict() for f in ctx.facts],
        "stats": kg.stats(),
        "elapsed_ms": int(elapsed * 1000),
        "graph": kg.snapshot(),
    }
    await hub.broadcast(sid, {"type": "turn", "payload": result})
    return result


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    sid = clean_sid(websocket.query_params.get("sid")) or new_sid()
    await hub.add(sid, websocket)
    s = sessions.get(sid)
    await websocket.send_json({
        "type": "hello",
        "payload": {"sid": sid, "stats": s.kg.stats(), "graph": s.kg.snapshot()},
    })
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "chat":
                text = (msg.get("payload") or {}).get("text", "").strip()
                if text:
                    await _process_turn(sid, text)
            elif msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await hub.remove(sid, websocket)


app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND / "index.html"))
