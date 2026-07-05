from __future__ import annotations

from .graph import Fact
from .llm import GroqClient
from .query import RetrievedContext, format_context_for_llm


RESPONDER_SYSTEM = """You are a warm, concise conversational assistant with a long-term memory: a knowledge graph of facts the user has told you.

You receive:
- CONVERSATION SO FAR: the last few turns. Use it to follow context and resolve references.
- RECALLED FACTS: stored facts relevant to the current message. Weave them in naturally; never list them back robotically. If empty, just be a friendly assistant getting to know the user.
- NEW FACTS: anything we just learned from the latest message. Acknowledge briefly ("got it") or weave it in.
- USER MESSAGE: what the user just said.

Rules:
- Sound human and curious, not corporate. Short sentences.
- Use the conversation to resolve references — if the user says "she", "he" or "it", you already know who or what from the earlier turns. Never ask "who do you mean?" when the answer is in the conversation.
- Never say "as an AI" or apologize for being a chatbot.
- If recalled facts conflict with new info, trust the new info and confirm the update.
- Recalled facts marked [no longer true] are past facts: speak of them in the past tense ("you used to live in Paris") and never present them as current.
- If the user asks what you remember, answer from RECALLED FACTS only — do not invent.
- Keep responses under 4 sentences unless the user asks for more."""


def template_response(ctx: RetrievedContext, new_facts: list[Fact], user_text: str) -> str:
    parts: list[str] = []
    text_lower = user_text.lower()
    asking_memory = any(q in text_lower for q in ["what do you", "what did i", "remember", "remind me", "recall", "what's my", "what is my"])

    if asking_memory and ctx.facts:
        bullets = []
        for f in ctx.facts[:5]:
            verb = f.predicate.replace("_", " ")
            subj = "you" if f.subject == "user" else f.subject
            bullets.append(f"{subj} {verb} {f.object}")
        return "Here's what I have on file:\n- " + "\n- ".join(bullets)

    if new_facts:
        latest = new_facts[0]
        verb = latest.predicate.replace("_", " ")
        if latest.subject == "user":
            parts.append(f"Got it — noted that you {verb} {latest.object}.")
        else:
            parts.append(f"Logged: {latest.subject} {verb} {latest.object}.")

    if ctx.facts and not asking_memory:
        relevant = ctx.facts[0]
        verb = relevant.predicate.replace("_", " ")
        if relevant.subject == "user" and relevant.predicate in ("struggles_with", "working_on", "studies"):
            parts.append(f"Last time you mentioned {verb} {relevant.object} — want to pick that back up?")

    if not parts:
        parts.append("Tell me more — I'm building a picture of what you're working on.")
    return " ".join(parts)


class Responder:
    def __init__(self, llm: GroqClient | None = None):
        self.llm = llm or GroqClient()

    async def reply(
        self,
        ctx: RetrievedContext,
        new_facts: list[Fact],
        user_text: str,
        history: str = "",
    ) -> str:
        if self.llm.available:
            try:
                recalled = format_context_for_llm(ctx)
                new_str = (
                    "\n".join(f"- {f.subject} {f.predicate} {f.object}" for f in new_facts)
                    if new_facts
                    else "(none)"
                )
                prompt = (
                    f"CONVERSATION SO FAR:\n{history or '(no earlier messages)'}\n\n"
                    f"RECALLED FACTS:\n{recalled}\n\n"
                    f"NEW FACTS (just learned):\n{new_str}\n\n"
                    f"USER MESSAGE: {user_text}"
                )
                return await self.llm.chat(RESPONDER_SYSTEM, prompt, temperature=0.6, max_tokens=300)
            except Exception:
                pass
        return template_response(ctx, new_facts, user_text)
