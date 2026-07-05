from __future__ import annotations

import json
import os
from typing import Any

import httpx


class GroqClient:
    """Thin async wrapper around the Groq OpenAI-compatible Chat Completions API."""

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "").strip()
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        max_tokens: int = 600,
    ) -> str:
        if not self.available:
            raise RuntimeError("GROQ_API_KEY not configured")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(self.BASE_URL, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]

    async def chat_json(self, system: str, user: str, **kw) -> dict:
        raw = await self.chat(system, user, json_mode=True, **kw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start : end + 1])
            return {}
