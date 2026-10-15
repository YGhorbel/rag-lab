"""Ollama Cloud chat client."""

from __future__ import annotations

import httpx

from src.config import Settings, get_settings
from src.generation.prompts import SYSTEM_PROMPT


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/api"):
        return f"{base}/chat"
    return f"{base}/api/chat"


def generate(
    prompt: str,
    model: str | None = None,
    settings: Settings | None = None,
    system: str | None = None,
) -> str:
    settings = settings or get_settings()
    model = model or settings.generator_model
    if "ollama.com" in settings.ollama_base_url and not settings.ollama_api_key:
        raise RuntimeError("OLLAMA_API_KEY is required for Ollama Cloud")

    headers = {"Content-Type": "application/json"}
    if settings.ollama_api_key:
        headers["Authorization"] = f"Bearer {settings.ollama_api_key}"

    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT if system is None else system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    if model.startswith("gpt-oss"):
        body["think"] = "medium"

    response = httpx.post(
        _chat_url(settings.ollama_base_url),
        headers=headers,
        json=body,
        timeout=120.0,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise RuntimeError(f"Ollama Cloud returned {exc.response.status_code}: {detail}") from None
    content = response.json()["message"].get("content") or ""
    if not content.strip():
        raise RuntimeError("model returned no final-channel answer")
    return content
