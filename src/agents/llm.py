"""Local LLM access through Ollama (no paid API). Returns None when no server is reachable,
so callers switch to the clearly labelled deterministic mode."""

from __future__ import annotations

import logging
from functools import lru_cache

import requests

from src.config import get_settings

log = logging.getLogger("gameops.llm")


def ollama_status(url: str | None = None, timeout: float = 1.5) -> dict:
    url = url or get_settings().ollama_url
    if not url:
        return {"available": False, "reason": "OLLAMA_URL not set"}
    try:
        r = requests.get(url.rstrip("/") + "/api/tags", timeout=timeout)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        want = get_settings().ollama_model
        if want not in models:
            return {"available": False, "reason": f"model {want} not pulled (have: {models})", "url": url}
        return {"available": True, "url": url, "model": want}
    except Exception as exc:  # connection refused, timeout, bad JSON
        return {"available": False, "reason": f"Ollama unreachable: {type(exc).__name__}", "url": url}


@lru_cache(maxsize=1)
def get_chat_model():
    status = ollama_status()
    if not status["available"]:
        log.info("LLM unavailable: %s", status["reason"])
        return None
    from langchain_ollama import ChatOllama
    return ChatOllama(base_url=status["url"], model=status["model"], temperature=0)
