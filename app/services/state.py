"""Loads and holds the read-only services used by the API (startup-time model loading)."""

from __future__ import annotations

import json
import logging

import joblib

from src.agents import tools
from src.agents.llm import ollama_status
from src.config import get_settings

log = logging.getLogger("gameops.api")


class Services:
    def __init__(self):
        self.errors: dict[str, str] = {}
        s = get_settings()
        try:
            self.store = tools.store()
        except Exception as exc:
            self.store, self.errors["analytics_db"] = None, str(exc)
        try:
            self.retriever = tools.retriever()
        except Exception as exc:
            self.retriever, self.errors["vector_index"] = None, str(exc)
        try:
            self.nlp = joblib.load(s.artifacts_dir / "nlp" / "nlp_models.joblib")
        except Exception as exc:
            self.nlp, self.errors["nlp_models"] = None, str(exc)
        self.metrics = {}
        for name, path in {"nlp": s.artifacts_dir / "nlp" / "nlp_metrics.json", "rag": s.artifacts_dir / "eval" / "rag_eval.json",
                           "agent": s.artifacts_dir / "eval" / "agent_eval.json", "pipeline": s.artifacts_dir / "pipeline_manifest.json"}.items():
            self.metrics[name] = json.loads(path.read_text()) if path.exists() else None

    def status(self) -> dict:
        llm = ollama_status()
        return {"analytics_db": {"loaded": self.store is not None, "error": self.errors.get("analytics_db")},
                "vector_index": {"loaded": self.retriever is not None, "error": self.errors.get("vector_index"),
                                 "embedder": self.retriever.reviews.manifest["embedder"] if self.retriever else None},
                "nlp_models": {"loaded": self.nlp is not None, "error": self.errors.get("nlp_models")},
                "llm": {"loaded": llm["available"], **llm}}


_SERVICES: Services | None = None


def load_services() -> Services:
    global _SERVICES
    _SERVICES = Services()
    if _SERVICES.errors:
        log.warning("startup with missing components: %s", _SERVICES.errors)
    return _SERVICES


def services() -> Services:
    return _SERVICES or load_services()
