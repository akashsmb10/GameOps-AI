"""Filtered semantic retrieval over the review and documentation indexes."""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.store import InvalidQuery
from src.config import get_settings
from src.nlp.embeddings import get_embedder

from .index import VectorIndex

# Minimum cosine score for the best hit before we answer at all. Set per embedder from
# separate calibration probes (scripts/calibrate_retrieval.py), NOT from the evaluation set.
DEFAULT_MIN_SCORE = {"spacy": 0.80, "sentence-transformers": 0.35, "tfidf-svd": 0.25}


class Retriever:
    def __init__(self, index_dir: Path | None = None, embedder=None):
        s = get_settings()
        self.dir = index_dir or s.artifacts_dir / "index"
        embedder = embedder or get_embedder(s.embedder, s.st_model)
        self.reviews = VectorIndex.load(self.dir, "reviews", embedder)
        self.docs = VectorIndex.load(self.dir, "docs", self.reviews.embedder)
        cal = self.dir / "retrieval_calibration.json"
        family = self.reviews.embedder.name.split(":")[0]
        self.min_score = json.loads(cal.read_text())["min_score"] if cal.exists() else DEFAULT_MIN_SCORE.get(family, 0.3)
        self.min_coverage = json.loads(cal.read_text()).get("min_coverage", 0.5) if cal.exists() else 0.5
        mode_file = self.dir / "retrieval_mode.json"  # chosen on the DEV questions (src/rag/select_mode.py)
        self.mode = json.loads(mode_file.read_text())["selected_mode"] if mode_file.exists() else "hybrid"
        m = self.reviews.meta
        self._game = m["game_id"].to_numpy()
        created = pd.to_datetime(m["created_at"])
        self._created = (created.dt.tz_localize(None) if created.dt.tz is not None else created).dt.date.to_numpy()
        self._neg = ~m["voted_up"].to_numpy()
        self.game_ids = set(m["game_id"].unique())
        self._name_words = {w for n in m["game_name"].unique() for w in str(n).lower().split()}
        self.last_latency_ms = 0.0

    def _allowed(self, game_id, start, end, sentiment) -> np.ndarray | None:
        mask = np.ones(len(self._game), dtype=bool)
        if game_id:
            if game_id not in self.game_ids:
                raise InvalidQuery(f"Unknown game_id '{game_id}'")
            mask &= self._game == game_id
        try:
            if start:
                mask &= self._created >= date.fromisoformat(str(start)[:10])
            if end:
                mask &= self._created <= date.fromisoformat(str(end)[:10])
        except ValueError as exc:
            raise InvalidQuery("Dates must be YYYY-MM-DD") from exc
        if sentiment == "negative":
            mask &= self._neg
        elif sentiment == "positive":
            mask &= ~self._neg
        elif sentiment not in (None, "", "any"):
            raise InvalidQuery("sentiment must be negative, positive or any")
        return None if mask.all() else np.flatnonzero(mask)

    def search_reviews(self, query: str, game_id=None, start=None, end=None, sentiment=None, top_k: int = 5) -> dict:
        if not query or not query.strip():
            raise InvalidQuery("query must not be empty")
        top_k = max(1, min(int(top_k), 20))
        t0 = time.perf_counter()
        res = self.reviews.search(query, top_k, self._allowed(game_id, start, end, sentiment), mode=self.mode)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        best = float(res["score"].max()) if len(res) else 0.0
        cov = self.reviews.coverage(query, ignore=self._name_words)
        # no-answer gate: enough of the question's vocabulary must exist in the reviews
        return {"results": res, "best_score": best, "coverage": round(cov, 3),
                "answerable": len(res) > 0 and cov >= self.min_coverage,
                "min_score": self.min_score, "mode": self.mode, "latency_ms": round(self.last_latency_ms, 2), "embedder": self.reviews.manifest["embedder"]}

    def search_docs(self, query: str, game_id=None, top_k: int = 3) -> dict:
        if not query or not query.strip():
            raise InvalidQuery("query must not be empty")
        m = self.docs.meta
        if game_id and game_id not in self.game_ids:
            raise InvalidQuery(f"Unknown game_id '{game_id}'")
        allowed = np.flatnonzero(m["game_id"].isna() | (m["game_id"] == game_id)) if game_id else None
        t0 = time.perf_counter()
        # docs are few and short: exact version/title words matter, so rank by summed scores
        res = self.docs.search(query, max(1, min(int(top_k), 10)), allowed, mode="sum")
        best = float(res["score"].max()) if len(res) else 0.0
        best_lex = float(res["lexical_score"].max()) if len(res) else 0.0
        # documents are answerable when they share real vocabulary with the question
        # (lexical >= 0.15) or score above the review threshold
        return {"results": res, "best_score": best, "answerable": len(res) > 0 and (best_lex >= 0.15 or best >= self.min_score),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    def update_notes(self, game_id: str) -> pd.DataFrame:
        m = self.docs.meta
        return m[(m["source_type"] == "update_note") & (m["game_id"] == game_id)].sort_values("created_at")
