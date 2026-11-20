"""Versioned API routes (/api/v1). All data access is read-only."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from app.schemas.models import (
    AgentRequest,
    AgentResponse,
    PredictRequest,
    PredictResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.services.state import services
from src.agents.graph import ask
from src.agents.llm import ollama_status
from src.config import CATEGORIES
from src.monitoring import telemetry

router = APIRouter()
def _date():
    # a NEW Query object per parameter: sharing one instance makes FastAPI reuse its name
    return Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")


def _store():
    st = services().store
    if st is None:
        raise HTTPException(503, "analytics database not built; run scripts/build_pipeline.py")
    return st


@router.get("/analytics/overview", tags=["analytics"])
def overview(game_id: str | None = None, start: str | None = _date(), end: str | None = _date()):
    st = _store()
    out = st.overview(game_id, start, end)
    out["games"] = [{"game_id": g, "game_name": n} for g, n in st.games]
    out["emerging_issues"] = st.emerging_issues(game_id, end=end)
    out["data_note"] = "Shares describe the collected reviews, not all players. Demo data is synthetic."
    return out


@router.get("/analytics/sentiment", tags=["analytics"])
def sentiment(game_id: str | None = None, start: str | None = _date(), end: str | None = _date(),
              freq: Literal["day", "week", "month"] = "week"):
    st = _store()
    ts = st.sentiment_timeseries(game_id, start, end, freq)
    ts["period"] = ts["period"].astype(str)
    return {"source_label": "player recommendation (voted_up)", "timeseries": ts.to_dict("records"),
            "by_game": st.sentiment_by_game(start, end).to_dict("records")}


@router.get("/analytics/complaints", tags=["analytics"])
def complaints(game_id: str | None = None, start: str | None = _date(), end: str | None = _date(),
               scope: Literal["negative", "all"] = "negative"):
    return _store().complaints(game_id, start, end, scope)


@router.get("/analytics/trends", tags=["analytics"])
def trends(game_id: str | None = None, start: str | None = _date(), end: str | None = _date(),
           freq: Literal["day", "week", "month"] = "week", scope: Literal["negative", "all"] = "negative"):
    df = _store().complaint_trends(game_id, start, end, freq, scope)
    df["period"] = df["period"].astype(str)
    return {"scope": scope, "freq": freq, "rows": df.to_dict("records")}


@router.post("/reviews/search", response_model=SearchResponse, tags=["retrieval"])
def search(req: SearchRequest):
    r = services().retriever
    if r is None:
        raise HTTPException(503, "vector index not built")
    out = r.search_reviews(req.query, req.game_id, req.start_date, req.end_date,
                           None if req.sentiment == "any" else req.sentiment, req.top_k)
    telemetry.record_retrieval(out["latency_ms"], out["best_score"], out["answerable"])
    hits = [SearchHit(review_id=x.source_id, game_id=x.game_id, game_name=x.game_name, created_at=str(x.created_at)[:19],
                      recommended=bool(x.voted_up), categories=[c for c in x.categories.split(",") if c], score=float(x.score),
                      dense_score=float(x.dense_score), lexical_score=float(x.lexical_score), text=x.text)
            for x in out["results"].itertuples()]
    return SearchResponse(answerable=out["answerable"], best_score=out["best_score"], min_score=out["min_score"],
                          mode=out["mode"], embedder=out["embedder"], latency_ms=out["latency_ms"], results=hits)


@router.post("/agent/query", response_model=AgentResponse, tags=["agent"])
async def agent_query(req: AgentRequest):
    out = await run_in_threadpool(ask, req.question)
    telemetry.record_agent(out)
    return AgentResponse(**{k: out[k] for k in ("run_id", "answer", "mode", "refused", "citations", "invalid_citations",
                                                "warnings", "tool_calls", "latency_ms")}, llm_status=ollama_status())


@router.post("/models/predict", response_model=PredictResponse, tags=["models"])
def predict(req: PredictRequest):
    nlp = services().nlp
    if nlp is None:
        raise HTTPException(503, "NLP models not built")
    specific = [c for c in CATEGORIES if c != "other"]
    out = {}
    for name, model in nlp["all_models"].items():
        pred = model.predict([req.text])[0]
        out[name] = [c for c, v in zip(specific, pred, strict=False) if v]
    p_neg = float(nlp["sentiment_model"].predict_proba([req.text])[0, 1])
    return PredictResponse(selected_complaint_model=nlp["selected"], complaint_categories=out,
                           model_sentiment="negative" if p_neg >= nlp["sentiment_threshold"] else "positive", p_negative=p_neg,
                           note="Model outputs; categories and sentiment were trained on SYNTHETIC demo labels.")


@router.get("/models/metrics", tags=["models"])
def metrics():
    return services().metrics


@router.get("/monitoring", tags=["ops"])
def monitoring(window_hours: float = Query(168, gt=0, le=24 * 90)):
    return telemetry.summary(window_hours)
