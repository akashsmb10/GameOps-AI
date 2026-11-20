"""The five approved, read-only tools of the GameOps Analyst agent.

Every tool:
* validates its arguments with a Pydantic schema (bad arguments -> structured error, no crash);
* only calls fixed, parameterised analytics queries or the vector retriever - there is no
  SQL or code execution path for model-generated text;
* returns JSON-serialisable data plus an `evidence_ids` list used for citation checking.
"""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.analytics.store import AnalyticsStore
from src.config import get_settings
from src.rag.retriever import Retriever


@lru_cache(maxsize=1)
def store() -> AnalyticsStore:
    return AnalyticsStore()


@lru_cache(maxsize=1)
def retriever() -> Retriever:
    return Retriever()


def _iso(v):
    if v in (None, ""):
        return None
    date.fromisoformat(str(v)[:10])  # raises ValueError -> pydantic error
    return str(v)[:10]


class SearchReviewsArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=300, description="What to look for in reviews")
    game_id: str | None = Field(None, description="Game id such as G001; omit for all games")
    start_date: str | None = Field(None, description="YYYY-MM-DD inclusive")
    end_date: str | None = Field(None, description="YYYY-MM-DD inclusive")
    sentiment: Literal["any", "negative", "positive"] = "any"
    top_k: int = Field(5, ge=1, le=10)

    @field_validator("start_date", "end_date")
    @classmethod
    def _dates(cls, v):
        return _iso(v)


class ComplaintArgs(BaseModel):
    game_id: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    scope: Literal["negative", "all"] = "negative"

    @field_validator("start_date", "end_date")
    @classmethod
    def _dates(cls, v):
        return _iso(v)


class CompareArgs(BaseModel):
    game_id: str | None = None
    period_a_start: str
    period_a_end: str
    period_b_start: str
    period_b_end: str

    @field_validator("period_a_start", "period_a_end", "period_b_start", "period_b_end")
    @classmethod
    def _dates(cls, v):
        return _iso(v)


class DocArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=300)
    game_id: str | None = None
    top_k: int = Field(3, ge=1, le=6)


class NoArgs(BaseModel):
    pass


def search_player_reviews(query: str, game_id: str | None = None, start_date: str | None = None,
                          end_date: str | None = None, sentiment: str = "any", top_k: int = 5) -> dict:
    """Semantic + lexical search over player reviews with optional game, date and sentiment filters."""
    a = SearchReviewsArgs(query=query, game_id=game_id, start_date=start_date, end_date=end_date, sentiment=sentiment, top_k=top_k)
    r = retriever().search_reviews(a.query, a.game_id, a.start_date, a.end_date, None if a.sentiment == "any" else a.sentiment, a.top_k)
    res = r["results"]
    reviews = [{"id": f"R:{x.source_id}", "game": x.game_name, "date": str(x.created_at)[:10], "recommended": bool(x.voted_up),
                "categories": x.categories.split(",") if x.categories else [], "score": round(float(x.score), 3), "text": x.text}
               for x in res.itertuples()] if r["answerable"] else []
    return {"reviews": reviews, "answerable": r["answerable"], "best_score": round(r["best_score"], 3),
            "min_score": r["min_score"], "retrieval_ms": r["latency_ms"], "evidence_ids": [x["id"] for x in reviews],
            "filters": a.model_dump(exclude={"query", "top_k"})}


def analyze_complaints(game_id: str | None = None, start_date: str | None = None, end_date: str | None = None,
                       scope: str = "negative") -> dict:
    """Complaint-category counts and shares (with 95% CIs) plus overall sentiment for a window."""
    a = ComplaintArgs(game_id=game_id, start_date=start_date, end_date=end_date, scope=scope)
    st = store()
    comp = st.complaints(a.game_id, a.start_date, a.end_date, a.scope)
    ov = st.overview(a.game_id, a.start_date, a.end_date)
    examples = []
    for c in [c["category"] for c in comp["categories"] if c["reviews"] > 0][:2]:
        for r in st.representative_reviews(c, a.game_id, a.start_date, a.end_date, limit=2).itertuples():
            examples.append({"id": f"R:{r.review_id}", "category": c, "game": r.game_name, "date": str(r.created_at)[:10], "text": r.review_text})
    return {"overview": ov, "complaints": comp, "examples": examples, "evidence_ids": [e["id"] for e in examples] + ["M:analyze_complaints"]}


def compare_feedback_trends(period_a_start: str, period_a_end: str, period_b_start: str, period_b_end: str,
                            game_id: str | None = None) -> dict:
    """Compare review counts, negative share and complaint shares between two periods (with p-values)."""
    a = CompareArgs(game_id=game_id, period_a_start=period_a_start, period_a_end=period_a_end,
                    period_b_start=period_b_start, period_b_end=period_b_end)
    cmp = store().compare_periods(a.game_id, a.period_a_start, a.period_a_end, a.period_b_start, a.period_b_end)
    return {**cmp, "evidence_ids": ["M:compare_feedback_trends"]}


def retrieve_game_documentation(query: str, game_id: str | None = None, top_k: int = 3) -> dict:
    """Retrieve update/patch notes and project documentation (metric definitions, data limitations)."""
    a = DocArgs(query=query, game_id=game_id, top_k=top_k)
    r = retriever()
    d = r.search_docs(a.query, a.game_id, a.top_k)
    docs = [{"id": f"D:{x.source_id}", "type": x.source_type, "title": getattr(x, "title", None), "text": x.text,
             "score": round(float(x.score), 3)} for x in d["results"].itertuples()] if d["answerable"] else []
    notes = []
    if a.game_id:
        notes = [{"id": f"D:{x.source_id}", "date": str(x.created_at)[:10], "text": x.text} for x in r.update_notes(a.game_id).itertuples()]
    ids = list(dict.fromkeys([x["id"] for x in docs] + [n["id"] for n in notes]))
    return {"documents": docs, "update_notes": notes, "answerable": bool(docs or notes), "evidence_ids": ids}


def get_data_quality_report() -> dict:
    """Dataset provenance, cleaning counts, model versions and evaluation summaries."""
    s = get_settings()
    dq = json.loads((s.processed_dir / "data_quality_report.json").read_text())
    manifest = json.loads((s.artifacts_dir / "pipeline_manifest.json").read_text())
    nlp = json.loads((s.artifacts_dir / "nlp" / "nlp_metrics.json").read_text())
    return {"data_quality": dq, "pipeline": manifest,
            "complaint_model": {"selected": nlp["selection"]["selected"], "label_source": nlp["label_source"],
                                "test_micro_f1": nlp["complaints_test"][nlp["selection"]["selected"]]["micro_f1"]},
            "evidence_ids": ["M:get_data_quality_report"]}


TOOLS = {
    "search_player_reviews": (search_player_reviews, SearchReviewsArgs),
    "analyze_complaints": (analyze_complaints, ComplaintArgs),
    "compare_feedback_trends": (compare_feedback_trends, CompareArgs),
    "retrieve_game_documentation": (retrieve_game_documentation, DocArgs),
    "get_data_quality_report": (get_data_quality_report, NoArgs),
}


def langchain_tools():
    """The same functions exposed as LangChain StructuredTools for LLM tool calling."""
    from langchain_core.tools import StructuredTool
    return [StructuredTool.from_function(fn, name=name, description=fn.__doc__, args_schema=schema)
            for name, (fn, schema) in TOOLS.items()]
