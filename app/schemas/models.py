"""Pydantic request/response schemas for the versioned API (v1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    version: str


class Ready(BaseModel):
    ready: bool
    components: dict[str, Any]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=300, examples=["servers keep disconnecting"])
    game_id: str | None = Field(None, examples=["G001"])
    start_date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    sentiment: Literal["any", "negative", "positive"] = "any"
    top_k: int = Field(5, ge=1, le=20)


class SearchHit(BaseModel):
    review_id: str
    game_id: str
    game_name: str
    created_at: str
    recommended: bool
    categories: list[str]
    score: float
    dense_score: float
    lexical_score: float
    text: str


class SearchResponse(BaseModel):
    answerable: bool
    best_score: float
    min_score: float
    mode: str
    embedder: str
    latency_ms: float
    results: list[SearchHit]


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, examples=["Why did negative reviews for Starfall Arena increase after update 2.1?"])


class AgentResponse(BaseModel):
    run_id: str
    answer: str | None
    mode: str | None
    refused: dict | None
    citations: list[str]
    invalid_citations: list[str]
    warnings: list[str]
    tool_calls: list[dict]
    latency_ms: float
    llm_status: dict


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000)


class PredictResponse(BaseModel):
    selected_complaint_model: str
    complaint_categories: dict[str, list[str]]
    model_sentiment: str
    p_negative: float
    note: str
