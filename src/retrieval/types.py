"""Shared retrieval result types."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.schemas.query import QueryFilters, QueryPlan


class RetrievalSource(str, Enum):
    EXACT = "exact"
    BM25 = "bm25"
    DENSE = "dense"
    FUSED = "fused"


class ScoredProduct(BaseModel):
    product_id: str
    score: float
    rank: int
    source: RetrievalSource
    payload: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    query: str
    mode: str
    plan: QueryPlan
    filters_applied: QueryFilters
    exact: list[ScoredProduct] = Field(default_factory=list)
    bm25: list[ScoredProduct] = Field(default_factory=list)
    dense: list[ScoredProduct] = Field(default_factory=list)
    fused: list[ScoredProduct] = Field(default_factory=list)
    latencies_ms: dict[str, float] = Field(default_factory=dict)
