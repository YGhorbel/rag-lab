"""Query plan and benchmark query schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class QueryIntent(str, Enum):
    EXACT_LOOKUP = "exact_lookup"
    FILTERED_PRODUCT_SEARCH = "filtered_product_search"
    SEMANTIC_DISCOVERY = "semantic_discovery"
    COMPARISON = "comparison"
    RELATIONSHIP = "relationship"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"
    CLARIFICATION_NEEDED = "clarification_needed"


class QueryType(str, Enum):
    EXACT_LOOKUP = "exact_lookup"
    ATTRIBUTE_CONSTRAINT = "attribute_constraint"
    SEMANTIC_NEED = "semantic_need"
    COMPARISON = "comparison"
    RELATIONSHIP = "relationship"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"
    AMBIGUOUS = "ambiguous"


class QueryFilters(BaseModel):
    brand: str | None = None
    category: str | None = None
    max_price: float | None = None
    min_price: float | None = None
    in_stock: bool | None = None
    min_battery_hours: float | None = None
    color: str | None = None
    sku: str | None = None
    compatible_with: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class QueryPlan(BaseModel):
    intent: QueryIntent
    keywords: list[str] = Field(default_factory=list)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    needs_clarification: bool = False
    compare_product_ids: list[str] = Field(default_factory=list)


class BenchmarkQuery(BaseModel):
    query_id: str
    query_text: str
    query_type: QueryType
    expected_product_ids: list[str] = Field(default_factory=list)
    required_filters: QueryFilters = Field(default_factory=QueryFilters)
    expected_facts: dict[str, Any] = Field(default_factory=dict)
    answerable: bool = True
    language: str = "en"
    difficulty: str = "medium"
    reasoning_path: str | None = None

    @model_validator(mode="after")
    def gold_consistency(self) -> "BenchmarkQuery":
        if self.answerable and not self.expected_product_ids:
            if self.query_type not in {QueryType.AMBIGUOUS, QueryType.UNANSWERABLE}:
                raise ValueError(
                    f"{self.query_id}: answerable queries need expected_product_ids "
                    "unless ambiguous/unanswerable"
                )
        return self
