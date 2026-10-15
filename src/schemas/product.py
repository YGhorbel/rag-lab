"""Canonical product record schema."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Product(BaseModel):
    product_id: str
    sku: str
    title: str
    brand: str
    category: str
    description: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    price: float
    currency: str = "EUR"
    in_stock: bool = True
    rating: float = 0.0
    tags: list[str] = Field(default_factory=list)
    compatible_with: list[str] = Field(default_factory=list)
    related_product_ids: list[str] = Field(default_factory=list)
    source_record: str = "synthetic_catalog_v1"
    updated_at: datetime

    @field_validator("price")
    @classmethod
    def price_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("price must be non-negative")
        return value

    @field_validator("rating")
    @classmethod
    def rating_range(cls, value: float) -> float:
        if not 0.0 <= value <= 5.0:
            raise ValueError("rating must be between 0 and 5")
        return value

    def to_retrieval_text(self) -> str:
        """Stable text representation for dense/lexical indexing (Phase 2+)."""
        attr_bits = []
        for key, val in sorted(self.attributes.items()):
            attr_bits.append(f"{key}: {val}")
        attrs = "; ".join(attr_bits)
        tags = ", ".join(self.tags)
        return (
            f"{self.title}. Brand: {self.brand}. Category: {self.category}. "
            f"{self.description} Attributes: {attrs}. Tags: {tags}. SKU: {self.sku}."
        )
