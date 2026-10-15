"""Grounded answer schema returned by the generator (Phase 5)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    product_id: str
    source_record: str | None = None
    fields: list[str] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    supported_facts: list[dict[str, Any]] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    abstained: bool = False
