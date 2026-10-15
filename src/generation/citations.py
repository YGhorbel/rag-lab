"""Citation validation against the evidence passed to the generator."""

from __future__ import annotations

from src.schemas.answer import GroundedAnswer


def validate_citations(answer: GroundedAnswer, evidence_ids: set[str]) -> bool:
    return all(citation.product_id in evidence_ids for citation in answer.citations)
