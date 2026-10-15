"""Reciprocal Rank Fusion."""

from __future__ import annotations

from src.retrieval.types import RetrievalSource, ScoredProduct


def reciprocal_rank_fusion(
    rank_lists: list[list[ScoredProduct]],
    k: int = 60,
    top_k: int = 50,
) -> list[ScoredProduct]:
    scores: dict[str, float] = {}
    for results in rank_lists:
        for item in results:
            # prefer explicit rank; fall back to list order
            rank = item.rank if item.rank > 0 else 1
            scores[item.product_id] = scores.get(item.product_id, 0.0) + 1.0 / (k + rank)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    fused: list[ScoredProduct] = []
    for rank, (pid, score) in enumerate(ordered, start=1):
        fused.append(
            ScoredProduct(
                product_id=pid,
                score=score,
                rank=rank,
                source=RetrievalSource.FUSED,
            )
        )
    return fused
