"""Retrieval metrics (generation-independent)."""

from __future__ import annotations

import math

from src.schemas.product import Product
from src.schemas.query import QueryFilters
from src.retrieval.filters import product_matches_filters


def recall_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top.intersection(relevant)) / len(set(relevant))


def precision_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    rel = set(relevant)
    return sum(1 for pid in top if pid in rel) / len(top)


def mrr(relevant: list[str], retrieved: list[str]) -> float:
    rel = set(relevant)
    for i, pid in enumerate(retrieved, start=1):
        if pid in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = 0.0
    for i, pid in enumerate(retrieved[:k], start=1):
        if pid in rel:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def exact_id_success(expected_ids: list[str], retrieved: list[str]) -> float:
    if not expected_ids:
        return 0.0
    return 1.0 if expected_ids[0] in retrieved[:1] else 0.0


def filter_correctness(
    retrieved_ids: list[str],
    products_by_id: dict[str, Product],
    filters: QueryFilters,
) -> float:
    """Fraction of retrieved products that satisfy hard filters. 1.0 if no filters."""
    active = any(
        getattr(filters, name) is not None
        for name in (
            "brand", "category", "max_price", "min_price", "in_stock",
            "min_battery_hours", "color", "sku", "compatible_with",
        )
    )
    if not active:
        return 1.0
    if not retrieved_ids:
        return 1.0
    ok = 0
    for pid in retrieved_ids:
        product = products_by_id.get(pid)
        if product and product_matches_filters(product, filters):
            ok += 1
    return ok / len(retrieved_ids)
