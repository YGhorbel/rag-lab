"""Structured filters for Qdrant payloads and in-memory post-filtering."""

from __future__ import annotations

from typing import Any

from src.retrieval.types import ScoredProduct
from src.schemas.product import Product
from src.schemas.query import QueryFilters


def product_matches_filters(product: Product, filters: QueryFilters) -> bool:
    if filters.sku and product.sku.upper() != filters.sku.upper():
        return False
    if filters.brand and product.brand.lower() != filters.brand.lower():
        return False
    if filters.category and product.category.lower() != filters.category.lower():
        return False
    if filters.max_price is not None and product.price > filters.max_price:
        return False
    if filters.min_price is not None and product.price < filters.min_price:
        return False
    if filters.in_stock is not None and product.in_stock != filters.in_stock:
        return False
    if filters.color is not None:
        color = str(product.attributes.get("color", "")).lower()
        if color != filters.color.lower():
            return False
    if filters.min_battery_hours is not None:
        battery = product.attributes.get("battery_hours")
        if battery is None or float(battery) < filters.min_battery_hours:
            return False
    if filters.compatible_with is not None:
        if filters.compatible_with not in product.compatible_with:
            # also check attribute
            if product.attributes.get("compatible_model") != filters.compatible_with:
                return False
    return True


def post_filter(
    scored: list[ScoredProduct],
    products_by_id: dict[str, Product],
    filters: QueryFilters,
) -> list[ScoredProduct]:
    if not _has_any_filter(filters):
        return scored
    kept: list[ScoredProduct] = []
    for item in scored:
        product = products_by_id.get(item.product_id)
        if product is None:
            continue
        if product_matches_filters(product, filters):
            kept.append(item)
    for i, item in enumerate(kept, start=1):
        item.rank = i
    return kept


def _has_any_filter(filters: QueryFilters) -> bool:
    return any(
        getattr(filters, name) is not None
        for name in (
            "brand",
            "category",
            "max_price",
            "min_price",
            "in_stock",
            "min_battery_hours",
            "color",
            "sku",
            "compatible_with",
        )
    )


def to_qdrant_filter(filters: QueryFilters) -> Any | None:
    """Build qdrant_client Filter from QueryFilters. Returns None if empty."""
    if not _has_any_filter(filters):
        return None

    from qdrant_client.http import models as qm

    must: list[Any] = []
    if filters.brand:
        must.append(qm.FieldCondition(key="brand", match=qm.MatchValue(value=filters.brand)))
    if filters.category:
        must.append(
            qm.FieldCondition(key="category", match=qm.MatchValue(value=filters.category))
        )
    if filters.sku:
        must.append(qm.FieldCondition(key="sku", match=qm.MatchValue(value=filters.sku)))
    if filters.color:
        must.append(qm.FieldCondition(key="color", match=qm.MatchValue(value=filters.color)))
    if filters.in_stock is not None:
        must.append(
            qm.FieldCondition(key="in_stock", match=qm.MatchValue(value=filters.in_stock))
        )
    if filters.max_price is not None:
        must.append(
            qm.FieldCondition(key="price", range=qm.Range(lte=float(filters.max_price)))
        )
    if filters.min_price is not None:
        must.append(
            qm.FieldCondition(key="price", range=qm.Range(gte=float(filters.min_price)))
        )
    if filters.min_battery_hours is not None:
        must.append(
            qm.FieldCondition(
                key="battery_hours",
                range=qm.Range(gte=float(filters.min_battery_hours)),
            )
        )
    if filters.compatible_with:
        must.append(
            qm.FieldCondition(
                key="compatible_with",
                match=qm.MatchValue(value=filters.compatible_with),
            )
        )
    return qm.Filter(must=must) if must else None


def apply_filters(candidates, filters: QueryFilters):
    raise NotImplementedError("Use post_filter or to_qdrant_filter")
