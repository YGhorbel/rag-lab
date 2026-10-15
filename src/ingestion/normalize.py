"""Normalize product records before indexing."""

from __future__ import annotations

from src.schemas.product import Product


def normalize_product(product: Product) -> Product:
    """Normalize categorical fields; full pipeline in Phase 2+."""
    return product.model_copy(
        update={
            "brand": product.brand.strip(),
            "category": product.category.strip().lower(),
            "currency": product.currency.upper(),
        }
    )
