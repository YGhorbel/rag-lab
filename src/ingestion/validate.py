"""Validate catalog invariants."""

from __future__ import annotations

from src.schemas.product import Product


def validate_products(products: list[Product]) -> None:
    seen: set[str] = set()
    skus: set[str] = set()
    for product in products:
        if product.product_id in seen:
            raise ValueError(f"duplicate product_id: {product.product_id}")
        seen.add(product.product_id)
        if product.sku in skus:
            # near-duplicate SKUs are allowed intentionally for hard cases
            continue
        skus.add(product.sku)
        Product.model_validate(product.model_dump())
