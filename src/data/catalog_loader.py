"""Load and validate canonical catalog JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import Settings, get_settings
from src.schemas.product import Product


def load_catalog(path: Path | None = None, settings: Settings | None = None) -> list[Product]:
    settings = settings or get_settings()
    catalog_path = path or settings.catalog_path
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    products: list[Product] = []
    seen: set[str] = set()
    with catalog_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                product = Product.model_validate(payload)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid product at line {line_no}: {exc}") from exc
            if product.product_id in seen:
                raise ValueError(f"Duplicate product_id: {product.product_id}")
            seen.add(product.product_id)
            products.append(product)
    return products


def products_by_id(products: list[Product]) -> dict[str, Product]:
    return {p.product_id: p for p in products}
