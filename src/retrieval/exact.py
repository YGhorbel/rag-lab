"""Exact SKU / product_id lookup index."""

from __future__ import annotations

from src.retrieval.types import RetrievalSource, ScoredProduct
from src.schemas.product import Product


class ExactIndex:
    def __init__(self, products: list[Product]) -> None:
        self._by_sku: dict[str, str] = {}
        self._by_id: dict[str, Product] = {}
        for product in products:
            self._by_sku[product.sku.upper()] = product.product_id
            self._by_id[product.product_id] = product

    def lookup(self, sku: str | None = None, product_id: str | None = None) -> list[ScoredProduct]:
        pid: str | None = None
        if product_id and product_id in self._by_id:
            pid = product_id
        elif sku:
            pid = self._by_sku.get(sku.upper())
        if not pid:
            return []
        return [
            ScoredProduct(
                product_id=pid,
                score=1.0,
                rank=1,
                source=RetrievalSource.EXACT,
                payload={"sku": self._by_id[pid].sku},
            )
        ]

    def lookup_from_text(self, query: str) -> list[ScoredProduct]:
        tokens = query.replace(",", " ").split()
        hits: list[ScoredProduct] = []
        seen: set[str] = set()
        for token in tokens:
            cleaned = token.strip().upper().rstrip(".,;:?!")
            if cleaned in self._by_sku and cleaned not in seen:
                seen.add(cleaned)
                hits.extend(self.lookup(sku=cleaned))
            # also try without trailing punctuation variants already handled
        for i, hit in enumerate(hits, start=1):
            hit.rank = i
        return hits
