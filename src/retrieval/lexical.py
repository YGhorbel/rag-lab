"""In-process BM25 lexical retrieval."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.retrieval.types import RetrievalSource, ScoredProduct
from src.schemas.product import Product

_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class LexicalIndex:
    def __init__(self, products: list[Product]) -> None:
        self.products = products
        self._ids = [p.product_id for p in products]
        corpus = [tokenize(p.to_retrieval_text()) for p in products]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def retrieve(self, query: str, top_k: int = 50) -> list[ScoredProduct]:
        if not self._bm25 or not query.strip():
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[ScoredProduct] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            if score <= 0:
                continue
            results.append(
                ScoredProduct(
                    product_id=self._ids[idx],
                    score=float(score),
                    rank=rank,
                    source=RetrievalSource.BM25,
                )
            )
        return results


def retrieve_lexical(query: str, top_k: int = 50, index: LexicalIndex | None = None):
    if index is None:
        raise NotImplementedError("Pass a LexicalIndex instance")
    return index.retrieve(query, top_k=top_k)
