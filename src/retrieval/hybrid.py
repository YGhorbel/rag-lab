"""Hybrid retrieval orchestration: exact + BM25 + dense + RRF."""

from __future__ import annotations

import time
from typing import Literal

from src.config import Settings, get_settings
from src.data.catalog_loader import load_catalog, products_by_id
from src.retrieval.dense import retrieve_dense
from src.retrieval.embeddings import Embedder, get_embedder
from src.retrieval.exact import ExactIndex
from src.retrieval.filters import post_filter
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.lexical import LexicalIndex
from src.retrieval.query_parser import CatalogVocab, parse_query
from src.retrieval.types import RetrievalTrace
from src.schemas.product import Product
from src.schemas.query import QueryFilters

RetrievalMode = Literal["exact", "bm25", "dense", "hybrid"]


class HybridRetriever:
    def __init__(
        self,
        products: list[Product],
        embedder: Embedder,
        settings: Settings | None = None,
        memory_dense=None,
        apply_hard_filters: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.products = products
        self.products_by_id = products_by_id(products)
        self.embedder = embedder
        self.exact = ExactIndex(products)
        self.lexical = LexicalIndex(products)
        self.vocab = CatalogVocab.from_products(products)
        self.memory_dense = memory_dense
        self.apply_hard_filters = apply_hard_filters

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        use_hash_embeddings: bool = False,
        memory_dense: bool = False,
        apply_hard_filters: bool = True,
    ) -> "HybridRetriever":
        settings = settings or get_settings()
        products = load_catalog(settings=settings)
        embedder = get_embedder(settings.embedding_model, use_hash=use_hash_embeddings)
        mem = None
        if memory_dense:
            from src.ingestion.index_builder import InMemoryDenseIndex

            mem = InMemoryDenseIndex(products, embedder)
        return cls(
            products,
            embedder,
            settings=settings,
            memory_dense=mem,
            apply_hard_filters=apply_hard_filters,
        )

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int | None = None,
        override_filters: QueryFilters | None = None,
    ) -> RetrievalTrace:
        top_k = top_k or self.settings.dense_top_k
        t0 = time.perf_counter()
        plan = parse_query(query, self.vocab)
        filters = override_filters or plan.filters
        # For pure semantic/exact modes, still honor override filters when provided
        latencies: dict[str, float] = {"parse": (time.perf_counter() - t0) * 1000}

        exact_hits = []
        bm25_hits = []
        dense_hits = []

        if mode in {"exact", "hybrid"}:
            t = time.perf_counter()
            if filters.sku:
                exact_hits = self.exact.lookup(sku=filters.sku)
            else:
                exact_hits = self.exact.lookup_from_text(query)
            if self.apply_hard_filters and mode != "exact":
                # exact SKU should survive even if other filters conflict? keep SKU identity
                pass
            elif self.apply_hard_filters and mode == "exact":
                exact_hits = post_filter(exact_hits, self.products_by_id, filters)
            latencies["exact"] = (time.perf_counter() - t) * 1000

        if mode in {"bm25", "hybrid"}:
            t = time.perf_counter()
            bm25_hits = self.lexical.retrieve(query, top_k=self.settings.lexical_top_k)
            if self.apply_hard_filters:
                bm25_hits = post_filter(bm25_hits, self.products_by_id, filters)
            latencies["bm25"] = (time.perf_counter() - t) * 1000

        if mode in {"dense", "hybrid"}:
            t = time.perf_counter()
            dense_filters = filters if self.apply_hard_filters else QueryFilters()
            dense_hits = retrieve_dense(
                query,
                self.embedder,
                top_k=self.settings.dense_top_k,
                filters=dense_filters,
                settings=self.settings,
                memory_index=self.memory_dense,
            )
            # post-filter as safety net (in case Qdrant filter omitted soft fields)
            if self.apply_hard_filters and self.memory_dense is None:
                dense_hits = post_filter(dense_hits, self.products_by_id, filters)
            latencies["dense"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        if mode == "exact":
            fused = exact_hits[:top_k]
        elif mode == "bm25":
            fused = bm25_hits[:top_k]
        elif mode == "dense":
            fused = dense_hits[:top_k]
        else:
            lists = [lst for lst in (exact_hits, bm25_hits, dense_hits) if lst]
            fused = reciprocal_rank_fusion(lists, top_k=top_k)
        latencies["fusion"] = (time.perf_counter() - t) * 1000
        latencies["total"] = (time.perf_counter() - t0) * 1000

        return RetrievalTrace(
            query=query,
            mode=mode,
            plan=plan,
            filters_applied=filters,
            exact=exact_hits,
            bm25=bm25_hits,
            dense=dense_hits,
            fused=fused,
            latencies_ms={k: round(v, 3) for k, v in latencies.items()},
        )


def retrieve_hybrid(query: str, top_k: int = 50):
    retriever = HybridRetriever.from_settings(memory_dense=True, use_hash_embeddings=True)
    return retriever.retrieve(query, mode="hybrid", top_k=top_k)
