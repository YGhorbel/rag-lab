"""Exact, BM25, RRF, filter unit tests."""

from src.config import Settings
from src.data.synthetic_generator import generate_catalog
from src.retrieval.embeddings import HashEmbedder
from src.retrieval.exact import ExactIndex
from src.retrieval.filters import post_filter, product_matches_filters
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.lexical import LexicalIndex
from src.retrieval.types import RetrievalSource, ScoredProduct
from src.schemas.query import QueryFilters
from src.data.catalog_loader import products_by_id
from src.ingestion.index_builder import InMemoryDenseIndex


def test_exact_sku_lookup():
    products = generate_catalog(Settings(catalog_size=20, dataset_seed=42))
    idx = ExactIndex(products)
    hits = idx.lookup(sku="AUDIO-X200-BLK")
    assert len(hits) == 1
    assert hits[0].product_id == "prod_000000"


def test_bm25_finds_sku_text():
    products = generate_catalog(Settings(catalog_size=50, dataset_seed=42))
    lex = LexicalIndex(products)
    hits = lex.retrieve("AUDIO-X200-BLK", top_k=5)
    assert hits
    assert any(h.product_id == "prod_000000" for h in hits)


def test_rrf_merges_lists():
    a = [
        ScoredProduct(product_id="a", score=1, rank=1, source=RetrievalSource.BM25),
        ScoredProduct(product_id="b", score=0.5, rank=2, source=RetrievalSource.BM25),
    ]
    b = [
        ScoredProduct(product_id="b", score=1, rank=1, source=RetrievalSource.DENSE),
        ScoredProduct(product_id="c", score=0.5, rank=2, source=RetrievalSource.DENSE),
    ]
    fused = reciprocal_rank_fusion([a, b], k=60, top_k=3)
    assert fused[0].product_id == "b"
    assert {x.product_id for x in fused} == {"a", "b", "c"}


def test_hard_price_filter_drops_expensive():
    products = generate_catalog(Settings(catalog_size=50, dataset_seed=42))
    by_id = products_by_id(products)
    expensive = next(p for p in products if p.price > 200)
    scored = [
        ScoredProduct(
            product_id=expensive.product_id,
            score=1.0,
            rank=1,
            source=RetrievalSource.DENSE,
        )
    ]
    filtered = post_filter(scored, by_id, QueryFilters(max_price=100))
    assert filtered == []
    assert not product_matches_filters(expensive, QueryFilters(max_price=100))


def test_hybrid_memory_dense_runs():
    products = generate_catalog(Settings(catalog_size=40, dataset_seed=42))
    embedder = HashEmbedder(dim=32)
    mem = InMemoryDenseIndex(products, embedder)
    retriever = HybridRetriever(
        products, embedder, memory_dense=mem, apply_hard_filters=True
    )
    trace = retriever.retrieve("Find SKU AUDIO-X200-BLK", mode="hybrid", top_k=5)
    assert trace.fused
    assert any(x.product_id == "prod_000000" for x in trace.fused)
