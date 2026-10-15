"""Regression: BM25/exact succeed on exact_lookup gold queries."""

from src.config import Settings
from src.data.benchmark_generator import generate_benchmark
from src.data.synthetic_generator import generate_catalog
from src.evaluation.metrics import recall_at_k
from src.retrieval.embeddings import HashEmbedder
from src.retrieval.hybrid import HybridRetriever
from src.ingestion.index_builder import InMemoryDenseIndex
from src.schemas.query import QueryType


def test_exact_lookup_recall_smoke():
    settings = Settings(catalog_size=200, dataset_seed=42, benchmark_size=64)
    products = generate_catalog(settings)
    queries = [
        q
        for q in generate_benchmark(products, settings, size=64)
        if q.query_type == QueryType.EXACT_LOOKUP
    ]
    assert queries
    embedder = HashEmbedder(32)
    retriever = HybridRetriever(
        products,
        embedder,
        memory_dense=InMemoryDenseIndex(products, embedder),
        apply_hard_filters=False,
    )
    recalls = []
    for q in queries:
        trace = retriever.retrieve(q.query_text, mode="bm25", top_k=5)
        retrieved = [s.product_id for s in trace.fused]
        recalls.append(recall_at_k(q.expected_product_ids, retrieved, 5))
    assert sum(recalls) / len(recalls) >= 0.8
