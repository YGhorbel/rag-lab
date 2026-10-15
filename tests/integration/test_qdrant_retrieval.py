"""Integration: Qdrant index + filtered dense search."""

from __future__ import annotations

import os

import pytest

from src.config import Settings
from src.data.synthetic_generator import generate_catalog
from src.ingestion.index_builder import build_qdrant_index
from src.retrieval.dense import retrieve_dense
from src.retrieval.embeddings import HashEmbedder
from src.schemas.query import QueryFilters

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def _qdrant_up() -> bool:
    try:
        import httpx

        r = httpx.get(f"{QDRANT_URL}/readyz", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _qdrant_up(), reason="Qdrant not running")
def test_qdrant_index_and_filter():
    settings = Settings(
        qdrant_url=QDRANT_URL,
        qdrant_collection="products_test_phase23",
        catalog_size=50,
        dataset_seed=42,
    )
    products = generate_catalog(settings)
    embedder = HashEmbedder(dim=64)
    n = build_qdrant_index(products, embedder, settings=settings, recreate=True)
    assert n == 50

    hits = retrieve_dense(
        "wireless headphones travel",
        embedder,
        top_k=10,
        filters=QueryFilters(max_price=200, in_stock=True, category="headphones"),
        settings=settings,
    )
    assert hits
    # payload filter correctness
    for h in hits:
        assert h.payload.get("price", 0) <= 200
        assert h.payload.get("in_stock") is True
        assert h.payload.get("category") == "headphones"
