"""Dense vector retrieval via Qdrant (or in-memory fallback)."""

from __future__ import annotations

from typing import Any

from src.config import Settings, get_settings
from src.retrieval.embeddings import Embedder
from src.retrieval.filters import to_qdrant_filter
from src.retrieval.types import RetrievalSource, ScoredProduct
from src.schemas.query import QueryFilters


def collection_vector_size(info: Any) -> int:
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        params = vectors.get("") or next(iter(vectors.values()))
        return int(params.size)
    return int(vectors.size)


def retrieve_dense(
    query: str,
    embedder: Embedder,
    top_k: int = 50,
    filters: QueryFilters | None = None,
    settings: Settings | None = None,
    memory_index=None,
) -> list[ScoredProduct]:
    settings = settings or get_settings()
    filters = filters or QueryFilters()

    if memory_index is not None:
        return memory_index.search(query, top_k=top_k, filters=filters)

    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url, check_compatibility=False)
    from qdrant_client.http import models as qm

    info = client.get_collection(collection_name=settings.qdrant_collection)
    stored = collection_vector_size(info)
    if stored != embedder.dim:
        raise RuntimeError(
            f"Qdrant collection {settings.qdrant_collection} has vector size {stored}, "
            f"but the embedder is {embedder.dim}-d. Reindex with rag-lab-index."
        )

    vector = embedder.embed([query])[0]
    qfilter = to_qdrant_filter(filters)
    search_params = qm.SearchParams(hnsw_ef=128)

    try:
        response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            query_filter=qfilter,
            limit=top_k,
            with_payload=True,
            search_params=search_params,
        )
        points = response.points
    except Exception:
        points = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            query_filter=qfilter,
            limit=top_k,
            with_payload=True,
            search_params=search_params,
        )

    results: list[ScoredProduct] = []
    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        pid = payload.get("product_id")
        if not pid:
            continue
        results.append(
            ScoredProduct(
                product_id=str(pid),
                score=float(point.score or 0.0),
                rank=rank,
                source=RetrievalSource.DENSE,
                payload=dict(payload),
            )
        )
    return results
