"""Build Qdrant dense index from canonical catalog products."""

from __future__ import annotations

import argparse
from typing import Any

from src.config import Settings, get_settings
from src.data.catalog_loader import load_catalog
from src.retrieval.embeddings import Embedder, get_embedder
from src.schemas.product import Product


def point_id(product: Product) -> int:
    """Stable Qdrant id from the numeric suffix of product_id (prod_000136 -> 136)."""
    return int(product.product_id.rsplit("_", 1)[-1])


def product_payload(product: Product) -> dict[str, Any]:
    battery = product.attributes.get("battery_hours")
    color = product.attributes.get("color")
    return {
        "product_id": product.product_id,
        "sku": product.sku,
        "brand": product.brand,
        "category": product.category,
        "price": float(product.price),
        "in_stock": bool(product.in_stock),
        "color": color,
        "battery_hours": float(battery) if battery is not None else None,
        "compatible_with": list(product.compatible_with),
        "title": product.title,
    }


def build_qdrant_index(
    products: list[Product],
    embedder: Embedder,
    settings: Settings | None = None,
    recreate: bool = True,
) -> int:
    settings = settings or get_settings()
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm

    client = QdrantClient(url=settings.qdrant_url, check_compatibility=False)
    collection = settings.qdrant_collection

    vectors_config = qm.VectorParams(
        size=embedder.dim,
        distance=qm.Distance.COSINE,
        hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
    )
    exists = client.collection_exists(collection)
    if recreate and exists:
        client.delete_collection(collection_name=collection)
        exists = False
    if not exists:
        client.create_collection(collection_name=collection, vectors_config=vectors_config)
        _create_payload_indexes(client, collection)

    batch_size = 64
    total = 0
    for start in range(0, len(products), batch_size):
        batch = products[start : start + batch_size]
        texts = [p.to_retrieval_text() for p in batch]
        vectors = embedder.embed(texts)
        points = []
        for product, vector in zip(batch, vectors):
            points.append(
                qm.PointStruct(
                    id=point_id(product),
                    vector=vector,
                    payload=product_payload(product),
                )
            )
        client.upsert(collection_name=collection, points=points)
        total += len(points)
    return total


def _create_payload_indexes(client: Any, collection: str) -> None:
    from qdrant_client.http import models as qm

    keyword = (
        "brand",
        "category",
        "sku",
        "color",
        "product_id",
        "compatible_with",
    )
    for field in keyword:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
    for field in ("price", "battery_hours"):
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=qm.PayloadSchemaType.FLOAT,
        )
    client.create_payload_index(
        collection_name=collection,
        field_name="in_stock",
        field_schema=qm.PayloadSchemaType.BOOL,
    )


class InMemoryDenseIndex:
    """Local cosine index for unit tests without Qdrant."""

    def __init__(self, products: list[Product], embedder: Embedder) -> None:
        self.products = products
        self.embedder = embedder
        self.ids = [p.product_id for p in products]
        self.vectors = embedder.embed([p.to_retrieval_text() for p in products])
        self.payloads = [product_payload(p) for p in products]

    def search(
        self,
        query: str,
        top_k: int = 50,
        qfilter: Any | None = None,
        products_by_id: dict[str, Product] | None = None,
        filters=None,
    ):
        from src.retrieval.filters import product_matches_filters
        from src.retrieval.types import RetrievalSource, ScoredProduct

        if not self.vectors:
            return []
        qv = self.embedder.embed([query])[0]
        scored: list[tuple[int, float]] = []
        for i, vec in enumerate(self.vectors):
            score = sum(a * b for a, b in zip(qv, vec))
            scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[ScoredProduct] = []
        rank = 0
        for idx, score in scored:
            product = self.products[idx]
            if filters is not None and not product_matches_filters(product, filters):
                continue
            rank += 1
            results.append(
                ScoredProduct(
                    product_id=product.product_id,
                    score=float(score),
                    rank=rank,
                    source=RetrievalSource.DENSE,
                    payload=self.payloads[idx],
                )
            )
            if rank >= top_k:
                break
        return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index catalog into Qdrant")
    parser.add_argument("--hash-embeddings", action="store_true", help="Use HashEmbedder")
    parser.add_argument("--no-recreate", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    products = load_catalog(settings=settings)
    embedder = get_embedder(settings.embedding_model, use_hash=args.hash_embeddings)
    count = build_qdrant_index(
        products,
        embedder,
        settings=settings,
        recreate=not args.no_recreate,
    )
    print(
        f"Indexed {count} products into {settings.qdrant_collection} "
        f"at {settings.qdrant_url} (dim={embedder.dim})"
    )


if __name__ == "__main__":
    main()
