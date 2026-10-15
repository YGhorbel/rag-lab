"""CLI retrieval benchmark runner."""

from __future__ import annotations

import argparse
import statistics
from typing import Any

from src.config import Settings, get_settings
from src.data.benchmark_generator import load_benchmark
from src.data.catalog_loader import load_catalog, products_by_id
from src.evaluation.metrics import (
    exact_id_success,
    filter_correctness,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.evaluation.reports import write_report
from src.retrieval.hybrid import HybridRetriever, RetrievalMode
from src.schemas.query import QueryType


def run_benchmark(
    mode: RetrievalMode = "hybrid",
    subset: str = "demo",
    k: int = 10,
    use_hash_embeddings: bool = True,
    memory_dense: bool = True,
    apply_hard_filters: bool = True,
    settings: Settings | None = None,
    limit: int | None = None,
    retriever: HybridRetriever | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    products = load_catalog(settings=settings)
    by_id = products_by_id(products)

    path = settings.demo_benchmark_path if subset == "demo" else settings.benchmark_path
    queries = load_benchmark(path)
    if limit is not None:
        queries = queries[:limit]

    if retriever is None:
        retriever = HybridRetriever.from_settings(
            settings=settings,
            use_hash_embeddings=use_hash_embeddings,
            memory_dense=memory_dense,
            apply_hard_filters=apply_hard_filters,
        )

    per_query: list[dict[str, Any]] = []
    recalls: list[float] = []
    precisions: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    filter_scores: list[float] = []
    exact_scores: list[float] = []
    latencies: list[float] = []

    for q in queries:
        # Skip ambiguous/unanswerable for ranking metrics aggregation but still record
        override = q.required_filters if apply_hard_filters else None
        # Prefer gold filters when evaluating filter correctness experiments
        if apply_hard_filters and any(
            getattr(q.required_filters, f) is not None
            for f in (
                "brand", "category", "max_price", "min_price", "in_stock",
                "min_battery_hours", "color", "sku", "compatible_with",
            )
        ):
            override = q.required_filters

        trace = retriever.retrieve(
            q.query_text,
            mode=mode,
            top_k=max(k, settings.dense_top_k),
            override_filters=override,
        )
        retrieved = [s.product_id for s in trace.fused]
        relevant = q.expected_product_ids

        r = recall_at_k(relevant, retrieved, k) if relevant else 0.0
        p = precision_at_k(relevant, retrieved, k) if relevant else 0.0
        m = mrr(relevant, retrieved) if relevant else 0.0
        n = ndcg_at_k(relevant, retrieved, k) if relevant else 0.0
        fc = filter_correctness(retrieved[:k], by_id, q.required_filters)
        ex = (
            exact_id_success(relevant, retrieved)
            if q.query_type == QueryType.EXACT_LOOKUP
            else None
        )

        if relevant and q.answerable and q.query_type not in {
            QueryType.AMBIGUOUS,
            QueryType.UNANSWERABLE,
        }:
            recalls.append(r)
            precisions.append(p)
            mrrs.append(m)
            ndcgs.append(n)
        filter_scores.append(fc)
        if ex is not None:
            exact_scores.append(ex)
        latencies.append(trace.latencies_ms.get("total", 0.0))

        per_query.append(
            {
                "query_id": q.query_id,
                "query_type": q.query_type.value,
                "language": q.language,
                "answerable": q.answerable,
                "recall_at_k": round(r, 4),
                "precision_at_k": round(p, 4),
                "mrr": round(m, 4),
                "ndcg_at_k": round(n, 4),
                "filter_correctness": round(fc, 4),
                "exact_id_success": ex,
                "latency_ms": trace.latencies_ms.get("total"),
                "retrieved_top": ",".join(retrieved[:k]),
            }
        )

    def avg(xs: list[float]) -> float:
        return round(statistics.mean(xs), 4) if xs else 0.0

    summary = {
        "mode": mode,
        "subset": subset,
        "k": k,
        "n_queries": len(queries),
        "catalog_version": settings.catalog_version,
        "embedding_model": "hash" if use_hash_embeddings else settings.embedding_model,
        "apply_hard_filters": apply_hard_filters,
        "metrics": {
            f"recall@{k}": avg(recalls),
            f"precision@{k}": avg(precisions),
            "mrr": avg(mrrs),
            f"ndcg@{k}": avg(ndcgs),
            "filter_correctness": avg(filter_scores),
            "exact_id_success": avg(exact_scores),
            "latency_ms_mean": avg(latencies),
        },
        "per_query": per_query,
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument(
        "--mode",
        choices=["exact", "bm25", "dense", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--subset", choices=["demo", "full"], default="demo")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--real-embeddings",
        action="store_true",
        help="Use configured SentenceTransformer model instead of HashEmbedder",
    )
    parser.add_argument(
        "--qdrant",
        action="store_true",
        help="Use Qdrant dense index instead of in-memory dense",
    )
    parser.add_argument("--no-filters", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    results = run_benchmark(
        mode=args.mode,
        subset=args.subset,
        k=args.k,
        use_hash_embeddings=not args.real_embeddings,
        memory_dense=not args.qdrant,
        apply_hard_filters=not args.no_filters,
        settings=settings,
        limit=args.limit,
    )
    path = write_report(results, settings.experiments_dir)
    print(json_dumps_metrics(results))
    print(f"Wrote report: {path}")


def json_dumps_metrics(results: dict[str, Any]) -> str:
    import json

    return json.dumps({"mode": results["mode"], "metrics": results["metrics"]}, indent=2)


if __name__ == "__main__":
    main()
