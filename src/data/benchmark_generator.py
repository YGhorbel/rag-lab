"""Generate gold benchmark queries from the synthetic catalog."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.config import Settings, get_settings
from src.data.catalog_loader import load_catalog, products_by_id
from src.schemas.product import Product
from src.schemas.query import BenchmarkQuery, QueryFilters, QueryType


def _exact_lookup(product: Product, qid: str, language: str) -> BenchmarkQuery:
    if language == "fr":
        text = f"Trouve le SKU {product.sku}."
    else:
        text = f"Find SKU {product.sku}."
    return BenchmarkQuery(
        query_id=qid,
        query_text=text,
        query_type=QueryType.EXACT_LOOKUP,
        expected_product_ids=[product.product_id],
        required_filters=QueryFilters(sku=product.sku),
        expected_facts={"sku": product.sku, "title": product.title},
        answerable=True,
        language=language,
        difficulty="easy",
    )


def _attribute_constraint(products: list[Product], rng: random.Random, qid: str, language: str) -> BenchmarkQuery | None:
    candidates = [
        p
        for p in products
        if p.category == "headphones"
        and p.in_stock
        and p.price <= 150
        and float(p.attributes.get("battery_hours", 0)) >= 30
        and str(p.attributes.get("color", "")).lower() == "black"
    ]
    if not candidates:
        return None
    expected = [p.product_id for p in candidates[:5]]
    if language == "fr":
        text = "Casque noir à moins de 150€ avec au moins 30 heures de batterie."
    else:
        text = "Black headphones under €150 with at least 30 hours of battery."
    return BenchmarkQuery(
        query_id=qid,
        query_text=text,
        query_type=QueryType.ATTRIBUTE_CONSTRAINT,
        expected_product_ids=expected,
        required_filters=QueryFilters(
            category="headphones",
            max_price=150,
            in_stock=True,
            min_battery_hours=30,
            color="black",
        ),
        expected_facts={"category": "headphones", "max_price": 150},
        answerable=True,
        language=language,
        difficulty="medium",
    )


def _semantic_need(products: list[Product], qid: str, language: str) -> BenchmarkQuery:
    travel = [
        p
        for p in products
        if p.category in {"headphones", "earbuds"} and "travel" in p.tags and p.in_stock
    ][:8]
    if not travel:
        travel = [p for p in products if p.category == "headphones"][:5]
    if language == "fr":
        text = "Je cherche un casque léger pour les longs vols."
    else:
        text = "I need lightweight headphones for long flights."
    return BenchmarkQuery(
        query_id=qid,
        query_text=text,
        query_type=QueryType.SEMANTIC_NEED,
        expected_product_ids=[p.product_id for p in travel],
        required_filters=QueryFilters(category="headphones", in_stock=True),
        expected_facts={"use_case": "travel"},
        answerable=True,
        language=language,
        difficulty="medium",
    )


def _comparison(products: list[Product], qid: str, language: str) -> BenchmarkQuery | None:
    by_id = products_by_id(products)
    a = by_id.get("prod_000000")
    b = by_id.get("prod_000001")
    if not a or not b:
        return None
    if language == "fr":
        text = "Compare le X200 et le X210 sur batterie, poids et réduction de bruit."
    else:
        text = "Compare the X200 and X210 on battery, weight, and noise cancellation."
    return BenchmarkQuery(
        query_id=qid,
        query_text=text,
        query_type=QueryType.COMPARISON,
        expected_product_ids=[a.product_id, b.product_id],
        required_filters=QueryFilters(),
        expected_facts={
            "battery_hours": {
                a.product_id: a.attributes.get("battery_hours"),
                b.product_id: b.attributes.get("battery_hours"),
            }
        },
        answerable=True,
        language=language,
        difficulty="medium",
    )


def _relationship(products: list[Product], qid: str, language: str) -> BenchmarkQuery:
    matches = [p for p in products if "device_family_7" in p.compatible_with][:10]
    if language == "fr":
        text = "Quels accessoires sont compatibles avec device family 7 ?"
    else:
        text = "Which accessories are compatible with device family 7?"
    return BenchmarkQuery(
        query_id=qid,
        query_text=text,
        query_type=QueryType.RELATIONSHIP,
        expected_product_ids=[p.product_id for p in matches],
        required_filters=QueryFilters(compatible_with="device_family_7"),
        expected_facts={"compatible_with": "device_family_7"},
        answerable=bool(matches),
        language=language,
        difficulty="hard",
        reasoning_path="filter compatible_with == device_family_7",
    )


def _multi_hop(products: list[Product], qid: str, language: str) -> BenchmarkQuery:
    matches = [
        p
        for p in products
        if "device_family_7" in p.compatible_with and p.price < 50 and p.in_stock
    ][:8]
    if language == "fr":
        text = (
            "Trouve un accessoire compatible avec l'appareil qui supporte device family 7 "
            "et coûte moins de 50€."
        )
    else:
        text = (
            "Find a compatible accessory for the device that supports device family 7 "
            "and costs under €50."
        )
    return BenchmarkQuery(
        query_id=qid,
        query_text=text,
        query_type=QueryType.MULTI_HOP,
        expected_product_ids=[p.product_id for p in matches],
        required_filters=QueryFilters(
            compatible_with="device_family_7", max_price=50, in_stock=True
        ),
        expected_facts={"max_price": 50},
        answerable=bool(matches),
        language=language,
        difficulty="hard",
        reasoning_path="compatible_with device_family_7 then price < 50",
    )


def _unanswerable(product: Product, qid: str, language: str) -> BenchmarkQuery:
    if language == "fr":
        text = f"Est-ce que {product.title} inclut une garantie de cinq ans ?"
    else:
        text = f"Does {product.title} include a five-year warranty?"
    return BenchmarkQuery(
        query_id=qid,
        query_text=text,
        query_type=QueryType.UNANSWERABLE,
        expected_product_ids=[],
        required_filters=QueryFilters(),
        expected_facts={"warranty_years": None},
        answerable=False,
        language=language,
        difficulty="easy",
    )


def _ambiguous(qid: str, language: str) -> BenchmarkQuery:
    if language == "fr":
        text = "Montre-moi le compact."
    else:
        text = "Show me the compact one."
    return BenchmarkQuery(
        query_id=qid,
        query_text=text,
        query_type=QueryType.AMBIGUOUS,
        expected_product_ids=[],
        required_filters=QueryFilters(),
        expected_facts={},
        answerable=True,
        language=language,
        difficulty="hard",
    )


def generate_benchmark(
    products: list[Product],
    settings: Settings | None = None,
    size: int | None = None,
) -> list[BenchmarkQuery]:
    settings = settings or get_settings()
    target = size or settings.benchmark_size
    rng = random.Random(settings.dataset_seed + 7)
    queries: list[BenchmarkQuery] = []
    languages = ["en", "fr"]

    builders = [
        "exact",
        "attribute",
        "semantic",
        "comparison",
        "relationship",
        "multi_hop",
        "unanswerable",
        "ambiguous",
    ]

    i = 0
    while len(queries) < target:
        kind = builders[i % len(builders)]
        language = languages[i % len(languages)]
        qid = f"q_{len(queries):04d}"
        product = products[rng.randint(0, len(products) - 1)]
        query: BenchmarkQuery | None = None

        if kind == "exact":
            query = _exact_lookup(product, qid, language)
        elif kind == "attribute":
            query = _attribute_constraint(products, rng, qid, language)
        elif kind == "semantic":
            query = _semantic_need(products, qid, language)
        elif kind == "comparison":
            query = _comparison(products, qid, language)
        elif kind == "relationship":
            query = _relationship(products, qid, language)
        elif kind == "multi_hop":
            query = _multi_hop(products, qid, language)
        elif kind == "unanswerable":
            query = _unanswerable(product, qid, language)
        elif kind == "ambiguous":
            query = _ambiguous(qid, language)

        i += 1
        if query is None:
            continue
        queries.append(query)

    return queries[:target]


def write_benchmark(queries: list[BenchmarkQuery], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for query in queries:
            fh.write(query.model_dump_json() + "\n")
    return path


def load_benchmark(path: Path) -> list[BenchmarkQuery]:
    items: list[BenchmarkQuery] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(BenchmarkQuery.model_validate(json.loads(line)))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid benchmark at line {line_no}: {exc}") from exc
    return items


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate gold benchmark JSONL")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--demo-size", type=int, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    products = load_catalog(settings=settings)
    bench = generate_benchmark(products, settings, size=args.size)
    demo_size = args.demo_size or settings.demo_set_size
    demo = bench[:demo_size]

    bpath = write_benchmark(bench, settings.benchmark_path)
    dpath = write_benchmark(demo, settings.demo_benchmark_path)
    print(f"Wrote {len(bench)} benchmark queries to {bpath}")
    print(f"Wrote {len(demo)} demo queries to {dpath}")


if __name__ == "__main__":
    main()
