"""Gold benchmark generator tests."""

from src.config import Settings
from src.data.benchmark_generator import generate_benchmark, load_benchmark, write_benchmark
from src.data.synthetic_generator import generate_catalog
from src.schemas.query import QueryType


def test_benchmark_gold_consistency(tmp_path):
    settings = Settings(
        data_processed_dir=tmp_path,
        data_benchmarks_dir=tmp_path,
        catalog_size=200,
        dataset_seed=42,
        benchmark_size=80,
        demo_set_size=20,
    )
    products = generate_catalog(settings)
    queries = generate_benchmark(products, settings, size=80)
    assert len(queries) == 80

    product_ids = {p.product_id for p in products}
    for q in queries:
        if q.answerable and q.query_type not in {QueryType.AMBIGUOUS, QueryType.UNANSWERABLE}:
            assert q.expected_product_ids, q.query_id
            for pid in q.expected_product_ids:
                assert pid in product_ids
        if not q.answerable:
            assert q.query_type == QueryType.UNANSWERABLE


def test_benchmark_has_required_query_types():
    settings = Settings(catalog_size=300, dataset_seed=42, benchmark_size=64)
    products = generate_catalog(settings)
    queries = generate_benchmark(products, settings, size=64)
    types = {q.query_type for q in queries}
    for required in [
        QueryType.EXACT_LOOKUP,
        QueryType.ATTRIBUTE_CONSTRAINT,
        QueryType.SEMANTIC_NEED,
        QueryType.COMPARISON,
        QueryType.RELATIONSHIP,
        QueryType.MULTI_HOP,
        QueryType.UNANSWERABLE,
        QueryType.AMBIGUOUS,
    ]:
        assert required in types


def test_languages_include_en_and_fr():
    settings = Settings(catalog_size=200, dataset_seed=42, benchmark_size=40)
    products = generate_catalog(settings)
    queries = generate_benchmark(products, settings, size=40)
    langs = {q.language for q in queries}
    assert "en" in langs and "fr" in langs


def test_write_load_benchmark(tmp_path):
    settings = Settings(catalog_size=100, dataset_seed=1, benchmark_size=16)
    products = generate_catalog(settings)
    queries = generate_benchmark(products, settings, size=16)
    path = write_benchmark(queries, tmp_path / "bench.jsonl")
    loaded = load_benchmark(path)
    assert len(loaded) == 16
