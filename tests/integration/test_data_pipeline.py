"""Integration: generate catalog then benchmark on disk."""

from src.config import Settings
from src.data.benchmark_generator import generate_benchmark, write_benchmark, load_benchmark
from src.data.catalog_loader import load_catalog
from src.data.synthetic_generator import generate_catalog, write_catalog


def test_end_to_end_data_pipeline(tmp_path):
    settings = Settings(
        data_processed_dir=tmp_path / "processed",
        data_benchmarks_dir=tmp_path / "benchmarks",
        catalog_size=120,
        dataset_seed=42,
        benchmark_size=40,
        demo_set_size=10,
    )
    products = generate_catalog(settings)
    write_catalog(products, settings)
    loaded = load_catalog(settings=settings)
    assert len(loaded) == 120

    queries = generate_benchmark(loaded, settings, size=40)
    write_benchmark(queries, settings.benchmark_path)
    write_benchmark(queries[:10], settings.demo_benchmark_path)

    assert len(load_benchmark(settings.benchmark_path)) == 40
    assert len(load_benchmark(settings.demo_benchmark_path)) == 10
    assert settings.catalog_meta_path.exists()
