"""Synthetic catalog generator tests."""

from src.config import Settings
from src.data.catalog_loader import load_catalog
from src.data.synthetic_generator import generate_catalog, write_catalog
from src.ingestion.validate import validate_products


def test_generate_catalog_size_and_unique_ids(tmp_path):
    settings = Settings(
        data_processed_dir=tmp_path,
        catalog_size=100,
        dataset_seed=42,
    )
    products = generate_catalog(settings)
    assert len(products) == 100
    ids = [p.product_id for p in products]
    assert len(ids) == len(set(ids))
    validate_products(products)


def test_catalog_reproducible_with_seed():
    a = Settings(catalog_size=50, dataset_seed=123)
    b = Settings(catalog_size=50, dataset_seed=123)
    pa = generate_catalog(a)
    pb = generate_catalog(b)
    assert [p.model_dump(mode="json") for p in pa] == [p.model_dump(mode="json") for p in pb]


def test_hard_cases_present():
    products = generate_catalog(Settings(catalog_size=50, dataset_seed=42))
    by_id = {p.product_id: p for p in products}
    assert by_id["prod_000000"].sku == "AUDIO-X200-BLK"
    assert by_id["prod_000001"].sku == "AUDIO-X210-BLK"
    assert "device_family_7" in by_id["prod_000003"].compatible_with


def test_write_and_load_roundtrip(tmp_path):
    settings = Settings(
        data_processed_dir=tmp_path,
        catalog_size=25,
        dataset_seed=7,
    )
    products = generate_catalog(settings)
    path = write_catalog(products, settings)
    loaded = load_catalog(path, settings)
    assert len(loaded) == 25
    assert loaded[0].product_id == products[0].product_id
