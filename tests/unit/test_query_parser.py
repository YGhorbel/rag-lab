"""Deterministic query parser tests."""

from src.config import Settings
from src.data.synthetic_generator import generate_catalog
from src.retrieval.query_parser import CatalogVocab, parse_query
from src.schemas.query import QueryIntent


def test_parse_sku_exact():
    products = generate_catalog(Settings(catalog_size=20, dataset_seed=42))
    vocab = CatalogVocab.from_products(products)
    plan = parse_query("Find SKU AUDIO-X200-BLK", vocab)
    assert plan.intent == QueryIntent.EXACT_LOOKUP
    assert plan.filters.sku == "AUDIO-X200-BLK"


def test_parse_price_and_category():
    products = generate_catalog(Settings(catalog_size=50, dataset_seed=42))
    vocab = CatalogVocab.from_products(products)
    plan = parse_query(
        "Black headphones under €150 with at least 30 hours of battery",
        vocab,
    )
    assert plan.filters.max_price == 150
    assert plan.filters.min_battery_hours == 30
    assert plan.filters.color == "black"
    assert plan.filters.category == "headphones"


def test_unknown_brand_not_invented():
    products = generate_catalog(Settings(catalog_size=30, dataset_seed=42))
    vocab = CatalogVocab.from_products(products)
    plan = parse_query("Show me FakeBrandX headphones", vocab)
    assert plan.filters.brand is None
