"""Schema validation tests."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.schemas.answer import GroundedAnswer
from src.schemas.product import Product
from src.schemas.query import BenchmarkQuery, QueryFilters, QueryType


def test_product_accepts_canonical_fields():
    product = Product(
        product_id="prod_000123",
        sku="AUDIO-X200-BLK",
        title="X200 Wireless Noise-Cancelling Headphones",
        brand="Northstar Audio",
        category="headphones",
        description="Over-ear wireless headphones",
        attributes={"battery_hours": 38, "color": "black"},
        price=129.99,
        currency="EUR",
        in_stock=True,
        rating=4.4,
        tags=["wireless"],
        compatible_with=["device_family_7"],
        related_product_ids=["prod_000124"],
        source_record="synthetic_catalog_v1",
        updated_at=datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc),
    )
    assert "AUDIO-X200-BLK" in product.to_retrieval_text()


def test_product_rejects_negative_price():
    with pytest.raises(ValidationError):
        Product(
            product_id="prod_x",
            sku="X",
            title="t",
            brand="b",
            category="c",
            description="d",
            price=-1,
            updated_at=datetime.now(timezone.utc),
        )


def test_benchmark_requires_expected_ids_when_answerable():
    with pytest.raises(ValidationError):
        BenchmarkQuery(
            query_id="q_bad",
            query_text="find something",
            query_type=QueryType.EXACT_LOOKUP,
            expected_product_ids=[],
            answerable=True,
        )


def test_unanswerable_allows_empty_expected():
    q = BenchmarkQuery(
        query_id="q_un",
        query_text="warranty?",
        query_type=QueryType.UNANSWERABLE,
        expected_product_ids=[],
        answerable=False,
    )
    assert q.answerable is False


def test_grounded_answer_defaults():
    ans = GroundedAnswer(answer="I cannot verify this from the catalog", abstained=True)
    assert ans.citations == []
    assert ans.abstained is True


def test_query_filters_optional():
    filters = QueryFilters(max_price=150, in_stock=True)
    assert filters.brand is None
