"""Dataset Explorer — inspect synthetic catalog and gold queries."""

from __future__ import annotations

import streamlit as st

from src.config import get_settings
from src.data.benchmark_generator import load_benchmark
from src.data.catalog_loader import load_catalog

st.set_page_config(page_title="Dataset Explorer", layout="wide")
st.title("Dataset Explorer")

settings = get_settings()

if not settings.catalog_path.exists():
    st.error("Catalog missing. Run: `rag-lab-generate-catalog`")
    st.stop()

@st.cache_data
def _products():
    return load_catalog()

products = _products()
st.metric("Products", len(products))

q = st.text_input("Search title / SKU / brand")
rows = products
if q:
    ql = q.lower()
    rows = [
        p
        for p in products
        if ql in p.title.lower() or ql in p.sku.lower() or ql in p.brand.lower()
    ]

st.caption(f"{len(rows)} products")
st.dataframe(
    [
        {
            "product_id": p.product_id,
            "sku": p.sku,
            "title": p.title,
            "brand": p.brand,
            "category": p.category,
            "price": p.price,
            "in_stock": p.in_stock,
        }
        for p in rows[:50]
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Gold queries")
if settings.demo_benchmark_path.exists():
    demo = load_benchmark(settings.demo_benchmark_path)
    st.write(f"{len(demo)} demo queries")
    st.dataframe(
        [
            {
                "query_id": b.query_id,
                "type": b.query_type.value,
                "language": b.language,
                "answerable": b.answerable,
                "text": b.query_text,
                "expected": ",".join(b.expected_product_ids[:5]),
            }
            for b in demo[:40]
        ],
        use_container_width=True,
    )
else:
    st.warning("Demo benchmark missing. Run: `rag-lab-generate-benchmark`")
