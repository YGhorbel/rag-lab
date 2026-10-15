"""Retrieval Inspector — stage traces."""

from __future__ import annotations

import streamlit as st

from src.retrieval.hybrid import HybridRetriever

st.set_page_config(page_title="Retrieval Inspector", layout="wide")
st.title("Retrieval Inspector")

query = st.text_input("Query", placeholder="Find SKU AUDIO-X200-BLK")
mode = st.selectbox("Mode", ["hybrid", "bm25", "dense", "exact"])


@st.cache_resource
def _retriever() -> HybridRetriever:
    return HybridRetriever.from_settings(
        use_hash_embeddings=False,
        memory_dense=False,
        apply_hard_filters=True,
    )


if st.button("Retrieve", type="primary"):
    if not query.strip():
        st.warning("Enter a query.")
        st.stop()
    try:
        trace = _retriever().retrieve(query, mode=mode, top_k=10)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Retrieval failed: {exc}")
        st.stop()

    st.caption(
        " · ".join(f"{name} {ms} ms" for name, ms in trace.latencies_ms.items())
    )
    active = {
        key: value
        for key, value in trace.filters_applied.model_dump().items()
        if value not in (None, {}, [])
    }
    if active:
        st.write(active)

    tabs = st.tabs(["Fused", "Exact", "BM25", "Dense"])
    for tab, items in zip(tabs, [trace.fused, trace.exact, trace.bm25, trace.dense]):
        with tab:
            st.dataframe(
                [
                    {
                        "rank": item.rank,
                        "product_id": item.product_id,
                        "score": round(item.score, 4),
                        "source": item.source.value,
                    }
                    for item in items[:10]
                ],
                use_container_width=True,
                hide_index=True,
            )
