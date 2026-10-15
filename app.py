"""RAG Lab Streamlit entrypoint."""

from __future__ import annotations

import streamlit as st

from src import __version__
from src.config import get_settings

st.set_page_config(
    page_title="RAG Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

settings = get_settings()

st.title("RAG Lab")
st.caption(f"Open-source e-commerce RAG workbench · v{__version__}")

st.markdown("Ask the catalog, inspect a retrieval trace, or run every retrieval mode on the gold set.")

col1, col2, col3 = st.columns(3)
col1.metric("Catalog", settings.catalog_version)
col2.metric("Embedder", settings.embedding_model.split("/")[-1])
col3.metric("Generator", settings.generator_model)

catalog_ok = settings.catalog_path.exists()
bench_ok = settings.benchmark_path.exists()
if not catalog_ok or not bench_ok:
    st.error("Catalog or benchmark file is missing. Generate both before using the pages.")
