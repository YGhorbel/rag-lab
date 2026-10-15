"""Evaluation Lab — all retrieval modes on the full gold set."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.evaluation.reports import write_report
from src.evaluation.run_benchmark import run_benchmark
from src.retrieval.hybrid import HybridRetriever

MODES = ["exact", "bm25", "dense", "hybrid"]

METRIC_HELP = [
    ("recall@10", "Share of gold products found in the top 10. Good ≥ 0.50 · weak < 0.20."),
    ("precision@10", "Share of the top 10 that are gold. No absolute bar here — compare modes."),
    ("mrr", "1 / rank of the first gold product. Good ≥ 0.70 · weak < 0.30."),
    ("ndcg@10", "Recall with higher weight on top ranks. Good ≥ 0.50 · weak < 0.20."),
    (
        "filter_correctness",
        "Share of top-10 hits that obey hard filters. Good ≥ 0.95 · otherwise weak.",
    ),
    (
        "exact_id_success",
        "SKU questions only: gold product is rank 1. Good ≥ 0.95 · otherwise weak.",
    ),
    ("latency_ms_mean", "Average milliseconds per query. Timing only — not colored."),
]

# (good_min, bad_max). None = leave black (no quality signal).
THRESHOLDS: dict[str, tuple[float | None, float | None]] = {
    "recall": (0.50, 0.20),
    "precision": (None, None),
    "mrr": (0.70, 0.30),
    "ndcg": (0.50, 0.20),
    "filter_correctness": (0.95, 0.95),
    "exact_id_success": (0.95, 0.95),
    "latency_ms_mean": (None, None),
}


def _metric_key(column: str) -> str:
    if column.startswith("recall"):
        return "recall"
    if column.startswith("precision"):
        return "precision"
    if column.startswith("ndcg"):
        return "ndcg"
    return column


def _tone(column: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "black"
    key = _metric_key(column)
    bounds = THRESHOLDS.get(key)
    if bounds is None:
        return "black"
    good_min, bad_max = bounds
    if good_min is None and bad_max is None:
        return "black"
    number = float(value)
    if good_min is not None and number >= good_min:
        return "green"
    if bad_max is not None and number < bad_max:
        return "red"
    if good_min is not None and bad_max is not None and good_min == bad_max:
        return "red"
    return "black"


def _color_cell(column: str, value: Any) -> str:
    tone = _tone(column, value)
    if tone == "green":
        return "color: #22c55e; font-weight: 600"
    if tone == "red":
        return "color: #f87171; font-weight: 600"
    # Empty style keeps Streamlit's theme text color (black was invisible in dark mode).
    return ""


def _style_metrics(frame: pd.DataFrame):
    def paint(series: pd.Series) -> list[str]:
        return [_color_cell(series.name, value) for value in series]

    return frame.style.apply(paint, axis=0)


st.set_page_config(page_title="Evaluation Lab", layout="wide")
st.title("Evaluation Lab")
st.caption("Runs exact, BM25, dense, and hybrid on the full gold set against Qdrant.")

with st.expander("Metric guide", expanded=True):
    for name, text in METRIC_HELP:
        st.markdown(f"**{name}** — {text}")
    st.caption(
        "Green = good · red = weak · default text = no absolute threshold or missing value."
    )

settings = get_settings()


@st.cache_resource
def _retriever() -> HybridRetriever:
    return HybridRetriever.from_settings(
        settings=settings,
        use_hash_embeddings=False,
        memory_dense=False,
        apply_hard_filters=True,
    )


if st.button("Run all", type="primary"):
    retriever = _retriever()
    rows = []
    details = {}
    progress = st.progress(0.0, text="Starting")
    for index, mode in enumerate(MODES, start=1):
        progress.progress((index - 1) / len(MODES), text=f"Running {mode}")
        results = run_benchmark(
            mode=mode,
            subset="full",
            k=10,
            use_hash_embeddings=False,
            memory_dense=False,
            apply_hard_filters=True,
            settings=settings,
            retriever=retriever,
        )
        write_report(results, settings.experiments_dir)
        rows.append({"mode": mode, "queries": results["n_queries"], **results["metrics"]})
        details[mode] = results["per_query"]
    progress.progress(1.0, text="Done")
    st.session_state["eval_rows"] = rows
    st.session_state["eval_details"] = details

rows = st.session_state.get("eval_rows")
if rows:
    st.subheader("All modes")
    summary = pd.DataFrame(rows)
    st.dataframe(_style_metrics(summary), width="stretch", hide_index=True)

    mode = st.selectbox("Per-query", MODES)
    detail = pd.DataFrame(st.session_state["eval_details"][mode])
    st.dataframe(_style_metrics(detail), width="stretch", hide_index=True)

reports = sorted(settings.experiments_dir.glob("eval_*.json"), reverse=True)
if reports and not rows:
    st.caption(f"Last saved report: {reports[0].name}")
    data = json.loads(reports[0].read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    if metrics:
        st.dataframe(
            _style_metrics(pd.DataFrame([{"mode": data.get("mode", "?"), **metrics}])),
            width="stretch",
            hide_index=True,
        )
