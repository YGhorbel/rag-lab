"""Ask the Catalog."""

from __future__ import annotations

import streamlit as st

from src.config import get_settings
from src.data.catalog_loader import load_catalog, products_by_id
from src.generation.answer_parser import parse_answer
from src.generation.citations import validate_citations
from src.generation.ollama_client import generate
from src.generation.prompts import build_user_prompt, format_evidence
from src.retrieval.hybrid import HybridRetriever

st.set_page_config(page_title="Ask the Catalog", layout="wide")
st.title("Ask the Catalog")

EXAMPLES: list[tuple[str, str]] = [
    ("en", "Find SKU LAPTO-X236-BLU3."),
    ("fr", "Casque noir à moins de 150€ avec au moins 30 heures de batterie."),
    ("en", "I need lightweight headphones for long flights."),
    ("fr", "Compare le X200 et le X210 sur batterie, poids et réduction de bruit."),
    ("en", "Which accessories are compatible with device family 7?"),
    ("fr", "Trouve un accessoire compatible avec l'appareil qui supporte device family 7 et coûte moins de 50€."),
    ("en", "Does X240 Cameras Graphite include a five-year warranty?"),
    ("fr", "Montre-moi le compact."),
]


def _apply_example() -> None:
    language, text = EXAMPLES[st.session_state.example_idx]
    st.session_state.question = text
    st.session_state.lang = language


if "question" not in st.session_state:
    st.session_state.question = EXAMPLES[0][1]
if "lang" not in st.session_state:
    st.session_state.lang = EXAMPLES[0][0]

st.selectbox(
    "Example",
    range(len(EXAMPLES)),
    format_func=lambda index: EXAMPLES[index][1],
    key="example_idx",
    on_change=_apply_example,
)
query = st.text_input("Question", key="question")
lang = st.selectbox("Answer language", ["en", "fr"], key="lang")


@st.cache_resource
def _retriever() -> HybridRetriever:
    return HybridRetriever.from_settings(
        use_hash_embeddings=False,
        memory_dense=False,
        apply_hard_filters=True,
    )


@st.cache_data
def _by_id():
    return products_by_id(load_catalog())


if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Enter a question.")
        st.stop()
    try:
        trace = _retriever().retrieve(query, mode="hybrid", top_k=8)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Retrieval failed: {exc}")
        st.stop()

    if trace.plan.needs_clarification:
        st.warning("This question is ambiguous. Add a category, brand, or constraint.")

    catalog = _by_id()
    shown = []
    for item in trace.fused[:8]:
        product = catalog.get(item.product_id)
        if product:
            shown.append((item, product))

    settings = get_settings()
    records = [
        product.model_dump(mode="json", exclude={"updated_at"}) for _, product in shown
    ]
    question = query.strip()
    if lang == "fr":
        question += "\n\nWrite the answer field in French."

    try:
        with st.spinner("Writing the answer…"):
            raw = generate(
                build_user_prompt(question, format_evidence(records, settings.context_budget_chars)),
                settings=settings,
            )
        answer = parse_answer(raw)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Generation failed: {exc}")
        st.stop()

    if answer.abstained:
        st.warning(answer.answer)
    else:
        st.markdown(answer.answer)

    evidence_ids = {product.product_id for _, product in shown}
    if answer.citations and not validate_citations(answer, evidence_ids):
        st.error("A citation points at a product that was not retrieved.")
    if answer.citations:
        st.caption(
            "Citations: "
            + ", ".join(f"`{citation.product_id}`" for citation in answer.citations)
        )
    if answer.missing_information:
        st.caption("Missing: " + ", ".join(answer.missing_information))

    st.subheader("Retrieved products")
    if not shown:
        st.info("No products matched.")
    else:
        st.dataframe(
            [
                {
                    "rank": item.rank,
                    "title": product.title,
                    "sku": product.sku,
                    "brand": product.brand,
                    "category": product.category,
                    "price_eur": product.price,
                    "in_stock": product.in_stock,
                    "product_id": product.product_id,
                }
                for item, product in shown
            ],
            use_container_width=True,
            hide_index=True,
        )
