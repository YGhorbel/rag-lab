"""Deterministic query understanding — no LLM-invented filters."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.schemas.product import Product
from src.schemas.query import QueryFilters, QueryIntent, QueryPlan

SKU_RE = re.compile(r"\b([A-Z]{2,8}-[A-Z0-9]{2,}(?:-[A-Z0-9]+)*)\b", re.IGNORECASE)
PRICE_MAX_RE = re.compile(
    r"(?:under|below|less than|moins de|à moins de|<)\s*€?\s*(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
PRICE_MAX_RE2 = re.compile(r"€\s*(\d+(?:[.,]\d+)?)\s*(?:or less|max|maximum)?", re.IGNORECASE)
PRICE_MIN_RE = re.compile(
    r"(?:over|above|more than|plus de|>)\s*€?\s*(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
BATTERY_RE = re.compile(
    r"(?:at least|minimum|min(?:imum)?|au moins)\s*(\d+)\s*(?:hours?|h\b|heures?)",
    re.IGNORECASE,
)
BATTERY_RE2 = re.compile(r"(\d+)\s*(?:hours?|h|heures?)\s*(?:of\s+)?battery", re.IGNORECASE)
IN_STOCK_RE = re.compile(r"\b(in stock|available|en stock|disponible)\b", re.IGNORECASE)
COLOR_WORDS = {
    "black", "white", "silver", "blue", "red", "green", "graphite",
    "noir", "blanc", "bleu", "rouge", "vert",
}
COLOR_MAP = {
    "noir": "black",
    "blanc": "white",
    "bleu": "blue",
    "rouge": "red",
    "vert": "green",
}


@dataclass
class CatalogVocab:
    brands: set[str]
    categories: set[str]
    brand_lower: dict[str, str]
    category_lower: dict[str, str]

    @classmethod
    def from_products(cls, products: list[Product]) -> "CatalogVocab":
        brands = {p.brand for p in products}
        categories = {p.category for p in products}
        return cls(
            brands=brands,
            categories=categories,
            brand_lower={b.lower(): b for b in brands},
            category_lower={c.lower(): c for c in categories},
        )


def _parse_float(raw: str) -> float:
    return float(raw.replace(",", "."))


def parse_query(query: str, vocab: CatalogVocab) -> QueryPlan:
    text = query.strip()
    lower = text.lower()
    filters = QueryFilters()
    keywords: list[str] = []

    sku_match = SKU_RE.search(text)
    if sku_match:
        filters.sku = sku_match.group(1).upper()

    for m in PRICE_MAX_RE.finditer(text):
        filters.max_price = _parse_float(m.group(1))
    if filters.max_price is None:
        m2 = PRICE_MAX_RE2.search(text)
        if m2:
            filters.max_price = _parse_float(m2.group(1))

    for m in PRICE_MIN_RE.finditer(text):
        filters.min_price = _parse_float(m.group(1))

    for m in BATTERY_RE.finditer(text):
        filters.min_battery_hours = float(m.group(1))
    if filters.min_battery_hours is None:
        m = BATTERY_RE2.search(text)
        if m:
            filters.min_battery_hours = float(m.group(1))

    if IN_STOCK_RE.search(text):
        filters.in_stock = True

    # colors
    for word in COLOR_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lower):
            filters.color = COLOR_MAP.get(word, word)
            break

    # category: longest match against vocab
    for cat_l, cat in sorted(vocab.category_lower.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(cat_l)}\b", lower) or cat_l.replace("_", " ") in lower:
            filters.category = cat
            break
    # French aliases
    fr_cats = {
        "casque": "headphones",
        "écouteurs": "earbuds",
        "ecouteurs": "earbuds",
        "ordinateur": "laptops",
        "portable": "laptops",
        "tablette": "tablets",
        "chargeur": "chargers",
        "coque": "cases",
        "accessoire": None,
    }
    if filters.category is None:
        for fr, cat in fr_cats.items():
            if fr in lower and cat and cat in vocab.categories:
                filters.category = cat
                break

    # brand: prefer multi-word brands first
    for brand_l, brand in sorted(vocab.brand_lower.items(), key=lambda x: -len(x[0])):
        if brand_l in lower:
            filters.brand = brand
            break

    # device family
    fam = re.search(r"device[_ ]family[_ ]?(\d+)", lower)
    if fam:
        filters.compatible_with = f"device_family_{fam.group(1)}"

    # keywords: leftover content words
    stop = {
        "the", "a", "an", "for", "with", "and", "or", "of", "to", "in", "on", "find",
        "show", "me", "i", "need", "je", "cherche", "un", "une", "les", "des", "de",
        "le", "la", "du", "under", "at", "least", "hours", "battery",
    }
    for tok in re.findall(r"[a-zA-Z0-9\-]+", lower):
        if tok not in stop and len(tok) > 2:
            keywords.append(tok)

    # intent
    if filters.sku:
        intent = QueryIntent.EXACT_LOOKUP
    elif "compare" in lower or "compare" in lower or "comparez" in lower or " vs " in lower:
        intent = QueryIntent.COMPARISON
    elif filters.compatible_with or "compatible" in lower or "accessoir" in lower:
        intent = QueryIntent.RELATIONSHIP
    elif any(
        getattr(filters, f) is not None
        for f in ("max_price", "min_price", "in_stock", "min_battery_hours", "color", "category", "brand")
    ):
        intent = QueryIntent.FILTERED_PRODUCT_SEARCH
    elif lower in {"show me the compact one", "montre-moi le compact."} or "compact one" in lower:
        intent = QueryIntent.CLARIFICATION_NEEDED
    else:
        intent = QueryIntent.SEMANTIC_DISCOVERY

    needs_clarification = intent == QueryIntent.CLARIFICATION_NEEDED

    return QueryPlan(
        intent=intent,
        keywords=keywords[:12],
        filters=filters,
        needs_clarification=needs_clarification,
    )
