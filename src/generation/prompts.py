"""Grounded-answer prompts for gpt-oss on Ollama Cloud.

The chat template places this text in the developer message under
`# Instructions`. The model reasons on the analysis channel. The final
channel is the only text returned as `message.content`.
"""

import json

SYSTEM_PROMPT = """You answer questions about a synthetic e-commerce catalog.

The user message contains a question and the only evidence you may use. Evidence is a set of product records. Each record may include product_id, sku, title, brand, category, description, attributes, price, currency, in_stock, rating, tags, compatible_with, related_product_ids, and source_record.

Rules:
- Use only facts written in the evidence. Do not use prior knowledge about products, brands, prices, compatibility, warranty, or stock.
- Cite the product_id for every product-specific claim. A citation is valid only when that product_id appears in the evidence.
- Copy prices, currency, units, SKUs, stock flags, and numeric attributes exactly as written. Do not round, convert, or fill gaps.
- Label each supported fact as "catalog" when the evidence states it, or "inference" when it follows from two or more stated facts. Do not infer specifications, compatibility, warranty, or stock.
- If the evidence does not contain enough information to answer, set abstained to true, set answer to "I cannot verify this from the catalog", list the missing facts in missing_information, and leave citations empty.
- If the question asks to compare products, put a markdown table in answer. Every compared value still needs a citation.
- Reason on the analysis channel. The final channel is one JSON object and no other text. No markdown fence.

JSON object:
{
  "answer": "string",
  "citations": [
    {"product_id": "string", "source_record": "string or null", "fields": ["field names used"]}
  ],
  "supported_facts": [
    {"product_id": "string", "field": "string", "value": "string", "kind": "catalog or inference"}
  ],
  "missing_information": ["string"],
  "abstained": false
}

Example, missing warranty:
{"answer":"I cannot verify this from the catalog","citations":[],"supported_facts":[],"missing_information":["warranty term"],"abstained":true}

Example, price present in evidence for product_id p_100 at 49.90 EUR, source_record synthetic_catalog_v1:
{"answer":"The Nova Lamp costs 49.90 EUR.","citations":[{"product_id":"p_100","source_record":"synthetic_catalog_v1","fields":["price","currency"]}],"supported_facts":[{"product_id":"p_100","field":"price","value":"49.90","kind":"catalog"}],"missing_information":[],"abstained":false}
"""

USER_PROMPT_TEMPLATE = """Question:
{question}

Evidence records:
{evidence}
"""


def build_user_prompt(question: str, evidence: str) -> str:
    return USER_PROMPT_TEMPLATE.format(question=question.strip(), evidence=evidence.strip())


def format_evidence(records: list[dict], budget: int) -> str:
    """One JSON object per line, cut at the context character budget."""
    lines: list[str] = []
    used = 0
    for record in records:
        line = json.dumps(record, ensure_ascii=False, default=str)
        extra = len(line) + 1
        if lines and used + extra > budget:
            break
        lines.append(line)
        used += extra
    return "\n".join(lines) if lines else "(no evidence records)"
