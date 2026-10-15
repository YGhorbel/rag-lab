"""Parse and validate grounded JSON answers."""

from __future__ import annotations

import json
import re

from src.schemas.answer import GroundedAnswer


def parse_answer(raw: str) -> GroundedAnswer:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("model output did not contain a JSON object")
        text = text[start : end + 1]
    return GroundedAnswer.model_validate(json.loads(text))
