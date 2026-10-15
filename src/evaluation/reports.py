"""Persist experiment reports."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_report(results: dict[str, Any], directory: Path, stem: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = stem or f"eval_{results.get('mode', 'run')}_{ts}"
    json_path = directory / f"{stem}.json"
    csv_path = directory / f"{stem}.csv"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = results.get("per_query", [])
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return json_path
