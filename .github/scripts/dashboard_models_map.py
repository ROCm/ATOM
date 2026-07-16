#!/usr/bin/env python3
"""Emit the dashboard's prefix→display-name map as a JS file.

The benchmark dashboard keys data by the HF model id's last path segment
(lowercased) plus the variant suffix. This script derives that map from the
catalog so display names stay in sync with models.json.

Usage:
    dashboard_models_map.py [CATALOG] [OUTPUT_JS]
    (defaults: .github/benchmark/models.json  /tmp/dashboard_models_map.js)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import load_variants  # noqa: E402


def build_map(catalog_path: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for m in load_variants(catalog_path):
        display, path, suffix = (
            m.get("display", ""),
            m.get("path", ""),
            m.get("suffix", ""),
        )
        if not display or not path:
            continue
        hf_name = path.split("/")[-1].lower()
        hf_key = hf_name + suffix.lower() if suffix else hf_name
        mapping[hf_key] = display
    return mapping


def main() -> int:
    catalog = sys.argv[1] if len(sys.argv) > 1 else ".github/benchmark/models.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/dashboard_models_map.js"
    mapping = build_map(catalog)
    Path(out).write_text(
        "window.MODELS_DISPLAY_MAP = " + json.dumps(mapping) + ";", encoding="utf-8"
    )
    print(f"Wrote {len(mapping)} display-name entries to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
