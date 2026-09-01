#!/usr/bin/env python3
"""Validate and normalize a github-action-benchmark data.js file.

github-action-benchmark reads data.js by passing everything after
``window.BENCHMARK_DATA = `` directly to ``JSON.parse``.  Keep the file valid
for that parser while preserving the existing formatting and history.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PREFIX = "window.BENCHMARK_DATA = "


def parse_data(text: str) -> tuple[Any, bool]:
    """Parse data.js and return ``(data, has_trailing_semicolon)``."""
    if not text.startswith(PREFIX):
        raise ValueError(f"missing {PREFIX!r} prefix")

    payload = text[len(PREFIX) :].strip()
    has_trailing_semicolon = payload.endswith(";")
    if has_trailing_semicolon:
        payload = payload[:-1].rstrip()

    data = json.loads(payload)
    if not isinstance(data, dict):
        raise TypeError("dashboard data must be a JSON object")

    entries = data.get("entries")
    if not isinstance(entries, dict):
        raise TypeError("dashboard data must contain an 'entries' object")
    if not entries or not any(
        isinstance(value, list) and value for value in entries.values()
    ):
        raise ValueError("dashboard data contains no benchmark entries")

    return data, has_trailing_semicolon


def normalize(text: str) -> tuple[str, bool]:
    """Validate text and remove only a terminal JavaScript semicolon."""
    _, has_trailing_semicolon = parse_data(text)
    if not has_trailing_semicolon:
        return text, False

    content_end = len(text.rstrip())
    normalized = text[: content_end - 1] + text[content_end:]
    return normalized, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_js", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Remove a terminal semicolon in place after validation.",
    )
    args = parser.parse_args()

    if not args.data_js.exists():
        print(f"{args.data_js}: does not exist; allowing first dashboard run")
        return 0

    try:
        original = args.data_js.read_text(encoding="utf-8")
        normalized, changed = normalize(original)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"{args.data_js}: invalid dashboard data: {exc}", file=sys.stderr)
        return 1

    if changed and args.write:
        args.data_js.write_text(normalized, encoding="utf-8")
        print(f"{args.data_js}: removed terminal semicolon")
    elif changed:
        print(f"{args.data_js}: valid but requires normalization")
    else:
        print(f"{args.data_js}: valid")

    return 0


if __name__ == "__main__":
    sys.exit(main())
