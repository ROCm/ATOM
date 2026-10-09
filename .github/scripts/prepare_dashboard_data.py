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

PREFIX = b"window.BENCHMARK_DATA = "


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def entry_count(data: Any) -> int:
    entries = data.get("entries")
    if not isinstance(entries, dict):
        raise TypeError("dashboard data must contain an 'entries' object")
    return sum(len(value) for value in entries.values() if isinstance(value, list))


def parse_data(text: bytes) -> tuple[Any, bool]:
    """Parse data.js and return ``(data, has_trailing_semicolon)``."""
    if not text.startswith(PREFIX):
        raise ValueError(f"missing {PREFIX.decode()!r} prefix")

    payload = text[len(PREFIX) :].strip()
    has_trailing_semicolon = False
    while payload.endswith(b";"):
        has_trailing_semicolon = True
        payload = payload[:-1].rstrip()

    data = json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)
    if not isinstance(data, dict):
        raise TypeError("dashboard data must be a JSON object")

    entry_count(data)

    return data, has_trailing_semicolon


def normalize(text: bytes) -> tuple[bytes, bool]:
    """Validate text and remove only a terminal JavaScript semicolon."""
    _, has_trailing_semicolon = parse_data(text)
    if not has_trailing_semicolon:
        return text, False

    content = text.rstrip()
    while content.endswith(b";"):
        content = content[:-1].rstrip()
    normalized = content + text[len(text.rstrip()) :]
    return normalized, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_js", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Remove a terminal semicolon in place after validation.",
    )
    parser.add_argument(
        "--min-entries",
        type=int,
        default=None,
        help="Require at least this many benchmark entries after validation.",
    )
    args = parser.parse_args()

    if not args.data_js.is_file():
        print(f"{args.data_js}: does not exist; allowing first dashboard run")
        print("entry_count=0")
        return 0

    try:
        original = args.data_js.read_bytes()
        normalized, changed = normalize(original)
        data, _ = parse_data(normalized)
        count = entry_count(data)
        if args.min_entries is not None and count < args.min_entries:
            raise ValueError(
                f"dashboard entry count decreased from {args.min_entries} to {count}"
            )
    except (OSError, TypeError, ValueError) as exc:
        print(f"{args.data_js}: invalid dashboard data: {exc}", file=sys.stderr)
        return 1

    if changed and args.write:
        args.data_js.write_bytes(normalized)
        print(f"{args.data_js}: removed terminal semicolon")
    elif changed:
        print(f"{args.data_js}: valid but requires normalization", file=sys.stderr)
        return 1
    else:
        print(f"{args.data_js}: valid")

    print(f"entry_count={count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
