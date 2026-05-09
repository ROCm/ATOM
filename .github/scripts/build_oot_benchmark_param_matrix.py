#!/usr/bin/env python3
"""Build the benchmark parameter matrix from separated workflow inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ALLOWED_CONCURRENCIES = (4, 8, 16, 32, 64, 128, 256, 512)


def _dedupe_preserve_order(values: list) -> list:
    deduped = []
    seen = set()
    for value in values:
        marker = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(value)
    return deduped


def parse_isl_osl_pairs(isl_osl_pairs_text: str) -> list[dict]:
    pairs = []
    for raw_pair in isl_osl_pairs_text.split(";"):
        pair = raw_pair.strip()
        if not pair:
            continue
        parts = [part.strip() for part in pair.split(",")]
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(
                f"Invalid ISL/OSL pair {pair!r}. Use semicolon-separated pairs like 1024,1024;2048,1024."
            )
        pairs.append(
            {
                "input_length": int(parts[0]),
                "output_length": int(parts[1]),
            }
        )

    if not pairs:
        raise ValueError("At least one ISL/OSL pair is required.")

    return _dedupe_preserve_order(pairs)


def parse_concurrency_values(concurrency_values_text: str) -> list[int]:
    values = []
    for raw_value in concurrency_values_text.split(","):
        value = raw_value.strip()
        if not value:
            continue
        if not value.isdigit():
            raise ValueError(
                f"Invalid concurrency {value!r}. Allowed values: 4,8,16,32,64,128,256,512"
            )
        parsed = int(value)
        if parsed not in ALLOWED_CONCURRENCIES:
            raise ValueError(
                f"Unsupported concurrency: {parsed}. Allowed values: 4,8,16,32,64,128,256,512"
            )
        values.append(parsed)

    if not values:
        raise ValueError("At least one concurrency value is required.")

    return _dedupe_preserve_order(values)


def parse_random_range_ratios(random_range_ratios_text: str) -> list[str]:
    text = random_range_ratios_text.strip()
    if not text:
        return ["0.8"]

    ratios = []
    for raw_ratio in text.split(","):
        ratio = raw_ratio.strip()
        if not ratio:
            continue
        try:
            float(ratio)
        except ValueError as exc:
            raise ValueError(
                f"Invalid random range ratio {ratio!r}. Use values like 0.8 or 1.0."
            ) from exc
        ratios.append(ratio)

    if not ratios:
        return ["0.8"]

    return _dedupe_preserve_order(ratios)


def build_param_matrix(
    *,
    isl_osl_pairs_text: str,
    concurrency_values_text: str,
    random_range_ratios_text: str,
) -> list[dict]:
    pairs = parse_isl_osl_pairs(isl_osl_pairs_text)
    concurrencies = parse_concurrency_values(concurrency_values_text)
    ratios = parse_random_range_ratios(random_range_ratios_text)

    matrix = []
    for pair in pairs:
        for concurrency in concurrencies:
            for ratio in ratios:
                matrix.append(
                    {
                        "input_length": pair["input_length"],
                        "output_length": pair["output_length"],
                        "concurrency": concurrency,
                        "random_range_ratio": ratio,
                    }
                )
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OOT benchmark parameter matrix")
    parser.add_argument("--isl-osl-pairs", required=True)
    parser.add_argument("--concurrency-values", required=True)
    parser.add_argument("--random-range-ratios", default="")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON file path. If omitted, only stdout key=value lines are printed.",
    )
    args = parser.parse_args()

    matrix = build_param_matrix(
        isl_osl_pairs_text=args.isl_osl_pairs,
        concurrency_values_text=args.concurrency_values,
        random_range_ratios_text=args.random_range_ratios,
    )

    matrix_json = json.dumps(matrix, separators=(",", ":"))
    print(f"matrix_json={matrix_json}")
    if args.output_json:
        Path(args.output_json).write_text(matrix_json, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
