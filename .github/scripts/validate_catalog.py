#!/usr/bin/env python3
"""Validate the accuracy catalogs against their JSON Schema + a few semantic rules.

Single source of truth = the catalog JSONs under .github/benchmark/. This script
is the T0 gate that keeps them well-formed: schema-valid shape (no typos / stray
fields) plus cross-field sanity that a pure schema can't express.

Run locally:  python .github/scripts/validate_catalog.py
Exit code 0 = all catalogs valid; 1 = at least one problem (details printed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

REPO_GITHUB = Path(__file__).resolve().parents[1]  # .github/
BENCH = REPO_GITHUB / "benchmark"
SCHEMA = BENCH / "schema" / "accuracy_catalog.schema.json"

ACCURACY_CATALOGS = [
    "models_accuracy.json",
    "oot_models_accuracy.json",
    "sglang_models_accuracy.json",
]


def _load(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _semantic_checks(entry: dict, idx: int) -> list[str]:
    """Cross-field rules the schema cannot express. Returns a list of problems."""
    problems: list[str] = []
    name = entry.get("model_name", f"#{idx}")

    # Every entry must declare a pass bar under exactly one of the two spellings
    # the catalogs use today (accuracy_threshold / accuracy_test_threshold). This
    # catches both omission and accidentally setting both during the pending
    # drift normalization.
    spellings = [
        k for k in ("accuracy_threshold", "accuracy_test_threshold") if k in entry
    ]
    if len(spellings) == 0:
        problems.append(
            f"[{name}] missing pass bar: set accuracy_threshold (or accuracy_test_threshold)"
        )
    elif len(spellings) == 2:
        problems.append(
            f"[{name}] has both accuracy_threshold and accuracy_test_threshold; keep one"
        )
    return problems


def validate_one(filename: str, validator: Draft202012Validator) -> list[str]:
    path = BENCH / filename
    if not path.exists():
        return [f"{filename}: missing"]

    try:
        data = _load(path)
    except json.JSONDecodeError as exc:
        return [f"{filename}: invalid JSON — {exc}"]

    problems: list[str] = []
    # schema errors (shape / required / enums / additionalProperties)
    for err in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in err.path) or "<root>"
        problems.append(f"{filename}: {loc}: {err.message}")

    # semantic errors (only meaningful once the shape is a list of objects)
    if isinstance(data, list):
        for idx, entry in enumerate(data):
            if isinstance(entry, dict):
                problems.extend(
                    f"{filename}: {p}" for p in _semantic_checks(entry, idx)
                )

    return problems


def main() -> int:
    if not SCHEMA.exists():
        print(f"ERROR: schema not found: {SCHEMA}", file=sys.stderr)
        return 1

    schema = _load(SCHEMA)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    all_problems: list[str] = []
    for filename in ACCURACY_CATALOGS:
        problems = validate_one(filename, validator)
        status = "OK" if not problems else f"{len(problems)} problem(s)"
        print(f"  {filename}: {status}")
        all_problems.extend(problems)

    if all_problems:
        print("\nCatalog validation FAILED:", file=sys.stderr)
        for p in all_problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print("\nAll accuracy catalogs valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
