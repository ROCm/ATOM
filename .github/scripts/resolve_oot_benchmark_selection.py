#!/usr/bin/env python3
"""Resolve ATOM vLLM benchmark model selections.

This helper keeps the workflow-facing model payload flat while allowing the
catalog to group TP variants by model family.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


ANCHOR_DATE = date(2026, 5, 6)
NIGHTLY_ROTATION = ("A", "B", "C")
NONE_CHOICE = "none"


def load_catalog(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark catalog must be a JSON object.")

    families = payload.get("families")
    if not isinstance(families, list):
        raise ValueError("Benchmark catalog must define a 'families' list.")

    for family in families:
        if not isinstance(family, dict):
            raise ValueError("Each benchmark family must be a JSON object.")
        if not family.get("choice_label"):
            raise ValueError("Each benchmark family must define 'choice_label'.")
        variants = family.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError(
                f"Benchmark family {family['choice_label']!r} must define variants."
            )
        for variant in variants:
            if "tp_size" not in variant or "prefix" not in variant or "display" not in variant:
                raise ValueError(
                    f"Benchmark family {family['choice_label']!r} has an incomplete variant."
                )
    return payload


def compute_rotated_nightly_group(run_created_at: str) -> str:
    if not run_created_at:
        raise ValueError("Scheduled selection requires a run creation timestamp.")

    beijing_tz = timezone(timedelta(hours=8))
    run_date = datetime.fromisoformat(run_created_at.replace("Z", "+00:00"))
    beijing_date = run_date.astimezone(beijing_tz).date()
    rotation_index = (beijing_date - ANCHOR_DATE).days % len(NIGHTLY_ROTATION)
    return NIGHTLY_ROTATION[rotation_index]


def _flatten_variant(family: dict, variant: dict) -> dict:
    flattened = {
        key: value for key, value in family.items() if key not in {"choice_label", "variants"}
    }
    flattened.update(variant)
    return flattened


def _supported_tp_sizes(family: dict) -> list[int]:
    return [int(variant["tp_size"]) for variant in family["variants"]]


def _normalize_tp_token(token: str) -> int:
    value = token.strip().lower()
    if value.startswith("tp"):
        value = value[2:]
    if not value.isdigit():
        raise ValueError(f"Invalid TP size token {token!r}. Use values like 4,8 or all.")
    return int(value)


def parse_requested_tp_sizes(tp_sizes_text: str, family: dict) -> list[int]:
    supported = _supported_tp_sizes(family)
    normalized = tp_sizes_text.strip()
    if not normalized:
        if len(supported) == 1:
            return supported
        raise ValueError(
            f"TP sizes are required when selecting {family['choice_label']}. Use values like 4,8 or all."
        )

    if normalized.lower() == "all":
        return supported

    requested: list[int] = []
    unsupported: list[int] = []
    for raw_token in normalized.split(","):
        token = raw_token.strip()
        if not token:
            continue
        tp_size = _normalize_tp_token(token)
        if tp_size not in supported:
            unsupported.append(tp_size)
            continue
        if tp_size not in requested:
            requested.append(tp_size)

    if unsupported:
        unsupported_text = ",".join(str(value) for value in unsupported)
        supported_text = ",".join(str(value) for value in supported)
        raise ValueError(
            f"{family['choice_label']} does not support TP sizes {unsupported_text}; "
            f"supported TP sizes: {supported_text}"
        )

    if not requested:
        raise ValueError(
            f"TP sizes are required when selecting {family['choice_label']}. Use values like 4,8 or all."
        )

    return requested


def resolve_manual_variants(catalog: dict, slot_inputs: list[tuple[str, str]]) -> list[dict]:
    families = catalog["families"]
    family_map = {family["choice_label"]: family for family in families}
    selected: list[dict] = []
    seen_prefixes: set[str] = set()

    for choice_label, tp_sizes_text in slot_inputs:
        label = (choice_label or "").strip()
        tp_sizes = (tp_sizes_text or "").strip()

        if not label or label == NONE_CHOICE:
            if tp_sizes:
                raise ValueError(
                    f"TP sizes were provided for an empty model slot: {tp_sizes!r}."
                )
            continue

        family = family_map.get(label)
        if family is None:
            available = ", ".join(sorted(family_map))
            raise ValueError(
                f"Unknown model family {label!r}. Available families: {available}"
            )

        requested_tps = set(parse_requested_tp_sizes(tp_sizes, family))
        for variant in family["variants"]:
            if int(variant["tp_size"]) not in requested_tps:
                continue
            prefix = variant["prefix"]
            if prefix in seen_prefixes:
                continue
            selected.append(_flatten_variant(family, variant))
            seen_prefixes.add(prefix)

    return selected


def resolve_scheduled_variants(catalog: dict, selected_group: str) -> list[dict]:
    group = selected_group.strip().upper()
    if not group:
        raise ValueError("Scheduled selection requires a nightly group.")

    selected: list[dict] = []
    for family in catalog["families"]:
        family_group = str(family.get("nightly_group", "")).upper()
        for variant in family["variants"]:
            variant_group = str(variant.get("nightly_group", family_group)).upper()
            if variant_group != group:
                continue
            selected.append(_flatten_variant(family, variant))

    if not selected:
        raise ValueError(f"No models configured for nightly benchmark group {group}.")

    return selected


def collect_manual_slot_inputs(max_slots: int) -> list[tuple[str, str]]:
    slots: list[tuple[str, str]] = []
    for slot_idx in range(1, max_slots + 1):
        slots.append(
            (
                os.environ.get(f"MODEL_SLOT_{slot_idx}", ""),
                os.environ.get(f"TP_SIZES_SLOT_{slot_idx}", ""),
            )
        )
    return slots


def print_selection_summary(selected: list[dict], *, selected_group: str = "") -> None:
    if selected_group:
        print(
            f"Scheduled nightly benchmark group: {selected_group}",
            file=sys.stderr,
        )
    if not selected:
        print("No benchmark model variants were selected.", file=sys.stderr)
        return

    print("Selected benchmark model variants:", file=sys.stderr)
    for model in selected:
        print(
            f" - {model['display']} ({model['prefix']})",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve OOT benchmark model variants")
    parser.add_argument(
        "--catalog",
        default=".github/benchmark/oot_benchmark_models.json",
        help="Path to the family-based OOT benchmark catalog JSON",
    )
    parser.add_argument(
        "--event",
        required=True,
        help="GitHub event name, for example schedule or workflow_dispatch",
    )
    parser.add_argument(
        "--schedule-created-at",
        default=os.environ.get("SCHEDULE_CREATED_AT", ""),
        help="Creation timestamp for scheduled run rotation, in ISO-8601 format",
    )
    parser.add_argument(
        "--selected-group",
        default=os.environ.get("SELECTED_GROUP", ""),
        help="Optional precomputed nightly group override",
    )
    parser.add_argument(
        "--max-slots",
        type=int,
        default=8,
        help="Maximum number of MODEL_SLOT_N / TP_SIZES_SLOT_N pairs to read",
    )
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)

    if args.event == "schedule":
        selected_group = args.selected_group or compute_rotated_nightly_group(
            args.schedule_created_at
        )
        selected = resolve_scheduled_variants(catalog, selected_group)
    else:
        selected_group = ""
        selected = resolve_manual_variants(
            catalog,
            collect_manual_slot_inputs(args.max_slots),
        )

    print_selection_summary(selected, selected_group=selected_group)
    print(f"models_json={json.dumps(selected, separators=(',', ':'))}")
    print(f"selected_group={selected_group}")
    print(f"has_enabled_models={'true' if selected else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
