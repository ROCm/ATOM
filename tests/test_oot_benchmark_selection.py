import importlib.util
import json
from pathlib import Path

import pytest


ATOM_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ATOM_ROOT / ".github/benchmark/oot_benchmark_models.json"
SCRIPT_PATH = ATOM_ROOT / ".github/scripts/resolve_oot_benchmark_selection.py"


def _load_catalog():
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert isinstance(
        payload, dict
    ), "Expected benchmark model catalog to use a family-based object schema."
    assert "families" in payload, "Expected benchmark model catalog to define families."
    return payload


def _load_selection_module():
    assert SCRIPT_PATH.exists(), "Expected benchmark selection resolver script to exist."
    spec = importlib.util.spec_from_file_location(
        "resolve_oot_benchmark_selection", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_groups_variants_by_family():
    catalog = _load_catalog()
    families = catalog["families"]

    qwen_fp8 = next(
        family
        for family in families
        if family["choice_label"] == "Qwen3.5-397B-A17B-FP8"
    )

    assert sorted(variant["tp_size"] for variant in qwen_fp8["variants"]) == [4, 8]
    assert [variant["prefix"] for variant in qwen_fp8["variants"]] == [
        "qwen3-5-397b-a17b-fp8-tp4",
        "qwen3-5-397b-a17b-fp8",
    ]


def test_manual_selection_resolves_selected_variant_labels():
    catalog = _load_catalog()
    resolver = _load_selection_module()

    selected = resolver.resolve_manual_variants(
        catalog,
        [
            "DeepSeek-R1 FP8 TP8",
            "Qwen3.5-397B-A17B-FP8 TP4",
            "Qwen3.5-397B-A17B-FP8 TP8",
        ],
    )

    assert [model["prefix"] for model in selected] == [
        "deepseek-r1-fp8",
        "qwen3-5-397b-a17b-fp8-tp4",
        "qwen3-5-397b-a17b-fp8",
    ]


def test_manual_selection_deduplicates_repeated_variant_labels():
    catalog = _load_catalog()
    resolver = _load_selection_module()

    selected = resolver.resolve_manual_variants(
        catalog,
        [
            "Qwen3-Next-80B-A3B-Instruct-FP8 TP1",
            "Qwen3-Next-80B-A3B-Instruct-FP8 TP2",
            "Qwen3-Next-80B-A3B-Instruct-FP8 TP2",
            "Qwen3-Next-80B-A3B-Instruct-FP8 TP4",
        ],
    )

    assert [model["prefix"] for model in selected] == [
        "qwen3-next-80b-a3b-instruct-fp8-tp1",
        "qwen3-next-80b-a3b-instruct-fp8-tp2",
        "qwen3-next-80b-a3b-instruct-fp8-tp4",
    ]


def test_manual_selection_rejects_unknown_variant_label():
    catalog = _load_catalog()
    resolver = _load_selection_module()

    with pytest.raises(
        ValueError, match="Unknown benchmark model variant choice"
    ):
        resolver.resolve_manual_variants(catalog, ["DeepSeek-R1 FP8 TP4"])


def test_scheduled_selection_expands_all_variants_in_selected_group():
    catalog = _load_catalog()
    resolver = _load_selection_module()

    selected = resolver.resolve_scheduled_variants(catalog, "A")

    assert {model["prefix"] for model in selected} == {
        "qwen3-5-397b-a17b-fp8",
        "qwen3-5-397b-a17b-fp8-tp4",
        "qwen3-5-397b-a17b-mxfp4",
        "qwen3-5-397b-a17b",
        "qwen3-5-397b-a17b-tp4",
        "qwen3-next-80b-a3b-instruct-fp8-tp1",
        "qwen3-next-80b-a3b-instruct-fp8-tp2",
        "qwen3-next-80b-a3b-instruct-fp8-tp4",
    }
