# SPDX-License-Identifier: MIT
"""Tests for the benchmark catalog (.github/scripts/catalog.py) and the
workflow's use of it. These guard the CI benchmark matrix against drift:

- build_args composes the server CLI in a fixed field order (synthetic inputs),
  plus a content-agnostic smoke pass over the real catalog
- build_cells reproduces the legacy effective matrix (concurrency bands ==
  the old hard-coded `exclude` block)
- result_filename keeps the dashboard/baseline naming contract
- workflow_dispatch model checkboxes stay in sync with the catalog prefixes
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / ".github" / "scripts"
CATALOG = str(REPO / ".github" / "benchmark" / "models.json")
WORKFLOW = REPO / ".github" / "workflows" / "atom-benchmark.yaml"

sys.path.insert(0, str(SCRIPTS))

import catalog  # noqa: E402
from build_benchmark_matrix import RESERVED_INPUTS  # noqa: E402

# Legacy hard-coded matrix `exclude` block (suffix, concurrency) pairs. The
# refactor must reproduce exactly this pruning via per-variant conc bands.
LEGACY_EXCLUDE = {
    ("-mtp3", 1),
    ("-mtp3", 2),
    ("-mtp3", 512),
    ("-mtp3", 1024),
    ("-dpa", 2),
    ("-dpa", 4),
    ("-dpa", 8),
    ("-dpa", 16),
    ("-dpa", 32),
    ("", 512),
    ("", 1024),
}


def test_build_args_composition():
    """build_args composes the CLI in a fixed order from structured fields plus
    verbatim config/variant extra_args. Uses synthetic inputs so it exercises
    the composition contract without coupling to real catalog content (which
    changes often)."""
    # Full: kv_cache_dtype -> tp -> config.extra_args -> variant.extra_args.
    assert (
        catalog.build_args(
            {"kv_cache_dtype": "fp8", "tp": 8, "extra_args": "--foo"},
            {"extra_args": "--bar"},
        )
        == "--kv_cache_dtype fp8 -tp 8 --foo --bar"
    )

    # tp omitted -> no -tp; default dtype fp8 when not set.
    assert catalog.build_args({}, {}) == "--kv_cache_dtype fp8"

    # trust_remote_code -> --trust-remote-code, before extra_args.
    assert (
        catalog.build_args({"tp": 4, "trust_remote_code": True}, {})
        == "--kv_cache_dtype fp8 -tp 4 --trust-remote-code"
    )

    # config.extra_args present, no variant.extra_args.
    assert (
        catalog.build_args({"kv_cache_dtype": "fp8", "tp": 8, "extra_args": "--x"}, {})
        == "--kv_cache_dtype fp8 -tp 8 --x"
    )


def test_build_args_smoke_over_real_catalog():
    """Every real (model, variant) pair produces a well-formed arg string.
    Content-agnostic: asserts shape only, so config edits never break it."""
    cat = catalog._load_catalog(CATALOG)
    for m, v in catalog._iter_variants(cat):
        args = catalog.build_args(m["config"], v)
        assert args.startswith("--kv_cache_dtype "), (m["display"], args)


def test_load_variants_shape():
    variants = catalog.load_variants(CATALOG)
    assert len(variants) == 22
    required = {
        "display",
        "path",
        "prefix",
        "args",
        "bench_args",
        "suffix",
        "runner",
        "env_vars",
        "conc_min",
        "conc_max",
    }
    for v in variants:
        assert required <= set(v)


# Variant suffixes that existed when the structured catalog replaced the
# hard-coded matrix `exclude` block. The migration guarantee is scoped to these;
# variants added later (e.g. -dpa-mtp3) are validated by the band invariants below.
LEGACY_SUFFIXES = {"", "-mtp3", "-dpa"}


def test_build_cells_matches_legacy_effective_matrix():
    """For the migrated suffixes, schedule cells == nightly grid × variants
    minus the legacy `exclude` block (proves the refactor changed nothing)."""
    cat = catalog._load_catalog(CATALOG)
    grid = [
        (sc["isl"], sc["osl"], c, sc["random_range_ratio"])
        for sc in cat["default_scenarios"]
        for c in sc["concurrency"]
    ]
    expected = {
        (v["prefix"], v["suffix"], i, o, c, r)
        for v in catalog.load_variants(CATALOG)
        if v["suffix"] in LEGACY_SUFFIXES
        for (i, o, c, r) in grid
        if (v["suffix"], c) not in LEGACY_EXCLUDE
    }
    got = {
        (c["prefix"], c["suffix"], c["isl"], c["osl"], c["conc"], c["ratio"])
        for c in catalog.build_cells(CATALOG)
        if c["suffix"] in LEGACY_SUFFIXES
    }
    assert got == expected


def test_cells_respect_conc_bands():
    # DP-attention variants run the high-concurrency band; everything else is
    # capped at 256. Keyed on the resolved server args so it stays correct as
    # new DP/non-DP variants are added.
    for c in catalog.build_cells(CATALOG):
        if "--enable-dp-attention" in c["server_args"]:
            assert c["conc"] >= 64
        else:
            assert c["conc"] <= 256


def test_result_filename_contract():
    cells = catalog.build_cells(CATALOG)
    by = {(c["prefix"], c["suffix"], c["isl"], c["osl"], c["conc"]): c for c in cells}
    c = by[("deepseek-v4-pro", "-dpa", 1024, 1024, 512)]
    assert c["result_filename"] == "deepseek-v4-pro-dpa-1024-1024-512-0.8"


def test_param_lists_override_and_conc_band():
    # c=512 only survives for the DP-attention variants (others capped at 256).
    cells = catalog.build_cells(
        CATALOG, param_lists="1024,1024,512,0.7", model_filter={"deepseek-v4-pro"}
    )
    assert sorted(c["suffix"] for c in cells) == ["-dpa", "-dpa-mtp3", "-dpa-tbo"]
    rfs = {c["result_filename"] for c in cells}
    assert "deepseek-v4-pro-dpa-1024-1024-512-0.7" in rfs
    assert "deepseek-v4-pro-dpa-mtp3-1024-1024-512-0.7" in rfs
    assert "deepseek-v4-pro-dpa-tbo-1024-1024-512-0.7" in rfs


def test_model_filter():
    cells = catalog.build_cells(CATALOG, model_filter={"glm-5-2-fp8"})
    assert {c["prefix"] for c in cells} == {"glm-5-2-fp8"}


def test_validate_dispatch_inputs_in_sync_and_drift():
    prefixes = {m["prefix"] for m in catalog._load_catalog(CATALOG)["models"]}
    assert catalog.validate_dispatch_inputs(CATALOG, prefixes) == []
    # missing a checkbox
    assert catalog.validate_dispatch_inputs(CATALOG, prefixes - {"glm-5-2-fp8"})
    # extra checkbox
    assert catalog.validate_dispatch_inputs(CATALOG, prefixes | {"ghost"})


def test_workflow_dispatch_inputs_match_catalog():
    """The 13 workflow_dispatch model toggles must equal the catalog prefixes."""
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load(WORKFLOW.read_text())
    # PyYAML parses the bare `on:` key as boolean True.
    on = wf.get("on", wf.get(True))
    dispatch_inputs = set(on["workflow_dispatch"]["inputs"])
    model_toggles = dispatch_inputs - RESERVED_INPUTS
    prefixes = {m["prefix"] for m in catalog._load_catalog(CATALOG)["models"]}
    assert model_toggles == prefixes


def test_scenario_tag():
    assert catalog.scenario_tag(1024, 1024) == "1k1k"
    assert catalog.scenario_tag(8192, 1024) == "8k1k"
    # Non-1024-multiple lengths fall back to an unambiguous tag.
    assert catalog.scenario_tag(1000, 1024) == "1000_1024"


def test_build_cell_configs_partitions_cells():
    """Configs are a lossless regrouping of build_cells: every cell appears in
    exactly one config (keyed by variant × scenario), expanded over concurrency."""
    import json

    cells = catalog.build_cells(CATALOG)
    configs = catalog.build_cell_configs(CATALOG)

    # Reconstruct the flat (variant, scenario, conc) set from configs.
    from_configs = set()
    for cfg in configs:
        conc_list = json.loads(cfg["concurrency"])
        assert conc_list == sorted(conc_list), "concurrency must be sorted"
        for conc in conc_list:
            from_configs.add(
                (cfg["prefix"], cfg["suffix"], cfg["isl"], cfg["osl"], conc)
            )
    from_cells = {
        (c["prefix"], c["suffix"], c["isl"], c["osl"], c["conc"]) for c in cells
    }
    assert from_configs == from_cells
    # Total cells preserved (no dup / drop).
    assert sum(len(json.loads(c["concurrency"])) for c in configs) == len(cells)


def test_build_cell_configs_matrix_under_github_limit():
    """Both fan-out levels must stay under GitHub's 256-jobs-per-matrix cap."""
    import json

    configs = catalog.build_cell_configs(CATALOG)
    assert len(configs) <= 256, "first-level (config) matrix exceeds 256"
    for cfg in configs:
        assert len(json.loads(cfg["concurrency"])) <= 256, "conc matrix exceeds 256"


def test_build_cell_configs_one_config_per_server_key():
    """Each config is a unique (variant, scenario) server-launch key."""
    configs = catalog.build_cell_configs(CATALOG)
    keys = [
        (c["model_path"], c["server_args"], c["env_vars"], c["isl"], c["osl"])
        for c in configs
    ]
    assert len(keys) == len(set(keys))
