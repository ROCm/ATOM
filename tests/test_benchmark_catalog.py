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

import catalog
from build_benchmark_matrix import RESERVED_INPUTS

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
    # Content-agnostic: assert the catalog produces at least one variant and
    # every variant carries the required fields. Deliberately does NOT pin the
    # variant count — that couples the test to catalog churn (models added /
    # removed) without testing any real invariant.
    variants = catalog.load_variants(CATALOG)
    assert variants, "catalog produced no variants"
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
    # Filtered to `random` the way every production caller is: a dispatch grid
    # does not apply to the agentic variants, so unfiltered they would come
    # through here carrying their own concurrency bands.
    cells = catalog.build_cells(
        CATALOG,
        param_lists="1024,1024,512,0.7",
        model_filter={"deepseek-v4-pro"},
        bench_kind_filter={"random"},
    )
    assert sorted(c["suffix"] for c in cells) == [
        "-dpa",
        "-dpa-dspark",
        "-dpa-mtp3",
        "-dpa-tbo",
    ]
    rfs = {c["result_filename"] for c in cells}
    assert "deepseek-v4-pro-dpa-1024-1024-512-0.7" in rfs
    assert "deepseek-v4-pro-dpa-dspark-1024-1024-512-0.7" in rfs
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
    """The workflow_dispatch model toggles must equal the catalog prefixes."""
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


def test_bench_kind_filter_keeps_the_two_workflows_apart():
    """The nightly must never pick up an agentic cell.

    An agentic cell replays for a full hour where a random cell is minutes, so
    the two live in separate workflows. Nothing enforces that but this field:
    the variants share the `deepseek-v4-pro` prefix, so a prefix filter alone
    puts them in the nightly. `atom-benchmark.yaml` passes no BENCH_KIND_FILTER
    and `build_benchmark_matrix.py` defaults it to `random`;
    `atom-agentic-benchmark.yaml` sets `aiperf_agentic`.
    """
    everything = catalog.build_cells(CATALOG)
    nightly = catalog.build_cells(CATALOG, bench_kind_filter={"random"})
    agentic = catalog.build_cells(CATALOG, bench_kind_filter={"aiperf_agentic"})

    assert {c["bench_kind"] for c in nightly} == {"random"}
    assert {c["bench_kind"] for c in agentic} == {"aiperf_agentic"}
    # A partition: every cell lands in exactly one side.
    assert len(nightly) + len(agentic) == len(everything)
    assert agentic, "catalog has no agentic variants -- did bench_kind get dropped?"


def test_agentic_variants_carry_what_the_recipe_requires():
    """Flags the InferenceX single-node recipe sets that the shared config does not.

    `--enable_prefix_caching` is the one that bites: the DeepSeek-V4-Pro config
    turns prefix caching OFF for every variant, and the agentic recipe needs it
    ON. `build_args` appends variant args after config args, so the positive
    form (underscores, matching the BooleanOptionalAction dest) wins -- but only
    if it is actually there.
    """
    agentic = catalog.build_cells(CATALOG, bench_kind_filter={"aiperf_agentic"})
    for cell in agentic:
        args = cell["server_args"]
        assert "--enable_prefix_caching" in args, cell["result_filename"]
        assert args.index("--no-enable_prefix_caching") < args.index(
            "--enable_prefix_caching"
        ), f"negation must come first to be overridden: {cell['result_filename']}"
        for flag in ("--cudagraph-mode FULL", "--method mtp", "-tp 8"):
            assert flag in args, f"{flag} missing from {cell['result_filename']}"

    dpa = [c for c in agentic if "--enable-dp-attention" in c["server_args"]]
    plain = [c for c in agentic if "--enable-dp-attention" not in c["server_args"]]
    assert dpa and plain, "expected both a TP band and a DPA band"
    # Small concurrency on TP, large on DPA -- the bands must not overlap.
    assert max(c["conc"] for c in plain) < min(c["conc"] for c in dpa)


def test_param_lists_does_not_overwrite_an_agentic_workload():
    """A dispatch grid describes the random sweep only.

    `param_lists` is (isl, osl, conc, ratio) and workflow_dispatch always sends
    one, defaulted. Applying it to an agentic variant would replace the whole
    trace-replay workload -- its concurrency bands, which are the point of the
    run -- with a single 1024/1024 c=128 cell, and silently: the run would look
    like it worked.
    """
    grid = "1024,1024,128,0.8"
    agentic = catalog.build_cells(
        CATALOG, param_lists=grid, bench_kind_filter={"aiperf_agentic"}
    )
    assert agentic
    assert {c["isl"] for c in agentic} != {1024}
    assert {c["conc"] for c in agentic} != {128}
    # Identical to what it resolves to with no grid at all.
    assert agentic == catalog.build_cells(CATALOG, bench_kind_filter={"aiperf_agentic"})

    # The random variants still take the grid.
    rnd = catalog.build_cells(CATALOG, param_lists=grid, bench_kind_filter={"random"})
    assert {c["isl"] for c in rnd} == {1024}
    assert {c["conc"] for c in rnd} == {128}


def test_dp_agentic_variant_carries_the_session_routing_env():
    """DP-attention needs session affinity, and only the DPA variant.

    Without `ATOM_DP_SESSION_AFFINITY` a conversation's turns land on different
    DP ranks, so the prefix KV from the previous turn is resident on a rank the
    next turn does not reach -- an agentic trace is nothing but multi-turn
    sessions, so the whole workload degrades to cold prefill. The TP variant has
    no ranks to scatter across and must not carry these.
    """
    dp_only = {
        "ATOM_DP_SESSION_AFFINITY=1",
        "ATOM_DP_LB_REQ_EQUIV=512",
        "ATOM_ENABLE_PREFILL_DELAYER=1",
        "ATOM_PREFILL_DECODE_INTERVAL=10",
        # `--enable-tbo` is never just a flag in this catalog: every TBO variant
        # pairs it with these two. Asserted alongside so the flag cannot travel
        # without them.
        "GPU_MAX_HW_QUEUES=5",
        "ATOM_NUMA_BIND=1",
    }
    for cell in catalog.build_cells(CATALOG, bench_kind_filter={"aiperf_agentic"}):
        env = set(cell["env_vars"].splitlines())
        args = cell["server_args"]
        name = cell["result_filename"]
        is_dpa = "--enable-dp-attention" in args
        if is_dpa:
            assert dp_only <= env, f"{name} missing {sorted(dp_only - env)}"
            assert "--enable-tbo" in args, name
        else:
            assert not (dp_only & env), f"{name} carries {sorted(dp_only & env)}"
            assert "--enable-tbo" not in args, name
        # Pinned so throughput is not read through a fluctuating accept rate.
        assert "--spec-decode-acceptance-rate 0.4966666667" in args, name
