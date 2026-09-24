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
    cells = catalog.build_cells(
        CATALOG, param_lists="1024,1024,512,0.7", model_filter={"deepseek-v4-pro"}
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


def _workflow():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text())


def test_cadence_splits_the_nightly_without_losing_cells():
    """The two crons must partition the catalog: every cell runs on exactly one.

    The point of the split is a shorter nightly, so a cell silently belonging to
    neither cadence would stop being benchmarked at all and nothing would say so.
    """
    everything = catalog.build_cells(CATALOG)
    nightly = catalog.build_cells(CATALOG, cadence="nightly")
    weekly = catalog.build_cells(CATALOG, cadence="weekly")

    assert nightly and weekly, "both cadences must produce cells"
    assert len(nightly) + len(weekly) == len(everything)

    def keys(cells):
        return {
            (c["prefix"], c["suffix"], c["isl"], c["osl"], c["conc"]) for c in cells
        }

    assert keys(nightly) | keys(weekly) == keys(everything)
    assert not (keys(nightly) & keys(weekly)), "a cell runs on one cadence only"


def test_nightly_dropped_1k1k_and_weekly_is_exactly_that():
    """The split as configured: 1k/1k moved off the nightly, nothing else did."""
    nightly = {
        (c["isl"], c["osl"]) for c in catalog.build_cells(CATALOG, cadence="nightly")
    }
    weekly = {
        (c["isl"], c["osl"]) for c in catalog.build_cells(CATALOG, cadence="weekly")
    }

    assert (1024, 1024) not in nightly
    assert weekly == {(1024, 1024)}


def test_untagged_scenarios_stay_nightly():
    """Adding a scenario must not need a `cadence` field to keep working.

    A model's or variant's own `scenarios` override the defaults, and none of
    them carries a tag today -- they have to land on the nightly, not vanish.
    """
    cat = catalog._load_catalog(CATALOG)
    tagged = {
        sc.get("cadence", catalog.DEFAULT_CADENCE) for sc in cat["default_scenarios"]
    }
    assert catalog.DEFAULT_CADENCE in tagged, "the default must remain reachable"

    overrides = [
        m["prefix"]
        for m in cat["models"]
        if m.get("scenarios") or any(v.get("scenarios") for v in m.get("variants", []))
    ]
    nightly = {c["prefix"] for c in catalog.build_cells(CATALOG, cadence="nightly")}
    for prefix in overrides:
        assert prefix in nightly, f"{prefix} overrides scenarios but runs on no cron"


def test_every_cron_produces_cells():
    """Each cadence the workflow can ask for must resolve to a real grid.

    `build_benchmark_matrix` fails the run on an empty schedule matrix, so a
    typo'd tag would take the whole nightly down -- catch it here instead.
    """
    on = _workflow().get("on", _workflow().get(True))
    crons = [c["cron"] for c in on["schedule"]]
    assert len(crons) == 2, "expected a nightly and a weekly cron"

    for cadence in ("nightly", "weekly"):
        assert catalog.build_cells(CATALOG, cadence=cadence), cadence


def test_weekly_cron_matches_the_cadence_expressions():
    """The weekly cron string is repeated in `run-name` and `env.CADENCE`.

    If they drift, a weekly run is titled `nightly`, and the baseline lookup --
    which matches on that title -- hands the next nightly a 1k/1k-only run to
    compare against. Every cell then reports no baseline, and nothing errors.
    """
    wf = _workflow()
    on = wf.get("on", wf.get(True))
    crons = [c["cron"] for c in on["schedule"]]
    weekly_crons = [c for c in crons if c.strip().endswith("0")]
    assert len(weekly_crons) == 1, f"expected one weekly cron, got {weekly_crons}"
    weekly = weekly_crons[0]

    for field, text in (
        ("run-name", wf["run-name"]),
        ("env.CADENCE", wf["env"]["CADENCE"]),
    ):
        assert weekly in text, f"{field} does not reference the weekly cron {weekly!r}"
        assert "weekly" in text and "nightly" in text, field


def test_agentic_dispatch_overrides_keep_the_server_recipe():
    import json

    from build_agentic_benchmark_matrix import build_configs

    defaults = build_configs()
    selected = build_configs(
        inputs={
            "models": "deepseek-v41-flash",
            "concurrency": "4,8",
            "duration_seconds": 1200,
        }
    )
    assert len(defaults) == len(selected) == 1
    assert selected[0]["server_args"] == defaults[0]["server_args"]
    assert selected[0]["bench_kind"] == "aiperf_agentic"
    assert json.loads(selected[0]["concurrency"]) == [4, 8]
    env = dict(line.split("=", 1) for line in selected[0]["env_vars"].splitlines())
    assert env["AIPERF_BENCHMARK_DURATION"] == "1200"
    assert env["ATOM_BUNDLE_REQUIRE_FULL"] == "1"


@pytest.mark.parametrize(
    "inputs",
    [
        {"models": "unknown-model"},
        {"concurrency": "0,4"},
        {"concurrency": "2,2"},
        {"concurrency": "2,,4"},
        {"concurrency": "9999"},
        {"duration_seconds": 899},
        {"duration_seconds": 3601},
        {"duration_seconds": 900.5},
        {"duration_seconds": True},
    ],
)
def test_agentic_bad_inputs_fail_before_allocating_gpu_jobs(inputs):
    from build_agentic_benchmark_matrix import build_configs

    with pytest.raises(ValueError):
        build_configs(inputs=inputs)


def test_agentic_schedule_uses_catalog_defaults(tmp_path, monkeypatch):
    import json

    from build_agentic_benchmark_matrix import build_configs, main

    output = tmp_path / "github-output"
    monkeypatch.setenv("EVENT_NAME", "schedule")
    monkeypatch.setenv("INPUTS_JSON", '{"models":"unknown","duration_seconds":1}')
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert main() == 0
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["has_cells"] == "true"
    assert json.loads(values["configs_json"]) == build_configs(
        inputs={"profile": "nightly"}
    )


def test_agentic_gpu_workflow_only_runs_manually_or_on_schedule():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (REPO / ".github/workflows/atom-agentic-benchmark.yaml").read_text()
    )
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"workflow_dispatch", "schedule"}
    assert workflow["jobs"]["benchmark"]["uses"] == (
        "./.github/workflows/benchmark-tmpl.yml"
    )


def test_agentic_nightly_matches_inferencex_3387_recipe():
    import json
    import shlex

    from build_agentic_benchmark_matrix import build_configs

    points = set()
    for config in build_configs(inputs={"profile": "nightly"}):
        args = shlex.split(config["server_args"])
        tp = int(args[args.index("-tp") + 1])
        captures = json.loads(args[args.index("--cudagraph-capture-sizes") + 1])
        env = dict(line.split("=", 1) for line in config["env_vars"].splitlines())
        assert config["image"] == "rocm/atom-dev:latest"
        assert env["HIP_VISIBLE_DEVICES"] == ",".join(str(i) for i in range(tp))
        assert env["AIPERF_BENCHMARK_DURATION"] == "3600"
        assert env["AIPERF_WARMUP_REQUESTS_PER_LANE"] == "5"
        assert env["ATOM_DSV41_BENCHMARK_SYNTHETIC"] == "1"
        assert "AIPERF_MAX_CONTEXT_LENGTH" not in env
        for flag, value in {
            "--spec-decode-acceptance-length": "3.51",
            "--num-speculative-tokens": "5",
            "--max-num-seqs": "128",
            "--max-num-batched-tokens": "16384",
            "--attn-prefill-chunk-size": "16384",
            "--state-checkpoint-interval-tokens": "8192",
            "--cudagraph-mode": "FULL",
            "--level": "3",
            "--tool-call-parser": "dsml_v41",
        }.items():
            assert args[args.index(flag) + 1] == value
        assert "--enforce-eager" not in args
        for conc in json.loads(config["concurrency"]):
            assert (tp, conc) not in points
            points.add((tp, conc))
            assert captures == (
                list(range(1, 33)) + [48, 64, 128]
                if conc == 32
                else [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 48, 64, 128]
            )
    assert points == {(2, c) for c in [1, 2, 8, 16, 32, 64]} | {
        (4, c) for c in [2, 8, 16, 32, 64]
    }


def test_agentic_nightly_subset_keeps_capture_recipe_without_duplicate_points():
    import json

    from build_agentic_benchmark_matrix import build_configs

    configs = build_configs(inputs={"profile": "nightly", "concurrency": "32"})
    assert len(configs) == 2
    assert all(json.loads(config["concurrency"]) == [32] for config in configs)
    with pytest.raises(ValueError, match="subset"):
        build_configs(inputs={"profile": "nightly", "concurrency": "4"})


def test_agentic_manual_default_stays_a_two_point_test():
    import json

    from build_agentic_benchmark_matrix import build_configs

    (config,) = build_configs()
    assert json.loads(config["concurrency"]) == [2, 4]
    assert "-tp 4" in config["server_args"]
    assert "--cudagraph-mode FULL" in config["server_args"]
    assert "AIPERF_BENCHMARK_DURATION=900" in config["env_vars"]
    assert config["image"] == "rocm/atom-dev:latest"


@pytest.mark.parametrize("profile", ["test", "nightly"])
def test_agentic_dispatch_replay_preserves_profile(tmp_path, monkeypatch, profile):
    import json

    from build_agentic_benchmark_matrix import build_configs, main

    output = tmp_path / "github-output"
    config_dir = tmp_path / "config"
    monkeypatch.setenv("EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("INPUTS_JSON", json.dumps({"profile": profile, "dry_run": True}))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv(
        "GITHUB_WORKFLOW_REF",
        "ROCm/ATOM/.github/workflows/atom-agentic-benchmark.yaml@refs/heads/test",
    )
    monkeypatch.setenv("AGENTIC_RUN_CONFIG_DIR", str(config_dir))
    assert main() == 0
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert json.loads(values["configs_json"]) == build_configs(
        inputs={"profile": profile}
    )
    dispatch = json.loads((config_dir / "dispatch-inputs.json").read_text())
    assert dispatch["profile"] == profile
    assert dispatch["dry_run"] is True
    assert dispatch["atom_commit"]
    assert "atom-agentic-benchmark.yaml" in (config_dir / "README.md").read_text()
