# SPDX-License-Identifier: MIT
"""Tests for .github/scripts/regression_rerun.py (regression report -> rerun matrix).

The regression report's "model" is the catalog display name of the variant
(benchmark-tmpl.yml stores it as benchmark_model_name), e.g.
"DeepSeek-V4-Pro DPA DSpark", and "model_id" is the served checkpoint dir
(e.g. "/models/deepseek-ai/DeepSeek-V4-Pro-DSpark"). The fixture entries below
use catalog identities and workload shapes from the historical report;
severity values are synthetic test data.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / ".github" / "scripts"
CATALOG = str(REPO / ".github" / "benchmark" / "models.json")

sys.path.insert(0, str(SCRIPTS))

import catalog
import regression_rerun

V4_SERVER = {"tp": 8, "kv_cache_dtype": "fp8"}
EPLB_CFG = "--eplb-config '{\"num_redundant_experts\": 0}'"

# Shapes copied from .github/benchmark/models.json: labels with spaces, a
# variant-level "path" override (DSpark draft checkpoint), and two variants
# that share a display name and suffix and differ only by concurrency band.
MODELS = {
    "default_scenarios": [],
    "models": [
        {
            "display": "DeepSeek-V4-Pro",
            "path": "deepseek-ai/DeepSeek-V4-Pro",
            "prefix": "deepseek-v4-pro",
            "runner": "atom-mi355-8gpu.predownload",
            "config": V4_SERVER,
            "variants": [
                {"label": "", "suffix": "", "conc_max": 256},
                {
                    "label": "DPA",
                    "suffix": "-dpa",
                    "extra_args": "--enable-dp-attention",
                    "conc_min": 64,
                    "conc_max": 2048,
                },
                {
                    "label": "DPA TBO",
                    "suffix": "-dpa-tbo",
                    "extra_args": "--enable-dp-attention --enable-tbo",
                    "conc_min": 256,
                    "conc_max": 2048,
                },
                {
                    "label": "DPA MTP3",
                    "suffix": "-dpa-mtp3",
                    "extra_args": "--method mtp --num-speculative-tokens 3 "
                    "--enable-dp-attention",
                    "bench_args": "--use-chat-template",
                    "conc_min": 64,
                    "conc_max": 1024,
                },
                {
                    "label": "DPA DSpark",
                    "suffix": "-dpa-dspark",
                    "path": "deepseek-ai/DeepSeek-V4-Pro-DSpark",
                    "extra_args": "--method dspark --enable-dp-attention",
                    "bench_args": "--use-chat-template",
                    "conc_min": 64,
                    "conc_max": 512,
                },
            ],
        },
        {
            "display": "DeepSeek-V4-Pro EPLB r0 MegaMoE",
            "path": "deepseek-ai/DeepSeek-V4-Pro",
            "prefix": "deepseek-v4-pro-eplb",
            "runner": "atom-mi355-8gpu.predownload",
            "config": V4_SERVER,
            "variants": [
                {
                    "label": "MegaMoE",
                    "suffix": "-mega",
                    "extra_args": EPLB_CFG,
                    "conc_min": 512,
                    "conc_max": 512,
                },
                {
                    "label": "MegaMoE",
                    "suffix": "-mega",
                    "extra_args": EPLB_CFG + " --gpu-memory-utilization 0.85",
                    "conc_min": 4096,
                    "conc_max": 4096,
                },
            ],
        },
        {
            "display": "DeepSeek-R1-0528-MXFP4",
            "path": "amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4",
            "prefix": "deepseek-r1-0528-mxfp4",
            "runner": "atom-mi355-8gpu.predownload",
            "config": {"tp": 4, "kv_cache_dtype": "fp8"},
            "variants": [
                {"label": "", "suffix": ""},
                {
                    "label": "MTP3",
                    "suffix": "-mtp3",
                    "extra_args": "--method mtp --num-speculative-tokens 3",
                    "conc_min": 1,
                },
            ],
        },
        {
            "display": "MiniMax-M3-MXFP4",
            "path": "amd/MiniMax-M3-MXFP4",
            "prefix": "m3-mxfp4",
            "runner": "atom-mi355-8gpu.predownload",
            "config": {"tp": 4, "kv_cache_dtype": "fp8"},
        },
    ],
}


def _reg(model, model_id, conc, tput_pct, tpot_pct, isl=8192, osl=1024):
    return {
        "backend": "ATOM",
        "model": model,
        "model_id": model_id,
        "isl": isl,
        "osl": osl,
        "conc": conc,
        "metrics": {
            "output_throughput": {"pct": tput_pct},
            "mean_tpot_ms": {"pct": tpot_pct},
        },
    }


# Catalog (HF) ids and the local dirs CI reports as model_id.
V4_ID = "deepseek-ai/DeepSeek-V4-Pro"
V4_DSPARK_ID = "deepseek-ai/DeepSeek-V4-Pro-DSpark"
R1_ID = "amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4"
M3_ID = "amd/MiniMax-M3-MXFP4"
V4 = "/models/" + V4_ID
V4_DSPARK = "/models/" + V4_DSPARK_ID
R1 = "/models/" + R1_ID
EPLB = "DeepSeek-V4-Pro EPLB r0 MegaMoE MegaMoE"

# Catalog identities and workload shapes model the historical report.
# Synthetic severity values put DPA DSpark / DPA MTP3 first, exercising
# variants the old matcher could not map.
REGRESSIONS_9A0A5202 = [
    _reg("DeepSeek-R1-0528-MXFP4 MTP3", R1, 4, -1.0, 1.0),
    _reg("DeepSeek-V4-Pro DPA", V4, 128, -6.0, 6.0),
    _reg("DeepSeek-V4-Pro DPA DSpark", V4_DSPARK, 64, -9.0, 9.0),
    _reg("DeepSeek-V4-Pro DPA DSpark", V4_DSPARK, 128, -7.0, 7.0),
    _reg("DeepSeek-V4-Pro DPA DSpark", V4_DSPARK, 256, -5.0, 5.0),
    _reg("DeepSeek-V4-Pro DPA MTP3", V4, 64, -8.0, 8.0),
    _reg("DeepSeek-V4-Pro DPA TBO", V4, 1024, -2.0, 2.0),
    _reg(EPLB, V4, 512, -3.0, 3.0),
    _reg("MiniMax-M3-MXFP4", "/models/" + M3_ID, 1, -4.0, 4.0),
]


@pytest.fixture
def models_path(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(MODELS), encoding="utf-8")
    return str(path)


def _report(tmp_path, regressions):
    path = tmp_path / "regression_report.json"
    path.write_text(json.dumps({"regressions": regressions}), encoding="utf-8")
    return str(path)


def _record(models_path, suffix, path, conc):
    for v in catalog.load_variants(models_path):
        if (
            v["suffix"] == suffix
            and v["path"] == path
            and v["conc_min"] <= conc <= v["conc_max"]
        ):
            return v
    raise AssertionError(f"no catalog record for {suffix} {path} {conc}")


@pytest.mark.parametrize(
    "entry, suffix, path",
    [
        (REGRESSIONS_9A0A5202[2], "-dpa-dspark", V4_DSPARK_ID),
        (REGRESSIONS_9A0A5202[5], "-dpa-mtp3", V4_ID),
        (REGRESSIONS_9A0A5202[6], "-dpa-tbo", V4_ID),
        (REGRESSIONS_9A0A5202[1], "-dpa", V4_ID),
        (REGRESSIONS_9A0A5202[0], "-mtp3", R1_ID),
        (REGRESSIONS_9A0A5202[8], "", M3_ID),
    ],
)
def test_report_entries_resolve_to_their_variant(models_path, entry, suffix, path):
    models = catalog.load_variants(models_path)
    found = regression_rerun._find_model_config(entry, models)
    assert found is not None, entry["model"]
    assert found == _record(models_path, suffix, path, entry["conc"])


def test_shared_display_name_resolves_by_concurrency_band(models_path):
    models = catalog.load_variants(models_path)
    for conc, extra in ((512, ""), (4096, " --gpu-memory-utilization 0.85")):
        entry = _reg(EPLB, V4, conc, -9.0, 9.0)
        found = regression_rerun._find_model_config(entry, models)
        assert found is not None
        assert found["args"].endswith(EPLB_CFG + extra)
    # Outside both bands, or without a concurrency, there is nothing to pick
    # between two different recipes.
    assert (
        regression_rerun._find_model_config(_reg(EPLB, V4, 64, -9, 9), models) is None
    )
    no_conc = {"model": EPLB, "model_id": V4}
    assert regression_rerun._find_model_config(no_conc, models) is None


def test_9a0a5202_report_yields_rerun_cells(tmp_path, models_path):
    # The nightly ran this with the default --top-n 3 and wrote 0 matrix cells.
    report = _report(tmp_path, REGRESSIONS_9A0A5202)
    matrix = regression_rerun.generate_matrix(report, models_path, top_n=3)

    cells = {c["prefix"]: c for c in matrix}
    assert set(cells) == {"deepseek-v4-pro-dpa-dspark", "deepseek-v4-pro-dpa-mtp3"}

    dspark = cells["deepseek-v4-pro-dpa-dspark"]
    assert dspark["model_path"] == V4_DSPARK_ID
    assert "--method dspark" in dspark["server_args"]
    assert [c["conc"] for c in json.loads(dspark["configs"])] == [64, 128]

    mtp = cells["deepseek-v4-pro-dpa-mtp3"]
    assert mtp["model_path"] == V4_ID
    assert "--method mtp" in mtp["server_args"]
    assert [c["conc"] for c in json.loads(mtp["configs"])] == [64]
    assert json.loads(mtp["configs"])[0]["bench_args"] == "--use-chat-template"


def test_unusable_regressions_do_not_use_up_top_n(tmp_path, models_path, capsys):
    unknown = _reg("Not-In-Catalog DPA", "/models/x/Not-In-Catalog", 64, -50.0, 60.0)
    no_isl = {**REGRESSIONS_9A0A5202[1], "isl": None}
    no_isl["metrics"] = {"output_throughput": {"pct": -49.0}}
    report = _report(tmp_path, [unknown, no_isl] + REGRESSIONS_9A0A5202)

    configs = regression_rerun.generate_configs(report, models_path, top_n=1)

    assert [(c["model_path"], c["conc"]) for c in configs] == [(V4_DSPARK_ID, 64)]
    err = capsys.readouterr().err
    assert "No model config found for Not-In-Catalog DPA" in err
    assert "missing isl/osl/conc" in err


@pytest.mark.parametrize(
    "display, model_id",
    [
        # A label the catalog no longer has must not fall back to the base
        # variant's server args.
        ("DeepSeek-V4-Pro DPA Renamed", V4),
        ("DeepSeek-V4-Pro DPA Renamed", ""),
        ("DeepSeek-V4-Pro-unknown", V4),
        # Display and checkpoint must agree.
        ("DeepSeek-V4-Pro DPA DSpark", V4),
        ("DeepSeek-V4-Pro", V4_DSPARK),
        ("DeepSeek-V4-Pro", "/models/other-org/DeepSeek-V4-Pro"),
    ],
)
def test_unknown_or_mismatched_identity_is_not_mapped(models_path, display, model_id):
    models = catalog.load_variants(models_path)
    entry = _reg(display, model_id, 128, -10.0, 10.0)
    assert regression_rerun._find_model_config(entry, models) is None


@pytest.mark.parametrize("model_id", [V4_ID, V4, V4 + "/"])
def test_hf_and_mounted_model_ids_match(models_path, model_id):
    models = catalog.load_variants(models_path)
    entry = _reg("DeepSeek-V4-Pro DPA", model_id, 128, -10.0, 10.0)
    found = regression_rerun._find_model_config(entry, models)
    assert found is not None
    assert (found["path"], found["suffix"]) == (V4_ID, "-dpa")


@pytest.mark.parametrize(
    "display, model_id, suffix",
    [
        ("DeepSeek-R1-0528-MXFP4-MTP-MoEFP4-mtp3", R1_ID, "-mtp3"),
        ("DeepSeek-R1-0528-MXFP4-MTP-MoEFP4-MTP3", R1, "-mtp3"),
        ("DeepSeek-R1-0528-MXFP4-MTP-MoEFP4-mtp3", "", "-mtp3"),
        ("DeepSeek-R1-0528-MXFP4-MTP-MoEFP4", "", ""),
    ],
)
def test_legacy_display_names_still_resolve(models_path, display, model_id, suffix):
    # Reports from results without benchmark_model_name use
    # "<path basename>[-mtp<N>]" (summarize._display_model).
    models = catalog.load_variants(models_path)
    entry = _reg(display, model_id, 4, -10.0, 10.0)
    found = regression_rerun._find_model_config(entry, models)
    assert found is not None
    assert found["suffix"] == suffix
    assert found["path"] == R1_ID


def test_cells_sharing_prefix_and_suffix_get_distinct_prefixes(tmp_path, models_path):
    regs = [_reg(EPLB, V4, 512, -9.0, 9.0), _reg(EPLB, V4, 4096, -9.0, 9.0)]
    matrix = regression_rerun.generate_matrix(_report(tmp_path, regs), models_path)
    assert [c["prefix"] for c in matrix] == [
        "deepseek-v4-pro-eplb-mega",
        "deepseek-v4-pro-eplb-mega-2",
    ]


def test_cell_prefixes_stay_unique_next_to_a_numbered_prefix(tmp_path):
    # Two bands of one variant share "a-mega"; another model's own prefix is
    # already "a-mega-2". The cell prefix names the uploaded artifacts, so no
    # two cells may share it, and a prefix that is already unique is kept.
    models = {
        "default_scenarios": [],
        "models": [
            {
                "display": "A",
                "path": "org/A",
                "prefix": "a",
                "runner": "cpu",
                "variants": [
                    {"label": "Mega", "suffix": "-mega", "conc_min": 1, "conc_max": 1},
                    {
                        "label": "Mega",
                        "suffix": "-mega",
                        "extra_args": "--band-2",
                        "conc_min": 2,
                        "conc_max": 2,
                    },
                ],
            },
            {"display": "B", "path": "org/B", "prefix": "a-mega-2", "runner": "cpu"},
        ],
    }
    models_path = tmp_path / "models.json"
    models_path.write_text(json.dumps(models), encoding="utf-8")
    regs = [
        _reg("A Mega", "/models/org/A", 1, -9.0, 9.0),
        _reg("A Mega", "/models/org/A", 2, -9.0, 9.0),
        _reg("B", "/models/org/B", 1, -9.0, 9.0),
    ]
    report = _report(tmp_path, regs)
    matrix = regression_rerun.generate_matrix(report, str(models_path))
    prefixes = {
        (c["model_path"], json.loads(c["configs"])[0]["conc"]): c["prefix"]
        for c in matrix
    }
    assert len(matrix) == 3
    assert len(set(prefixes.values())) == 3
    assert prefixes[("org/A", 1)] == "a-mega"
    assert prefixes[("org/B", 1)] == "a-mega-2"


def test_every_real_catalog_variant_resolves_to_itself():
    # Content-agnostic over the real catalog: a regression on any variant, as
    # summarize.py reports it (catalog display name, local model dir), maps
    # back to that same variant record.
    models = catalog.load_variants(CATALOG)
    for v in models:
        entry = _reg(v["display"], "/models/" + v["path"], v["conc_max"], -10.0, 10.0)
        found = regression_rerun._find_model_config(entry, models)
        assert found is not None, v["display"]
        assert (found["path"], found["suffix"], found["args"]) == (
            v["path"],
            v["suffix"],
            v["args"],
        ), v["display"]
