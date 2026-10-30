# SPDX-License-Identifier: MIT
# The online-quant argument of the M3 MoE smoke model. With moe=True and the experts online-quantized
# to ptpc_fp8, gfx950 warmup died in ck_moe_stage2 ("device_gemm ... does not support this GEMM
# problem"): per-token FP8 experts with Swiglu select aiter's generic 2-stage CK path, which no M3
# recipe uses. ATOM's own M3 recipe excludes the whole ``*block_sparse_moe`` from online quant (the
# real checkpoints carry MXFP4/MXFP8 experts), and the production expert path is MXFP4.
#
# These tests are torch-free: ATOM's real ``_matches_exclude`` is extracted from atom/config.py with
# ast, so what each layer name does under the generated config is decided by ATOM's own matcher.
# Whether the chosen combination RUNS on gfx950 is not tested here (no GPU).
import ast
import fnmatch
import json
import re
from pathlib import Path

import pytest

from atom.model_ops.gemm_fusion import test_utils as TU

CONFIG_PY = Path(__file__).resolve().parents[2] / "atom/config.py"


def _real_matches_exclude():
    tree = ast.parse(CONFIG_PY.read_text(encoding="utf-8"))
    fn = next(
        f
        for c in tree.body
        if isinstance(c, ast.ClassDef)
        for f in c.body
        if isinstance(f, ast.FunctionDef) and f.name == "_matches_exclude"
    )
    fn.decorator_list = []  # a staticmethod in the class; a plain function here
    module = ast.Module(
        [ast.parse("from __future__ import annotations").body[0], fn], []
    )
    ns = {"re": re, "fnmatch": fnmatch}
    exec(  # noqa: S102 - ATOM's real matcher, extracted by ast
        compile(ast.fix_missing_locations(module), str(CONFIG_PY), "exec"), ns
    )
    return ns["_matches_exclude"]


def _excluded(name, cfg):
    match = _real_matches_exclude()
    return any(match(name, pat) for pat in cfg["exclude_layer"])


def _layer_pattern(name, cfg):
    """The ``layer_quant_config`` entry ATOM would apply (config.py: a glob is fnmatch-ed)."""
    for pattern, scheme in cfg.get("layer_quant_config", {}).items():
        if fnmatch.fnmatch(name, pattern):
            return scheme
    return None


MOE = "model.layers.1.block_sparse_moe"
FUSION_LINEARS = [  # what the fusion path consumes: must keep the global ptpc_fp8
    "model.layers.1.self_attn.qkv_proj",
    "model.layers.1.self_attn.o_proj",
    "model.layers.0.mlp.gate_up_proj",
    "model.layers.0.mlp.down_proj",
]


@pytest.mark.parametrize("variant", ["mxfp4", "bf16"])
def test_both_variants_keep_ptpc_fp8_for_the_linears_the_fusion_path_consumes(variant):
    cfg = json.loads(TU.m3_online_quant_config(variant))
    assert cfg["global_quant_config"] == "ptpc_fp8"
    for name in FUSION_LINEARS:
        assert not _excluded(name, cfg), name
        assert (
            _layer_pattern(name, cfg) is None
        ), name  # no override: the global applies
    for name in ("lm_head", "model.embed_tokens"):
        assert _excluded(name, cfg), name


def test_the_bf16_variant_is_the_whole_block_exclusion_of_atoms_own_m3_recipe():
    cfg = json.loads(TU.m3_online_quant_config("bf16"))
    assert "*block_sparse_moe" in cfg["exclude_layer"]  # recipes/MiniMax-M3.md
    assert "layer_quant_config" not in cfg
    for name in (
        f"{MOE}.experts",
        f"{MOE}.shared_experts.gate_up_proj",
        f"{MOE}.shared_experts.down_proj",
        f"{MOE}.gate",
        f"language_model.{MOE}.experts",  # the vl layout's prefix
    ):
        assert _excluded(name, cfg), name


def test_the_mxfp4_variant_quantizes_only_the_experts_and_leaves_the_rest_alone():
    cfg = json.loads(TU.m3_online_quant_config("mxfp4"))
    assert cfg["layer_quant_config"] == {"*block_sparse_moe.experts*": "mxfp4"}
    for name in (f"{MOE}.experts", f"language_model.{MOE}.experts"):
        assert not _excluded(
            name, cfg
        ), name  # the exclusion must not shadow the override
        assert _layer_pattern(name, cfg) == "mxfp4", name
    for name in (
        f"{MOE}.shared_experts.gate_up_proj",
        f"{MOE}.shared_experts.down_proj",
    ):
        assert _excluded(
            name, cfg
        ), name  # shared experts stay unquantized (MiniMaxM3MLP)
    assert "*block_sparse_moe" not in cfg["exclude_layer"]


@pytest.mark.parametrize("variant", ["ptpc_fp8", "fp8", "", None])
def test_ptpc_fp8_experts_are_not_offered(variant):
    """The combination that fails in ck_moe_stage2 on gfx950 cannot be requested by accident."""
    with pytest.raises(ValueError, match="ck_moe_stage2"):
        TU.m3_online_quant_config(variant)


def test_serve_args_adds_the_flag_only_when_asked_and_as_one_argv_element():
    base = TU.serve_args("D")
    assert "--online_quant_config" not in base  # existing evidence argv is unchanged
    argv = TU.serve_args("D", online_quant="mxfp4")
    i = argv.index("--online_quant_config")
    assert json.loads(argv[i + 1]) == json.loads(TU.m3_online_quant_config("mxfp4"))
    assert argv[:i] == base  # nothing else moved
    with pytest.raises(ValueError):
        TU.serve_args("D", online_quant="ptpc_fp8")
