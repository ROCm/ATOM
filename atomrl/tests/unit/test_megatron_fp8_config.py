# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""B1: quant_config -> TransformerConfig FP8 fields.

The engine hard-coded ``bf16=True`` with no FP8 reference at all. These pin the
mapping, and above all pin that a BF16 run is unchanged -- the refactor must not
alter a single field for the configuration everything currently runs.
"""

import pytest
from atomrl.engine.training.megatron_native_engine import MegatronNativeEngine
from atomrl.quantization.fp8_config import FP8Config, megatron_fp8_kwargs

# --- BF16 must be untouched --------------------------------------------------


def test_bf16_produces_no_fp8_fields():
    """The load-bearing case: today's runs must not change at all."""
    assert megatron_fp8_kwargs(FP8Config(precision="bf16"), num_layers=48) == {}


def test_engine_helper_returns_empty_without_a_precision():
    """Absent, empty and whitespace all mean "not requested", not "default fp8"."""
    for ec in (
        {},
        {"fp8_precision": ""},
        {"fp8_precision": "   "},
        {"fp8_precision": None},
    ):
        assert MegatronNativeEngine._fp8_kwargs(ec, num_layers=48, pp=1) == {}


# --- format and recipe --------------------------------------------------------


def test_blockwise_is_the_default_recipe():
    """DeepSeek checkpoints ship block-128 fp8, so training in that scheme avoids
    a conversion nobody has specified."""
    kw = megatron_fp8_kwargs(FP8Config(precision="fp8"), num_layers=48)
    assert kw["fp8_recipe"] == "blockwise"
    assert kw["fp8"] == "e4m3"


@pytest.mark.parametrize(
    "precision,expected",
    [
        ("fp8", "e4m3"),
        ("e4m3", "e4m3"),
        ("float8", "e4m3"),
        ("hybrid_fp8", "hybrid"),
        ("e5m2", "hybrid"),
    ],
)
def test_precision_maps_to_a_te_format(precision, expected):
    assert (
        megatron_fp8_kwargs(FP8Config(precision=precision), num_layers=48)["fp8"]
        == expected
    )


def test_recipe_is_passed_through():
    kw = megatron_fp8_kwargs(
        FP8Config(precision="fp8", recipe="tensorwise"), num_layers=48
    )
    assert kw["fp8_recipe"] == "tensorwise"


def test_amax_fields_are_not_set():
    """Delayed-scaling concepts. Blockwise/tensorwise derive scales each step, and
    RL recalibrates every step anyway because the policy weights move."""
    kw = megatron_fp8_kwargs(FP8Config(precision="fp8"), num_layers=48)
    assert "fp8_amax_history_len" not in kw
    assert "fp8_amax_compute_algo" not in kw
    assert "fp8_param" not in kw  # storing fp8 masters is a separate decision


# --- keep-outs ----------------------------------------------------------------


def test_layer_keepouts_map_to_megatrons_fields():
    kw = megatron_fp8_kwargs(
        FP8Config(
            precision="fp8", num_first_layers_in_bf16=2, num_last_layers_in_bf16=1
        ),
        num_layers=48,
    )
    assert kw["first_last_layers_bf16"] is True
    assert kw["num_layers_at_start_in_bf16"] == 2
    assert kw["num_layers_at_end_in_bf16"] == 1


def test_no_keepout_fields_when_none_requested():
    """Setting first_last_layers_bf16=False explicitly is not the same as leaving
    it alone; leave Megatron's defaults in place."""
    kw = megatron_fp8_kwargs(FP8Config(precision="fp8"), num_layers=48)
    assert "first_last_layers_bf16" not in kw


def test_keepout_beyond_a_pipeline_stage_is_refused_here():
    """Caught by the mapper with the computed limit, rather than deep in Megatron's
    config validation where the message does not mention pp."""
    with pytest.raises(ValueError, match="exceeds layers per pipeline stage"):
        megatron_fp8_kwargs(
            FP8Config(precision="fp8", num_first_layers_in_bf16=20),
            num_layers=48,
            pp_size=4,
        )  # 12 layers per stage


def test_delayed_scaling_with_keepouts_is_refused():
    """Megatron rejects the combination, and `delayed` is its default recipe -- so
    a caller who sets keep-outs without choosing a recipe would otherwise fail
    late with a message that does not mention the recipe."""
    with pytest.raises(ValueError, match="delayed"):
        megatron_fp8_kwargs(
            FP8Config(precision="fp8", recipe="delayed", num_first_layers_in_bf16=1),
            num_layers=48,
        )


# --- the engine helper reads what the worker writes ---------------------------


def test_engine_helper_consumes_the_worker_keys():
    """Pins the contract between _build_engine_config and the engine. A rename on
    either side silently reverts FP8 to off, which no other test would catch."""
    ec = {
        "fp8_precision": "hybrid_fp8",
        "fp8_recipe": "blockwise",
        "fp8_num_first_layers_in_bf16": 1,
        "fp8_num_last_layers_in_bf16": 1,
    }
    kw = MegatronNativeEngine._fp8_kwargs(ec, num_layers=48, pp=1)
    assert kw["fp8"] == "hybrid"
    assert kw["fp8_recipe"] == "blockwise"
    assert kw["first_last_layers_bf16"] is True


def test_produced_kwargs_build_a_real_transformer_config():
    """Accepting our dict is Megatron's call, not ours -- assert it, do not assume."""
    torch = pytest.importorskip("torch")
    # Megatron refuses FP8 MoE configs without transformer-engine >= 1.7.
    pytest.importorskip("transformer_engine", exc_type=ImportError)
    import torch.nn.functional as F
    from megatron.core.transformer.transformer_config import TransformerConfig

    kw = megatron_fp8_kwargs(
        FP8Config(
            precision="fp8",
            recipe="blockwise",
            num_first_layers_in_bf16=1,
            num_last_layers_in_bf16=1,
        ),
        num_layers=48,
    )
    cfg = TransformerConfig(
        num_layers=48,
        hidden_size=2048,
        num_attention_heads=32,
        num_query_groups=4,
        ffn_hidden_size=6144,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        normalization="RMSNorm",
        bf16=True,
        params_dtype=torch.bfloat16,
        num_moe_experts=128,
        moe_ffn_hidden_size=768,
        moe_grouped_gemm=True,
        moe_router_topk=8,
        moe_token_dispatcher_type="alltoall",
        **kw,
    )
    assert cfg.fp8 == "e4m3" and cfg.fp8_recipe == "blockwise"
    assert cfg.first_last_layers_bf16 is True
