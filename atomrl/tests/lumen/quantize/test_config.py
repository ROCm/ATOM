# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for atomrl.lumen.quantize.config — QuantConfig, enums, and helpers.

Covers:
  - QuantFormat enum values
  - ScalingType enum values
  - AmaxAlgo enum values
  - QuantConfig defaults
  - QuantConfig.from_str with various aliases
  - QuantConfig properties: torch_dtype, fp8_max, is_quantized, recipe
  - get_fp8_max / get_fp8_max_bwd consistency
  - Edge cases: format=HYBRID bwd dtype differs from fwd
"""

import pytest
from atomrl.lumen.quantize.config import (
    AmaxAlgo,
    QuantConfig,
    QuantFormat,
    ScalingType,
    get_fp8_max,
    get_fp8_max_bwd,
)

# ===================================================================
# Enum coverage
# ===================================================================


class TestQuantFormat:
    def test_all_values(self):
        assert set(QuantFormat) == {
            QuantFormat.FP8_E4M3,
            QuantFormat.FP8_E5M2,
            QuantFormat.HYBRID,
            QuantFormat.MXFP8,
            QuantFormat.MXFP4,
            QuantFormat.FP4,
        }

    def test_string_values(self):
        assert QuantFormat.FP8_E4M3.value == "fp8_e4m3"
        assert QuantFormat.MXFP8.value == "mxfp8"


class TestScalingType:
    def test_all_values(self):
        assert set(ScalingType) == {
            ScalingType.DYNAMIC,
            ScalingType.DELAYED,
            ScalingType.BLOCKWISE,
            ScalingType.BLOCKWISE2D,
            ScalingType.PER_TOKEN,
            ScalingType.NONE,
        }


class TestAmaxAlgo:
    def test_all_values(self):
        assert set(AmaxAlgo) == {AmaxAlgo.MAX, AmaxAlgo.MOST_RECENT}


# ===================================================================
# QuantConfig defaults
# ===================================================================


class TestQuantConfigDefaults:
    def test_default_format(self):
        cfg = QuantConfig()
        assert cfg.format == QuantFormat.FP8_E4M3

    def test_default_scaling(self):
        cfg = QuantConfig()
        assert cfg.scaling == ScalingType.DELAYED

    def test_default_block_size(self):
        cfg = QuantConfig()
        assert cfg.block_size == 32

    def test_default_history_len(self):
        cfg = QuantConfig()
        assert cfg.history_len == 16

    def test_default_use_sdma(self):
        cfg = QuantConfig()
        assert cfg.use_sdma is False

    def test_default_quantize_activation(self):
        cfg = QuantConfig()
        assert cfg.quantize_activation is True

    def test_default_fp8_wgrad(self):
        cfg = QuantConfig()
        assert cfg.fp8_wgrad is True

    def test_default_quantize_grad(self):
        cfg = QuantConfig()
        assert cfg.quantize_grad is None


# ===================================================================
# QuantConfig.from_str
# ===================================================================


class TestQuantConfigFromStr:
    def test_basic(self):
        cfg = QuantConfig.from_str("fp8_e4m3", "delayed")
        assert cfg.format == QuantFormat.FP8_E4M3
        assert cfg.scaling == ScalingType.DELAYED

    def test_mxfp8_blockwise(self):
        cfg = QuantConfig.from_str("mxfp8", "blockwise")
        assert cfg.format == QuantFormat.MXFP8
        assert cfg.scaling == ScalingType.BLOCKWISE

    def test_blockwise2d(self):
        cfg = QuantConfig.from_str("fp8_e4m3", "blockwise2d")
        assert cfg.format == QuantFormat.FP8_E4M3
        assert cfg.scaling == ScalingType.BLOCKWISE2D

    def test_scaling_alias_current(self):
        cfg = QuantConfig.from_str("fp8_e4m3", "current")
        assert cfg.scaling == ScalingType.DYNAMIC

    def test_scaling_alias_no_quant(self):
        cfg = QuantConfig.from_str("fp8_e4m3", "no_quant")
        assert cfg.scaling == ScalingType.NONE

    def test_amax_algo_string(self):
        cfg = QuantConfig.from_str("fp8_e4m3", "delayed", amax_algo="most_recent")
        assert cfg.amax_algo == AmaxAlgo.MOST_RECENT

    def test_extra_kwargs(self):
        cfg = QuantConfig.from_str("fp8_e4m3", "delayed", margin=2, history_len=8)
        assert cfg.margin == 2
        assert cfg.history_len == 8


# ===================================================================
# Properties
# ===================================================================


class TestQuantConfigProperties:
    def test_torch_dtype_not_none_for_fp8(self):
        cfg = QuantConfig(format=QuantFormat.FP8_E4M3)
        assert cfg.torch_dtype is not None

    def test_torch_dtype_bwd_hybrid_differs(self):
        cfg = QuantConfig(format=QuantFormat.HYBRID)
        assert cfg.torch_dtype != cfg.torch_dtype_bwd

    def test_fp8_max_positive(self):
        cfg = QuantConfig(format=QuantFormat.FP8_E4M3)
        assert cfg.fp8_max > 0

    def test_is_quantized_true(self):
        cfg = QuantConfig()
        assert cfg.is_quantized is True

    def test_is_quantized_false(self):
        cfg = QuantConfig(scaling=ScalingType.NONE)
        assert cfg.is_quantized is False

    def test_recipe_delayed(self):
        cfg = QuantConfig(scaling=ScalingType.DELAYED)
        assert cfg.recipe == "delayed"

    def test_recipe_none(self):
        cfg = QuantConfig(scaling=ScalingType.NONE)
        assert cfg.recipe == "none"

    def test_recipe_mxfp8(self):
        cfg = QuantConfig(format=QuantFormat.MXFP8, scaling=ScalingType.BLOCKWISE)
        assert cfg.recipe == "mxfp8"


# ===================================================================
# MXFP4 block size
# ===================================================================


class TestMXFP4BlockSize:
    """MXFP4's block size is fixed by the format, not a config knob.

    A config carrying the FP8 default of 128 computed scales over 128 elements
    while every consumer read them as 32-element blocks -- not a precision
    trade-off but wrong arithmetic, and nothing checked.
    """

    def test_mxfp4_defaults_to_32(self):
        assert QuantConfig(format=QuantFormat.MXFP4).block_size == 32

    def test_mxfp4_refuses_another_block_size(self):
        with pytest.raises(ValueError, match="32-element blocks"):
            QuantConfig(format=QuantFormat.MXFP4, block_size=128)

    def test_from_str_mxfp4_refuses_another_block_size(self):
        with pytest.raises(ValueError, match="32-element blocks"):
            QuantConfig.from_str("mxfp4", "blockwise", block_size=128)

    def test_fp8_still_takes_any_block_size(self):
        assert (
            QuantConfig(format=QuantFormat.FP8_E4M3, block_size=128).block_size == 128
        )

    def test_lumen_config_uses_format_default_but_refuses_explicit_mismatch(self):
        from atomrl.lumen.config import LumenConfig

        assert (
            LumenConfig(format="mxfp4", scaling="blockwise").quant_config.block_size
            == 32
        )
        assert (
            LumenConfig(format="fp8_e4m3", scaling="blockwise").quant_config.block_size
            == 128
        )
        with pytest.raises(ValueError, match="32-element blocks"):
            _ = LumenConfig(
                format="mxfp4",
                scaling="blockwise",
                block_size=128,
            ).quant_config

    def test_fp8_param_manager_is_refused_before_mxfp4_patching(self):
        from atomrl.lumen.config import LumenConfig
        from torch import nn

        cfg = LumenConfig(
            format="mxfp4",
            scaling="blockwise",
            fp8_param_manager=True,
        )
        with pytest.raises(ValueError, match="fp8_param_manager cannot be combined"):
            cfg.enable(nn.Linear(32, 32))


# ===================================================================
# get_fp8_max helpers
# ===================================================================


def test_fp8_max_e4m3_positive():
    assert get_fp8_max(QuantFormat.FP8_E4M3) > 0


def test_fp8_max_e5m2_positive():
    assert get_fp8_max(QuantFormat.FP8_E5M2) > 0


def test_fp8_max_bwd_hybrid_uses_e5m2():
    fwd = get_fp8_max(QuantFormat.HYBRID)
    bwd = get_fp8_max_bwd(QuantFormat.HYBRID)
    assert bwd == get_fp8_max(QuantFormat.FP8_E5M2)
    assert fwd == get_fp8_max(QuantFormat.FP8_E4M3)
