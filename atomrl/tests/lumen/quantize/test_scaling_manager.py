# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for atomrl.lumen.quantize.scaling_manager — ScalingManager lifecycle.

Covers:
  - Construction from QuantConfig and legacy kwargs
  - get_scale returns None for blockwise/mxfp8/per_token recipes
  - get_scale returns a tensor for delayed/dynamic recipes
  - update_amax records history
  - quantize round-trip: delayed, dynamic, blockwise — returns FP8Descriptor; numerical correctness vs golden
  - FP8 param lifecycle: register, mark stale, re-quantize
  - Gradient quantization: quantize_grad_tensor static method — SNR check
  - quantize_bwd_delayed: cross-iteration delayed scaling for blockwise2d backward
  - reset clears all tracked state
  - Benchmarks: quantize throughput for various recipes and tensor sizes
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops"))
from atomrl.lumen.quantize.config import (
    AmaxAlgo,
    QuantConfig,
    QuantFormat,
    ScalingType,
)
from atomrl.lumen.quantize.descriptor import FP8Descriptor
from atomrl.lumen.quantize.scaling_manager import ScalingManager
from conftest import (
    compute_snr,
    delayed_scale_ref,
    fp8_blockwise_quant_dequant_ref,
    fp8_dynamic_scale_ref,
    fp8_quant_dequant_ref,
)

# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    def test_from_config(self):
        cfg = QuantConfig(format=QuantFormat.FP8_E4M3, scaling=ScalingType.DELAYED)
        mgr = ScalingManager(cfg)
        assert mgr.recipe == "delayed"

    def test_from_legacy_recipe(self):
        mgr = ScalingManager(recipe="delayed")
        assert mgr.recipe == "delayed"

    def test_default_construction(self):
        mgr = ScalingManager()
        assert mgr.recipe == "delayed"

    def test_mxfp8_recipe(self):
        cfg = QuantConfig(format=QuantFormat.MXFP8, scaling=ScalingType.BLOCKWISE)
        mgr = ScalingManager(cfg)
        assert mgr.recipe == "mxfp8"


# ===================================================================
# get_scale behaviour per recipe
# ===================================================================


class TestGetScale:
    def test_blockwise_returns_none(self):
        mgr = ScalingManager(recipe="blockwise")
        t = torch.randn(4, 8, device="cuda")
        assert mgr.get_scale("test", t) is None

    def test_delayed_returns_tensor(self):
        mgr = ScalingManager(recipe="delayed")
        t = torch.randn(4, 8, device="cuda")
        scale = mgr.get_scale("test", t)
        assert isinstance(scale, torch.Tensor)
        assert scale.numel() == 1

    def test_dynamic_returns_tensor(self):
        mgr = ScalingManager(recipe="dynamic")
        t = torch.randn(4, 8, device="cuda")
        scale = mgr.get_scale("test", t)
        assert isinstance(scale, torch.Tensor)
        assert scale.numel() == 1

    def test_dynamic_scale_matches_golden(self):
        """Dynamic scale should equal amax / fp8_max (golden reference)."""
        mgr = ScalingManager(recipe="dynamic")
        torch.manual_seed(0)
        t = torch.randn(32, 64, device="cuda")
        scale = mgr.get_scale("t", t)
        fp8_max = torch.finfo(mgr.fp8_dtype).max
        golden_scale = fp8_dynamic_scale_ref(t, fp8_max)
        assert torch.allclose(
            scale.cpu().float(), torch.tensor(golden_scale.item()), rtol=1e-3
        )

    def test_none_recipe_returns_none(self):
        cfg = QuantConfig(scaling=ScalingType.NONE)
        mgr = ScalingManager(cfg)
        t = torch.randn(4, 8, device="cuda")
        assert mgr.get_scale("test", t) is None


# ===================================================================
# Amax history
# ===================================================================


class TestAmaxHistory:
    def test_update_amax_records(self):
        mgr = ScalingManager(recipe="delayed")
        t = torch.randn(4, 8, device="cuda") * 2.0
        mgr.update_amax("test_tensor", t)
        assert len(mgr.amax_history["test_tensor"]) == 1

    def test_update_amax_multiple(self):
        mgr = ScalingManager(recipe="delayed", history_len=4)
        for i in range(6):
            t = torch.randn(4, 8, device="cuda") * (i + 1)
            mgr.update_amax("test_tensor", t)
        assert len(mgr.amax_history["test_tensor"]) == 4

    def test_delayed_scale_uses_history(self):
        mgr = ScalingManager(recipe="delayed")
        for i in range(3):
            t = torch.ones(4, device="cuda") * (i + 1)
            mgr.update_amax("t", t)
        scale = mgr.get_scale("t", torch.ones(4, device="cuda"))
        assert isinstance(scale, torch.Tensor)

    def test_most_recent_algo(self):
        cfg = QuantConfig(scaling=ScalingType.DELAYED, amax_algo=AmaxAlgo.MOST_RECENT)
        mgr = ScalingManager(cfg)
        mgr.update_amax("t", torch.tensor([1.0], device="cuda"))
        mgr.update_amax("t", torch.tensor([100.0], device="cuda"))
        scale = mgr.get_scale("t", torch.ones(1, device="cuda"))
        assert isinstance(scale, torch.Tensor)


# ===================================================================
# Quantize round-trip
# ===================================================================


class TestQuantize:
    def test_delayed_quantize(self):
        mgr = ScalingManager(recipe="delayed")
        t = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        desc = mgr.quantize("w", t)
        assert isinstance(desc, FP8Descriptor)
        assert desc.data is not None
        assert desc.scale is not None

    def test_delayed_quantize_round_trip_snr(self):
        """Delayed quant→dequant round-trip should have high SNR vs original."""
        mgr = ScalingManager(recipe="delayed")
        torch.manual_seed(42)
        t = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        desc = mgr.quantize("w", t)
        fp8_t, scale = desc.data, desc.scale
        reconstructed = fp8_t.to(torch.bfloat16) * scale
        snr = compute_snr(t, reconstructed)
        assert snr > 10, f"Delayed quant round-trip SNR too low: {snr:.1f} dB"

    def test_dynamic_quantize(self):
        mgr = ScalingManager(recipe="dynamic")
        t = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        desc = mgr.quantize("w", t)
        assert isinstance(desc, FP8Descriptor)
        assert desc.data is not None
        assert desc.scale is not None

    def test_dynamic_quantize_matches_golden(self):
        """Dynamic quant output should closely match pure-PyTorch golden."""
        mgr = ScalingManager(recipe="dynamic")
        torch.manual_seed(42)
        t = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        desc = mgr.quantize("w", t)
        fp8_t, scale = desc.data, desc.scale
        reconstructed = fp8_t.to(torch.bfloat16) * scale

        golden, _ = fp8_quant_dequant_ref(t, fp8_dtype=mgr.fp8_dtype)
        snr = compute_snr(golden, reconstructed)
        assert snr > 20, f"Dynamic quant vs golden SNR: {snr:.1f} dB"

    def test_delayed_scale_matches_golden(self):
        """Delayed scale from history should match golden reference."""
        mgr = ScalingManager(recipe="delayed", history_len=4)
        fp8_max = torch.finfo(mgr.fp8_dtype).max
        for val in [1.0, 2.0, 3.0]:
            mgr.update_amax("t", torch.tensor([val], device="cuda"))
        scale = mgr.get_scale("t", torch.ones(1, device="cuda"))
        amax_history = list(mgr.amax_history["t"])
        golden_quant_scale = delayed_scale_ref(amax_history, fp8_max, margin=0)
        assert torch.allclose(
            scale.cpu().float(),
            torch.tensor(1.0 / golden_quant_scale),
            rtol=1e-3,
        )

    def test_blockwise_quantize_matches_golden(self):
        """Blockwise quant→dequant round-trip should match golden reference."""
        mgr = ScalingManager(recipe="blockwise")
        torch.manual_seed(42)
        t = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        desc = mgr.quantize("w", t)
        fp8_t, fp8_scales = desc.data, desc.scale

        reconstructed = fp8_t.to(torch.bfloat16)
        if fp8_scales is not None:
            if fp8_scales.shape != reconstructed.shape:
                block_size = mgr.config.block_size if hasattr(mgr, "config") else 128
                fp8_scales_expanded = fp8_scales.repeat_interleave(block_size, dim=-1)[
                    :, : reconstructed.shape[-1]
                ]
                reconstructed = reconstructed * fp8_scales_expanded
            else:
                reconstructed = reconstructed * fp8_scales

        golden = fp8_blockwise_quant_dequant_ref(t, fp8_dtype=mgr.fp8_dtype)
        snr = compute_snr(golden, reconstructed)
        assert snr > 15, f"Blockwise quant vs golden SNR: {snr:.1f} dB"

    def test_none_recipe_passthrough(self):
        cfg = QuantConfig(scaling=ScalingType.NONE)
        mgr = ScalingManager(cfg)
        t = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        result = mgr.quantize("w", t)
        assert result is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_quantize_returns_descriptor():
    mgr = ScalingManager(QuantConfig(scaling=ScalingType.DELAYED))
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    result = mgr.quantize("test_weight", x)
    assert isinstance(result, FP8Descriptor)


# ===================================================================
# FP8 param lifecycle
# ===================================================================


class TestFP8ParamLifecycle:
    def test_register_and_count(self):
        mgr = ScalingManager(recipe="delayed")
        param = torch.nn.Parameter(torch.randn(4, 8, device="cuda"))
        mgr.register_fp8_param("layer.weight", param)
        assert mgr.num_fp8_params == 1

    def test_mark_stale(self):
        mgr = ScalingManager(recipe="delayed")
        param = torch.nn.Parameter(torch.randn(4, 8, device="cuda"))
        mgr.register_fp8_param("layer.weight", param)
        mgr.mark_fp8_params_stale()
        assert "layer.weight" in mgr._fp8_param_stale

    def test_check_and_mark_stale(self):
        mgr = ScalingManager(recipe="delayed")
        param = torch.nn.Parameter(torch.randn(4, 8, device="cuda"))
        mgr.register_fp8_param("layer.weight", param)
        mgr.check_and_mark_fp8_stale(0)
        assert "layer.weight" in mgr._fp8_param_stale
        mgr._fp8_param_stale.clear()
        mgr.check_and_mark_fp8_stale(0)
        assert "layer.weight" not in mgr._fp8_param_stale

    def test_mxfp4_is_refused(self):
        """The param cache is an FP8 mechanism; no MXFP4 path reads it back.

        Under MXFP4 it was not inert: it spent a quantization pass per weight
        per optimizer step and stored row-wise scales where an MXFP4 consumer
        expects 2D tiles, so the first reader to appear would have gotten a
        silently wrong layout rather than an error.
        """
        cfg = QuantConfig(format=QuantFormat.MXFP4, scaling=ScalingType.BLOCKWISE)
        mgr = ScalingManager(cfg)
        model = torch.nn.Linear(8, 8)
        model._quant_tensor_id = "layer.weight"

        with pytest.raises(ValueError, match="does not apply to MXFP4"):
            mgr.enable_fp8_params(model)

    def test_fp8_weight_cache_is_refused_for_mxfp4(self):
        """Same defect as the param cache, in the sibling feature.

        store_weights_fp8 leaves per-tensor FP8 with a scalar scale on the
        module, and the forward hands a populated cache straight to the GEMM --
        _mxfp4_cached_weight passes it through and quantize_input is skipped.
        The MXFP4 GEMM then got e4m3 bytes where it wanted packed FP4 and died
        unpacking an empty shape, several frames from the cause.
        """
        import atomrl.lumen.quantize as quant
        from atomrl.lumen.quantize.config import QuantConfig as _QC

        model = torch.nn.Linear(64, 64).cuda().to(torch.bfloat16)
        quant.enable(
            model, config=_QC(format=QuantFormat.MXFP4, scaling=ScalingType.BLOCKWISE)
        )

        with pytest.raises(ValueError, match="does not apply to MXFP4"):
            quant.store_weights_fp8(model)


# ===================================================================
# Gradient quantization (static method)
# ===================================================================


class TestGradQuantStatic:
    def test_none_is_identity(self):
        t = torch.randn(4, 8, device="cuda")
        result = ScalingManager.quantize_grad_tensor(t, None)
        assert result is t

    def test_fp8_round_trip(self):
        t = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        result = ScalingManager.quantize_grad_tensor(t, "fp8")
        assert result.shape == t.shape
        assert result.dtype == t.dtype

    def test_fp8_round_trip_snr(self):
        """FP8 gradient quant should have high SNR vs original."""
        torch.manual_seed(42)
        t = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        result = ScalingManager.quantize_grad_tensor(t, "fp8")
        snr = compute_snr(t, result)
        assert snr > 10, f"Grad quant SNR: {snr:.1f} dB"

    def test_fp8_round_trip_matches_golden(self):
        """Gradient quant should match pure-PyTorch golden reference."""
        torch.manual_seed(42)
        t = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        result = ScalingManager.quantize_grad_tensor(t, "fp8")
        golden, _ = fp8_quant_dequant_ref(t)
        snr = compute_snr(golden, result)
        assert snr > 20, f"Grad quant vs golden SNR: {snr:.1f} dB"

    def test_mxfp4_ignores_an_fp8_recipe_block_size(self, monkeypatch):
        import atomrl.lumen.quantize.scaling_manager as sm_mod

        seen = {}

        def _round(tensor, block_size):
            seen["block_size"] = block_size
            return tensor

        monkeypatch.setattr(sm_mod, "_round_to_mxfp4", _round)
        tensor = torch.zeros(2, 32)
        assert (
            ScalingManager.quantize_grad_tensor(
                tensor,
                "mxfp4",
                block_size=128,
            )
            is tensor
        )
        assert seen["block_size"] == 32

    def test_invalid_raises(self):
        t = torch.randn(4, 8, device="cuda")
        with pytest.raises(ValueError):
            ScalingManager.quantize_grad_tensor(t, "bogus")


# ===================================================================
# quantize_bwd_delayed — cross-iteration delayed scaling for blockwise2d
# ===================================================================


class TestQuantizeBwdDelayed:
    """Tests for ScalingManager.quantize_bwd_delayed (blockwise2d backward path)."""

    def test_returns_fp8_and_scale(self):
        mgr = ScalingManager(recipe="delayed")
        t = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        fp8_t, scale = mgr.quantize_bwd_delayed("grad_bwd", t)
        assert fp8_t.dtype == mgr.fp8_dtype_bwd
        assert fp8_t.shape == t.shape
        assert scale.numel() == 1

    def test_first_call_uses_current_amax(self):
        """Without history, scale should be derived from the current tensor."""
        mgr = ScalingManager(recipe="delayed")
        torch.manual_seed(42)
        t = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        _fp8_t, scale = mgr.quantize_bwd_delayed("grad", t)

        fp8_max = mgr._fp8_max_bwd
        expected_amax = t.abs().amax()
        expected_scale = mgr._compute_scale(expected_amax, fp8_max)
        assert torch.allclose(scale, expected_scale, rtol=1e-5)

    def test_records_amax_in_history(self):
        mgr = ScalingManager(recipe="delayed")
        t = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
        mgr.quantize_bwd_delayed("grad", t)
        assert len(mgr.amax_history["grad"]) == 1

    def test_subsequent_call_uses_history(self):
        """Second call should use the amax recorded from the first call."""
        mgr = ScalingManager(recipe="delayed")
        t1 = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16) * 2.0
        mgr.quantize_bwd_delayed("grad", t1)

        t2 = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16) * 0.5
        _, scale2 = mgr.quantize_bwd_delayed("grad", t2)

        fp8_max = mgr._fp8_max_bwd
        expected_amax = t1.abs().amax()
        expected_scale = mgr._compute_scale(expected_amax, fp8_max)
        assert torch.allclose(scale2, expected_scale, rtol=1e-5)

    def test_most_recent_algo(self):
        cfg = QuantConfig(scaling=ScalingType.DELAYED, amax_algo=AmaxAlgo.MOST_RECENT)
        mgr = ScalingManager(cfg)
        t1 = torch.ones(8, 16, device="cuda", dtype=torch.bfloat16) * 10.0
        mgr.quantize_bwd_delayed("grad", t1)

        t2 = torch.ones(8, 16, device="cuda", dtype=torch.bfloat16) * 1.0
        mgr.quantize_bwd_delayed("grad", t2)

        t3 = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        _, scale3 = mgr.quantize_bwd_delayed("grad", t3)

        fp8_max = mgr._fp8_max_bwd
        expected_amax = t2.abs().amax()
        expected_scale = mgr._compute_scale(expected_amax, fp8_max)
        assert torch.allclose(scale3, expected_scale, rtol=1e-5)

    def test_max_algo_uses_global_max(self):
        cfg = QuantConfig(scaling=ScalingType.DELAYED, amax_algo=AmaxAlgo.MAX)
        mgr = ScalingManager(cfg)
        t1 = torch.ones(8, 16, device="cuda", dtype=torch.bfloat16) * 10.0
        mgr.quantize_bwd_delayed("grad", t1)

        t2 = torch.ones(8, 16, device="cuda", dtype=torch.bfloat16) * 1.0
        mgr.quantize_bwd_delayed("grad", t2)

        t3 = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        _, scale3 = mgr.quantize_bwd_delayed("grad", t3)

        fp8_max = mgr._fp8_max_bwd
        expected_amax = torch.stack([t1.abs().amax(), t2.abs().amax()]).amax()
        expected_scale = mgr._compute_scale(expected_amax.to("cuda"), fp8_max)
        assert torch.allclose(scale3, expected_scale, rtol=1e-5)

    def test_round_trip_snr(self):
        """quantize_bwd_delayed → dequant round-trip should have decent SNR."""
        mgr = ScalingManager(recipe="delayed")
        torch.manual_seed(42)
        t = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        fp8_t, scale = mgr.quantize_bwd_delayed("grad", t)
        reconstructed = fp8_t.to(torch.bfloat16) * scale
        snr = compute_snr(t, reconstructed)
        assert snr > 10, f"quantize_bwd_delayed round-trip SNR too low: {snr:.1f} dB"

    def test_history_respects_maxlen(self):
        mgr = ScalingManager(recipe="delayed", history_len=4)
        for i in range(10):
            t = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16) * (i + 1)
            mgr.quantize_bwd_delayed("grad", t)
        assert len(mgr.amax_history["grad"]) == 4

    def test_independent_tensor_ids(self):
        mgr = ScalingManager(recipe="delayed")
        t1 = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16) * 5.0
        t2 = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16) * 0.1
        mgr.quantize_bwd_delayed("grad_a", t1)
        mgr.quantize_bwd_delayed("grad_b", t2)

        assert len(mgr.amax_history["grad_a"]) == 1
        assert len(mgr.amax_history["grad_b"]) == 1
        # History stores Python floats (``.item()``), not tensors.
        assert float(mgr.amax_history["grad_a"][-1]) != float(
            mgr.amax_history["grad_b"][-1]
        )

    def test_zero_tensor_no_nan(self):
        """Zero input should not produce NaN/Inf in scale or output."""
        mgr = ScalingManager(recipe="delayed")
        t = torch.zeros(16, 32, device="cuda", dtype=torch.bfloat16)
        fp8_t, scale = mgr.quantize_bwd_delayed("grad_zero", t)
        assert not torch.any(torch.isnan(fp8_t.float()))
        assert not torch.any(torch.isinf(scale))


# ===================================================================
# ScalingManager with blockwise2d recipe
# ===================================================================


class TestBlockwise2DRecipe:
    """Verify ScalingManager routes blockwise2d through blockwise kernels."""

    def test_get_scale_returns_none(self):
        mgr = ScalingManager(recipe="blockwise2d")
        t = torch.randn(4, 8, device="cuda")
        assert mgr.get_scale("test", t) is None

    def test_quantize_matches_blockwise(self):
        """blockwise2d quantize should produce same result as blockwise."""
        torch.manual_seed(42)
        mgr_bw = ScalingManager(recipe="blockwise")
        mgr_bw2d = ScalingManager(recipe="blockwise2d")
        t = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)

        desc_bw = mgr_bw.quantize("w", t)
        desc_bw2d = mgr_bw2d.quantize("w", t)

        assert torch.equal(desc_bw.data, desc_bw2d.data)
        scale_bw, scale_bw2d = desc_bw.scale, desc_bw2d.scale
        if scale_bw is not None and scale_bw2d is not None:
            assert torch.allclose(scale_bw, scale_bw2d)


# ===================================================================
# Reset
# ===================================================================


def test_reset_clears_state():
    mgr = ScalingManager(recipe="delayed")
    t = torch.randn(4, 8, device="cuda")
    mgr.update_amax("t", t)
    mgr.quantize("t", t)
    mgr.reset()
    assert len(mgr.amax_history) == 0
    assert len(mgr.scale_cache) == 0
    assert len(mgr._fp8_param_cache) == 0


# ===================================================================
# Benchmarks
# ===================================================================


class TestQuantizeBenchmark:
    """Throughput benchmarks for quantization recipes."""

    @pytest.mark.parametrize("recipe", ["dynamic", "delayed", "blockwise"])
    @pytest.mark.parametrize("size", [(128, 256), (1024, 4096), (4096, 4096)])
    def test_quantize_throughput(self, recipe, size):
        M, N = size
        mgr = ScalingManager(recipe=recipe)
        t = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        for _ in range(3):
            mgr.quantize("bench", t)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 20

        start.record()
        for _ in range(iters):
            mgr.quantize("bench", t)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / iters
        total_bytes = M * N * 2
        bw_gb_s = (total_bytes / (elapsed_ms / 1000.0)) / (1024**3)
        print(
            f"\n[Quantize] recipe={recipe} size={M}x{N}: {elapsed_ms:.3f}ms, {bw_gb_s:.1f} GB/s"
        )


# ===================================================================
# quantize_block_2d_fp8 — golden reference tests
# ===================================================================


def _quantize_block_2d_fp8_golden(tensor_bshd, block_m, block_n, fp8_dtype):
    """Pure-PyTorch golden reference for 2D block FP8 quantization.

    Input: [B, S, H, D] (BSHD).  Internally works in [B, H, S, D].
    Returns (tensor_fp8, scale_inv) matching ScalingManager.quantize_block_2d_fp8.
    """
    t = tensor_bshd.permute(0, 2, 1, 3).float()  # [B, H, S, D]
    B, H, S, D = t.shape
    MAX_FP8 = torch.finfo(fp8_dtype).max

    t_blocked = t.reshape(B, H, S // block_m, block_m, D // block_n, block_n)
    tile_max = t_blocked.abs().amax(dim=(3, 5))  # [B, H, S//bm, D//bn]
    tile_max = torch.where(tile_max == 0, MAX_FP8, tile_max)
    scale = MAX_FP8 / tile_max

    scale_expanded = scale[:, :, :, None, :, None].expand_as(t_blocked)
    t_quant = (t_blocked * scale_expanded).clamp(-MAX_FP8, MAX_FP8).to(fp8_dtype)
    t_quant = t_quant.reshape(B, H, S, D).permute(0, 2, 1, 3).contiguous()

    return t_quant, (1.0 / scale).to(torch.float32).contiguous()


class TestQuantizeBlock2DFP8:
    def _get_fp8_dtype(self):
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        return _get_float8_e4m3()

    def test_output_shapes(self):
        fp8_dtype = self._get_fp8_dtype()
        B, S, H, D = 2, 128, 4, 64
        t = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        t_fp8, scale_inv = ScalingManager.quantize_block_2d_fp8(t, 64, 32, fp8_dtype)
        assert t_fp8.shape == (B, S, H, D)
        assert t_fp8.dtype == fp8_dtype
        assert scale_inv.shape == (B, H, S // 64, D // 32)
        assert scale_inv.dtype == torch.float32

    def test_matches_golden(self):
        fp8_dtype = self._get_fp8_dtype()
        torch.manual_seed(42)
        B, S, H, D = 2, 128, 4, 64
        t = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        t_fp8, scale_inv = ScalingManager.quantize_block_2d_fp8(t, 64, 32, fp8_dtype)
        golden_fp8, golden_scale = _quantize_block_2d_fp8_golden(t, 64, 32, fp8_dtype)

        assert torch.equal(t_fp8, golden_fp8)
        assert torch.allclose(scale_inv, golden_scale, rtol=1e-5)

    def test_round_trip_snr(self):
        fp8_dtype = self._get_fp8_dtype()
        torch.manual_seed(42)
        B, S, H, D = 2, 256, 8, 128
        block_m, block_n = 64, 64
        t = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        t_fp8, scale_inv = ScalingManager.quantize_block_2d_fp8(
            t, block_m, block_n, fp8_dtype
        )

        t_bhsd = t_fp8.permute(0, 2, 1, 3).float()  # [B, H, S, D]
        Bx, Hx, Sx, Dx = t_bhsd.shape
        t_blocked = t_bhsd.reshape(
            Bx, Hx, Sx // block_m, block_m, Dx // block_n, block_n
        )
        s_expanded = scale_inv[:, :, :, None, :, None].expand_as(t_blocked)
        reconstructed = (t_blocked * s_expanded).reshape(Bx, Hx, Sx, Dx)
        reconstructed = reconstructed.permute(0, 2, 1, 3).to(torch.bfloat16)

        snr = compute_snr(t, reconstructed)
        assert snr > 15, f"2D block FP8 round-trip SNR too low: {snr:.1f} dB"

    @pytest.mark.parametrize("block_m,block_n", [(32, 32), (64, 64), (128, 64)])
    def test_various_block_sizes(self, block_m, block_n):
        fp8_dtype = self._get_fp8_dtype()
        B, S, H, D = 1, 256, 2, 128
        t = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        t_fp8, scale_inv = ScalingManager.quantize_block_2d_fp8(
            t, block_m, block_n, fp8_dtype
        )
        assert t_fp8.shape == (B, S, H, D)
        assert scale_inv.shape == (B, H, S // block_m, D // block_n)

    def test_zero_tensor_handled(self):
        fp8_dtype = self._get_fp8_dtype()
        B, S, H, D = 1, 64, 2, 64
        t = torch.zeros(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        _t_fp8, scale_inv = ScalingManager.quantize_block_2d_fp8(t, 32, 32, fp8_dtype)
        assert not torch.any(torch.isnan(scale_inv))
        assert not torch.any(torch.isinf(scale_inv))


# ===================================================================
# Blockwise2DScaleManager — lifecycle tests
# ===================================================================


from atomrl.lumen.quantize.scaling_manager import Blockwise2DScaleManager


class TestBlockwise2DScaleManager:
    def _get_fp8_dtype(self):
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        return _get_float8_e4m3()

    def test_construction_defaults(self):
        mgr = Blockwise2DScaleManager()
        assert mgr.block_m == 64
        assert mgr.block_n == 64

    def test_construction_custom(self):
        mgr = Blockwise2DScaleManager(block_m=32, block_n=128)
        assert mgr.block_m == 32
        assert mgr.block_n == 128

    def test_quantize_and_cache(self):
        fp8_dtype = self._get_fp8_dtype()
        mgr = Blockwise2DScaleManager(block_m=64, block_n=64)
        B, S, H, D = 2, 128, 4, 64
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

        q_fp8, k_fp8, v_fp8, q_s, k_s, v_s = mgr.quantize_and_cache(q, k, v, fp8_dtype)

        assert q_fp8.dtype == fp8_dtype
        assert k_fp8.dtype == fp8_dtype
        assert v_fp8.dtype == fp8_dtype
        assert q_s.shape == (B, H, S // 64, D // 64)
        assert k_s.shape == (B, H, S // 64, D // 64)
        assert v_s.shape == (B, H, S // 64, D // 64)

    def test_get_cached_returns_same_tensors(self):
        fp8_dtype = self._get_fp8_dtype()
        mgr = Blockwise2DScaleManager(block_m=64, block_n=64)
        B, S, H, D = 1, 128, 2, 64
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

        q_fp8, k_fp8, v_fp8, q_s, k_s, v_s = mgr.quantize_and_cache(q, k, v, fp8_dtype)
        cached = mgr.get_cached()

        assert cached[0] is q_fp8
        assert cached[1] is k_fp8
        assert cached[2] is v_fp8
        assert cached[3] is q_s
        assert cached[4] is k_s
        assert cached[5] is v_s

    def test_get_cached_before_quantize_raises(self):
        mgr = Blockwise2DScaleManager()
        with pytest.raises(RuntimeError, match="quantize_and_cache"):
            mgr.get_cached()

    def test_do_scale_lifecycle(self):
        mgr = Blockwise2DScaleManager()
        assert mgr.get_do_scale() is None

        do_scale = torch.tensor([0.5], device="cuda")
        mgr.cache_do_scale(do_scale)
        cached_do = mgr.get_do_scale()
        assert cached_do is not None
        assert torch.equal(cached_do, do_scale)
        assert not cached_do.requires_grad

    def test_clear(self):
        fp8_dtype = self._get_fp8_dtype()
        mgr = Blockwise2DScaleManager(block_m=64, block_n=64)
        B, S, H, D = 1, 128, 2, 64
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        mgr.quantize_and_cache(q, k, v, fp8_dtype)
        mgr.cache_do_scale(torch.tensor([1.0], device="cuda"))
        mgr.clear()

        assert mgr._q_fp8 is None
        assert mgr._do_scale is None
        with pytest.raises(RuntimeError):
            mgr.get_cached()

    def test_quantize_and_cache_matches_static(self):
        """Manager's quantize_and_cache should produce identical results
        to calling ScalingManager.quantize_block_2d_fp8 three times."""
        fp8_dtype = self._get_fp8_dtype()
        torch.manual_seed(123)
        mgr = Blockwise2DScaleManager(block_m=64, block_n=64)
        B, S, H, D = 2, 128, 4, 64
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

        q_fp8, k_fp8, v_fp8, q_s, k_s, v_s = mgr.quantize_and_cache(q, k, v, fp8_dtype)

        ref_q, ref_qs = ScalingManager.quantize_block_2d_fp8(q, 64, 64, fp8_dtype)
        ref_k, ref_ks = ScalingManager.quantize_block_2d_fp8(k, 64, 64, fp8_dtype)
        ref_v, ref_vs = ScalingManager.quantize_block_2d_fp8(v, 64, 64, fp8_dtype)

        assert torch.equal(q_fp8, ref_q)
        assert torch.equal(k_fp8, ref_k)
        assert torch.equal(v_fp8, ref_v)
        assert torch.allclose(q_s, ref_qs)
        assert torch.allclose(k_s, ref_ks)
        assert torch.allclose(v_s, ref_vs)


class TestGetScaleSdma:
    """Verify get_scale routes to SDMA when _use_sdma is True."""

    def test_delayed_uses_sdma_when_enabled(self):
        from unittest import mock as umock

        mgr = ScalingManager(recipe="delayed")
        mgr._use_sdma = True
        mgr._dp_group = "fake_dp_group"
        mgr.config.reduce_amax = True
        mgr.amax_history["x"].append(torch.tensor(2.0, device="cuda"))

        with umock.patch.object(
            mgr,
            "_reduce_single_amax_sdma",
            return_value=torch.tensor(2.0, device="cuda"),
        ) as mock_sdma:
            scale = mgr.get_scale("x", torch.randn(4, 8, device="cuda"))
            mock_sdma.assert_called_once()
            assert scale is not None

    def test_dynamic_uses_sdma_when_enabled(self):
        from unittest import mock as umock

        mgr = ScalingManager(recipe="dynamic")
        mgr._use_sdma = True
        mgr._dp_group = "fake_dp_group"
        mgr.config.reduce_amax = True

        with umock.patch.object(
            mgr,
            "_reduce_single_amax_sdma",
            return_value=torch.tensor(2.0, device="cuda"),
        ) as mock_sdma:
            scale = mgr.get_scale("x", torch.randn(4, 8, device="cuda"))
            mock_sdma.assert_called_once()
            assert scale is not None

    def test_delayed_uses_dist_when_sdma_disabled(self):
        from unittest import mock as umock

        mgr = ScalingManager(recipe="delayed")
        mgr._use_sdma = False
        mgr._dp_group = "fake_dp_group"
        mgr.config.reduce_amax = True
        mgr.amax_history["x"].append(torch.tensor(2.0, device="cuda"))

        with umock.patch("torch.distributed.all_reduce") as mock_dist:
            mgr.get_scale("x", torch.randn(4, 8, device="cuda"))
            mock_dist.assert_called_once()

    def test_reduce_single_amax_sdma_casts_to_float32(self):
        """Verify _reduce_single_amax_sdma casts input to float32."""
        from unittest import mock as umock

        mgr = ScalingManager(recipe="delayed")
        mgr._use_sdma = True

        fake_result = torch.tensor([3.0], device="cuda")
        with (
            umock.patch(
                "atomrl.lumen.ops.sdma.sdma_allgather_max", return_value=fake_result
            ) as mock_fn,
            umock.patch("atomrl.lumen.ops.sdma.SdmaAllgather"),
        ):
            amax_bf16 = torch.tensor(2.0, device="cuda", dtype=torch.bfloat16)
            result = mgr._reduce_single_amax_sdma(amax_bf16)
            call_args = mock_fn.call_args[0][0]
            assert call_args.dtype == torch.float32
            assert result.item() == pytest.approx(3.0)
