# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for atomrl.lumen.modules.attention — LumenAttention nn.Module.

Covers:
  - Construction with default parameters
  - Construction with FP8 backend (aiter_triton_fp8 etc.)
  - forward() non-FP8 path dispatches correctly
  - forward() FP8 path dispatches correctly
  - forward() with grad_quant_type
  - Module parameter introspection (no learnable params)
  - Causal mode forwarding
  - return_lse forwarding
  - Various backend_type / quant_type combos

Reference: attention_ref from conftest (pure PyTorch BSHD).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops"))
from conftest import AttnConfig, attention_ref, compute_snr


def _make_tensors(config, batch_size, dtype, device, requires_grad=False):
    torch.cuda.manual_seed(42)
    q = (
        torch.randn(
            batch_size,
            config.seqlen_q,
            config.num_head_q,
            config.head_dim_qk,
            device=device,
            dtype=dtype,
        )
        * 0.02
    ).requires_grad_(requires_grad)
    k = (
        torch.randn(
            batch_size,
            config.seqlen_kv,
            config.num_head_kv,
            config.head_dim_qk,
            device=device,
            dtype=dtype,
        )
        * 0.02
    ).requires_grad_(requires_grad)
    v = (
        torch.randn(
            batch_size,
            config.seqlen_kv,
            config.num_head_kv,
            config.head_dim_v,
            device=device,
            dtype=dtype,
        )
        * 0.02
    ).requires_grad_(requires_grad)
    return q, k, v


# ===================================================================
# Construction
# ===================================================================


class TestLumenAttentionConstruction:
    def test_default_params(self):
        from atomrl.lumen.modules.attention import LumenAttention

        attn = LumenAttention()
        assert attn.dropout_p == 0.0
        assert attn.causal is False
        assert attn.return_lse is False

    def test_no_learnable_params(self):
        from atomrl.lumen.modules.attention import LumenAttention

        attn = LumenAttention()
        params = list(attn.parameters())
        assert len(params) == 0

    def test_fp8_flag(self):
        from atomrl.lumen.modules.attention import LumenAttention

        attn = LumenAttention(backend_type="aiter_triton_fp8")
        assert attn._is_fp8 is True

    def test_quant_type_sets_fp8(self):
        from atomrl.lumen.modules.attention import LumenAttention

        attn = LumenAttention(quant_type="blockwise")
        assert attn._is_fp8 is True

    def test_non_fp8_flag(self):
        from atomrl.lumen.modules.attention import LumenAttention

        attn = LumenAttention(backend_type="aiter_triton")
        assert attn._is_fp8 is False

    def test_csrc_backend_with_aiter(self):
        from atomrl.lumen.modules.attention import LumenAttention

        attn = LumenAttention(backend_type="aiter_csrc")
        assert attn.backend_type == "aiter_csrc"


# ===================================================================
# Forward: non-FP8 path
# ===================================================================


class TestLumenAttentionForward:
    def test_triton_forward(self):
        from atomrl.lumen.modules.attention import LumenAttention

        config = AttnConfig(128, 128, 8, 8, 64, 64)
        attn = LumenAttention(backend_type="aiter_triton", softmax_scale=64**-0.5)
        q, k, v = _make_tensors(config, 2, torch.bfloat16, "cuda")
        out = attn(q, k, v)
        assert out.shape == (2, 128, 8, 64)

        out_ref = attention_ref(q, k, v, 64**-0.5)
        snr = compute_snr(out_ref, out)
        assert snr > 20, f"LumenAttention triton fwd SNR: {snr:.1f} dB"

    def test_causal_forward(self):
        from atomrl.lumen.modules.attention import LumenAttention

        config = AttnConfig(128, 128, 8, 8, 64, 64)
        attn = LumenAttention(
            backend_type="aiter_triton", causal=True, softmax_scale=64**-0.5
        )
        q, k, v = _make_tensors(config, 2, torch.bfloat16, "cuda")
        out = attn(q, k, v)
        assert out.shape == (2, 128, 8, 64)

        out_ref = attention_ref(q, k, v, 64**-0.5, causal=True)
        snr = compute_snr(out_ref, out)
        assert snr > 20, f"LumenAttention causal fwd SNR: {snr:.1f} dB"

    def test_return_lse(self):
        from atomrl.lumen.modules.attention import LumenAttention

        attn = LumenAttention(
            backend_type="aiter_triton", return_lse=True, softmax_scale=64**-0.5
        )
        config = AttnConfig(128, 128, 8, 8, 64, 64)
        q, k, v = _make_tensors(config, 2, torch.bfloat16, "cuda")
        result = attn(q, k, v)
        assert isinstance(result, tuple)
        out, lse = result
        assert out.shape == (2, 128, 8, 64)
        assert lse.dtype == torch.float32

    def test_gqa_forward_snr(self):
        """Grouped-query attention (HQ > HKV) forward should match golden."""
        from atomrl.lumen.modules.attention import LumenAttention

        config = AttnConfig(128, 128, 16, 4, 64, 64)
        sm_scale = 64**-0.5
        attn = LumenAttention(backend_type="aiter_triton", softmax_scale=sm_scale)
        q, k, v = _make_tensors(config, 2, torch.bfloat16, "cuda")
        out = attn(q, k, v)
        assert out.shape == (2, 128, 16, 64)

        out_ref = attention_ref(q, k, v, sm_scale)
        snr = compute_snr(out_ref, out)
        assert snr > 20, f"GQA fwd SNR: {snr:.1f} dB"

    def test_causal_backward_grad_snr(self):
        """Causal backward gradients should match golden reference."""
        from atomrl.lumen.modules.attention import LumenAttention

        sm_scale = 64**-0.5

        torch.cuda.manual_seed(99)
        q_ref = (
            torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k_ref = (
            torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v_ref = (
            torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        out_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale, causal=True)
        out_ref.float().mean().backward()

        torch.cuda.manual_seed(99)
        q = (
            torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k = (
            torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v = (
            torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        attn = LumenAttention(
            backend_type="aiter_triton", causal=True, softmax_scale=sm_scale
        )
        out = attn(q, k, v)
        out.float().mean().backward()

        assert q.grad is not None and k.grad is not None and v.grad is not None
        dq_snr = compute_snr(q_ref.grad, q.grad)
        dk_snr = compute_snr(k_ref.grad, k.grad)
        dv_snr = compute_snr(v_ref.grad, v.grad)
        assert dq_snr > 15, f"Causal dQ SNR: {dq_snr:.1f} dB"
        assert dk_snr > 15, f"Causal dK SNR: {dk_snr:.1f} dB"
        assert dv_snr > 15, f"Causal dV SNR: {dv_snr:.1f} dB"

    def test_forward_backward_grad_snr(self):
        """Non-FP8 backward gradients should match pure-PyTorch reference."""
        from atomrl.lumen.modules.attention import LumenAttention

        sm_scale = 64**-0.5

        torch.cuda.manual_seed(42)
        q_ref = (
            torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k_ref = (
            torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v_ref = (
            torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        out_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale)
        out_ref.float().mean().backward()

        torch.cuda.manual_seed(42)
        q = (
            torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k = (
            torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v = (
            torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        attn = LumenAttention(backend_type="aiter_triton", softmax_scale=sm_scale)
        out = attn(q, k, v)
        out.float().mean().backward()

        assert q.grad is not None and k.grad is not None and v.grad is not None
        dq_snr = compute_snr(q_ref.grad, q.grad)
        dk_snr = compute_snr(k_ref.grad, k.grad)
        dv_snr = compute_snr(v_ref.grad, v.grad)
        assert dq_snr > 15, f"Triton dQ SNR: {dq_snr:.1f} dB"
        assert dk_snr > 15, f"Triton dK SNR: {dk_snr:.1f} dB"
        assert dv_snr > 15, f"Triton dV SNR: {dv_snr:.1f} dB"


# ===================================================================
# Forward: FP8 path
# ===================================================================


class TestLumenAttentionFP8:
    def test_fp8_blockwise_forward(self):
        from atomrl.lumen.modules.attention import LumenAttention

        _config = AttnConfig(128, 128, 8, 8, 64, 64)
        attn = LumenAttention(quant_type="blockwise", softmax_scale=64**-0.5)
        q, k, v = _make_tensors(_config, 2, torch.bfloat16, "cuda")
        out = attn(q, k, v)
        assert out.shape == (2, 128, 8, 64)

        out_ref = attention_ref(q, k, v, 64**-0.5)
        snr = compute_snr(out_ref, out)
        assert snr > 15, f"FP8 blockwise fwd SNR: {snr:.1f} dB"

    @pytest.mark.parametrize(
        "quant_type", ["blockwise", "dynamic", "delayed", "per_token"]
    )
    def test_fp8_quant_types_snr(self, quant_type):
        """All FP8 quant types should produce output with reasonable SNR vs golden."""
        from atomrl.lumen.modules.attention import LumenAttention

        _config = AttnConfig(128, 128, 8, 8, 64, 64)
        attn = LumenAttention(quant_type=quant_type, softmax_scale=64**-0.5)
        q, k, v = _make_tensors(_config, 2, torch.bfloat16, "cuda")
        out = attn(q, k, v)
        assert out.shape == (2, 128, 8, 64)

        out_ref = attention_ref(q, k, v, 64**-0.5)
        snr = compute_snr(out_ref, out)
        assert snr > 12, f"FP8 {quant_type} fwd SNR too low: {snr:.1f} dB"

    @pytest.mark.parametrize("quant_type", ["blockwise", "dynamic"])
    def test_fp8_forward_backward_grad_snr(self, quant_type):
        """FP8 backward gradients (dQ, dK, dV) should be numerically close to BF16 reference."""
        from atomrl.lumen.modules.attention import LumenAttention

        sm_scale = 64**-0.5

        torch.cuda.manual_seed(0)
        q_ref = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k_ref = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v_ref = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        out_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale)
        out_ref.float().mean().backward()

        torch.cuda.manual_seed(0)
        q = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        attn = LumenAttention(quant_type=quant_type, softmax_scale=sm_scale)
        out = attn(q, k, v)
        out.float().mean().backward()

        assert q.grad is not None and k.grad is not None and v.grad is not None
        dq_snr = compute_snr(q_ref.grad, q.grad)
        dk_snr = compute_snr(k_ref.grad, k.grad)
        dv_snr = compute_snr(v_ref.grad, v.grad)
        assert dq_snr > 8, f"FP8 {quant_type} dQ SNR: {dq_snr:.1f} dB"
        assert dk_snr > 8, f"FP8 {quant_type} dK SNR: {dk_snr:.1f} dB"
        assert dv_snr > 8, f"FP8 {quant_type} dV SNR: {dv_snr:.1f} dB"

    def test_grad_quant_type_snr(self):
        """grad_quant_type='fp8' backward should still be numerically reasonable."""
        from atomrl.lumen.modules.attention import LumenAttention

        sm_scale = 64**-0.5

        torch.cuda.manual_seed(7)
        q_ref = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k_ref = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v_ref = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        out_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale)
        out_ref.float().mean().backward()

        torch.cuda.manual_seed(7)
        q = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        attn = LumenAttention(
            backend_type="aiter_triton",
            softmax_scale=sm_scale,
            grad_quant_type="fp8",
        )
        out = attn(q, k, v)
        out.float().mean().backward()

        assert q.grad is not None and k.grad is not None and v.grad is not None
        dq_snr = compute_snr(q_ref.grad, q.grad)
        dk_snr = compute_snr(k_ref.grad, k.grad)
        dv_snr = compute_snr(v_ref.grad, v.grad)
        assert dq_snr > 8, f"grad_quant dQ SNR: {dq_snr:.1f} dB"
        assert dk_snr > 8, f"grad_quant dK SNR: {dk_snr:.1f} dB"
        assert dv_snr > 8, f"grad_quant dV SNR: {dv_snr:.1f} dB"


# ===================================================================
# Benchmarks
# ===================================================================


class TestLumenAttentionBlockwise2D:
    """End-to-end correctness tests for blockwise2d FP8 attention."""

    def test_blockwise2d_forward_snr(self):
        """blockwise2d forward output should match BF16 golden with reasonable SNR."""
        from atomrl.lumen.modules.attention import LumenAttention

        sm_scale = 64**-0.5
        config = AttnConfig(128, 128, 8, 8, 64, 64)
        attn = LumenAttention(quant_type="blockwise2d", softmax_scale=sm_scale)
        q, k, v = _make_tensors(config, 2, torch.bfloat16, "cuda")
        out = attn(q, k, v)
        assert out.shape == (2, 128, 8, 64)

        out_ref = attention_ref(q, k, v, sm_scale)
        snr = compute_snr(out_ref, out)
        assert snr > 12, f"blockwise2d fwd SNR: {snr:.1f} dB"

    def test_blockwise2d_forward_backward_grad_snr(self):
        """blockwise2d backward gradients should be numerically close to BF16 reference."""
        from atomrl.lumen.modules.attention import LumenAttention

        sm_scale = 64**-0.5

        torch.cuda.manual_seed(0)
        q_ref = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k_ref = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v_ref = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        out_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale)
        out_ref.float().mean().backward()

        torch.cuda.manual_seed(0)
        q = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        attn = LumenAttention(quant_type="blockwise2d", softmax_scale=sm_scale)
        out = attn(q, k, v)
        out.float().mean().backward()

        assert q.grad is not None and k.grad is not None and v.grad is not None
        dq_snr = compute_snr(q_ref.grad, q.grad)
        dk_snr = compute_snr(k_ref.grad, k.grad)
        dv_snr = compute_snr(v_ref.grad, v.grad)
        assert dq_snr > 8, f"blockwise2d dQ SNR: {dq_snr:.1f} dB"
        assert dk_snr > 8, f"blockwise2d dK SNR: {dk_snr:.1f} dB"
        assert dv_snr > 8, f"blockwise2d dV SNR: {dv_snr:.1f} dB"

    def test_blockwise2d_with_scale_manager(self):
        """blockwise2d with Blockwise2DScaleManager should cache Q/K/V and dO scale."""
        from atomrl.lumen.ops.attention.attention import attention_fp8_quant
        from atomrl.lumen.quantize.scaling_manager import Blockwise2DScaleManager

        mgr = Blockwise2DScaleManager(block_m=64, block_n=64)
        sm_scale = 64**-0.5

        torch.cuda.manual_seed(42)
        q = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)

        out = attention_fp8_quant(
            q,
            k,
            v,
            softmax_scale=sm_scale,
            causal=False,
            backend_type="aiter_triton_fp8",
            quant_type="blockwise2d",
            scale_manager=mgr,
        )
        assert out.shape == (1, 128, 4, 64)

        cached = mgr.get_cached()
        assert cached[0] is not None  # q_fp8
        assert cached[3] is not None  # q_scale

        out.float().mean().backward()

        assert (
            mgr.get_do_scale() is not None
        ), "dO scale should be cached after backward"

    def test_blockwise2d_do_scale_reuse_across_iterations(self):
        """dO scale from first backward should be available in second iteration."""
        from atomrl.lumen.ops.attention.attention import attention_fp8_quant
        from atomrl.lumen.quantize.scaling_manager import Blockwise2DScaleManager

        mgr = Blockwise2DScaleManager(block_m=64, block_n=64)
        sm_scale = 64**-0.5

        for iteration in range(2):
            torch.cuda.manual_seed(42 + iteration)
            q = (
                torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
            ).requires_grad_(True)
            k = (
                torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
            ).requires_grad_(True)
            v = (
                torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
            ).requires_grad_(True)

            out = attention_fp8_quant(
                q,
                k,
                v,
                softmax_scale=sm_scale,
                causal=False,
                backend_type="aiter_triton_fp8",
                quant_type="blockwise2d",
                scale_manager=mgr,
            )
            out.float().mean().backward()

            if iteration == 0:
                assert mgr.get_do_scale() is not None
                first_do_scale = mgr.get_do_scale().clone()
            else:
                current_do_scale = mgr.get_do_scale()
                assert current_do_scale is not None
                assert current_do_scale.shape == first_do_scale.shape

    def test_blockwise2d_causal_forward_snr(self):
        """blockwise2d with causal mask should match BF16 golden."""
        from atomrl.lumen.modules.attention import LumenAttention

        sm_scale = 64**-0.5
        config = AttnConfig(128, 128, 8, 8, 64, 64)
        attn = LumenAttention(
            quant_type="blockwise2d", softmax_scale=sm_scale, causal=True
        )
        q, k, v = _make_tensors(config, 2, torch.bfloat16, "cuda")
        out = attn(q, k, v)
        assert out.shape == (2, 128, 8, 64)

        out_ref = attention_ref(q, k, v, sm_scale, causal=True)
        snr = compute_snr(out_ref, out)
        assert snr > 12, f"blockwise2d causal fwd SNR: {snr:.1f} dB"


class TestBlockwiseRecomputeOptimization:
    """Tests for blockwise forward/backward recompute with custom block sizes."""

    def test_custom_block_sizes_forward_snr(self):
        from atomrl.lumen.ops.attention.attention import attention_fp8_quant

        sm_scale = 64**-0.5
        config = AttnConfig(128, 128, 4, 4, 64, 64)

        for block_m, block_n in [(32, 32), (128, 128)]:
            q, k, v = _make_tensors(config, 1, torch.bfloat16, "cuda")
            out_ref = attention_ref(q, k, v, sm_scale)
            out = attention_fp8_quant(
                q,
                k,
                v,
                softmax_scale=sm_scale,
                causal=False,
                backend_type="aiter_triton_fp8",
                quant_type="mxfp8",
                block_m_fwd=block_m,
                block_n_fwd=block_n,
            )
            snr = compute_snr(out_ref, out)
            assert (
                snr > 10
            ), f"block_m={block_m} block_n={block_n} fwd SNR: {snr:.1f} dB"

    def test_asymmetric_bwd_block_sizes(self):
        from atomrl.lumen.ops.attention.attention import attention_fp8_quant

        sm_scale = 64**-0.5

        torch.cuda.manual_seed(0)
        q_ref = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k_ref = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v_ref = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        out_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale)
        out_ref.float().mean().backward()

        torch.cuda.manual_seed(0)
        q = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        out = attention_fp8_quant(
            q,
            k,
            v,
            softmax_scale=sm_scale,
            causal=False,
            backend_type="aiter_triton_fp8",
            quant_type="mxfp8",
            block_m_dq_bwd=32,
            block_n_dq_bwd=64,
            block_m_dkv_bwd=64,
            block_n_dkv_bwd=32,
        )
        out.float().mean().backward()

        assert q.grad is not None and k.grad is not None and v.grad is not None
        dq_snr = compute_snr(q_ref.grad, q.grad)
        dk_snr = compute_snr(k_ref.grad, k.grad)
        dv_snr = compute_snr(v_ref.grad, v.grad)
        assert dq_snr > 6, f"asymmetric dQ SNR: {dq_snr:.1f} dB"
        assert dk_snr > 6, f"asymmetric dK SNR: {dk_snr:.1f} dB"
        assert dv_snr > 6, f"asymmetric dV SNR: {dv_snr:.1f} dB"

    @pytest.mark.parametrize("quant_block_size", [64, 128, 256])
    def test_quant_block_size_variations(self, quant_block_size):
        from atomrl.lumen.ops.attention.attention import attention_fp8_quant

        head_dim = 256
        seqlen = 256
        sm_scale = head_dim**-0.5

        torch.cuda.manual_seed(0)
        q_ref = (
            torch.randn(1, seqlen, 4, head_dim, device="cuda", dtype=torch.bfloat16)
            * 0.02
        ).requires_grad_(True)
        k_ref = (
            torch.randn(1, seqlen, 4, head_dim, device="cuda", dtype=torch.bfloat16)
            * 0.02
        ).requires_grad_(True)
        v_ref = (
            torch.randn(1, seqlen, 4, head_dim, device="cuda", dtype=torch.bfloat16)
            * 0.02
        ).requires_grad_(True)
        out_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale)
        out_ref.float().mean().backward()

        torch.cuda.manual_seed(0)
        q = (
            torch.randn(1, seqlen, 4, head_dim, device="cuda", dtype=torch.bfloat16)
            * 0.02
        ).requires_grad_(True)
        k = (
            torch.randn(1, seqlen, 4, head_dim, device="cuda", dtype=torch.bfloat16)
            * 0.02
        ).requires_grad_(True)
        v = (
            torch.randn(1, seqlen, 4, head_dim, device="cuda", dtype=torch.bfloat16)
            * 0.02
        ).requires_grad_(True)
        out = attention_fp8_quant(
            q,
            k,
            v,
            softmax_scale=sm_scale,
            causal=False,
            backend_type="aiter_triton_fp8",
            quant_type="mxfp8",
            quant_block_size=quant_block_size,
            block_m_fwd=quant_block_size,
            block_n_fwd=quant_block_size,
            block_m_dq_bwd=quant_block_size,
            block_n_dq_bwd=quant_block_size,
            block_m_dkv_bwd=quant_block_size,
            block_n_dkv_bwd=quant_block_size,
        )
        out.float().mean().backward()

        assert q.grad is not None and k.grad is not None and v.grad is not None
        dq_snr = compute_snr(q_ref.grad, q.grad)
        dk_snr = compute_snr(k_ref.grad, k.grad)
        dv_snr = compute_snr(v_ref.grad, v.grad)
        out_snr = compute_snr(out_ref, out)
        assert (
            out_snr > 10
        ), f"quant_block_size={quant_block_size} fwd SNR: {out_snr:.1f} dB"
        assert (
            dq_snr > 6
        ), f"quant_block_size={quant_block_size} dQ SNR: {dq_snr:.1f} dB"
        assert (
            dk_snr > 6
        ), f"quant_block_size={quant_block_size} dK SNR: {dk_snr:.1f} dB"
        assert (
            dv_snr > 6
        ), f"quant_block_size={quant_block_size} dV SNR: {dv_snr:.1f} dB"

    def test_blockwise2d_multi_iteration_scale_convergence(self):
        from atomrl.lumen.ops.attention.attention import attention_fp8_quant
        from atomrl.lumen.quantize.scaling_manager import Blockwise2DScaleManager

        mgr = Blockwise2DScaleManager(block_m=64, block_n=64)
        sm_scale = 64**-0.5
        snrs = []

        for iteration in range(3):
            torch.cuda.manual_seed(42 + iteration)
            q = (
                torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
            ).requires_grad_(True)
            k = (
                torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
            ).requires_grad_(True)
            v = (
                torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
            ).requires_grad_(True)

            out_ref = attention_ref(q, k, v, sm_scale)
            out = attention_fp8_quant(
                q,
                k,
                v,
                softmax_scale=sm_scale,
                causal=False,
                backend_type="aiter_triton_fp8",
                quant_type="blockwise2d",
                scale_manager=mgr,
            )
            snr = compute_snr(out_ref, out)
            snrs.append(snr)

            if iteration > 0:
                assert mgr.get_do_scale() is not None
            out.float().mean().backward()
            assert mgr.get_do_scale() is not None

        assert all(s > 10 for s in snrs), f"SNRs across iterations: {snrs}"

    def test_blockwise2d_with_grad_quant_type(self):
        from atomrl.lumen.ops.attention.attention import attention_fp8_quant

        sm_scale = 64**-0.5

        torch.cuda.manual_seed(0)
        q_ref = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k_ref = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v_ref = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        out_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale)
        out_ref.float().mean().backward()

        torch.cuda.manual_seed(0)
        q = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        k = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        v = (
            torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)
        out = attention_fp8_quant(
            q,
            k,
            v,
            softmax_scale=sm_scale,
            causal=False,
            backend_type="aiter_triton_fp8",
            quant_type="blockwise2d",
            grad_quant_type="fp8",
        )
        out.float().mean().backward()

        assert q.grad is not None and k.grad is not None and v.grad is not None
        dq_snr = compute_snr(q_ref.grad, q.grad)
        dk_snr = compute_snr(k_ref.grad, k.grad)
        dv_snr = compute_snr(v_ref.grad, v.grad)
        assert dq_snr > 6, f"blockwise2d+grad fp8 dQ SNR: {dq_snr:.1f} dB"
        assert dk_snr > 6, f"blockwise2d+grad fp8 dK SNR: {dk_snr:.1f} dB"
        assert dv_snr > 6, f"blockwise2d+grad fp8 dV SNR: {dv_snr:.1f} dB"

    def test_blockwise_recompute_throughput(self):
        from atomrl.lumen.ops.attention.attention import attention_fp8_quant

        sm_scale = 64**-0.5
        config = AttnConfig(128, 128, 4, 4, 64, 64)
        q, k, v = _make_tensors(config, 1, torch.bfloat16, "cuda", requires_grad=True)

        for _ in range(3):
            out = attention_fp8_quant(
                q,
                k,
                v,
                softmax_scale=sm_scale,
                causal=False,
                backend_type="aiter_triton_fp8",
                quant_type="mxfp8",
            )
            out.float().mean().backward()
            q.grad = None
            k.grad = None
            v.grad = None
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 10

        start.record()
        for _ in range(iters):
            out = attention_fp8_quant(
                q,
                k,
                v,
                softmax_scale=sm_scale,
                causal=False,
                backend_type="aiter_triton_fp8",
                quant_type="mxfp8",
            )
            out.float().mean().backward()
            q.grad = None
            k.grad = None
            v.grad = None
        end.record()
        torch.cuda.synchronize()
        default_ms = start.elapsed_time(end) / iters

        q2, k2, v2 = _make_tensors(
            config, 1, torch.bfloat16, "cuda", requires_grad=True
        )
        for _ in range(3):
            out = attention_fp8_quant(
                q2,
                k2,
                v2,
                softmax_scale=sm_scale,
                causal=False,
                backend_type="aiter_triton_fp8",
                quant_type="mxfp8",
                block_m_fwd=32,
                block_n_fwd=32,
            )
            out.float().mean().backward()
            q2.grad = None
            k2.grad = None
            v2.grad = None
        torch.cuda.synchronize()

        start.record()
        for _ in range(iters):
            out = attention_fp8_quant(
                q2,
                k2,
                v2,
                softmax_scale=sm_scale,
                causal=False,
                backend_type="aiter_triton_fp8",
                quant_type="mxfp8",
                block_m_fwd=32,
                block_n_fwd=32,
            )
            out.float().mean().backward()
            q2.grad = None
            k2.grad = None
            v2.grad = None
        end.record()
        torch.cuda.synchronize()
        custom_ms = start.elapsed_time(end) / iters

        print(
            f"\n[BlockwiseRecompute] default: {default_ms:.2f}ms, custom (32,32): {custom_ms:.2f}ms"
        )


class TestLumenAttentionBenchmark:
    """Attention forward throughput benchmarks."""

    @pytest.mark.parametrize(
        "seqlen,heads,dim",
        [(512, 8, 64), (1024, 8, 128), (2048, 32, 128), (4096, 32, 128)],
    )
    def test_triton_forward_throughput(self, seqlen, heads, dim):
        from atomrl.lumen.modules.attention import LumenAttention

        attn = LumenAttention(backend_type="aiter_triton", softmax_scale=dim**-0.5)
        config = AttnConfig(seqlen, seqlen, heads, heads, dim, dim)
        q, k, v = _make_tensors(config, 1, torch.bfloat16, "cuda")

        for _ in range(3):
            attn(q, k, v)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 10

        start.record()
        for _ in range(iters):
            attn(q, k, v)
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / iters
        flops = 4 * seqlen * seqlen * heads * dim
        tflops = (flops / (avg_ms / 1000.0)) / 1e12
        print(
            f"\n[Attn] seq={seqlen} heads={heads} dim={dim}: {avg_ms:.2f}ms, {tflops:.1f} TFLOPS"
        )
