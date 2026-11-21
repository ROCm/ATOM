# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

pytest.importorskip("triton")

DEVICE = "cuda"
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ── Pure-PyTorch references (test-only) ─────────────────────────────────────


def _manual_moe_reference(
    hidden_states, expert_weights, topk_ids, topk_weights, num_experts, k
):
    """Reference: per-expert matmul without fusion.

    AITER layout: expert_weights is [E, N, K] where N = intermediate_dim,
    K = hidden_dim.  The kernel computes ``x @ W.T`` per expert.

    hidden_states:   [num_tokens, hidden_dim]
    expert_weights:  [num_experts, intermediate_dim, hidden_dim]
    topk_ids:        [num_tokens, k]
    topk_weights:    [num_tokens, k]

    Returns: [num_tokens, k, intermediate_dim]
    """
    num_tokens = hidden_states.shape[0]
    intermediate_dim = expert_weights.shape[1]
    out = torch.zeros(
        num_tokens,
        k,
        intermediate_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    for i in range(num_tokens):
        for j in range(k):
            eid = topk_ids[i, j].item()
            w = topk_weights[i, j]
            # W is [N, K], matmul: [K] @ [N, K].T = [N]
            out[i, j] = w * (hidden_states[i] @ expert_weights[eid].T)

    return out


# ── Assertion / probe tests ────────────────────────────────────────────────


class TestAssertions:
    """Verify assertion guards."""


class TestConfig:
    """Verify default configuration."""

    def test_default_config_values(self):
        from atomrl.lumen.ops.moe.fused_moe import _DEFAULT_MOE_CONFIG

        assert _DEFAULT_MOE_CONFIG["BLOCK_SIZE_M"] == 64
        assert _DEFAULT_MOE_CONFIG["BLOCK_SIZE_N"] == 64
        assert _DEFAULT_MOE_CONFIG["BLOCK_SIZE_K"] == 32
        assert _DEFAULT_MOE_CONFIG["GROUP_SIZE_M"] == 8


class TestAPISignature:
    """Verify public API shape."""

    def test_fused_moe_triton_signature(self):
        import inspect

        from atomrl.lumen.ops.moe.fused_moe import fused_moe_triton

        params = list(inspect.signature(fused_moe_triton).parameters.keys())
        for name in [
            "hidden_states",
            "expert_weights",
            "topk_ids",
            "topk_weights",
            "num_experts",
            "k",
            "use_fp8",
            "config",
        ]:
            assert name in params
        assert "block_size" not in params

    def test_align_tokens_callable(self):
        from atomrl.lumen.ops.moe.fused_moe import _align_tokens

        assert callable(_align_tokens)


class TestExports:
    """Verify module exports."""

    def test_fused_moe_triton_exported(self):
        from atomrl.lumen.ops.moe import fused_moe_triton

        assert callable(fused_moe_triton)

    def test_routing_functions_still_exported(self):
        from atomrl.lumen.ops.moe import fused_permute, fused_topk, fused_unpermute

        assert callable(fused_topk)
        assert callable(fused_permute)
        assert callable(fused_unpermute)


# ── Numerical correctness tests (require CUDA + AITER) ─────────────────────


@_CUDA
class TestFusedMoeCorrectness:
    """Numerical correctness against manual per-expert matmul."""

    @pytest.mark.parametrize(
        "num_tokens,hidden_dim,intermediate_dim,num_experts,k",
        [
            (8, 64, 128, 4, 2),
            (16, 128, 256, 8, 2),
            (4, 32, 64, 4, 1),
        ],
    )
    def test_matches_reference(
        self, num_tokens, hidden_dim, intermediate_dim, num_experts, k
    ):
        from atomrl.lumen.ops.moe.fused_moe import fused_moe_triton

        hidden = torch.randn(
            num_tokens, hidden_dim, device=DEVICE, dtype=torch.bfloat16
        )
        expert_w = torch.randn(
            num_experts,
            intermediate_dim,
            hidden_dim,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, k), device=DEVICE, dtype=torch.int32
        )
        topk_weights = torch.softmax(
            torch.randn(num_tokens, k, device=DEVICE), dim=-1
        ).float()

        out = fused_moe_triton(hidden, expert_w, topk_ids, topk_weights, num_experts, k)
        ref = _manual_moe_reference(
            hidden.float(), expert_w.float(), topk_ids, topk_weights, num_experts, k
        )

        assert out.shape == (num_tokens, k, intermediate_dim)
        # Relaxed tolerance: bf16 GEMM accumulates in reduced precision;
        # triton fused kernel may fuse multiply-add differently than PyTorch.
        torch.testing.assert_close(out.float(), ref, atol=0.1, rtol=0.1)

    def test_output_shape(self):
        from atomrl.lumen.ops.moe.fused_moe import fused_moe_triton

        num_tokens, hidden_dim, intermediate_dim, num_experts, k = 8, 64, 128, 4, 2
        hidden = torch.randn(
            num_tokens, hidden_dim, device=DEVICE, dtype=torch.bfloat16
        )
        expert_w = torch.randn(
            num_experts,
            intermediate_dim,
            hidden_dim,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, k), device=DEVICE, dtype=torch.int32
        )
        topk_weights = torch.softmax(
            torch.randn(num_tokens, k, device=DEVICE), dim=-1
        ).float()

        out = fused_moe_triton(hidden, expert_w, topk_ids, topk_weights, num_experts, k)
        assert out.shape == (num_tokens, k, intermediate_dim)
        assert out.dtype == torch.bfloat16

    def test_zero_weights_give_zero_output(self):
        from atomrl.lumen.ops.moe.fused_moe import fused_moe_triton

        num_tokens, hidden_dim, intermediate_dim, num_experts, k = 4, 64, 128, 4, 2
        hidden = torch.randn(
            num_tokens, hidden_dim, device=DEVICE, dtype=torch.bfloat16
        )
        expert_w = torch.randn(
            num_experts,
            intermediate_dim,
            hidden_dim,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        topk_ids = torch.zeros(num_tokens, k, device=DEVICE, dtype=torch.int32)
        topk_weights = torch.zeros(num_tokens, k, device=DEVICE)

        out = fused_moe_triton(hidden, expert_w, topk_ids, topk_weights, num_experts, k)
        torch.testing.assert_close(
            out.float(),
            torch.zeros_like(out, dtype=torch.float32),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_single_expert_equals_matmul(self):
        """With k=1 and weight=1, should equal x @ W.T."""
        from atomrl.lumen.ops.moe.fused_moe import fused_moe_triton

        num_tokens, hidden_dim, intermediate_dim, num_experts = 8, 64, 128, 4
        hidden = torch.randn(
            num_tokens, hidden_dim, device=DEVICE, dtype=torch.bfloat16
        )
        expert_w = torch.randn(
            num_experts,
            intermediate_dim,
            hidden_dim,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        expert_id = 2
        topk_ids = torch.full(
            (num_tokens, 1), expert_id, device=DEVICE, dtype=torch.int32
        )
        topk_weights = torch.ones(num_tokens, 1, device=DEVICE)

        out = fused_moe_triton(hidden, expert_w, topk_ids, topk_weights, num_experts, 1)
        ref = (hidden.float() @ expert_w[expert_id].float().T).unsqueeze(1)

        assert out.shape == (num_tokens, 1, intermediate_dim)
        torch.testing.assert_close(out.float(), ref, atol=0.1, rtol=0.1)

    def test_mul_routed_weight_false(self):
        """With mul_routed_weight=False, output should be unscaled x @ W.T."""
        from atomrl.lumen.ops.moe.fused_moe import fused_moe_triton

        num_tokens, hidden_dim, intermediate_dim, num_experts = 8, 64, 128, 4
        hidden = torch.randn(
            num_tokens, hidden_dim, device=DEVICE, dtype=torch.bfloat16
        )
        expert_w = torch.randn(
            num_experts,
            intermediate_dim,
            hidden_dim,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        expert_id = 1
        topk_ids = torch.full(
            (num_tokens, 1), expert_id, device=DEVICE, dtype=torch.int32
        )
        topk_weights = torch.full((num_tokens, 1), 0.5, device=DEVICE)

        out = fused_moe_triton(
            hidden,
            expert_w,
            topk_ids,
            topk_weights,
            num_experts,
            1,
            mul_routed_weight=False,
        )
        ref = (hidden.float() @ expert_w[expert_id].float().T).unsqueeze(1)

        assert out.shape == (num_tokens, 1, intermediate_dim)
        torch.testing.assert_close(out.float(), ref, atol=0.1, rtol=0.1)

    def test_finite_output(self):
        from atomrl.lumen.ops.moe.fused_moe import fused_moe_triton

        num_tokens, hidden_dim, intermediate_dim, num_experts, k = 16, 128, 256, 8, 2
        hidden = torch.randn(
            num_tokens, hidden_dim, device=DEVICE, dtype=torch.bfloat16
        )
        expert_w = torch.randn(
            num_experts,
            intermediate_dim,
            hidden_dim,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, k), device=DEVICE, dtype=torch.int32
        )
        topk_weights = torch.softmax(
            torch.randn(num_tokens, k, device=DEVICE), dim=-1
        ).float()

        out = fused_moe_triton(hidden, expert_w, topk_ids, topk_weights, num_experts, k)
        assert torch.isfinite(out).all()


@_CUDA
class TestAlignTokensCorrectness:
    """Test _align_tokens helper correctness."""

    def test_output_shapes(self):
        from atomrl.lumen.ops.moe.fused_moe import _align_tokens

        num_tokens, k, num_experts, block_size = 8, 2, 4, 128
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, k), device=DEVICE, dtype=torch.int32
        )

        sorted_ids, expert_ids, num_tokens_post_pad = _align_tokens(
            topk_ids, num_experts, block_size
        )

        assert sorted_ids.dtype == torch.int32
        assert expert_ids.dtype == torch.int32
        assert num_tokens_post_pad.dtype == torch.int32
        assert num_tokens_post_pad.shape == (1,)

    def test_post_pad_at_least_numel(self):
        """Padded count should be >= original count."""
        from atomrl.lumen.ops.moe.fused_moe import _align_tokens

        num_tokens, k, num_experts, block_size = 16, 2, 4, 64
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, k), device=DEVICE, dtype=torch.int32
        )

        _, _, num_tokens_post_pad = _align_tokens(topk_ids, num_experts, block_size)
        assert num_tokens_post_pad.item() >= num_tokens * k

    def test_post_pad_aligned_to_block_size(self):
        from atomrl.lumen.ops.moe.fused_moe import _align_tokens

        block_size = 64
        topk_ids = torch.randint(0, 4, (8, 2), device=DEVICE, dtype=torch.int32)
        _, _, num_tokens_post_pad = _align_tokens(topk_ids, 4, block_size)
        assert num_tokens_post_pad.item() % block_size == 0
