# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################
"""
Tests for attention features with zero prior test coverage:
  - Sliding window attention (window_size parameter, csrc backend only)
  - Attention bias (arbitrary bias tensor)
  - ALiBi slopes (linear position bias)

Reference: attention_ref from conftest (pure PyTorch BSHD).

SNR thresholds (empirical, MI300X):
  - BF16 with bias/alibi:  >= 18 dB (bias adds a few ops vs vanilla attention)
  - Sliding window fwd:    >= 18 dB (csrc backend)
"""

import math

import atomrl.lumen.ops.attention as attn_ops
import pytest
import torch
from conftest import AttnConfig, compute_snr

# ---------------------------------------------------------------------------
# Hardware / backend detection
# ---------------------------------------------------------------------------


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_AITER = _CUDA
_CSRC = _CUDA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensors(
    config: AttnConfig,
    batch_size=2,
    dtype=torch.bfloat16,
    device="cuda",
    requires_grad=False,
):
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
    sm_scale = config.head_dim_qk**-0.5
    return q, k, v, sm_scale


def _attention_ref_with_bias(q, k, v, sm_scale, bias=None, causal=False):
    """Pure-PyTorch reference that adds an explicit bias to attention scores."""
    B, SQ, HQ, D = q.shape
    _, SK, HK, _ = k.shape
    DV = v.shape[-1]
    gqa_ratio = HQ // HK

    q_f32, k_f32, v_f32 = q.float(), k.float(), v.float()
    if gqa_ratio > 1:
        k_f32 = k_f32.unsqueeze(3).expand(B, SK, HK, gqa_ratio, D).reshape(B, SK, HQ, D)
        v_f32 = (
            v_f32.unsqueeze(3).expand(B, SK, HK, gqa_ratio, DV).reshape(B, SK, HQ, DV)
        )

    q_t = q_f32.transpose(1, 2)
    k_t = k_f32.transpose(1, 2)
    v_t = v_f32.transpose(1, 2)

    attn = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale

    if bias is not None:
        attn = attn + bias.float()

    if causal:
        row_idx = torch.arange(SQ, device=q.device).unsqueeze(1)
        col_idx = torch.arange(SK, device=q.device).unsqueeze(0)
        col_offset = SQ - SK
        mask = row_idx >= (col_offset + col_idx)
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v_t)
    return out.transpose(1, 2).to(q.dtype)


def _attention_ref_with_window(
    q, k, v, sm_scale, window_left, window_right, causal=False
):
    """Pure-PyTorch reference with sliding window masking."""
    B, SQ, HQ, D = q.shape
    _, SK, HK, _ = k.shape
    DV = v.shape[-1]
    gqa_ratio = HQ // HK

    q_f32, k_f32, v_f32 = q.float(), k.float(), v.float()
    if gqa_ratio > 1:
        k_f32 = k_f32.unsqueeze(3).expand(B, SK, HK, gqa_ratio, D).reshape(B, SK, HQ, D)
        v_f32 = (
            v_f32.unsqueeze(3).expand(B, SK, HK, gqa_ratio, DV).reshape(B, SK, HQ, DV)
        )

    q_t = q_f32.transpose(1, 2)
    k_t = k_f32.transpose(1, 2)
    v_t = v_f32.transpose(1, 2)

    attn = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale

    row_idx = torch.arange(SQ, device=q.device).unsqueeze(1)
    col_idx = torch.arange(SK, device=q.device).unsqueeze(0)
    offset = SQ - SK

    mask = torch.ones(SQ, SK, dtype=torch.bool, device=q.device)
    if window_left >= 0:
        mask &= (row_idx + offset - col_idx) <= window_left
    if window_right >= 0:
        mask &= (col_idx - row_idx - offset) <= window_right
    if causal:
        mask &= row_idx >= (offset + col_idx)

    attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v_t)
    return out.transpose(1, 2).to(q.dtype)


def _get_alibi_slopes(nheads, device):
    """Generate ALiBi slopes for nheads.

    Returns shape (nheads,) — the standard 1-D form accepted by both AITER
    CK and Triton kernels.  Lumen's attention() expands to (B, H) internally.
    """
    closest_pow2 = 2 ** math.floor(math.log2(nheads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_pow2) - 3))),
        dtype=torch.float32,
        device=device,
    )
    powers = torch.arange(1, 1 + closest_pow2, dtype=torch.int32, device=device)
    slopes = torch.pow(base, powers.float())
    if closest_pow2 != nheads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_pow2) - 3))),
            dtype=torch.float32,
            device=device,
        )
        num_remaining = min(closest_pow2, nheads - closest_pow2)
        extra_powers = torch.arange(
            1, 1 + 2 * num_remaining, 2, dtype=torch.int32, device=device
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers.float())])
    return slopes[:nheads]


def _attention_ref_with_alibi(q, k, v, sm_scale, alibi_slopes, causal=False):
    """Pure-PyTorch reference with ALiBi position bias.

    alibi_slopes: (1, nheads) or (nheads,)
    """
    if alibi_slopes.dim() == 2:
        alibi_slopes = alibi_slopes.squeeze(0)
    B, SQ, HQ, D = q.shape
    _, SK, HK, _ = k.shape
    DV = v.shape[-1]
    gqa_ratio = HQ // HK

    q_f32, k_f32, v_f32 = q.float(), k.float(), v.float()
    if gqa_ratio > 1:
        k_f32 = k_f32.unsqueeze(3).expand(B, SK, HK, gqa_ratio, D).reshape(B, SK, HQ, D)
        v_f32 = (
            v_f32.unsqueeze(3).expand(B, SK, HK, gqa_ratio, DV).reshape(B, SK, HQ, DV)
        )

    q_t = q_f32.transpose(1, 2)
    k_t = k_f32.transpose(1, 2)
    v_t = v_f32.transpose(1, 2)

    attn = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale

    row_idx = torch.arange(SQ, device=q.device).unsqueeze(1).float()
    col_idx = torch.arange(SK, device=q.device).unsqueeze(0).float()
    offset = SQ - SK
    dist = torch.abs(row_idx + offset - col_idx)
    alibi_bias = -alibi_slopes.view(1, -1, 1, 1) * dist.unsqueeze(0).unsqueeze(0)
    attn = attn + alibi_bias

    if causal:
        row_i = torch.arange(SQ, device=q.device).unsqueeze(1)
        col_i = torch.arange(SK, device=q.device).unsqueeze(0)
        mask = row_i >= (SQ - SK + col_i)
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v_t)
    return out.transpose(1, 2).to(q.dtype)


# =========================================================================
# Sliding Window Attention (csrc backend only)
# =========================================================================

WINDOW_CONFIGS = [
    AttnConfig(128, 128, 8, 8, 64, 64),
    AttnConfig(256, 256, 8, 8, 64, 64),
]
WINDOW_IDS = [repr(c) for c in WINDOW_CONFIGS]


@_CUDA
@_CSRC
class TestSlidingWindowForward:
    """Sliding window attention forward correctness via csrc backend."""

    @pytest.mark.parametrize("config", WINDOW_CONFIGS, ids=WINDOW_IDS)
    @pytest.mark.parametrize("window_left", [31, 63, 127])
    def test_causal_sliding_window(self, config, window_left):
        q, k, v, sm_scale = _make_tensors(config)
        ref = _attention_ref_with_window(q, k, v, sm_scale, window_left, 0, causal=True)
        out = attn_ops.attention(
            q,
            k,
            v,
            softmax_scale=sm_scale,
            causal=True,
            window_size=(window_left, 0),
            backend_type="aiter_csrc",
        )
        snr = compute_snr(ref, out)
        assert snr > 18, f"Sliding window causal SNR too low: {snr:.1f} dB"

    @pytest.mark.parametrize("config", WINDOW_CONFIGS[:1], ids=WINDOW_IDS[:1])
    def test_symmetric_window(self, config):
        q, k, v, sm_scale = _make_tensors(config)
        window = 31
        ref = _attention_ref_with_window(
            q, k, v, sm_scale, window, window, causal=False
        )
        out = attn_ops.attention(
            q,
            k,
            v,
            softmax_scale=sm_scale,
            causal=False,
            window_size=(window, window),
            backend_type="aiter_csrc",
        )
        snr = compute_snr(ref, out)
        assert snr > 18, f"Symmetric sliding window SNR too low: {snr:.1f} dB"

    @pytest.mark.parametrize("config", WINDOW_CONFIGS[:1], ids=WINDOW_IDS[:1])
    def test_full_window_matches_vanilla(self, config):
        """window_size=(-1,-1) should match regular attention."""
        q, k, v, sm_scale = _make_tensors(config)
        out_full = attn_ops.attention(
            q,
            k,
            v,
            softmax_scale=sm_scale,
            causal=True,
            window_size=(-1, -1),
            backend_type="aiter_csrc",
        )
        out_vanilla = attn_ops.attention(
            q,
            k,
            v,
            softmax_scale=sm_scale,
            causal=True,
            backend_type="aiter_csrc",
        )
        torch.testing.assert_close(out_full, out_vanilla, atol=1e-6, rtol=1e-5)


@_CUDA
@_CSRC
class TestSlidingWindowBackward:

    def test_sliding_window_backward_runs(self):
        config = AttnConfig(128, 128, 8, 8, 64, 64)
        q, k, v, sm_scale = _make_tensors(config, requires_grad=True)
        out = attn_ops.attention(
            q,
            k,
            v,
            softmax_scale=sm_scale,
            causal=True,
            window_size=(63, 0),
            backend_type="aiter_csrc",
        )
        out.sum().backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None


# =========================================================================
# Attention Bias
# =========================================================================

BIAS_CONFIGS = [
    AttnConfig(128, 128, 8, 8, 64, 64),
    AttnConfig(256, 256, 16, 4, 64, 64),
]
BIAS_IDS = [repr(c) for c in BIAS_CONFIGS]


@_CUDA
@_AITER
class TestAttentionBiasForward:
    """Attention with explicit bias tensor."""

    @pytest.mark.parametrize("config", BIAS_CONFIGS, ids=BIAS_IDS)
    @pytest.mark.parametrize("backend", ["aiter_csrc", "aiter_triton"])
    def test_bias_forward_correctness(self, config, backend):
        q, k, v, sm_scale = _make_tensors(config)
        bias = (
            torch.randn(
                1,
                config.num_head_q,
                config.seqlen_q,
                config.seqlen_kv,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.1
        )
        ref = _attention_ref_with_bias(q, k, v, sm_scale, bias=bias)
        out = attn_ops.attention(
            q, k, v, softmax_scale=sm_scale, bias=bias, backend_type=backend
        )
        snr = compute_snr(ref, out)
        assert snr > 18, f"Attention bias fwd SNR too low ({backend}): {snr:.1f} dB"

    @pytest.mark.parametrize("config", BIAS_CONFIGS[:1], ids=BIAS_IDS[:1])
    def test_zero_bias_matches_no_bias(self, config):
        q, k, v, sm_scale = _make_tensors(config)
        # Use (1,1,sq,sk) so both paths hit the same CK backend (per-head
        # bias would route to Triton while no-bias stays on CK, making a
        # strict tolerance impossible due to BF16 kernel differences).
        bias = torch.zeros(
            1, 1, config.seqlen_q, config.seqlen_kv, device="cuda", dtype=torch.float32
        )
        out_bias = attn_ops.attention(q, k, v, softmax_scale=sm_scale, bias=bias)
        out_no_bias = attn_ops.attention(q, k, v, softmax_scale=sm_scale)
        torch.testing.assert_close(out_bias, out_no_bias, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("config", BIAS_CONFIGS[:1], ids=BIAS_IDS[:1])
    def test_causal_with_bias(self, config):
        q, k, v, sm_scale = _make_tensors(config)
        bias = (
            torch.randn(
                1,
                config.num_head_q,
                config.seqlen_q,
                config.seqlen_kv,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.1
        )
        ref = _attention_ref_with_bias(q, k, v, sm_scale, bias=bias, causal=True)
        out = attn_ops.attention(
            q, k, v, softmax_scale=sm_scale, bias=bias, causal=True
        )
        snr = compute_snr(ref, out)
        assert snr > 18, f"Causal attention with bias SNR too low: {snr:.1f} dB"


@_CUDA
@_AITER
class TestAttentionBiasBackward:

    def test_bias_backward_runs(self):
        config = AttnConfig(128, 128, 8, 8, 64, 64)
        q, k, v, sm_scale = _make_tensors(config, requires_grad=True)
        bias = (
            torch.randn(
                1,
                config.num_head_q,
                config.seqlen_q,
                config.seqlen_kv,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.1
        )
        out = attn_ops.attention(q, k, v, softmax_scale=sm_scale, bias=bias)
        out.sum().backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None


# =========================================================================
# ALiBi Slopes
# =========================================================================

ALIBI_CONFIGS = [
    AttnConfig(128, 128, 8, 8, 64, 64),
    AttnConfig(256, 256, 16, 4, 64, 64),
]
ALIBI_IDS = [repr(c) for c in ALIBI_CONFIGS]


@_CUDA
@_AITER
class TestALiBiForward:
    """Attention with ALiBi linear position bias."""

    @pytest.mark.parametrize("config", ALIBI_CONFIGS, ids=ALIBI_IDS)
    @pytest.mark.parametrize("backend", ["aiter_csrc", "aiter_triton"])
    def test_alibi_forward_correctness(self, config, backend):
        q, k, v, sm_scale = _make_tensors(config)
        slopes = _get_alibi_slopes(config.num_head_q, q.device)
        ref = _attention_ref_with_alibi(q, k, v, sm_scale, slopes)
        out = attn_ops.attention(
            q, k, v, softmax_scale=sm_scale, alibi_slopes=slopes, backend_type=backend
        )
        snr = compute_snr(ref, out)
        assert snr > 18, f"ALiBi fwd SNR too low ({backend}): {snr:.1f} dB"

    @pytest.mark.parametrize("config", ALIBI_CONFIGS[:1], ids=ALIBI_IDS[:1])
    def test_alibi_causal(self, config):
        q, k, v, sm_scale = _make_tensors(config)
        slopes = _get_alibi_slopes(config.num_head_q, q.device)
        ref = _attention_ref_with_alibi(q, k, v, sm_scale, slopes, causal=True)
        out = attn_ops.attention(
            q, k, v, softmax_scale=sm_scale, alibi_slopes=slopes, causal=True
        )
        snr = compute_snr(ref, out)
        assert snr > 18, f"ALiBi causal fwd SNR too low: {snr:.1f} dB"

    @pytest.mark.parametrize("config", ALIBI_CONFIGS[:1], ids=ALIBI_IDS[:1])
    def test_zero_slopes_matches_vanilla(self, config):
        q, k, v, sm_scale = _make_tensors(config)
        slopes = torch.zeros(config.num_head_q, device=q.device, dtype=torch.float32)
        out_alibi = attn_ops.attention(
            q, k, v, softmax_scale=sm_scale, alibi_slopes=slopes
        )
        out_vanilla = attn_ops.attention(q, k, v, softmax_scale=sm_scale)
        torch.testing.assert_close(out_alibi, out_vanilla, atol=1e-5, rtol=1e-4)


@_CUDA
@_AITER
class TestALiBiBackward:

    def test_alibi_backward_runs(self):
        config = AttnConfig(128, 128, 8, 8, 64, 64)
        q, k, v, sm_scale = _make_tensors(config, requires_grad=True)
        slopes = _get_alibi_slopes(config.num_head_q, q.device)
        out = attn_ops.attention(q, k, v, softmax_scale=sm_scale, alibi_slopes=slopes)
        out.sum().backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_alibi_backward_snr(self):
        config = AttnConfig(128, 128, 8, 8, 64, 64)
        slopes = _get_alibi_slopes(config.num_head_q, "cuda")

        torch.cuda.manual_seed(42)
        q_ref, k_ref, v_ref, sm_scale = _make_tensors(config, requires_grad=True)
        ref = _attention_ref_with_alibi(q_ref, k_ref, v_ref, sm_scale, slopes)
        ref.sum().backward()

        torch.cuda.manual_seed(42)
        q, k, v, _ = _make_tensors(config, requires_grad=True)
        out = attn_ops.attention(q, k, v, softmax_scale=sm_scale, alibi_slopes=slopes)
        out.sum().backward()

        assert compute_snr(q_ref.grad, q.grad) > 15, "ALiBi dQ SNR too low"
        assert compute_snr(k_ref.grad, k.grad) > 15, "ALiBi dK SNR too low"
        assert compute_snr(v_ref.grad, v.grad) > 15, "ALiBi dV SNR too low"
