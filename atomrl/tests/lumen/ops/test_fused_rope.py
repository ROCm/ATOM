# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import pytest
import torch

DEVICE = "cuda"
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ── Pure-PyTorch reference (test-only, not part of Lumen) ───────────────────


def _rope_reference_neox(x, cos, sin):
    """Pure-PyTorch NeoX-style RoPE: split half, rotate, concat.

    x:   [B, H, S, D]
    cos: [S, D//2]
    sin: [S, D//2]
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D//2]
    s = sin.unsqueeze(0).unsqueeze(0)
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return torch.cat((out1, out2), dim=-1)


def _make_cos_sin(seq_len, head_dim, device="cpu"):
    """Generate cos/sin frequency tables."""
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    dim = torch.arange(0, head_dim, 2, device=device).float()
    freq = 1.0 / (10000.0 ** (dim / head_dim))
    angles = pos * freq
    return torch.cos(angles), torch.sin(angles)


# ── Assertion / probe tests (run without CUDA) ─────────────────────────────


class TestAssertions:
    """Verify that functions assert when AITER is unavailable."""

    def test_apply_rotary_requires_aiter(self):
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb

        x = torch.randn(2, 4, 16, 64)
        cos, sin = _make_cos_sin(16, 64)
        with pytest.raises(AssertionError):
            apply_rotary_pos_emb(x, cos, sin)

    def test_apply_rotary_rejects_non_4d(self):
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb

        x = torch.randn(16, 64)
        cos, sin = _make_cos_sin(16, 64)
        with pytest.raises(AssertionError):
            apply_rotary_pos_emb(x, cos, sin)

    def test_fused_rope_requires_aiter(self):
        from atomrl.lumen.ops.rope import fused_rope

        q = torch.randn(2, 4, 16, 64)
        k = torch.randn(2, 4, 16, 64)
        cos, sin = _make_cos_sin(16, 64)
        with pytest.raises(AssertionError):
            fused_rope(q, k, cos, sin)

    def test_rope_2d_requires_aiter(self):
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb_2d

        x = torch.randn(2, 64, 4, 32)
        cos_h = sin_h = cos_w = sin_w = torch.randn(8, 16)
        with pytest.raises(AssertionError):
            apply_rotary_pos_emb_2d(x, cos_h, sin_h, cos_w, sin_w, 8, 8)

    def test_rope_3d_requires_aiter(self):
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb_3d

        x = torch.randn(1, 32, 4, 64)
        grid_sizes = torch.tensor([[4, 4, 2]])
        freqs = torch.randn(32, 32, dtype=torch.cfloat)
        with pytest.raises(AssertionError):
            apply_rotary_pos_emb_3d(x, grid_sizes, freqs)


class TestLayoutAdapters:
    """Test BHSD <-> SBHD layout conversion."""

    def test_roundtrip(self):
        from atomrl.lumen.ops.rope import _bhsd_to_sbhd, _sbhd_to_bhsd

        B, H, S, D = 2, 4, 16, 64
        x = torch.randn(B, H, S, D)
        x_sbhd = _bhsd_to_sbhd(x)
        assert x_sbhd.shape == (S, B, H, D)
        x_back = _sbhd_to_bhsd(x_sbhd)
        torch.testing.assert_close(x, x_back)

    def test_contiguous(self):
        from atomrl.lumen.ops.rope import _bhsd_to_sbhd, _sbhd_to_bhsd

        x = torch.randn(2, 4, 16, 64)
        assert _bhsd_to_sbhd(x).is_contiguous()
        assert _sbhd_to_bhsd(_bhsd_to_sbhd(x)).is_contiguous()


# ── Numerical correctness tests (require CUDA + AITER) ─────────────────────


@_CUDA
class TestRoPECorrectness:
    """Numerical correctness against pure-PyTorch reference."""

    @pytest.mark.parametrize(
        "B,H,S,D", [(1, 1, 8, 32), (2, 4, 16, 64), (1, 8, 128, 128)]
    )
    def test_apply_rotary_matches_reference(self, B, H, S, D):
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb

        x = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)
        cos, sin = _make_cos_sin(S, D, device=DEVICE)

        out = apply_rotary_pos_emb(x, cos, sin)
        ref = _rope_reference_neox(x, cos, sin)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_norm_preservation(self):
        """RoPE is a rotation — L2 norm per vector should be preserved."""
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb

        B, H, S, D = 2, 4, 32, 64
        x = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)
        cos, sin = _make_cos_sin(S, D, device=DEVICE)

        out = apply_rotary_pos_emb(x, cos, sin)
        torch.testing.assert_close(
            x.norm(dim=-1),
            out.norm(dim=-1),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_identity_at_zero_angles(self):
        """cos=1, sin=0 should give identity."""
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb

        B, H, S, D = 2, 4, 16, 64
        x = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)
        cos = torch.ones(S, D // 2, device=DEVICE)
        sin = torch.zeros(S, D // 2, device=DEVICE)

        out = apply_rotary_pos_emb(x, cos, sin)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_double_rotation_matches_sum(self):
        """Applying RoPE with angle θ then angle φ == applying once with θ+φ."""
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb

        B, H, S, D = 1, 2, 16, 64
        x = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)

        theta = torch.linspace(0, 1, S * (D // 2), device=DEVICE).reshape(S, D // 2)
        phi = torch.linspace(0, 0.5, S * (D // 2), device=DEVICE).reshape(S, D // 2)

        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        cos_p, sin_p = torch.cos(phi), torch.sin(phi)
        cos_tp = torch.cos(theta + phi)
        sin_tp = torch.sin(theta + phi)

        out_two_steps = apply_rotary_pos_emb(
            apply_rotary_pos_emb(x, cos_t, sin_t),
            cos_p,
            sin_p,
        )
        out_one_step = apply_rotary_pos_emb(x, cos_tp, sin_tp)

        torch.testing.assert_close(out_two_steps, out_one_step, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_precision(self, dtype):
        """Verify correctness in half-precision (common in training)."""
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb

        B, H, S, D = 2, 4, 16, 64
        x = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
        cos, sin = _make_cos_sin(S, D, device=DEVICE)

        out = apply_rotary_pos_emb(x, cos, sin)
        ref = _rope_reference_neox(x.float(), cos, sin).to(dtype)

        # Relaxed tolerance: bf16 mantissa is 7 bits → ~1e-2 rounding;
        # kernel may accumulate in reduced precision internally.
        torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)


@_CUDA
class TestFusedRoPECorrectness:
    """Verify fused_rope applies identical rotation to Q and K."""

    def test_q_k_match_individual(self):
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb, fused_rope

        B, H, S, D = 2, 4, 16, 64
        q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)
        cos, sin = _make_cos_sin(S, D, device=DEVICE)

        q_rot, k_rot = fused_rope(q, k, cos, sin)
        q_ref = apply_rotary_pos_emb(q, cos, sin)
        k_ref = apply_rotary_pos_emb(k, cos, sin)

        torch.testing.assert_close(q_rot, q_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_rot, k_ref, atol=1e-5, rtol=1e-5)

    def test_gqa_different_num_heads(self):
        """Q and K can have different head counts (GQA)."""
        from atomrl.lumen.ops.rope import fused_rope

        B, S, D = 2, 16, 64
        q = torch.randn(B, 8, S, D, device=DEVICE)
        k = torch.randn(B, 2, S, D, device=DEVICE)
        cos, sin = _make_cos_sin(S, D, device=DEVICE)

        q_rot, k_rot = fused_rope(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert torch.isfinite(q_rot).all()
        assert torch.isfinite(k_rot).all()

    def test_interleaved_gptj_style(self):
        """Smoke-test interleaved=True (GPT-J style rotation)."""
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb

        B, H, S, D = 2, 4, 16, 64
        x = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)
        cos, sin = _make_cos_sin(S, D, device=DEVICE)

        out = apply_rotary_pos_emb(x, cos, sin, interleaved=True)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        # Norm should still be preserved (it's a rotation in a different basis)
        torch.testing.assert_close(
            x.norm(dim=-1),
            out.norm(dim=-1),
            atol=1e-4,
            rtol=1e-4,
        )


# ── HF-style reference for the autograd path (full cos/sin [S, D]) ──────────


def _rope_hf_full(x, cos, sin):
    """HF reference: x*cos + rotate_half(x)*sin. cos/sin are full [S, D]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos[None, None] + rot * sin[None, None]


def _make_cos_sin_full(seq_len, head_dim, device="cpu"):
    """Full [S, D] cos/sin (NeoX: front half duplicated), as HF produces."""
    cos_h, sin_h = _make_cos_sin(seq_len, head_dim, device=device)  # [S, D/2]
    return torch.cat([cos_h, cos_h], dim=-1), torch.cat([sin_h, sin_h], dim=-1)


@_CUDA
class TestRoPEAutograd:
    """apply_rotary_qk_autograd: zero-copy fwd + bwd vs HF reference.

    The training path feeds AITER's stride-aware rope kernel BHSD-view inputs and
    has it write a contiguous BHSD output (no permute().contiguous() round trip),
    so verify fwd + dQ/dK match the HF reference and outputs/grads are contiguous.
    """

    @pytest.mark.parametrize("Hq,Hk", [(4, 4), (8, 2)])  # MHA and GQA
    def test_fwd_bwd_matches_reference(self, Hq, Hk):
        from atomrl.lumen.ops.rope import apply_rotary_qk_autograd

        B, S, D = 1, 64, 128
        cos, sin = _make_cos_sin_full(S, D, device=DEVICE)

        q = torch.randn(B, Hq, S, D, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, Hk, S, D, device=DEVICE, dtype=torch.float32)
        go_q, go_k = torch.randn_like(q), torch.randn_like(k)

        # reference
        qr = q.clone().requires_grad_(True)
        kr = k.clone().requires_grad_(True)
        (_rope_hf_full(qr, cos, sin) * go_q).sum().backward()
        (_rope_hf_full(kr, cos, sin) * go_k).sum().backward()

        # lumen autograd (zero-copy)
        ql = q.clone().requires_grad_(True)
        kl = k.clone().requires_grad_(True)
        oq, ok = apply_rotary_qk_autograd(ql, kl, cos, sin)
        (oq * go_q).sum().backward()
        (ok * go_k).sum().backward()

        torch.testing.assert_close(oq, _rope_hf_full(q, cos, sin), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(ok, _rope_hf_full(k, cos, sin), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(ql.grad, qr.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(kl.grad, kr.grad, atol=1e-4, rtol=1e-4)

    def test_outputs_and_grads_contiguous(self):
        """Zero-copy path must still hand attention/optimizer contiguous tensors."""
        from atomrl.lumen.ops.rope import apply_rotary_qk_autograd

        B, Hq, Hk, S, D = 1, 8, 2, 64, 128
        cos, sin = _make_cos_sin_full(S, D, device=DEVICE)
        q = torch.randn(B, Hq, S, D, device=DEVICE, requires_grad=True)
        k = torch.randn(B, Hk, S, D, device=DEVICE, requires_grad=True)

        oq, ok = apply_rotary_qk_autograd(q, k, cos, sin)
        assert oq.is_contiguous() and ok.is_contiguous()
        (oq.sum() + ok.sum()).backward()
        assert q.grad.is_contiguous() and k.grad.is_contiguous()

    def test_bf16(self):
        from atomrl.lumen.ops.rope import apply_rotary_qk_autograd

        B, Hq, Hk, S, D = 1, 8, 2, 64, 128
        cos, sin = _make_cos_sin_full(S, D, device=DEVICE)
        q = torch.randn(
            B, Hq, S, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            B, Hk, S, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True
        )

        oq, ok = apply_rotary_qk_autograd(q, k, cos, sin)
        ref_q = _rope_hf_full(q.float(), cos.float(), sin.float()).bfloat16()
        torch.testing.assert_close(oq, ref_q, atol=5e-2, rtol=5e-2)
        (oq.sum() + ok.sum()).backward()
        assert q.grad is not None and k.grad is not None
