# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Benchmark 4 — RoPE fusion latency reduction via Lumen AITER kernels.

Exercises all Lumen RoPE variants:

  * **apply_rotary_pos_emb**:  Standard 1D RoPE (NeoX / GPT-J style).
  * **fused_rope**:            Q+K fused RoPE (single dispatch for both).
  * **apply_rotary_pos_emb_2d**: Vision 2D RoPE (height × width).
  * **apply_rotary_pos_emb_3d**: Video 3D RoPE (temporal × spatial).
  * **GQA**:                   Fused RoPE with different Q / KV head counts.
  * **Interleaved**:           GPT-J interleaved vs NeoX half-split.

Run::

    python -m benchmarks.bench_rope_fusion
    pytest benchmarks/bench_rope_fusion.py -v -s
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.bench_utils import (
    BenchResult,
    cuda_timer,
    print_report,
    require_aiter,
    require_cuda,
)
from benchmarks.conftest import AITER, CUDA

# ---------------------------------------------------------------------------
# Default dimensions
# ---------------------------------------------------------------------------
B, H, D = 2, 32, 128
ROTARY_DIM = D


def _make_cos_sin(seqlen: int, rotary_dim: int, device="cuda", dtype=torch.bfloat16):
    """Create cos/sin frequency tables."""
    positions = torch.arange(seqlen, device=device, dtype=torch.float32)
    dim_half = rotary_dim // 2
    freqs = 1.0 / (
        10000.0
        ** (torch.arange(0, dim_half, device=device, dtype=torch.float32) / dim_half)
    )
    angles = torch.outer(positions, freqs)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _pytorch_rope_neox(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Pure PyTorch NeoX-style RoPE (decomposed into individual ops).

    This is the naive unfused implementation that the AITER Triton kernel
    replaces. Uses 4+ separate ops: slice, multiply, cat, add.
    """
    d = x.shape[-1]
    d2 = d // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos_exp = cos[: x.shape[-2], :].unsqueeze(0).unsqueeze(0)
    sin_exp = sin[: x.shape[-2], :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([-x2, x1], dim=-1)
    return (
        x * cos_exp.repeat(1, 1, 1, 2)[..., :d]
        + rotated * sin_exp.repeat(1, 1, 1, 2)[..., :d]
    )


def _pytorch_rope_2d(
    x: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    """Naive PyTorch 2D RoPE matching AITER _rope_fwd_2d_kernel_neox.

    x: [B, H*W, n_heads, D].  The kernel splits D into two halves:
      - first half rotated by height freqs (NeoX quarter-pairing)
      - second half rotated by width freqs (NeoX quarter-pairing)
    cos_h/sin_h: [img_h, D//2],  cos_w/sin_w: [img_w, D//2].
    """
    B_dim, _wh, n_heads, d = x.shape
    d2 = d // 2
    q = d2 // 2
    x_first = x[..., :d2].reshape(B_dim, img_h, img_w, n_heads, d2)
    x_second = x[..., d2:].reshape(B_dim, img_h, img_w, n_heads, d2)

    cos_h_exp = cos_h[:img_h, :].reshape(1, img_h, 1, 1, d2)
    sin_h_exp = sin_h[:img_h, :].reshape(1, img_h, 1, 1, d2)
    x_a, x_b = x_first[..., :q], x_first[..., q:]
    rot_h = torch.cat([-x_b, x_a], dim=-1)
    out_h = (x_first * cos_h_exp + rot_h * sin_h_exp).reshape(B_dim, _wh, n_heads, d2)

    cos_w_exp = cos_w[:img_w, :].reshape(1, 1, img_w, 1, d2)
    sin_w_exp = sin_w[:img_w, :].reshape(1, 1, img_w, 1, d2)
    x_c, x_d = x_second[..., :q], x_second[..., q:]
    rot_w = torch.cat([-x_d, x_c], dim=-1)
    out_w = (x_second * cos_w_exp + rot_w * sin_w_exp).reshape(B_dim, _wh, n_heads, d2)

    return torch.cat([out_h, out_w], dim=-1)


# ---------------------------------------------------------------------------
# 0. Fused AITER RoPE vs naive PyTorch RoPE (M3 acceptance criterion)
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestRoPEFusedVsPyTorch:
    """Compare Lumen fused AITER RoPE kernel vs decomposed PyTorch ops.

    This directly addresses M3 acceptance: 'Fused RoPE shows latency reduction.'
    The naive PyTorch implementation uses slice + mul + cat + add (4+ ops /
    kernel launches); the AITER kernel fuses all of these into a single launch.
    """

    # Expected: Lumen fused RoPE should be 2-5x faster than the naive PyTorch
    # decomposition. The PyTorch path launches 4+ kernels (slice, mul, cat, add)
    # with intermediate tensor allocations; the fused kernel does one read of
    # cos/sin, one read/write of x, and produces the result in a single pass.
    @pytest.mark.parametrize("seqlen", [512, 2048, 4096, 8192])
    def test_fused_vs_pytorch(self, seqlen):
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb

        x = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
        cos, sin = _make_cos_sin(seqlen, ROTARY_DIM)

        r_fused = cuda_timer(
            lambda: apply_rotary_pos_emb(x, cos, sin),
            label=f"Lumen fused RoPE S={seqlen}",
        )
        r_pytorch = cuda_timer(
            lambda: _pytorch_rope_neox(x, cos, sin),
            label=f"PyTorch decomposed RoPE S={seqlen}",
        )

        speedup = r_pytorch.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup_vs_pytorch"] = round(speedup, 2)
        print_report(f"Fused vs PyTorch RoPE S={seqlen}", [r_fused, r_pytorch])

    # Expected: Fused Q+K RoPE should be 3-6x faster than applying PyTorch
    # decomposed RoPE to Q and K separately (8+ kernel launches vs 1).
    def test_fused_qk_vs_pytorch(self):
        from atomrl.lumen.ops.rope import fused_rope

        seqlen = 2048
        q = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
        cos, sin = _make_cos_sin(seqlen, ROTARY_DIM)

        r_fused = cuda_timer(
            lambda: fused_rope(q, k, cos, sin),
            label="Lumen fused_rope Q+K",
        )

        def _pytorch_both():
            _pytorch_rope_neox(q, cos, sin)
            _pytorch_rope_neox(k, cos, sin)

        r_pytorch = cuda_timer(_pytorch_both, label="PyTorch RoPE Q + K separate")

        speedup = r_pytorch.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup_vs_pytorch"] = round(speedup, 2)
        print_report("Fused Q+K vs PyTorch Separate", [r_fused, r_pytorch])


# ---------------------------------------------------------------------------
# 1. Standard 1D RoPE — sequence length sweep
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestRoPE1D:
    """apply_rotary_pos_emb: AITER Triton kernel for standard NeoX RoPE."""

    # Expected: Latency scales linearly with sequence length because RoPE is
    # an element-wise operation (O(S*H*D)). The AITER Triton kernel fuses the
    # sin/cos multiply and rotate into a single kernel, avoiding the 4+ separate
    # PyTorch ops (slice, mul, cat, add) of a naive implementation.
    @pytest.mark.parametrize("seqlen", [128, 512, 2048, 4096, 8192])
    def test_seqlen_sweep(self, seqlen):
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb

        x = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
        cos, sin = _make_cos_sin(seqlen, ROTARY_DIM)

        r_fused = cuda_timer(
            lambda: apply_rotary_pos_emb(x, cos, sin),
            label=f"Lumen fused S={seqlen}",
        )
        r_naive = cuda_timer(
            lambda: _pytorch_rope_neox(x, cos, sin),
            label=f"PyTorch decomposed S={seqlen}",
        )
        speedup = r_naive.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup"] = round(speedup, 2)
        print_report(f"RoPE 1D S={seqlen}", [r_fused, r_naive])

    # Expected: Both NeoX and GPT-J styles should have similar latency because
    # the AITER Triton kernel handles the index permutation internally. NeoX
    # splits the head dim in half (x[:d/2], x[d/2:]); GPT-J interleaves
    # (x[::2], x[1::2]). The kernel absorbs this layout difference at no cost.
    def test_neox_vs_gptj(self):
        """Compare NeoX (half-split) vs GPT-J (interleaved) style."""
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb

        seqlen = 2048
        x = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
        cos, sin = _make_cos_sin(seqlen, ROTARY_DIM)

        r_neox = cuda_timer(
            lambda: apply_rotary_pos_emb(x, cos, sin, interleaved=False),
            label="RoPE NeoX (interleaved=False)",
        )
        r_gptj = cuda_timer(
            lambda: apply_rotary_pos_emb(x, cos, sin, interleaved=True),
            label="RoPE GPT-J (interleaved=True)",
        )
        diff = (r_gptj.avg_ms - r_neox.avg_ms) / max(r_neox.avg_ms, 1e-6) * 100
        r_gptj.extra["vs_neox"] = f"{diff:+.1f}%"
        print_report("RoPE: NeoX vs GPT-J Style", [r_neox, r_gptj])


# ---------------------------------------------------------------------------
# 2. Fused Q+K RoPE
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestFusedRoPEQK:
    """fused_rope: apply RoPE to Q and K in one call."""

    # Expected: fused_rope (Q+K in one dispatch) should be ~1.5-2x faster than
    # calling apply_rotary_pos_emb twice. The fused kernel reads cos/sin tables
    # once and applies rotation to both Q and K in a single pass, saving one
    # full kernel launch and one round of cos/sin memory reads.
    @pytest.mark.parametrize("seqlen", [512, 2048, 4096])
    def test_fused_qk(self, seqlen):
        from atomrl.lumen.ops.rope import fused_rope

        q = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
        cos, sin = _make_cos_sin(seqlen, ROTARY_DIM)

        r_fused = cuda_timer(
            lambda: fused_rope(q, k, cos, sin),
            label=f"Lumen fused Q+K S={seqlen}",
        )

        def _pytorch():
            _pytorch_rope_neox(q, cos, sin)
            _pytorch_rope_neox(k, cos, sin)

        r_pytorch = cuda_timer(_pytorch, label=f"PyTorch RoPE Q+K S={seqlen}")

        speedup = r_pytorch.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup"] = round(speedup, 2)
        print_report(f"Fused vs PyTorch Q+K S={seqlen}", [r_fused, r_pytorch])


# ---------------------------------------------------------------------------
# 3. GQA RoPE (different Q/KV head counts)
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestRoPEGQA:
    """Fused RoPE with GQA configurations."""

    # Expected: Latency with fewer KV heads (h_kv=1,4) should be noticeably
    # lower than h_kv=32 (MHA) because the K tensor is smaller. With h_kv=1
    # (MQA), K has H/32 the elements of Q. The fused kernel processes Q and K
    # in parallel threads; fewer K elements = less total work.
    @pytest.mark.parametrize("h_kv", [1, 4, 8, 32])
    def test_gqa_head_configs(self, h_kv):
        from atomrl.lumen.ops.rope import fused_rope

        seqlen = 2048
        q = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, h_kv, seqlen, D, device="cuda", dtype=torch.bfloat16)
        cos, sin = _make_cos_sin(seqlen, ROTARY_DIM)

        r_fused = cuda_timer(
            lambda: fused_rope(q, k, cos, sin),
            label=f"Lumen fused GQA H_q={H} H_kv={h_kv}",
        )

        def _pytorch():
            _pytorch_rope_neox(q, cos, sin)
            _pytorch_rope_neox(k, cos, sin)

        r_pytorch = cuda_timer(_pytorch, label=f"PyTorch RoPE Q({H})+K({h_kv})")

        speedup = r_pytorch.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup"] = round(speedup, 2)
        print_report(f"RoPE GQA H_kv={h_kv}", [r_fused, r_pytorch])


# ---------------------------------------------------------------------------
# 4. Vision 2D RoPE
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestRoPE2D:
    """apply_rotary_pos_emb_2d: 2D RoPE for vision models (ViT etc.)."""

    # Expected: 2D RoPE applies separate rotations for height and width
    # dimensions in a single kernel launch. Latency should scale with spatial
    # resolution (H*W). The fused kernel avoids the 2 separate 1D RoPE calls
    # and the intermediate reshape that a naive 2D implementation would need.
    @pytest.mark.parametrize("img_size", [(8, 8), (16, 16), (32, 32)])
    def test_2d_rope_sizes(self, img_size):
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb_2d

        img_h, img_w = img_size
        spatial = img_h * img_w
        n_heads = 16
        dim = 64

        x = torch.randn(B, spatial, n_heads, dim, device="cuda", dtype=torch.bfloat16)
        cos_h, sin_h = _make_cos_sin(img_h, dim)
        cos_w, sin_w = _make_cos_sin(img_w, dim)

        r_fused = cuda_timer(
            lambda: apply_rotary_pos_emb_2d(
                x, cos_h, sin_h, cos_w, sin_w, img_h, img_w
            ),
            label=f"Lumen fused 2D {img_h}x{img_w}",
        )
        r_naive = cuda_timer(
            lambda: _pytorch_rope_2d(x, cos_h, sin_h, cos_w, sin_w, img_h, img_w),
            label=f"PyTorch decomposed 2D {img_h}x{img_w}",
        )
        speedup = r_naive.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup"] = round(speedup, 2)
        print_report(f"RoPE 2D {img_h}x{img_w}", [r_fused, r_naive])


# ---------------------------------------------------------------------------
# 5. Video 3D RoPE
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestRoPE3D:
    """apply_rotary_pos_emb_3d: 3D RoPE for video models."""

    # Expected: 3D RoPE fuses temporal (T) and spatial (H, W) rotations into
    # one kernel. The head dimension is split into 3 equal parts, each rotated
    # by the corresponding axis frequencies. Without fusion, this would require
    # 3 separate rotation passes plus complex slicing, making the fused version
    # significantly faster for video model workloads (T*H*W tokens).
    def test_3d_rope(self):
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb_3d

        T, H_img, W_img = 4, 8, 8
        spatial = T * H_img * W_img
        n_heads = 16
        dim = 192  # must be divisible by 3 for 3D

        x = torch.randn(B, spatial, n_heads, dim, device="cuda", dtype=torch.float32)
        grid_sizes = torch.tensor([[T, H_img, W_img]], dtype=torch.int32, device="cuda")

        half_dim = dim // 2
        freqs_per_axis = half_dim // 3
        freqs = torch.randn(
            spatial, 3, freqs_per_axis, device="cuda", dtype=torch.float32
        )
        freqs_complex = torch.view_as_complex(
            torch.stack([freqs.cos(), freqs.sin()], dim=-1)
        )

        r_fused = cuda_timer(
            lambda: apply_rotary_pos_emb_3d(x, grid_sizes, freqs_complex),
            label=f"Lumen fused 3D {T}x{H_img}x{W_img}",
        )

        def _pytorch_3d():
            c_total = dim // 2
            c1 = c_total - 2 * (c_total // 3)
            c2 = c_total // 3
            c3 = c_total // 3
            x_complex = torch.view_as_complex(
                x.reshape(B, spatial, n_heads, c_total, 2)
            )
            ft = freqs_complex[:, 0, :c1].unsqueeze(0).unsqueeze(2)
            fh = freqs_complex[:, 1, :c2].unsqueeze(0).unsqueeze(2)
            fw = freqs_complex[:, 2, :c3].unsqueeze(0).unsqueeze(2)
            out_t = x_complex[..., :c1] * ft
            out_h = x_complex[..., c1 : c1 + c2] * fh
            out_w = x_complex[..., c1 + c2 : c1 + c2 + c3] * fw
            out_complex = torch.cat([out_t, out_h, out_w], dim=-1)
            return torch.view_as_real(out_complex).reshape(B, spatial, n_heads, dim)

        r_naive = cuda_timer(
            _pytorch_3d, label=f"PyTorch decomposed 3D {T}x{H_img}x{W_img}"
        )

        speedup = r_naive.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup"] = round(speedup, 2)
        print_report(f"RoPE 3D {T}x{H_img}x{W_img}", [r_fused, r_naive])


# ---------------------------------------------------------------------------
# 6. Dtype comparison
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestRoPEDtype:
    """RoPE performance across dtypes."""

    # Expected: float16 and bfloat16 should have nearly identical latency
    # because both are 16-bit and the Triton kernel uses the same memory
    # bandwidth. Any difference reflects register-level instruction throughput
    # differences (e.g., bfloat16 may have slightly different FMA latency on
    # some AMD architectures).
    @pytest.fixture()
    def _dtype_baseline(self):
        """Collect results for cross-dtype comparison."""
        self._dtype_results = {}
        yield
        if len(self._dtype_results) >= 2:
            dtypes = sorted(self._dtype_results.keys(), key=str)
            baseline = self._dtype_results[dtypes[0]]
            for dt in dtypes[1:]:
                r = self._dtype_results[dt]
                diff = (r.avg_ms - baseline.avg_ms) / max(baseline.avg_ms, 1e-6) * 100
                r.extra[f"vs_{dtypes[0]}"] = f"{diff:+.1f}%"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_dtype_latency(self, dtype):
        from atomrl.lumen.ops.rope import apply_rotary_pos_emb

        seqlen = 2048
        x = torch.randn(B, H, seqlen, D, device="cuda", dtype=dtype)
        cos, sin = _make_cos_sin(seqlen, ROTARY_DIM, dtype=dtype)

        r = cuda_timer(
            lambda: apply_rotary_pos_emb(x, cos, sin),
            label=f"RoPE S={seqlen} dtype={dtype}",
        )
        print_report(f"RoPE dtype={dtype}", [r])
        assert r.avg_ms > 0


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def main():
    require_cuda()
    require_aiter()

    from atomrl.lumen.ops.rope import apply_rotary_pos_emb, fused_rope

    results: list[BenchResult] = []

    # Fused vs PyTorch baseline
    for seqlen in [512, 2048, 4096, 8192]:
        x = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
        cos, sin = _make_cos_sin(seqlen, ROTARY_DIM)

        r_fused = cuda_timer(
            lambda cos=cos, sin=sin, x=x: apply_rotary_pos_emb(x, cos, sin),
            label=f"Lumen fused S={seqlen}",
        )
        r_pytorch = cuda_timer(
            lambda cos=cos, sin=sin, x=x: _pytorch_rope_neox(x, cos, sin),
            label=f"PyTorch decomposed S={seqlen}",
        )
        speedup = r_pytorch.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup_vs_pytorch"] = round(speedup, 2)
        results.extend([r_fused, r_pytorch])

    # Fused Q+K
    seqlen = 2048
    q = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
    cos, sin = _make_cos_sin(seqlen, ROTARY_DIM)
    results.append(
        cuda_timer(lambda: fused_rope(q, k, cos, sin), label="fused_rope Q+K S=2048")
    )

    # GQA variants — run MHA baseline first, then compare
    gqa_results = []
    k_mha = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.bfloat16)
    r_mha = cuda_timer(
        lambda: fused_rope(q, k_mha, cos, sin), label=f"fused_rope MHA H_kv={H}"
    )
    gqa_results.append(r_mha)
    for h_kv in [1, 4, 8]:
        k_gqa = torch.randn(B, h_kv, seqlen, D, device="cuda", dtype=torch.bfloat16)
        r = cuda_timer(
            lambda k_gqa=k_gqa: fused_rope(q, k_gqa, cos, sin),
            label=f"fused_rope GQA H_kv={h_kv}",
        )
        diff = (r.avg_ms - r_mha.avg_ms) / max(r_mha.avg_ms, 1e-6) * 100
        r.extra["vs_MHA"] = f"{diff:+.1f}%"
        gqa_results.append(r)
    results.extend(gqa_results)

    print_report("Lumen RoPE Fusion Benchmarks", results)


if __name__ == "__main__":
    main()
