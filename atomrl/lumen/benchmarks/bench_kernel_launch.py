# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Benchmark 1 — Single-GPU kernel-launch reduction via Lumen fused ops.

Exercises Lumen's fused-kernel features that reduce kernel-launch overhead:

  * **FP8 quantized_linear**: All 7 scaling modes (delayed, dynamic,
    per_token, blockwise, blockwise2d, mxfp8, none/BF16).
  * **FP8 activation store**: ``LumenGatedMLP(fp8_activation_store=True)``
    vs ``fp8_activation_store=False``.
  * **Attention backends**: ``aiter_csrc`` vs ``aiter_triton``.
  * **Fused RMSNorm + GEMM pipeline**: norm → quant → GEMM in one flow.
  * **Fusion latency**: fused RoPE, fused RMSNorm+quant, fused GatedMLP,
    fused MLP, and full norm+quant+GEMM pipeline vs decomposed equivalents.

Run::

    python -m benchmarks.bench_kernel_launch
    pytest benchmarks/bench_kernel_launch.py -v -s
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
# Model dimensions (Llama 3.1 8B)
# ---------------------------------------------------------------------------
B, S = 2, 2048
H, D = 32, 128
H_KV = 8
HIDDEN = H * D  # 4096
FFN_HIDDEN = 14336


# ---------------------------------------------------------------------------
# 1. FP8 quantized_linear — all scaling modes
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestQuantizedLinearScalingModes:
    """Compare latency of quantized_linear across all 7 scaling modes."""

    # Expected: FP8 modes (delayed/dynamic/per_token/blockwise) should have
    # comparable or slightly higher latency than BF16 ("none"), because the
    # quantize step adds overhead but the FP8 GEMM itself is faster on
    # memory-bound shapes. Per-token and blockwise have extra scaling
    # computation. MXFP8 may be slowest due to microscaling conversion.
    @pytest.mark.parametrize(
        "scaling_type",
        [
            "none",
            "delayed",
            "dynamic",
            "per_token",
            "blockwise",
            "blockwise2d",
            "mxfp8",
        ],
    )
    def test_scaling_mode_latency(self, scaling_type):
        from atomrl.lumen.ops.quantize.linear import quantized_linear
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        x = torch.randn(B * S, HIDDEN, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(FFN_HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(FFN_HIDDEN, device="cuda", dtype=torch.bfloat16)

        r = cuda_timer(
            lambda: quantized_linear(
                x,
                w,
                bias,
                scaling_type=scaling_type,
                fp8_dtype=fp8_dtype,
            ),
            label=f"quantized_linear scaling={scaling_type}",
        )
        print_report(f"quantized_linear scaling={scaling_type}", [r])
        assert r.avg_ms > 0

    # Expected: "none" (BF16) is the baseline. "delayed"/"dynamic" add ~5-15%
    # overhead for per-tensor quant but unlock FP8 GEMM throughput. "per_token"
    # adds row-wise scaling. "blockwise" adds block-level granularity. The
    # overhead_vs_bf16 metric shows the quant cost relative to pure BF16 GEMM.
    def test_scaling_modes_comparison(self):
        """Side-by-side comparison of all modes."""
        from atomrl.lumen.ops.quantize.linear import quantized_linear
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        x = torch.randn(B * S, HIDDEN, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(FFN_HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16)

        results = []
        for mode in [
            "none",
            "delayed",
            "dynamic",
            "per_token",
            "blockwise",
            "blockwise2d",
            "mxfp8",
        ]:
            r = cuda_timer(
                lambda m=mode: quantized_linear(
                    x, w, scaling_type=m, fp8_dtype=fp8_dtype
                ),
                label=f"scaling={mode}",
            )
            results.append(r)

        bf16_ms = results[0].avg_ms  # "none" = BF16 baseline
        for r in results[1:]:
            overhead = (r.avg_ms - bf16_ms) / max(bf16_ms, 1e-6) * 100
            r.extra["overhead_vs_bf16"] = f"{overhead:+.1f}%"

        print_report("quantized_linear: All Scaling Modes", results)


# ---------------------------------------------------------------------------
# 2. FP8 activation store — memory / latency tradeoff
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestFP8ActivationStore:
    """LumenGatedMLP with fp8_activation_store=True vs False."""

    def _make_mlp(self, fp8_store: bool):
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP

        return LumenGatedMLP(
            HIDDEN,
            FFN_HIDDEN,
            activation="swiglu",
            bias=False,
            fp8_activation_store=fp8_store,
        ).to(device="cuda", dtype=torch.bfloat16)

    # Expected: Forward latency should be similar or slightly faster with
    # fp8_activation_store=True because storing activations in FP8 reduces
    # memory traffic. The quant overhead is small relative to the GEMM cost.
    def test_forward_latency(self):
        mlp_bf16 = self._make_mlp(False)
        mlp_fp8 = self._make_mlp(True)
        x = torch.randn(B, S, HIDDEN, device="cuda", dtype=torch.bfloat16)

        r_bf16 = cuda_timer(lambda: mlp_bf16(x), label="GatedMLP fp8_store=False (fwd)")
        r_fp8 = cuda_timer(lambda: mlp_fp8(x), label="GatedMLP fp8_store=True (fwd)")

        overhead = (r_fp8.avg_ms - r_bf16.avg_ms) / max(r_bf16.avg_ms, 1e-6) * 100
        r_fp8.extra["vs_bf16"] = f"{overhead:+.1f}%"
        print_report("FP8 Activation Store: Forward", [r_bf16, r_fp8])

    # Expected: Backward pass with fp8_activation_store=True should be faster
    # because saved activations are in FP8 (1 byte) instead of BF16 (2 bytes),
    # halving the memory reload cost during backward. The dequant cost is small
    # compared to the memory bandwidth savings on large activation tensors.
    def test_backward_latency(self):
        mlp_bf16 = self._make_mlp(False)
        mlp_fp8 = self._make_mlp(True)
        x_bf16 = torch.randn(
            B, S, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        x_fp8 = x_bf16.detach().clone().requires_grad_(True)

        def _fwd_bwd(mlp, x):
            out = mlp(x)
            out.sum().backward()
            if x.grad is not None:
                x.grad = None
            mlp.zero_grad()

        r_bf16 = cuda_timer(
            lambda: _fwd_bwd(mlp_bf16, x_bf16),
            label="GatedMLP fp8_store=False (fwd+bwd)",
        )
        r_fp8 = cuda_timer(
            lambda: _fwd_bwd(mlp_fp8, x_fp8), label="GatedMLP fp8_store=True (fwd+bwd)"
        )

        speedup = r_bf16.avg_ms / max(r_fp8.avg_ms, 1e-6)
        overhead = (r_fp8.avg_ms - r_bf16.avg_ms) / max(r_bf16.avg_ms, 1e-6) * 100
        r_fp8.extra["vs_bf16"] = f"{overhead:+.1f}%"
        r_fp8.extra["speedup"] = round(speedup, 2)
        print_report("FP8 Activation Store: Forward + Backward", [r_bf16, r_fp8])

    # Expected: ~1.5-2x memory reduction for activations saved during forward.
    # FP8 stores activations at 1 byte/element vs BF16's 2 bytes/element.
    # The gate/up projection activations in a GatedMLP are the dominant
    # memory consumers, so FP8 storage roughly halves peak activation memory.
    def test_activation_memory(self):
        """Compare peak memory with and without FP8 activation storage."""
        from benchmarks.bench_utils import format_bytes, track_cuda_memory

        x = torch.randn(
            B, S, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        mlp_bf16 = self._make_mlp(False)
        with track_cuda_memory() as mem_bf16:
            out = mlp_bf16(x)
            out.sum().backward()
        mlp_bf16.zero_grad()
        if x.grad is not None:
            x.grad = None
        del out
        torch.cuda.empty_cache()

        mlp_fp8 = self._make_mlp(True)
        x2 = x.detach().clone().requires_grad_(True)
        with track_cuda_memory() as mem_fp8:
            out = mlp_fp8(x2)
            out.sum().backward()

        reduction = mem_bf16["peak_delta"] / max(mem_fp8["peak_delta"], 1)
        print(f"\n  BF16 activation peak delta: {format_bytes(mem_bf16['peak_delta'])}")
        print(f"  FP8  activation peak delta: {format_bytes(mem_fp8['peak_delta'])}")
        print(f"  Memory reduction: {reduction:.2f}x")


# ---------------------------------------------------------------------------
# 4. Attention backends
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestAttentionBackends:
    """Compare attention backend_type='aiter_csrc' vs 'aiter_triton'."""

    def _make_qkv(self, seqlen=S):
        q = torch.randn(B, seqlen, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, seqlen, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, seqlen, H, D, device="cuda", dtype=torch.bfloat16)
        return q, k, v

    # Expected: aiter_csrc (Composable Kernel / HIP) is typically faster than
    # aiter_triton for standard causal attention, because CK kernels are
    # hand-tuned assembly with tighter memory access patterns. Triton backend
    # offers more flexibility (e.g., custom masking) but at a latency cost.
    def test_csrc_vs_triton_causal(self):
        from atomrl.lumen.ops.attention import attention

        q, k, v = self._make_qkv()

        r_csrc = cuda_timer(
            lambda: attention(q, k, v, causal=True, backend_type="aiter_csrc"),
            label="attention causal (aiter_csrc)",
        )
        r_triton = cuda_timer(
            lambda: attention(q, k, v, causal=True, backend_type="aiter_triton"),
            label="attention causal (aiter_triton)",
        )
        speedup = r_triton.avg_ms / max(r_csrc.avg_ms, 1e-6)
        r_csrc.extra["speedup_vs_triton"] = round(speedup, 2)
        print_report("Attention: CK csrc vs Triton (causal)", [r_csrc, r_triton])

    # Expected: GQA (32 Q heads, 8 KV heads) reduces KV memory traffic by 4x
    # vs MHA. Both backends should show lower latency than equivalent MHA.
    # CK csrc may still be faster due to native GQA tile scheduling; Triton
    # handles GQA via head broadcast which adds slight overhead.
    def test_gqa_backends(self):
        """GQA: 32 Q heads, 8 KV heads."""
        from atomrl.lumen.ops.attention import attention

        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, H_KV, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, H_KV, D, device="cuda", dtype=torch.bfloat16)

        r_csrc = cuda_timer(
            lambda: attention(q, k, v, causal=True, backend_type="aiter_csrc"),
            label="attention GQA (aiter_csrc)",
        )
        r_triton = cuda_timer(
            lambda: attention(q, k, v, causal=True, backend_type="aiter_triton"),
            label="attention GQA (aiter_triton)",
        )
        speedup = r_triton.avg_ms / max(r_csrc.avg_ms, 1e-6)
        r_csrc.extra["speedup_vs_triton"] = round(speedup, 2)
        print_report("Attention: GQA (32Q/8KV)", [r_csrc, r_triton])

    # Expected: Sliding window (256 tokens) should be faster than full causal
    # attention for long sequences. Full causal is O(S^2); sliding window is
    # O(S*W) where W=256. At S=2048, this is ~8x less computation, though
    # kernel overhead means actual speedup is typically 2-4x.
    def test_sliding_window(self):
        """Sliding window attention (window_size != (-1,-1))."""
        from atomrl.lumen.ops.attention import attention

        q, k, v = self._make_qkv()
        window = (256, 256)

        r_full = cuda_timer(
            lambda: attention(q, k, v, causal=True, backend_type="aiter_csrc"),
            label="attention full causal (csrc)",
        )
        r_window = cuda_timer(
            lambda: attention(
                q, k, v, causal=True, window_size=window, backend_type="aiter_csrc"
            ),
            label=f"attention window={window} (csrc)",
        )
        speedup = r_full.avg_ms / max(r_window.avg_ms, 1e-6)
        r_window.extra["speedup_vs_full"] = round(speedup, 2)
        print_report("Attention: Sliding Window vs Full", [r_full, r_window])

    # Expected: Latency scales quadratically with sequence length for causal
    # attention. Doubling S should ~4x the latency on compute-bound shapes.
    # The "auto" backend lets Lumen's dispatch pick the fastest kernel for
    # each (S, H, D) configuration, demonstrating adaptive backend selection.
    @pytest.mark.parametrize("seqlen", [512, 1024, 2048, 4096])
    def test_seqlen_sweep(self, seqlen):
        """Attention latency sweep across sequence lengths."""
        from atomrl.lumen.ops.attention import attention

        q, k, v = self._make_qkv(seqlen)
        r = cuda_timer(
            lambda: attention(q, k, v, causal=True),
            label=f"attention S={seqlen} (auto backend)",
        )
        print_report(f"Attention S={seqlen}", [r])
        assert r.avg_ms > 0


# ---------------------------------------------------------------------------
# 5. Fused norm + GEMM pipeline
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestFusedNormGEMMPipeline:
    """RMSNorm → quantized_linear pipeline: measure combined latency."""

    # Expected: rmsnorm + gemm_bf16 in sequence measures the baseline latency
    # of a normalization-then-linear pipeline. The AITER rmsnorm fuses the
    # norm into a single kernel (vs PyTorch's multi-op decomposition), so the
    # pipeline latency is dominated by the GEMM.
    def test_rmsnorm_plus_gemm_bf16(self):
        from atomrl.lumen.ops.normalization.rmsnorm import rmsnorm
        from atomrl.lumen.ops.quantize.linear import gemm_bf16

        x = torch.randn(B * S, HIDDEN, device="cuda", dtype=torch.bfloat16)
        norm_w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
        gemm_w = torch.randn(FFN_HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16)

        def _separate():
            normed = rmsnorm(x, norm_w)
            return gemm_bf16(normed, gemm_w)

        r_sep = cuda_timer(_separate, label="rmsnorm + gemm_bf16 (separate)")
        print_report("Norm + GEMM Pipeline (BF16)", [r_sep])
        assert r_sep.avg_ms > 0

    # Expected: "dynamic" and "per_token" modes add quant cost after rmsnorm
    # but the FP8 GEMM is faster on memory-bandwidth-limited shapes. "blockwise"
    # adds block-level scaling overhead. Overall pipeline latency with FP8 should
    # be competitive with pure BF16 ("none") thanks to halved GEMM operand size.
    def test_rmsnorm_plus_fp8_gemm(self):
        """RMSNorm → FP8 quantized_linear with different scaling modes."""
        from atomrl.lumen.ops.normalization.rmsnorm import rmsnorm
        from atomrl.lumen.ops.quantize.linear import quantized_linear
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        x = torch.randn(B * S, HIDDEN, device="cuda", dtype=torch.bfloat16)
        norm_w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
        gemm_w = torch.randn(FFN_HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16)

        results = []
        for mode in ["none", "dynamic", "per_token", "blockwise"]:

            def _pipeline(m=mode):
                normed = rmsnorm(x, norm_w)
                return quantized_linear(
                    normed, gemm_w, scaling_type=m, fp8_dtype=fp8_dtype
                )

            r = cuda_timer(_pipeline, label=f"rmsnorm + quantized_linear({mode})")
            results.append(r)

        print_report("Norm + FP8 GEMM Pipeline", results)


# ---------------------------------------------------------------------------
# 6. Fusion latency: fused ops vs decomposed equivalents
# ---------------------------------------------------------------------------


@CUDA
@AITER
class TestFusionLatency:
    """Latency comparison of Lumen fused ops vs decomposed equivalents.

    Measures the wall-clock benefit of AITER-backed fused kernels over
    manually sequenced PyTorch operations for several common pipelines.
    """

    # Expected: fused_rope dispatches two AITER Triton RoPE kernels (Q, K)
    # which are individually faster than pure-PyTorch rope math (sin/cos
    # broadcast multiply + interleave).  Speedup comes from fewer kernel
    # launches and fused memory passes.
    def test_fused_rope_vs_manual(self):
        """Fused RoPE (AITER Triton) vs manual sin/cos rotation."""
        from atomrl.lumen.ops.rope import fused_rope

        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, H_KV, D, device="cuda", dtype=torch.bfloat16)
        cos = torch.randn(1, S, 1, D, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(1, S, 1, D, device="cuda", dtype=torch.bfloat16)

        def _manual():
            x1_q, x2_q = q[..., : D // 2], q[..., D // 2 :]
            rq = torch.cat((-x2_q, x1_q), dim=-1)
            q_rot = q * cos + rq * sin
            x1_k, x2_k = k[..., : D // 2], k[..., D // 2 :]
            rk = torch.cat((-x2_k, x1_k), dim=-1)
            k_rot = k * cos[:, :, :1, :] + rk * sin[:, :, :1, :]
            return q_rot, k_rot

        r_manual = cuda_timer(_manual, label="manual RoPE (sin/cos)")

        r_fused = cuda_timer(
            lambda: fused_rope(
                q, k, cos.squeeze(0).squeeze(-2), sin.squeeze(0).squeeze(-2)
            ),
            label="fused_rope (AITER Triton)",
        )

        speedup = r_manual.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup"] = round(speedup, 2)
        print_report("Fused RoPE vs Manual", [r_manual, r_fused])

    # Expected: rmsnorm_with_quant fuses norm + FP8 quantization into a
    # single kernel pass (for dynamic/per_token/blockwise modes), avoiding
    # the extra memory round-trip of writing BF16 normed output then
    # re-reading it for quantization.
    def test_fused_rmsnorm_quant_vs_separate(self):
        """Fused RMSNorm+quant vs separate rmsnorm then manual FP8 cast."""
        from atomrl.lumen.ops.normalization.rmsnorm import rmsnorm, rmsnorm_with_quant
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        x = torch.randn(B * S, HIDDEN, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)

        def _separate():
            normed = rmsnorm(x, w)
            return normed.to(fp8_dtype)

        r_sep = cuda_timer(_separate, label="rmsnorm + manual cast to FP8")

        results = [r_sep]
        for mode in ["dynamic", "per_token", "blockwise"]:
            r = cuda_timer(
                lambda m=mode: rmsnorm_with_quant(
                    x, w, eps=1e-6, scaling_type=m, fp8_dtype=fp8_dtype
                ),
                label=f"rmsnorm_with_quant({mode})",
            )
            speedup = r_sep.avg_ms / max(r.avg_ms, 1e-6)
            r.extra["speedup"] = round(speedup, 2)
            results.append(r)

        print_report("Fused RMSNorm + Quant vs Separate", results)

    # Expected: fused_gated_mlp uses AITER's ff_a16w16_fused_gated kernel
    # which performs gate_proj, up_proj, SwiGLU activation, and down_proj
    # in fewer kernel launches than the decomposed path. The fused kernel
    # also benefits from register-level intermediate reuse.
    def test_fused_gated_mlp_vs_decomposed(self):
        """AITER fused_gated_mlp vs manual GEMMs + SwiGLU."""
        from atomrl.lumen.ops.mlp.fused_mlp import fused_gated_mlp
        from atomrl.lumen.ops.quantize.linear import gemm_bf16

        x = torch.randn(B * S, HIDDEN, device="cuda", dtype=torch.bfloat16)
        w_gate = (
            torch.randn(FFN_HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        w_up = (
            torch.randn(FFN_HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        w_down = (
            torch.randn(HIDDEN, FFN_HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
        )

        def _decomposed():
            gate = gemm_bf16(x, w_gate)
            up = gemm_bf16(x, w_up)
            act = torch.nn.functional.silu(gate) * up
            return gemm_bf16(act, w_down)

        r_decomposed = cuda_timer(_decomposed, label="manual gate+up+silu+down")

        r_fused = cuda_timer(
            lambda: fused_gated_mlp(x, w_up, w_gate, w_down, activation="swiglu"),
            label="fused_gated_mlp (AITER)",
        )

        speedup = r_decomposed.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup"] = round(speedup, 2)
        print_report("Fused GatedMLP vs Decomposed", [r_decomposed, r_fused])

    # Expected: fused_mlp uses AITER's ff_a16w16_fused_ungated kernel
    # to combine up_proj + activation + down_proj.  The benefit is smaller
    # than gated MLP because there are fewer intermediate tensors.
    def test_fused_mlp_vs_decomposed(self):
        """AITER fused_mlp vs manual GEMMs + GELU."""
        from atomrl.lumen.ops.mlp.fused_mlp import fused_mlp
        from atomrl.lumen.ops.quantize.linear import gemm_bf16

        x = torch.randn(B * S, HIDDEN, device="cuda", dtype=torch.bfloat16)
        w_up = (
            torch.randn(FFN_HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        w_down = (
            torch.randn(HIDDEN, FFN_HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
        )

        def _decomposed():
            up = gemm_bf16(x, w_up)
            act = torch.nn.functional.gelu(up)
            return gemm_bf16(act, w_down)

        r_decomposed = cuda_timer(_decomposed, label="manual up+gelu+down")

        r_fused = cuda_timer(
            lambda: fused_mlp(x, w_up, w_down, activation="gelu"),
            label="fused_mlp (AITER)",
        )

        speedup = r_decomposed.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup"] = round(speedup, 2)
        print_report("Fused MLP vs Decomposed", [r_decomposed, r_fused])

    # Expected: The full transformer-block pre-norm pipeline (RMSNorm →
    # FP8 quant → GEMM) benefits from norm+quant fusion that eliminates
    # one global memory round-trip.  Comparing the end-to-end latency
    # of the fused pipeline against the decomposed version.
    def test_norm_quant_gemm_pipeline(self):
        """Full pipeline: RMSNorm → FP8 quant → GEMM (fused vs decomposed).

        Both paths use the same pre-quantized FP8 weight so the comparison
        isolates norm+quant fusion (one fewer global-memory round-trip) from
        weight quantization overhead.
        """
        from atomrl.lumen.ops.normalization.rmsnorm import rmsnorm, rmsnorm_with_quant
        from atomrl.lumen.ops.quantize.linear import (
            dispatch_gemm,
            quantize_input,
        )
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        fp8_dtype = _get_float8_e4m3()
        x = torch.randn(B * S, HIDDEN, device="cuda", dtype=torch.bfloat16)
        norm_w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
        gemm_w = torch.randn(FFN_HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16)

        w_fp8, w_scale = quantize_input(gemm_w, "dynamic", fp8_dtype)

        def _decomposed():
            normed = rmsnorm(x, norm_w)
            normed_2d = normed.reshape(-1, normed.shape[-1]).contiguous()
            a_fp8, a_scale = quantize_input(normed_2d, "dynamic", fp8_dtype)
            return dispatch_gemm(a_fp8, w_fp8, a_scale, w_scale, "dynamic")

        r_decomposed = cuda_timer(
            _decomposed, label="rmsnorm + quant + GEMM (separate)"
        )

        def _fused():
            normed_fp8, a_scale = rmsnorm_with_quant(
                x,
                norm_w,
                eps=1e-6,
                scaling_type="dynamic",
                fp8_dtype=fp8_dtype,
            )
            normed_2d = normed_fp8.reshape(-1, normed_fp8.shape[-1]).contiguous()
            return dispatch_gemm(normed_2d, w_fp8, a_scale, w_scale, "dynamic")

        r_fused = cuda_timer(_fused, label="rmsnorm_with_quant + GEMM (fused)")

        speedup = r_decomposed.avg_ms / max(r_fused.avg_ms, 1e-6)
        r_fused.extra["speedup"] = round(speedup, 2)
        print_report(
            "Norm+Quant+GEMM Pipeline: Fused vs Decomposed", [r_decomposed, r_fused]
        )


# ---------------------------------------------------------------------------
# 7. Kernel-launch count: fused vs unfused (M1 acceptance criterion)
# ---------------------------------------------------------------------------


def _count_kernels(fn, warmup=2, active=1):
    """Run *fn* under torch.profiler and return the number of GPU kernel launches."""
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(active):
            fn()
            torch.cuda.synchronize()

    kernel_count = sum(
        evt.count
        for evt in prof.key_averages()
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.count > 0
    )
    return kernel_count, prof.key_averages()


@CUDA
@AITER
class TestKernelLaunchCount:
    """Measure actual GPU kernel-launch counts for fused vs unfused paths.

    Directly addresses M1 acceptance: 'Single-GPU benchmarks show measurable
    kernel-launch reduction.' Uses torch.profiler to count distinct CUDA
    kernels launched by fused Lumen ops vs decomposed alternatives.
    """

    # Expected: LumenGatedMLP fuses gate_proj + up_proj + activation + down_proj
    # into fewer kernels than doing separate GEMMs + activation. The fused path
    # should show a measurable reduction in kernel count.
    def test_gated_mlp_kernel_count(self):
        from atomrl.lumen.modules.fused_mlp import LumenGatedMLP
        from atomrl.lumen.ops.quantize.linear import gemm_bf16

        x = torch.randn(B * S, HIDDEN, device="cuda", dtype=torch.bfloat16)
        mlp = LumenGatedMLP(
            HIDDEN,
            FFN_HIDDEN,
            activation="swiglu",
            bias=False,
        ).to(device="cuda", dtype=torch.bfloat16)

        def _fused():
            mlp(x)

        w_gate = mlp.w_gate.data
        w_up = mlp.w_up.data
        w_down = mlp.w_down.data

        def _unfused():
            gate = gemm_bf16(x, w_gate)
            up = gemm_bf16(x, w_up)
            act = torch.nn.functional.silu(gate) * up
            return gemm_bf16(act, w_down)

        n_fused, _ = _count_kernels(_fused)
        n_unfused, _ = _count_kernels(_unfused)

        reduction = n_unfused - n_fused
        pct = reduction / max(n_unfused, 1) * 100
        sep = "=" * 56
        print(f"\n{sep}")
        print("  GatedMLP Kernel Launch Analysis")
        print(sep)
        print(f"  LumenGatedMLP (fused):    {n_fused:3d} kernels")
        print(f"  Manual gate+up+down:      {n_unfused:3d} kernels")
        print("  ─────────────────────────────────")
        print(f"  Reduction:                {reduction:3d} kernels  ({pct:.0f}% fewer)")
        print(sep)
        assert (
            n_fused < n_unfused
        ), f"Fused ({n_fused}) should use fewer kernels than unfused ({n_unfused})"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def main():
    require_cuda()
    require_aiter()

    from atomrl.lumen.modules.fused_mlp import LumenGatedMLP
    from atomrl.lumen.ops.attention import attention
    from atomrl.lumen.ops.quantize.linear import quantized_linear
    from atomrl.lumen.quantize.config import _get_float8_e4m3

    fp8_dtype = _get_float8_e4m3()
    results: list[BenchResult] = []

    # FP8 scaling modes
    x = torch.randn(B * S, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(FFN_HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16)
    bf16_ms = None
    for mode in [
        "none",
        "delayed",
        "dynamic",
        "per_token",
        "blockwise",
        "blockwise2d",
        "mxfp8",
    ]:
        r = cuda_timer(
            lambda m=mode: quantized_linear(x, w, scaling_type=m, fp8_dtype=fp8_dtype),
            label=f"quantized_linear({mode})",
        )
        if mode == "none":
            bf16_ms = r.avg_ms
        elif bf16_ms is not None:
            overhead = (r.avg_ms - bf16_ms) / max(bf16_ms, 1e-6) * 100
            r.extra["vs_bf16"] = f"{overhead:+.1f}%"
        results.append(r)

    # Attention backends
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    attn_results = []
    for be in ["aiter_csrc", "aiter_triton"]:
        r = cuda_timer(
            lambda b=be: attention(q, k, v, causal=True, backend_type=b),
            label=f"attention causal ({be})",
        )
        attn_results.append(r)
    if len(attn_results) == 2:
        speedup = attn_results[1].avg_ms / max(attn_results[0].avg_ms, 1e-6)
        attn_results[0].extra["speedup_vs_triton"] = round(speedup, 2)
    results.extend(attn_results)

    # GatedMLP fp8 activation store
    mlp_results = []
    for fp8_store in [False, True]:
        mlp = LumenGatedMLP(
            HIDDEN,
            FFN_HIDDEN,
            activation="swiglu",
            bias=False,
            fp8_activation_store=fp8_store,
        ).to(device="cuda", dtype=torch.bfloat16)
        x_mlp = torch.randn(B, S, HIDDEN, device="cuda", dtype=torch.bfloat16)
        r = cuda_timer(
            lambda mlp=mlp, x_mlp=x_mlp: mlp(x_mlp),
            label=f"GatedMLP fp8_store={fp8_store}",
        )
        mlp_results.append(r)
    if len(mlp_results) == 2:
        overhead = (
            (mlp_results[1].avg_ms - mlp_results[0].avg_ms)
            / max(mlp_results[0].avg_ms, 1e-6)
            * 100
        )
        mlp_results[1].extra["vs_bf16"] = f"{overhead:+.1f}%"
    results.extend(mlp_results)

    print_report("Lumen Single-GPU Feature Benchmarks", results)


if __name__ == "__main__":
    main()
