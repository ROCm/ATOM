"""Instrument hipb_mm to collect all unique GEMM shapes during training.

Usage: Run inside the training container with LUMEN_COLLECT_GEMM_SHAPES=1.
Shapes are written to /tmp/gemm_shapes.csv after 2 training steps.

This patches aiter.hipb_mm to log (M, N, K, dtype_a, dtype_b, out_dtype)
for each call, then aggregates into unique shapes with call counts.
"""

import csv
import os
from collections import defaultdict

_shape_counts = defaultdict(int)
_original_hipb_mm = None
_step_count = 0


def _patched_hipb_mm(
    mat1,
    mat2,
    solution_index,
    bias=None,
    out_dtype=None,
    scaleA=None,
    scaleB=None,
    scaleOut=None,
    bpreshuffle=None,
):
    M, K_a = mat1.shape
    _K_b, N = mat2.shape
    key = (
        M,
        N,
        K_a,
        str(mat1.dtype),
        str(mat2.dtype),
        str(out_dtype) if out_dtype is not None else "None",
        bias is not None,
        scaleA is not None,
    )
    _shape_counts[key] += 1
    return _original_hipb_mm(
        mat1,
        mat2,
        solution_index,
        bias=bias,
        out_dtype=out_dtype,
        scaleA=scaleA,
        scaleB=scaleB,
        scaleOut=scaleOut,
        bpreshuffle=bpreshuffle,
    )


def install():
    global _original_hipb_mm
    if os.environ.get("LUMEN_COLLECT_GEMM_SHAPES", "0") != "1":
        return

    import aiter.ops.gradlib as _gl

    _original_hipb_mm = _gl.hipb_mm
    _gl.hipb_mm = _patched_hipb_mm

    import aiter

    if hasattr(aiter, "hipb_mm"):
        aiter.hipb_mm = _patched_hipb_mm

    print("[GemmShapes] Instrumentation installed", flush=True)


def dump(path="/tmp/gemm_shapes.csv"):
    if not _shape_counts:
        return
    rows = sorted(_shape_counts.items(), key=lambda x: -x[1])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "M",
                "N",
                "K",
                "dtype_a",
                "dtype_b",
                "out_dtype",
                "has_bias",
                "has_scale",
                "count",
            ]
        )
        for key, cnt in rows:
            w.writerow(list(key) + [cnt])
    print(f"[GemmShapes] Wrote {len(rows)} unique shapes to {path}", flush=True)


if __name__ == "__main__":
    print("This module is meant to be imported, not run directly.")
    print("Manually computing Llama2-70B GEMM shapes for TP=1 DP=8:")
    print()

    H = 8192  # hidden
    FFN = 28672  # ffn_hidden
    HEADS = 64
    KV_HEADS = 8
    SEQ = 8192
    MBS = 1
    M = SEQ * MBS  # = 8192

    head_dim = H // HEADS  # 128
    qkv_out = H + 2 * (KV_HEADS * head_dim)  # 8192 + 2*1024 = 10240
    # But Megatron uses a single fused QKV linear:
    # linear_qkv: (H -> qkv_out) = (8192 -> 10240)

    shapes = {}

    # Forward GEMMs (Y = X @ W^T, so hipb_mm(X, W.t()) where W is (N,K))
    # hipb_mm sees: mat1=(M,K), mat2=(K,N) after .t() view

    # Attention QKV
    shapes["attn_qkv_fwd"] = (M, qkv_out, H)  # 8192 x 10240 x 8192
    shapes["attn_qkv_dgrad"] = (M, H, qkv_out)  # 8192 x 8192 x 10240

    # Attention output projection
    shapes["attn_proj_fwd"] = (M, H, H)  # 8192 x 8192 x 8192
    shapes["attn_proj_dgrad"] = (M, H, H)  # same

    # MLP FC1 (gate + up, fused)
    shapes["mlp_fc1_fwd"] = (M, FFN * 2, H)  # 8192 x 57344 x 8192
    # Wait - with SwiGLU, FC1 produces gate+up = 2*FFN
    # Actually Megatron's FC1 for SwiGLU outputs 2*ffn_hidden
    # But with TP=1, the full output is 2*28672 = 57344
    shapes["mlp_fc1_dgrad"] = (M, H, FFN * 2)  # 8192 x 8192 x 57344

    # MLP FC2
    shapes["mlp_fc2_fwd"] = (M, H, FFN)  # 8192 x 8192 x 28672
    shapes["mlp_fc2_dgrad"] = (M, FFN, H)  # 8192 x 28672 x 8192

    # LoRA GEMMs (tiny)
    RANK = 16
    shapes["lora_A_fwd"] = (M, RANK, H)  # 8192 x 16 x 8192
    shapes["lora_B_fwd"] = (M, H, RANK)  # 8192 x 8192 x 16
    shapes["lora_A_wgrad"] = (RANK, H, M)  # 16 x 8192 x 8192 (grad.T @ input)
    shapes["lora_B_wgrad"] = (H, RANK, M)  # 8192 x 16 x 8192

    print(f"{'Name':<25} {'M':>6} {'N':>6} {'K':>6}  {'TFLOPS':>8}")
    print("-" * 60)
    total_flops = 0
    for name, (m, n, k) in shapes.items():
        flops = 2 * m * n * k
        tflops = flops / 1e12
        total_flops += flops
        print(f"{name:<25} {m:>6} {n:>6} {k:>6}  {tflops:>8.3f}")

    print("-" * 60)
    print(f"{'Total per layer':<25} {'':>6} {'':>6} {'':>6}  {total_flops/1e12:>8.3f}")
    print(f"{'× 80 layers':<25} {'':>6} {'':>6} {'':>6}  {total_flops*80/1e12:>8.3f}")
    print()
    print("Core GEMM shapes for hipBLASLt tuning (FP8 input, BF16 output):")
    print()
    core = [
        ("attn_qkv_fwd", 8192, 10240, 8192),
        ("attn_qkv_dgrad", 8192, 8192, 10240),
        ("attn_proj_fwd", 8192, 8192, 8192),
        ("attn_proj_dgrad", 8192, 8192, 8192),
        ("mlp_fc1_fwd", 8192, 57344, 8192),
        ("mlp_fc1_dgrad", 8192, 8192, 57344),
        ("mlp_fc2_fwd", 8192, 8192, 28672),
        ("mlp_fc2_dgrad", 8192, 28672, 8192),
    ]
    for name, m, n, k in core:
        print(f"  {name}: M={m}, N={n}, K={k}")
