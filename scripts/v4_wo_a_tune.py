"""Micro-bench autotune for V4 wo_a FP8 BMM kernel.

Sweeps (BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M) for the V4 wo_a shape and
reports best per (M=prefill_S, M=decode_S) regime.

Usage on GPU node:
    python /shared/amdgpu/home/zufa_yu_qle/ATOM/scripts/v4_wo_a_tune.py
"""

import time
import torch
from itertools import product

from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as fp8_bmm,
)


def dynamic_per_batched_tensor_quant(x, dtype=torch.float8_e4m3fn):
    DTYPE_MAX = torch.finfo(dtype).max
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-10)
    scale = DTYPE_MAX / amax
    x_scl_sat = (x * scale).clamp(min=-DTYPE_MAX, max=DTYPE_MAX)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


# V4 wo_a shape on tp=8
B = 2          # n_local_groups
K = 4096       # d_per_group = n_local_heads * head_dim / n_local_groups
N = 1024       # o_lora_rank
GROUP_SIZE = 128

# Test M values: representative of prefill / chunk / decode batches
M_LIST = [4, 8, 16, 64, 256, 512, 1024]

# Configs to sweep
SWEEP = list(
    product(
        [16, 32, 64, 128],          # BLOCK_SIZE_M
        [32, 64, 128, 256],         # BLOCK_SIZE_N
        [1, 2, 4, 8],               # GROUP_SIZE_M (L2 grouping, not batch)
    )
)


def benchmark(M, config_dict=None, n_iter=50, n_warmup=10):
    torch.manual_seed(0)
    x = torch.randn(M, B, K, dtype=torch.bfloat16, device="cuda")
    w_bf16 = torch.randn(B, N, K, dtype=torch.bfloat16, device="cuda")
    w_fp8, w_scale = dynamic_per_batched_tensor_quant(w_bf16)

    # warmup
    for _ in range(n_warmup):
        _ = fp8_bmm(
            x, w_fp8, w_scale,
            group_size=GROUP_SIZE,
            transpose_bm_in=True, transpose_bm=True,
            config=config_dict,
        )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = fp8_bmm(
            x, w_fp8, w_scale,
            group_size=GROUP_SIZE,
            transpose_bm_in=True, transpose_bm=True,
            config=config_dict,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e6  # μs


def make_config(bm, bn, gm):
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": GROUP_SIZE,   # fixed to per-token group size
        "GROUP_SIZE_M": gm,
        "kpack": 1,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
    }


def main():
    print(f"V4 wo_a FP8 BMM autotune sweep")
    print(f"  shape: B={B}, K={K}, N={N}, group_size={GROUP_SIZE}")
    print(f"  configs to sweep: {len(SWEEP)}")
    print()

    # Baseline: kernel's default _get_config
    print("===== Baseline (kernel default _get_config) =====")
    print(f"{'M':>6} {'us':>10}")
    baseline = {}
    for M in M_LIST:
        try:
            t = benchmark(M, config_dict=None, n_iter=30, n_warmup=5)
            baseline[M] = t
            print(f"{M:>6} {t:>10.1f}")
        except Exception as e:
            print(f"{M:>6} FAIL: {e}")
            baseline[M] = float("inf")
    print()

    # Per-M best config
    for M in M_LIST:
        print(f"===== M={M} sweep =====")
        results = []
        for bm, bn, gm in SWEEP:
            if bm > max(M, 16):  # skip block size > rounded M
                continue
            cfg = make_config(bm, bn, gm)
            try:
                t = benchmark(M, config_dict=cfg, n_iter=20, n_warmup=3)
                results.append((t, bm, bn, gm))
            except Exception:
                pass
        results.sort()
        baseline_t = baseline[M]
        print(f"  baseline: {baseline_t:>8.1f}us")
        for t, bm, bn, gm in results[:5]:
            speedup = baseline_t / t
            mark = "***" if t < baseline_t else "   "
            print(f"  {mark} bm={bm:3} bn={bn:3} gm={gm}  {t:>8.1f}us  ({speedup:.2f}x)")
        print()


if __name__ == "__main__":
    main()
