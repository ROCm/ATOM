"""Micro-bench wo_a kernel options head-to-head: einsum vs G-loop blockscale.

Companion to scripts/v4_wo_a_tune.py --microbench (which compares BMM vs einsum).
This script measures the alternative G-loop blockscale path used by PR #677.

Usage on GPU node:
    python /shared/amdgpu/home/zufa_yu_qle/ATOM/scripts/v4_wo_a_microbench_gloop.py
"""

import torch
import triton

from atom.model_ops.linear import gemm_a8w8_blockscale_preshuffle_impl
from aiter import QuantType, get_hip_quant


# V4 wo_a tp=8 shape
B = 2
K = 4096
N = 1024
GROUP_SIZE = 128

M_LIST = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def main():
    quant_func = get_hip_quant(QuantType.per_1x128)
    print("V4 wo_a G-loop FP8 blockscale vs BF16 einsum micro-bench")
    print(f"  shape: B={B}, K={K}, N={N}, group_size={GROUP_SIZE}")
    print(f"  M values: {M_LIST}")
    print()
    print(f"{'M':>5} {'einsum (us)':>14} {'G-loop (us)':>14} {'speedup':>10}")
    print(f"{'':>5} {'-' * 14:>14} {'-' * 14:>14} {'-' * 10:>10}")

    for M in M_LIST:
        torch.manual_seed(0)
        o = torch.randn(M, B, K, dtype=torch.bfloat16, device="cuda")
        w_bf16 = torch.randn(B, N, K, dtype=torch.bfloat16, device="cuda")

        # Pre-quantize the wo_a weight as a single (B*N, K) tensor matching
        # how LinearBase stores it post-load.
        w_bf16_flat = w_bf16.reshape(B * N, K).contiguous()
        w_fp8, w_scale = quant_func(
            w_bf16_flat, quant_dtype=torch.float8_e4m3fn, scale=None
        )

        # einsum baseline
        einsum_ms = triton.testing.do_bench(
            lambda: torch.einsum("sgd,grd->sgr", o, w_bf16),
            warmup=25,
            rep=100,
        )

        # G-loop blockscale (PR #677's path)
        def run_gloop():
            outs = []
            for g in range(B):
                x_g = o[:, g, :].contiguous()
                x_q, x_s = quant_func(
                    x_g,
                    quant_dtype=torch.float8_e4m3fn,
                    scale=None,
                    transpose_scale=True,
                )
                W_g = w_fp8[g * N : (g + 1) * N]
                S_g = w_scale[g * (N // 128) : (g + 1) * (N // 128)]
                y_g = gemm_a8w8_blockscale_preshuffle_impl(
                    x_q,
                    W_g,
                    x_s,
                    S_g,
                    dtype=torch.bfloat16,
                    prefix=f"wo_a.g{g}",
                )
                outs.append(y_g)
            return torch.stack(outs, dim=1)

        gloop_ms = triton.testing.do_bench(run_gloop, warmup=25, rep=100)
        speedup = einsum_ms / gloop_ms if gloop_ms > 0 else float("inf")
        print(
            f"{M:>5} {einsum_ms*1000:>12.2f}   {gloop_ms*1000:>12.2f}   "
            f"{speedup:>8.2f}x"
        )

    print()
    print("note: G-loop = act_quant + GEMM × G_local (B). Includes per-group")
    print("      Python overhead. cudagraph would amortize the loop overhead.")


if __name__ == "__main__":
    main()
