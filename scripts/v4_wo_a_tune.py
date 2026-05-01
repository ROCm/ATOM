"""Tune the aiter Triton FP8 BMM kernel for V4 wo_a shape.

Sweeps configs at our shape (B=2, K=4096, N=1024) and writes a JSON config
file in aiter's expected location. Next V4 run will auto-pick it up via
the kernel's _get_config(M, N, K) lookup.

Usage on GPU node:
    python /shared/amdgpu/home/zufa_yu_qle/ATOM/scripts/v4_wo_a_tune.py
    # or to also write the JSON config to aiter:
    python /shared/amdgpu/home/zufa_yu_qle/ATOM/scripts/v4_wo_a_tune.py --write
"""

import argparse
import json
import os
import time
from itertools import product
from pathlib import Path

import torch
import triton

from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as fp8_bmm,
)
import aiter.ops.triton as _aiter_triton

# V4 wo_a shape on tp=8
B = 2          # n_local_groups
K = 4096       # d_per_group
N = 1024       # o_lora_rank
GROUP_SIZE = 128

# M ranges that matter:
#   decode: 1, 4, 8, 16   (--max-num-seqs 4 → batch up to ~16 with cudagraph rounding)
#   prefill: 64, 256, 512, 1024  (chunked prefill chunks)
M_LIST = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# kernel's _get_config bucketing breakpoints (mirror existing JSON keys)
M_BUCKETS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

# Configs to sweep — keep BLOCK_SIZE_K = group_size = 128 (kernel constraint)
SWEEP = list(
    product(
        [16, 32, 64, 128, 256],     # BLOCK_SIZE_M
        [16, 32, 64, 128, 256],     # BLOCK_SIZE_N
        [1, 2, 4, 8, 16],           # GROUP_SIZE_M
        [4, 8],                     # num_warps
        [1, 2, 3],                  # num_stages
        [0, 1, 2, 3, 4],            # waves_per_eu
    )
)


def make_config(bm, bn, gm, num_warps, num_stages, waves_per_eu):
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "GROUP_SIZE_M": gm,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "waves_per_eu": waves_per_eu,
        "matrix_instr_nonkdim": 16,
        "kpack": 2,
        "cache_modifier": ".cg",
    }


def dynamic_per_batched_tensor_quant(x, dtype=torch.float8_e4m3fn):
    DTYPE_MAX = torch.finfo(dtype).max
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-10)
    scale = DTYPE_MAX / amax
    x_scl_sat = (x * scale).clamp(min=-DTYPE_MAX, max=DTYPE_MAX)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def benchmark(M, config_dict=None, n_iter=30, n_warmup=5):
    torch.manual_seed(0)
    x = torch.randn(M, B, K, dtype=torch.bfloat16, device="cuda")
    w_bf16 = torch.randn(B, N, K, dtype=torch.bfloat16, device="cuda")
    w_fp8, w_scale = dynamic_per_batched_tensor_quant(w_bf16)

    def run():
        return fp8_bmm(
            x, w_fp8, w_scale,
            group_size=GROUP_SIZE,
            transpose_bm_in=True, transpose_bm=True,
            config=config_dict,
        )

    try:
        for _ in range(n_warmup):
            run()
        torch.cuda.synchronize()
        # Use triton.testing.do_bench for proper measurement (steady-state)
        ms = triton.testing.do_bench(run, warmup=10, rep=50)
        return ms * 1e3  # μs
    except Exception:
        return float("inf")


def find_bucket(M):
    """Map M to its bucket label per aiter's _get_config convention."""
    for bp in M_BUCKETS:
        if M <= bp:
            return f"M_LEQ_{bp}"
    return f"M_GREATER_THAN_{M_BUCKETS[-1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="Write tuned JSON to aiter configs dir")
    ap.add_argument("--quick", action="store_true",
                    help="Coarse sweep (only 2 num_warps × 1 num_stages)")
    ap.add_argument("--m-list", nargs="+", type=int, default=M_LIST,
                    help="M values to tune (decode + prefill samples)")
    args = ap.parse_args()

    sweep = SWEEP
    if args.quick:
        sweep = [c for c in sweep
                 if c[3] in (8,) and c[4] in (2,) and c[5] in (1,)]
    print(f"V4 wo_a FP8 BMM tune sweep")
    print(f"  shape: B={B}, K={K}, N={N}, group_size={GROUP_SIZE}")
    print(f"  configs to sweep per M: {len(sweep)}")
    print(f"  M values: {args.m_list}")
    print()

    # Baseline: kernel default
    print("===== Baseline (kernel _get_config default) =====")
    print(f"{'M':>6} {'us':>10}")
    baseline = {}
    for M in args.m_list:
        t = benchmark(M, config_dict=None)
        baseline[M] = t
        print(f"{M:>6} {t:>10.2f}")
    print()

    # Per-M best config
    best_per_m = {}
    for M in args.m_list:
        print(f"===== M={M} sweep ({len(sweep)} configs) =====")
        results = []
        for cfg_tuple in sweep:
            bm, bn, gm, nw, ns, we = cfg_tuple
            # skip configs where BLOCK_SIZE_M is way bigger than M
            # (still allow bm=16 for tiny M since 16 is min)
            if bm > max(M, 16) * 2:
                continue
            cfg = make_config(bm, bn, gm, nw, ns, we)
            t = benchmark(M, config_dict=cfg, n_iter=15, n_warmup=3)
            results.append((t, cfg, cfg_tuple))
        results.sort(key=lambda r: r[0])
        baseline_t = baseline[M]
        print(f"  baseline: {baseline_t:>8.2f}us")
        for t, _, cfg_tup in results[:5]:
            speedup = baseline_t / t if t > 0 else 0
            mark = "***" if t < baseline_t * 0.95 else "   "
            bm, bn, gm, nw, ns, we = cfg_tup
            print(f"  {mark} bm={bm:3} bn={bn:3} gm={gm} nw={nw} ns={ns} we={we}  "
                  f"{t:>8.2f}us  ({speedup:.2f}x)")
        if results:
            best_per_m[M] = (results[0][0], results[0][1])
        print()

    # Bucket best configs by aiter's M_LEQ_* convention
    bucketed = {}
    for M, (t, cfg) in best_per_m.items():
        bucket = find_bucket(M)
        # take fastest within bucket
        if bucket not in bucketed or bucketed[bucket][0] > t:
            bucketed[bucket] = (t, cfg)

    # Print summary + format as aiter JSON
    print("===== Best per bucket =====")
    aiter_json = {}
    for bucket in sorted(bucketed.keys(), key=lambda b: (
        int(b.split("_LEQ_")[1]) if "_LEQ_" in b else 1 << 30
    )):
        t, cfg = bucketed[bucket]
        aiter_json[bucket] = cfg
        print(f"  {bucket}: {t:.2f}us  config={cfg}")
    print()

    # Where aiter expects the JSON
    aiter_root = Path(_aiter_triton.__file__).parent
    target_path = (
        aiter_root
        / "configs"
        / "gemm"
        / f"gfx942-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N={N}-K={K}.json"
    )

    if args.write:
        target_path.write_text(json.dumps(aiter_json, indent=4))
        print(f"WROTE tuned config to: {target_path}")
        print(f"Next V4 run will auto-pick this via _get_config(M, N={N}, K={K}).")
    else:
        print(f"DRY RUN. To write: rerun with --write")
        print(f"Target path would be: {target_path}")
        print(f"JSON content preview:")
        print(json.dumps(aiter_json, indent=4))


if __name__ == "__main__":
    main()
