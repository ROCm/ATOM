"""Sweep gemm_a8w8 (per-Tensor FP8 path, gfx1201) across:
  - 4 Mistral-3 shapes: qkv (6144x4096), o (4096x4096), gate_up (28672x4096), down (4096x14336)
  - 6 batch sizes: 1, 2, 4, 8, 16, 32
  - 4 candidate configs (current pinned + 3 alternatives)

Goal: find if the current bs=1-tuned config is still optimal at higher bs.

Output: per (shape, bs), best config and time vs current pinned.
"""

import os

os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")

import torch
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

torch.manual_seed(0)
DEV = "cuda"
fp8 = torch.float8_e4m3fn

SHAPES = [
    ("qkv", 6144, 4096),
    ("o", 4096, 4096),
    ("gate_up", 28672, 4096),
    ("down", 4096, 14336),
]

BS_LIST = [1, 2, 4, 8, 16, 32]


# Candidate configs to test. Current pinned configs (from _gfx1201_gemm_a8w8_config):
def cfg_pinned(N, K):
    if N >= 16384:  # gate_up
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "NUM_KSPLIT": 1,
            "num_warps": 8,
            "num_stages": 2,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
            "cache_modifier": None,
            "SPLITK_BLOCK_SIZE": 4096,
            "_label": "pin_M64_N64",
        }
    if K >= 8192:  # down
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "NUM_KSPLIT": 1,
            "num_warps": 8,
            "num_stages": 2,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
            "cache_modifier": None,
            "SPLITK_BLOCK_SIZE": K,
            "_label": "pin_M16_N128",
        }
    return {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "NUM_KSPLIT": 1,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
        "cache_modifier": None,
        "SPLITK_BLOCK_SIZE": 4096,
        "_label": "pin_M16_N128",
    }


# Alternatives to test at higher bs — bigger M tile:
def cfg_alts(N, K):
    # SPLITK_BLOCK_SIZE must be >= K (with NUM_KSPLIT=1) for correctness —
    # otherwise the kernel only processes the first SPLITK_BLOCK_SIZE columns
    # of K and silently produces wrong output. Use K directly.
    splitk = max(K, 4096)

    def base(M_, Nn, K_, gm, nw):
        return {
            "BLOCK_SIZE_M": M_,
            "BLOCK_SIZE_N": Nn,
            "BLOCK_SIZE_K": K_,
            "GROUP_SIZE_M": gm,
            "NUM_KSPLIT": 1,
            "num_warps": nw,
            "num_stages": 2,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
            "cache_modifier": None,
            "SPLITK_BLOCK_SIZE": splitk,
        }

    cands = [
        {**base(32, 128, 128, 1, 8), "_label": "M32_N128"},
        {**base(64, 128, 128, 1, 8), "_label": "M64_N128"},
        {**base(64, 64, 128, 1, 8), "_label": "M64_N64"},
        {**base(16, 256, 128, 1, 8), "_label": "M16_N256"},
        {**base(32, 64, 128, 1, 8), "_label": "M32_N64"},
    ]
    return cands


WARMUP, REPS = 5, 30


def bench(cfg, M, N, K):
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV) * 0.1
    w = torch.randn(N, K, dtype=torch.bfloat16, device=DEV) * 0.1
    x_q = x.clamp(-448, 448).to(fp8)
    w_q = w.clamp(-448, 448).to(fp8)
    x_scale = torch.ones(M, 1, dtype=torch.float32, device=DEV)
    w_scale = torch.ones(1, N, dtype=torch.float32, device=DEV)

    cfg_clean = {k: v for k, v in cfg.items() if not k.startswith("_")}

    # Correctness check vs reference (BF16 matmul of dequant'd FP8)
    try:
        x_bf = x_q.to(torch.float32).to(torch.bfloat16)
        w_bf = w_q.to(torch.float32).to(torch.bfloat16)
        y_ref = x_bf @ w_bf.T
        y = gemm_a8w8(
            x_q, w_q, x_scale, w_scale, dtype=torch.bfloat16, config=cfg_clean
        )
        torch.cuda.synchronize()
        if (y - y_ref).abs().max().item() > 0.5:
            return None, "WRONG_OUTPUT"
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"

    # Warmup
    try:
        for _ in range(WARMUP):
            _ = gemm_a8w8(
                x_q, w_q, x_scale, w_scale, dtype=torch.bfloat16, config=cfg_clean
            )
        torch.cuda.synchronize()
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"

    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPS):
        _ = gemm_a8w8(
            x_q, w_q, x_scale, w_scale, dtype=torch.bfloat16, config=cfg_clean
        )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / REPS * 1000, None  # us


print(
    f"{'shape':<10} {'bs':>3}  {'pinned':<14} {'best':<14} {'best_us':>9}  {'pin_us':>9}  {'gain':>6}"
)
print("-" * 88)
for name, N, K in SHAPES:
    pinned = cfg_pinned(N, K)
    cands = [pinned] + cfg_alts(N, K)
    for bs in BS_LIST:
        results = []
        first_err = None
        for cfg in cands:
            t, err = bench(cfg, bs, N, K)
            if err:
                if first_err is None:
                    first_err = (cfg["_label"], err)
                continue
            results.append((cfg["_label"], t))
        if not results:
            err_lbl, err_msg = first_err
            print(f"{name:<10} {bs:>3}  ALL FAILED  first: {err_lbl}: {err_msg[:60]}")
            continue
        results.sort(key=lambda x: x[1])
        pin_us = next(t for lbl, t in results if lbl == pinned["_label"])
        best_lbl, best_us = results[0]
        gain = 100 * (pin_us - best_us) / pin_us
        print(
            f"{name:<10} {bs:>3}  {pinned['_label']:<14} {best_lbl:<14} {best_us:>9.1f}  {pin_us:>9.1f}  {gain:>5.1f}%"
        )
