# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Microbench prefix-cache MLA prefill: dequant+mla_prefill_fwd vs gather+fmha.

Determines the total_kv crossover point where the new path becomes faster (or
at least not slower) than the old materialized path. Run once to pick a
defensible ATOM_MLA_PREFILL_KV_THRESHOLD.

DSR1 shape, TP=8 → per-rank nhead=16. Workload mimics prefix-cache HIT:
short new q_len (8), variable shared cached prefix per request, variable
batch size; total_kv = batch * (prefix_len + q_len).

Usage:
  python -m atom.examples.bench_mla_prefix_paths
"""

import argparse
import math
import statistics

import torch

import aiter
from aiter import dtypes
from aiter.mla import mla_prefill_fwd
from aiter.ops.triton.gather_kv_b_proj import gather_kv_b_proj
from aiter import flash_attn_varlen_func

# DSR1 dims (per-rank TP=8).
NHEAD = 16
KV_LORA = 512
QK_ROPE = 64
QK_NOPE = 128
V_DIM = 128
QK = KV_LORA + QK_ROPE  # 576
NON_ABS_QK = QK_NOPE + QK_ROPE  # 192


def _sync_time(fn, iters: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def run_one(batch: int, prefix_len: int, q_len: int, warmup=5, iters=20):
    device = torch.device("cuda")
    sm_scale = 1.0 / math.sqrt(QK)
    total_q = batch * q_len
    total_kv = batch * (prefix_len + q_len)

    cu_q = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_q[1:] = q_len
    cu_q = torch.cumsum(cu_q, dim=0).to(torch.int32)
    cu_kv = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_kv[1:] = prefix_len + q_len
    cu_kv = torch.cumsum(cu_kv, dim=0).to(torch.int32)
    kv_last_page = torch.ones(batch, dtype=torch.int32, device=device)

    # Paged fp8 KV cache (block_size=1; ATOM MLA uses block_size=1).
    num_page = total_kv
    kv_buf_fp8 = (torch.randn(num_page, 1, 1, QK, device=device) * 0.1).to(dtypes.fp8)
    kv_indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    k_scale = torch.ones([1], dtype=torch.float32, device=device)

    # ---- Path A (OLD): non-absorbed q + gather_kv_b_proj + flash_attn_varlen ----
    q_old = torch.randn(total_q, NHEAD, NON_ABS_QK, device=device, dtype=dtypes.bf16)
    # kv_b_proj weight [NHEAD * (QK_NOPE + V_DIM), KV_LORA] bf16 (unquantized).
    kv_b_w = (
        torch.randn(
            NHEAD * (QK_NOPE + V_DIM), KV_LORA, device=device, dtype=dtypes.bf16
        )
        * 0.02
    )

    def run_old():
        k_full = torch.empty(
            total_kv, NHEAD, NON_ABS_QK, device=device, dtype=dtypes.bf16
        )
        v_full = torch.empty(total_kv, NHEAD, V_DIM, device=device, dtype=dtypes.bf16)
        gather_kv_b_proj(
            kv_buf_fp8.view(num_page, 1, QK),
            k_scale,
            cu_kv,
            kv_indices,
            cu_kv,  # kv_prefix_sum_context_lens — same as cu_kv when 1 page per token
            kv_b_w,
            None,  # no scale
            k_full,
            v_full,
        )
        out = flash_attn_varlen_func(
            q=q_old,
            k=k_full,
            v=v_full,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_kv,
            max_seqlen_q=q_len,
            max_seqlen_k=prefix_len + q_len,
            min_seqlen_q=q_len,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=True,
        )
        return out

    # ---- Path B (NEW): absorbed q + dequant gather + mla_prefill_fwd ----
    q_new = torch.randn(total_q, NHEAD, QK, device=device, dtype=dtypes.bf16)

    def run_new():
        gathered = kv_buf_fp8.view(num_page, QK)[kv_indices.long()]
        gathered_bf16 = (gathered.to(torch.float32) * k_scale).to(dtypes.bf16)
        kv_buf = gathered_bf16.view(-1, 1, 1, QK)
        kv_idx = torch.arange(total_kv, dtype=torch.int32, device=device)
        out = torch.empty(total_q, NHEAD, V_DIM, device=device, dtype=dtypes.bf16)
        mla_prefill_fwd(
            q_new,
            kv_buf,
            out,
            cu_q,
            cu_kv,
            kv_idx,
            kv_last_page,
            q_len,
            sm_scale,
            0.0,
            None,
        )
        return out

    # Warmup
    for _ in range(warmup):
        run_old()
        run_new()
    torch.cuda.synchronize()

    t_old = _sync_time(run_old, iters)
    t_new = _sync_time(run_new, iters)
    return t_old, t_new, total_kv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-len", type=int, default=4096)
    parser.add_argument("--q-len", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    print(
        f"\nDSR1 shape, nhead={NHEAD} (TP=8 per-rank), q_len={args.q_len}, "
        f"prefix_len={args.prefix_len}\n"
    )
    header = f"{'batch':>6} {'total_kv':>10} {'OLD (ms)':>12} {'NEW (ms)':>12} {'winner':>10} {'speedup':>10}"
    print(header)
    print("-" * len(header))

    results = []
    for batch in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        try:
            t_old, t_new, total_kv = run_one(
                batch, args.prefix_len, args.q_len, iters=args.iters
            )
        except torch.cuda.OutOfMemoryError as e:
            print(
                f"{batch:>6} {batch*(args.prefix_len+args.q_len):>10}  OOM (OLD): {type(e).__name__}"
            )
            torch.cuda.empty_cache()
            # try NEW only
            try:
                _, t_new, total_kv = run_one(
                    batch, args.prefix_len, args.q_len, iters=args.iters
                )
                # Hack: re-run path B only by zeroing the old timing
                print(
                    f"{batch:>6} {total_kv:>10} {'OOM':>12} {t_new:>12.3f} {'NEW':>10} {'inf':>10}"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"{batch:>6}  OOM both paths")
            torch.cuda.empty_cache()
            continue
        winner = "NEW" if t_new < t_old else "OLD"
        speedup = t_old / t_new
        print(
            f"{batch:>6} {total_kv:>10} {t_old:>12.3f} {t_new:>12.3f} {winner:>10} {speedup:>10.2f}x"
        )
        results.append((batch, total_kv, t_old, t_new))
        torch.cuda.empty_cache()

    # Crossover (first total_kv where NEW <= OLD)
    print("\n--- crossover ---")
    for batch, total_kv, t_old, t_new in results:
        if t_new <= t_old:
            print(
                f"  NEW wins starting at batch={batch}, total_kv={total_kv} "
                f"(OLD {t_old:.2f} ms, NEW {t_new:.2f} ms)"
            )
            break
    else:
        print("  NEW never wins in tested range")


if __name__ == "__main__":
    main()
