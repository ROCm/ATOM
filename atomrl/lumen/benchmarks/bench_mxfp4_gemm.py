# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 GEMM backend selection benchmark.

Exercises ``gemm_mxfp4_dispatch`` at Llama 3.1 8B linear shapes across the three
AITER MXFP4 kernels Lumen can reach, to show what each buys and why the choice
has to be made per shape rather than globally.

Expected effect: the prebuilt A4W4 ASM/CK kernels win everywhere they are tuned,
since they come with a per-shape kernel and split-K choice. Between the two
Triton kernels, the shuffled layout roughly halves GEMM time on the two MLP
projections -- the packed FP4 weight is large enough that the GEMM is
weight-streaming bound and coalesced tile reads dominate the shuffle prologue --
while the plain kernel stays ahead on the attention projections, where the weight
is small and the prologue is not amortised.

Run:
    pytest benchmarks/bench_mxfp4_gemm.py -v -s
"""

import os

import pytest
import torch
from atomrl.lumen.ops.quantize import mxfp4_autotune
from atomrl.lumen.ops.quantize.linear import (
    _gemm_mxfp4_aiter,
    _gemm_mxfp4_aiter_asm,
    _gemm_mxfp4_aiter_preshuffle,
    _mxfp4_asm_eligible,
    _mxfp4_preshuffle_eligible,
    gemm_mxfp4_dispatch,
)
from atomrl.lumen.ops.quantize.ops import convert_to_mxfp4, convert_to_mxfp4_2d

from benchmarks.bench_utils import cuda_timer, print_report_with_table
from benchmarks.conftest import AITER

HIDDEN = 4096
FFN_HIDDEN = 14336
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128

MXFP4_BLOCK = 32
TOKENS = 8192

LAYERS = [
    ("qkv_proj", NUM_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM, HIDDEN),
    ("o_proj", HIDDEN, HIDDEN),
    ("gate_up_proj", 2 * FFN_HIDDEN, HIDDEN),
    ("down_proj", HIDDEN, FFN_HIDDEN),
]


def _quantized_operands(M, N, K):
    a_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_hp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    a_fp4, a_s = convert_to_mxfp4(a_hp, block_size=MXFP4_BLOCK, axis=-1, use_sr=False)
    w_fp4, w_s = convert_to_mxfp4_2d(w_hp, block_size=MXFP4_BLOCK, use_sr=False)
    del a_hp, w_hp
    return a_fp4, w_fp4, a_s, w_s


@AITER
@pytest.mark.parametrize("layer,N,K", LAYERS, ids=[layer for layer, _, _ in LAYERS])
def test_mxfp4_gemm_backend_choice(layer, N, K):
    """Compare both AITER MXFP4 kernels and confirm the dispatcher picks the faster."""
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("torch.float4_e2m1fn_x2 unavailable in this PyTorch build")

    M = TOKENS
    a_fp4, w_fp4, a_s, w_s = _quantized_operands(M, N, K)

    # Settle the autotune decision up front. Its measurement allocates output
    # buffers for every candidate, and letting that happen inside the timed run
    # charges the dispatcher for one-time work it does not do per call.
    gemm_mxfp4_dispatch(a_fp4, w_fp4, a_s, w_s)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    results = [
        cuda_timer(
            lambda a_bf16=a_bf16, w_bf16=w_bf16: torch.mm(a_bf16, w_bf16.t()),
            label=f"{layer} bf16",
        ),
    ]
    del a_bf16, w_bf16
    torch.cuda.empty_cache()

    results.append(
        cuda_timer(
            lambda: _gemm_mxfp4_aiter(a_fp4, w_fp4, a_s, w_s),
            label=f"{layer} mxfp4 plain",
        )
    )
    try:
        results.append(
            cuda_timer(
                lambda: _gemm_mxfp4_aiter_preshuffle(a_fp4, w_fp4, a_s, w_s),
                label=f"{layer} mxfp4 shuffled",
            )
        )
    except (RuntimeError, NotImplementedError, AssertionError) as e:
        pytest.skip(f"shuffled MXFP4 GEMM unavailable: {e}")

    asm_ok = _mxfp4_asm_eligible(a_fp4, w_fp4)
    asm = None
    if asm_ok:
        results.append(
            cuda_timer(
                lambda: _gemm_mxfp4_aiter_asm(a_fp4, w_fp4, a_s, w_s),
                label=f"{layer} mxfp4 asm",
            )
        )
        asm = results[-1].avg_ms

    results.append(
        cuda_timer(
            lambda: gemm_mxfp4_dispatch(a_fp4, w_fp4, a_s, w_s),
            label=f"{layer} mxfp4 dispatch",
        )
    )
    print_report_with_table(f"MXFP4 GEMM  {layer}  M={M} N={N} K={K}", results)

    plain, shuffled, dispatch = results[1].avg_ms, results[2].avg_ms, results[-1].avg_ms
    times = {"plain": plain, "shuffled": shuffled}
    if asm is not None:
        times["asm"] = asm

    # Which backend the dispatcher settled on is a property of the autotuner, not
    # of the static thresholds, so ask it rather than re-deriving it.
    chosen = mxfp4_autotune.cached((M, N, K)) or "plain"
    print(
        f"  asm_eligible={asm_ok}  "
        f"shuffle_eligible={_mxfp4_preshuffle_eligible(a_fp4, w_fp4)}  "
        f"plain={plain:.3f}ms  shuffled={shuffled:.3f}ms  "
        f"asm={'n/a' if asm is None else f'{asm:.3f}ms'}  "
        f"dispatch={dispatch:.3f}ms  chose={chosen}"
    )

    expected = times.get(chosen, plain)
    assert (
        dispatch <= expected * 1.20
    ), f"dispatch {dispatch:.3f}ms far above its chosen backend {chosen} {expected:.3f}ms"
    # Autotune is allowed to be wrong by the margin it uses to avoid churn, but
    # not more: a decision worse than plain by a wide gap means the measurement,
    # not the ranking, is at fault.
    assert (
        dispatch <= plain * 1.20
    ), f"dispatch {dispatch:.3f}ms ({chosen}) is far worse than plain {plain:.3f}ms"


@AITER
def test_mxfp4_gemm_layer_total():
    """Sum the four projections to show the per-layer forward GEMM effect."""
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("torch.float4_e2m1fn_x2 unavailable in this PyTorch build")

    M = TOKENS
    totals = {"bf16": 0.0, "plain": 0.0, "shuffled": 0.0, "dispatch": 0.0}

    for layer, N, K in LAYERS:
        a_fp4, w_fp4, a_s, w_s = _quantized_operands(M, N, K)
        a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

        totals["bf16"] += cuda_timer(
            lambda a_bf16=a_bf16, w_bf16=w_bf16: torch.mm(a_bf16, w_bf16.t())
        ).avg_ms
        del a_bf16, w_bf16
        torch.cuda.empty_cache()

        totals["plain"] += cuda_timer(
            lambda a_fp4=a_fp4, a_s=a_s, w_fp4=w_fp4, w_s=w_s: _gemm_mxfp4_aiter(
                a_fp4, w_fp4, a_s, w_s
            )
        ).avg_ms
        # What step 1 shipped: shuffled Triton where it wins, plain otherwise.
        step1 = (
            _gemm_mxfp4_aiter_preshuffle
            if _mxfp4_preshuffle_eligible(a_fp4, w_fp4)
            else _gemm_mxfp4_aiter
        )
        totals["shuffled"] += cuda_timer(
            lambda a_fp4=a_fp4, a_s=a_s, step1=step1, w_fp4=w_fp4, w_s=w_s: step1(
                a_fp4, w_fp4, a_s, w_s
            )
        ).avg_ms
        totals["dispatch"] += cuda_timer(
            lambda a_fp4=a_fp4, a_s=a_s, w_fp4=w_fp4, w_s=w_s: gemm_mxfp4_dispatch(
                a_fp4, w_fp4, a_s, w_s
            )
        ).avg_ms

        del a_fp4, w_fp4, a_s, w_s
        torch.cuda.empty_cache()

    print(f"\n  per-layer forward GEMM total, M={M}")
    for k, v in totals.items():
        print(f"    {k:<10} {v:7.3f} ms   ({totals['bf16'] / v:.2f}x vs bf16)")
    print(
        f"    dispatch is {totals['plain'] / totals['dispatch']:.2f}x the plain path, "
        f"{totals['shuffled'] / totals['dispatch']:.2f}x the shuffled-Triton path"
    )


if __name__ == "__main__":
    os.environ.setdefault("LUMEN_BENCH_ITERS", "30")
    pytest.main([__file__, "-v", "-s"])
