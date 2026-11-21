# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-layer MXFP4 training GEMM cost, across models and backend-selection policies.

``bench_mxfp4_gemm.py`` covers the forward GEMMs of one model. This one covers a
whole training layer -- all three GEMMs of every projection -- for several models,
and compares the three ways of choosing a backend:

    plain      always the plain Triton kernel (what Lumen did before any of this)
    static     the hand-measured byte thresholds
    autotune   whichever legal backend is actually fastest for the shape

The point of the comparison is that the static thresholds were fitted to Llama
3.1 8B and do not transfer: Qwen3-8B's MLP weights are 24 MiB against Llama's
28 MiB, which puts them on the wrong side of a constant chosen for the other
model.

Run:
    pytest benchmarks/bench_mxfp4_gemm_models.py -v -s
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
    _mxfp4_asm_supported,
    _mxfp4_preshuffle_eligible,
    _mxfp4_preshuffle_supported,
)
from atomrl.lumen.ops.quantize.ops import convert_to_mxfp4

from benchmarks.bench_utils import cuda_timer
from benchmarks.conftest import AITER

MXFP4_BLOCK = 32
TOKENS = 8192

# name: (hidden, intermediate, q_dim, kv_dim)
MODELS = {
    "Qwen3-0.6B": (1024, 3072, 16 * 128, 8 * 128),
    "Qwen3-8B": (4096, 12288, 32 * 128, 8 * 128),
    "Llama-3.1-8B": (4096, 14336, 32 * 128, 8 * 128),
}


def _projections(hidden, inter, q_dim, kv_dim):
    return [
        ("q_proj", q_dim, hidden),
        ("k_proj", kv_dim, hidden),
        ("v_proj", kv_dim, hidden),
        ("o_proj", hidden, q_dim),
        ("gate_proj", inter, hidden),
        ("up_proj", inter, hidden),
        ("down_proj", hidden, inter),
    ]


def _layer_gemms(hidden, inter, q_dim, kv_dim, tokens):
    """Every (M, N, K) a training step issues for one transformer layer.

    Each linear contributes three: the forward GEMM, and the two backward ones,
    which permute the dims -- dgrad swaps the output and reduction widths, wgrad
    reduces over the token count.
    """
    out = []
    for name, n_out, k_in in _projections(hidden, inter, q_dim, kv_dim):
        out.append((f"{name}.fprop", tokens, n_out, k_in))
        out.append((f"{name}.dgrad", tokens, k_in, n_out))
        out.append((f"{name}.wgrad", n_out, k_in, tokens))
    return out


def _asm_kernel_only(a_fp4, w_fp4, a_s, w_s):
    """The ASM GEMM with the operand layout already built.

    Lumen rebuilds that layout on every call, because the weight is requantized
    each step and nothing upstream produces the tiled/swizzled form. Timing the
    kernel without it shows how much of the ASM path's cost is the prologue --
    that is the headroom a fused cast+shuffle quantize kernel would recover.
    """
    import aiter
    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.triton.utils._triton.arch_info import get_arch
    from atomrl.lumen.ops.quantize.linear import (
        _MXFP4_SCALE_SHUFFLE_TILING,
        _pad_and_swizzle_mxfp4_scale,
    )

    arch = get_arch()
    tiling = _MXFP4_SCALE_SHUFFLE_TILING[arch]
    w_shuf = shuffle_weight(w_fp4, layout=(16, 16))
    sa = _pad_and_swizzle_mxfp4_scale(a_s, arch, tiling)
    sw = _pad_and_swizzle_mxfp4_scale(w_s, arch, tiling)
    return lambda: aiter.gemm_a4w4(a_fp4, w_shuf, sa, sw, dtype=torch.bfloat16)


def _time_backends(M, N, K):
    """ms/iter for each backend that can legally run this shape."""
    a_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_hp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    a_fp4, a_s = convert_to_mxfp4(a_hp, block_size=MXFP4_BLOCK, axis=-1, use_sr=False)
    w_fp4, w_s = convert_to_mxfp4(w_hp, block_size=MXFP4_BLOCK, axis=-1, use_sr=False)
    del a_hp, w_hp
    torch.cuda.empty_cache()

    times = {
        "plain": cuda_timer(
            lambda a_fp4=a_fp4, a_s=a_s, w_fp4=w_fp4, w_s=w_s: _gemm_mxfp4_aiter(
                a_fp4, w_fp4, a_s, w_s
            )
        ).avg_ms
    }
    legal_static = "plain"

    if _mxfp4_preshuffle_supported(a_fp4, w_fp4):
        times["shuffled"] = cuda_timer(
            lambda a_fp4=a_fp4, a_s=a_s, w_fp4=w_fp4, w_s=w_s: _gemm_mxfp4_aiter_preshuffle(
                a_fp4, w_fp4, a_s, w_s
            )
        ).avg_ms
        if _mxfp4_preshuffle_eligible(a_fp4, w_fp4):
            legal_static = "shuffled"
    if _mxfp4_asm_supported(a_fp4, w_fp4):
        times["asm"] = cuda_timer(
            lambda a_fp4=a_fp4, a_s=a_s, w_fp4=w_fp4, w_s=w_s: _gemm_mxfp4_aiter_asm(
                a_fp4, w_fp4, a_s, w_s
            )
        ).avg_ms
        times["asm_nolayout"] = cuda_timer(
            _asm_kernel_only(a_fp4, w_fp4, a_s, w_s)
        ).avg_ms
        if _mxfp4_asm_eligible(a_fp4, w_fp4):
            legal_static = "asm"

    del a_fp4, w_fp4, a_s, w_s
    torch.cuda.empty_cache()
    return times, legal_static


@AITER
@pytest.mark.parametrize("model", list(MODELS), ids=list(MODELS))
def test_mxfp4_layer_backend_policies(model):
    """Sum a training layer's GEMMs under each backend-selection policy."""
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("torch.float4_e2m1fn_x2 unavailable in this PyTorch build")

    hidden, inter, q_dim, kv_dim = MODELS[model]
    totals = {"plain": 0.0, "static": 0.0, "autotune": 0.0, "no_prologue": 0.0}
    picks = {"static": {}, "autotune": {}}
    measured = {}

    print(f"\n  {model}  tokens={TOKENS}")
    print(
        f"  {'gemm':<18} {'M':>6} {'N':>6} {'K':>6} {'wMiB':>6} "
        f"{'plain':>8} {'shuf':>8} {'asm':>8} {'asm-noprol':>11}  "
        f"{'static':<9} {'auto':<9}"
    )

    for label, M, N, K in _layer_gemms(hidden, inter, q_dim, kv_dim, TOKENS):
        key = (M, N, K)
        if key not in measured:
            times, static_pick = _time_backends(M, N, K)
            # Mirror the dispatcher: only leave the plain kernel for a clear win.
            auto_pick = min(
                (n for n in times if n != "asm_nolayout"), key=lambda n: times[n]
            )
            if times[auto_pick] * mxfp4_autotune._SWITCH_MARGIN > times["plain"]:
                auto_pick = "plain"
            measured[key] = (times, static_pick, auto_pick)
        # The dispatcher caches per shape, so repeated shapes reuse the decision
        # rather than re-rolling it against measurement noise.
        times, static_pick, auto_pick = measured[key]

        totals["plain"] += times["plain"]
        totals["static"] += times[static_pick]
        totals["autotune"] += times[auto_pick]
        totals["no_prologue"] += min(
            times[auto_pick], times.get("asm_nolayout", float("inf"))
        )
        for k, v in (("static", static_pick), ("autotune", auto_pick)):
            picks[k][v] = picks[k].get(v, 0) + 1

        def show(name, width=8, times=times):
            return f"{times[name]:{width}.3f}" if name in times else f"{'-':>{width}}"

        print(
            f"  {label:<18} {M:>6} {N:>6} {K:>6} {N * K / 2 / 1024 / 1024:>6.1f} "
            f"{show('plain')} {show('shuffled')} {show('asm')} "
            f"{show('asm_nolayout', 11)}  {static_pick:<9} {auto_pick:<9}"
        )

    print(f"\n  per-layer training GEMM total, {model}:")
    for k in ("plain", "static", "autotune"):
        speedup = totals["plain"] / totals[k]
        chosen = "  ".join(f"{n}={c}" for n, c in sorted(picks.get(k, {}).items()))
        print(f"    {k:<11} {totals[k]:7.3f} ms  ({speedup:.2f}x vs plain)   {chosen}")
    print(
        f"    {'no-prologue':<11} {totals['no_prologue']:7.3f} ms  "
        f"({totals['plain'] / totals['no_prologue']:.2f}x vs plain)   "
        f"what a fused cast+shuffle quantize kernel would leave on the table"
    )
    print(
        f"    autotune is {totals['static'] / totals['autotune']:.2f}x "
        f"the static-threshold policy"
    )

    # Measuring can only ever match or beat a fixed threshold.
    assert (
        totals["autotune"] <= totals["static"] * 1.02
    ), f"autotune {totals['autotune']:.3f}ms worse than static {totals['static']:.3f}ms"


if __name__ == "__main__":
    os.environ.setdefault("LUMEN_BENCH_ITERS", "30")
    pytest.main([__file__, "-v", "-s"])
