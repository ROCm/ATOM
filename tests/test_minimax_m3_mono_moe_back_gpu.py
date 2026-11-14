# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono back half (``kernels/moe_back.py``: router + a4w4 routed experts + MXFP8 shared expert + FFN reduce) for 32 / 64 rows
against the torch oracle of tests/test_minimax_m3_mono_moe_a4w4_gpu.py (f64 from the logical weights, the kernel's own routing),
single rank (gfx950, 1 card). ``run_case(n, seed)`` -> (got, want, details); ``python tests/test_minimax_m3_mono_moe_back_gpu.py
[--n 32,64] [--seed 0] [--route hot|uniform]`` prints rel L2 per row and the max over rows.
"""

import argparse
import os
import sys

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("flydsl")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="needs gfx950",
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_minimax_m3_mono_moe_a4w4_gpu as A4

from atom.models.minimax_m3.mono.config import (
    HIDDEN,
    MOE_SLOTS,
    N_ROUTED,
    TOP_K,
)
from atom.models.minimax_m3.mono.kernels.moe_back import (
    BACK_ABI,
    back_scratch_layout,
    back_sym_layout,
    build_moe_back_kernel,
)
from atom.mono.runtime.uncached import uncached_bytes

DEV, BF = A4.DEV, A4.BF


def make_inputs(n, seed, route="uniform"):
    """(x_moe, h_in, g_post, gate, bias): x_moe = the normalized MoE input the front half hands over."""
    g = torch.Generator(device=DEV).manual_seed(seed * 7 + n)
    h_in = (torch.randn(n, HIDDEN, device=DEV, generator=g) * 3).to(BF)
    g_post = (torch.randn(HIDDEN, device=DEV, generator=g) * 0.3).to(BF)
    gate = (torch.randn(N_ROUTED, HIDDEN, device=DEV, generator=g) * 0.02).to(BF)
    # "uniform": zero bias (77-87 / 109-118 distinct experts at n = 32 / 64); "skew": bias dominates (18-26 distinct); "hot": 6 experts
    bias = torch.randn(N_ROUTED, device=DEV, generator=g) * (
        0.0 if route == "uniform" else 0.5
    )
    if route == "hot":  # a hot expert set: many rows per expert (multi-tile experts)
        bias = torch.zeros(N_ROUTED, device=DEV)
        bias[torch.randperm(N_ROUTED, device=DEV, generator=g)[:6]] = 3.0
    x_moe = A4.norm(h_in, g_post).to(BF).contiguous()
    return x_moe, h_in, g_post, gate, bias


def launch_args(
    n, x_moe, gate, bias, ex, ar_out, scratch, sym, peers, ids_t, layer=3, rank=0, tl=0
):
    p = lambda t: t.data_ptr()
    return BACK_ABI.pack(
        {
            "x_moe": p(x_moe), "w_gate": p(gate), "bias": p(bias), "w13": p(ex.w13), "s13": p(ex.s13), "w2": p(ex.w2),
            "s2": p(ex.s2), "w13s": p(ex.w13s), "s13s": p(ex.k13), "w2s": p(ex.w2s), "s2s": p(ex.k2), "ar_out": p(ar_out),
            "scratch": scratch.data_ptr(), "sym": sym.data_ptr(), "peers": p(peers), "rank": rank, "layer": layer,
            "batch_ids": p(ids_t), "tl": tl,
        }
    )  # fmt: skip


def production(x_moe, ex, ids, w):
    """The production MoE of this rank (aiter fused_moe, a16w4 routed experts, then the MXFP8 shared expert: bf16 sum) on the kernel's own routing
    ``ids`` [n, 4] / weights ``w`` [n, 4], from the same tensors the mono kernel reads. f32 [n, HIDDEN].
    """
    import importlib

    from aiter import ActivationType, QuantType, dtypes

    from atom.model_ops.blockscale import native_quant_linear
    from atom.model_ops.swiglu_oai import swiglu_oai_split

    fm = importlib.import_module("aiter.fused_moe")
    from atom.models.minimax_m3.mono.config import INTER

    e = N_ROUTED
    w1 = ex.w13.view(e, 2 * INTER, HIDDEN // 2).view(dtypes.fp4x2)
    w1.is_shuffled = True
    w2 = ex.w2.view(e, HIDDEN, INTER // 2).view(dtypes.fp4x2)
    w2.is_shuffled = True
    s1 = ex.s13.view(torch.float8_e8m0fnu).view(e, 2 * INTER, -1)
    s2 = ex.s2.view(torch.float8_e8m0fnu).view(e, HIDDEN, -1)
    routed = fm.fused_moe(
        x_moe, w1, w2, w.contiguous().float(), ids.to(torch.int32).contiguous(), None, ActivationType.Swiglu, QuantType.per_1x32,
        False, s1, s2, None, None, dtype=dtypes.bf16, swiglu_limit=A4.SWIGLU_LIMIT,
    )  # fmt: skip
    gu_s = ex.k13.view(
        torch.float8_e8m0fnu
    )  # the shared expert's scales stay row-major (weights.py ``_mxfp8``)
    dn_s = ex.k2.view(torch.float8_e8m0fnu)
    gu = native_quant_linear(
        x_moe,
        ex.w13s.view(dtypes.fp8),
        gu_s,
        weight_group_rows=1,
        weight_preshuffled=True,
    )
    mid = swiglu_oai_split(gu, 1.702, 1.0, A4.SWIGLU_LIMIT)
    sh = native_quant_linear(
        mid, ex.w2s.view(dtypes.fp8), dn_s, weight_group_rows=1, weight_preshuffled=True
    )
    return (routed + sh * A4.SHARED_WEIGHT).float()


def run_case(n, seed=0, route="uniform", prod=False, dead=0, **build_kw):
    ex = A4.experts_of(seed)
    x_moe, h_in, g_post, gate, bias = make_inputs(n, seed, route)
    z = lambda *shape, dt=torch.uint8: torch.zeros(*shape, dtype=dt, device=DEV)
    ar_out = z(n, HIDDEN, dt=BF)
    scratch = uncached_bytes(back_scratch_layout(n)["_bytes"], torch.device(DEV))
    sym = uncached_bytes(back_sym_layout(1, n)["_bytes"], torch.device(DEV))
    peers = torch.tensor([sym.data_ptr()], dtype=torch.int64, device=DEV)
    ids_t = torch.arange(n, dtype=torch.int32, device=DEV)
    if (
        dead
    ):  # the last ``dead`` rows are CUDA-graph pad rows (batch id -1): compared on the live rows only
        ids_t[n - dead :] = -1
    scratch.zero_()  # an uncached buffer may hold an earlier run's tagged pairs
    sym.zero_()
    launch = build_moe_back_kernel(
        1, A4.ROUTE_SCALE, A4.SHARED_WEIGHT, A4.SWIGLU_LIMIT, tokens=n, **build_kw
    )
    launch(*launch_args(n, x_moe, gate, bias, ex, ar_out, scratch, sym, peers, ids_t))
    torch.cuda.synchronize()
    off = back_scratch_layout(n)["sel"]
    words = scratch[off : off + 2 * MOE_SLOTS * n * 8].view(torch.int32)[0::2]
    sel = words.view(n, 2, MOE_SLOTS)
    ids = sel[:, 0, :TOP_K].long()
    w = sel[:, 1, :TOP_K].contiguous().view(torch.float32)
    assert (ids >= 0).all() and (ids < N_ROUTED).all(), ids
    got = ar_out.float()
    want = A4.moe_oracle(
        h_in, gate, bias, g_post, ex, None, (ids, w), act="a4w4", prod_round=True, S=n
    )
    rel = lambda a, b: ((a - b).norm(dim=-1) / b.norm(dim=-1))
    same_set, wdiff = A4._route_check(h_in, gate, bias, g_post, ids, w)
    cnt = torch.bincount(ids.flatten(), minlength=N_ROUTED)
    prod_rel = None
    if prod:
        pr = production(x_moe, ex, ids, w)
        prod_rel = rel(got, pr).cpu()
    details = {
        "rel_vs_production": prod_rel,
        "rel": rel(got, want).cpu()[: n - dead],
        "identical": (got == want).float().mean().item(),
        "route_match": bool(same_set.all()),
        "route_weight_maxdiff": wdiff,
        "distinct": int((cnt > 0).sum()),
        "max_rows_per_expert": int(cnt.max()),
        "units": int(((cnt + 15) // 16).sum()),
    }
    return got.cpu(), want.cpu(), details


def run_layers(n, layers=8, **build_kw):
    """``layers`` back-to-back launches (layer tags 3 ..) over ONE scratch / peer buffer zeroed once, as a step's layers are: other inputs, other routing
    (uniform / skew / hot in turn) and two alternating expert sets a layer. Each layer against its own oracle: stale flags or scratch from the layer
    before would show here. -> [(rel L2 median, max, identical bits)] a layer."""
    exs = [A4.experts_of(0), A4.experts_of(1)]
    scratch = uncached_bytes(back_scratch_layout(n)["_bytes"], torch.device(DEV))
    sym = uncached_bytes(back_sym_layout(1, n)["_bytes"], torch.device(DEV))
    peers = torch.tensor([sym.data_ptr()], dtype=torch.int64, device=DEV)
    ids_t = torch.arange(n, dtype=torch.int32, device=DEV)
    scratch.zero_(), sym.zero_()
    launch = build_moe_back_kernel(
        1, A4.ROUTE_SCALE, A4.SHARED_WEIGHT, A4.SWIGLU_LIMIT, tokens=n, **build_kw
    )
    out = []
    for i in range(layers):
        ex = exs[i % 2]
        x_moe, h_in, g_post, gate, bias = make_inputs(
            n, 100 + i, ("uniform", "skew", "hot")[i % 3]
        )
        ar_out = torch.zeros(n, HIDDEN, dtype=BF, device=DEV)
        launch(
            *launch_args(
                n,
                x_moe,
                gate,
                bias,
                ex,
                ar_out,
                scratch,
                sym,
                peers,
                ids_t,
                layer=3 + i,
            )
        )
        torch.cuda.synchronize()
        off = back_scratch_layout(n)["sel"]
        sel = (
            scratch[off : off + 2 * MOE_SLOTS * n * 8]
            .view(torch.int32)[0::2]
            .view(n, 2, MOE_SLOTS)
        )
        ids, w = sel[:, 0, :TOP_K].long(), sel[:, 1, :TOP_K].contiguous().view(
            torch.float32
        )
        assert (ids >= 0).all() and (ids < N_ROUTED).all(), (i, ids)
        want = A4.moe_oracle(
            h_in,
            gate,
            bias,
            g_post,
            ex,
            None,
            (ids, w),
            act="a4w4",
            prod_round=True,
            S=n,
        )
        got = ar_out.float()
        rel = ((got - want).norm(dim=-1) / want.norm(dim=-1)).cpu()
        out.append(
            (rel.median().item(), rel.max().item(), (got == want).float().mean().item())
        )
    return out


@pytest.mark.parametrize("route", ["uniform", "skew", "hot"])
@pytest.mark.parametrize("n", [32, 64])
def test_back_matches_the_oracle(n, route):
    got, _want, d = run_case(n, route=route)
    assert torch.isfinite(got).all() and got.abs().max() > 0
    assert d["rel"].median() <= 2e-3 and d["rel"].max() <= 1e-2, (d["rel"], d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="32,64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--route", default="uniform,skew,hot")
    ap.add_argument("--prod", type=int, default=0)
    ap.add_argument(
        "--dead", type=int, default=0, help="the last N rows are pad rows (batch id -1)"
    )
    ap.add_argument(
        "--layers",
        type=int,
        default=0,
        help="N > 0: N back-to-back layers over one scratch instead of the single cases",
    )
    ap.add_argument(
        "--kw", default="", help="build kwargs, comma-separated ints (steal=1)"
    )
    a = ap.parse_args()
    kw = {k: int(v) for k, _, v in (t.partition("=") for t in a.kw.split(",") if t)}
    for n in [int(x) for x in a.n.split(",")]:
        if a.layers:
            for i, (med, mx, ident) in enumerate(run_layers(n, a.layers, **kw)):
                print(
                    f"n={n} layer {3 + i} ({('uniform', 'skew', 'hot')[i % 3]}): rel L2 median {med:.2e} max {mx:.2e} identical {ident:.4f}",
                    flush=True,
                )
            continue
        for route in a.route.split(","):
            got, want, d = run_case(
                n, a.seed, route, prod=bool(a.prod), dead=a.dead, **kw
            )
            r = d["rel"]
            print(
                f"n={n} route={route} | rel L2 vs oracle median {r.median():.2e} max {r.max():.2e} | identical {d['identical']:.4f} | "
                f"router match {d['route_match']} (w {d['route_weight_maxdiff']:.1e}) | distinct {d['distinct']} "
                f"max rows/expert {d['max_rows_per_expert']} units {d['units']}"
                + (
                    f" | rel L2 vs production aiter MoE+shared: median {d['rel_vs_production'].median():.3f} max {d['rel_vs_production'].max():.3f}"
                    if d["rel_vs_production"] is not None
                    else ""
                ),
                flush=True,
            )
