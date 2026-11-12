# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono K4's MoE with a4w4 routed experts (``moe_a4w4``) against a torch oracle (gfx950; single-rank layer kernel, 1 card).
Same harness and oracle structure as tests/test_minimax_m3_mono_moe_a16w4.py (zero attention, the kernel's own routing, f64 / f32 from the
logical weights); the routed experts' xn and mid are quantized MXFP4 (per 1x32, E8M0 RoundUp = aiter's a4w4 convention, RNE e2m1, the CPU
rule ``fp4_quant_kernel_rule`` of test_minimax_m3_mono_moe_a4w4.py) before the GEMVs; slots rounded to bf16 per (expert, token) as the
a16w4 path does (``prod_round``); the shared expert stays MXFP8. ``run_case(S, act, prod_round, seed)`` -> (got, oracle, details), ``act`` in
a4w4 / a16w4 / a8w4; ``python tests/test_minimax_m3_mono_moe_a4w4_gpu.py [--S 1,4,16] [--acts a4w4,a16w4,a8w4]`` prints a table with the
rel L2 of the kernel against the oracle of every act (which semantics it implements) and against a16w4 (the production path).
"""

import argparse
import inspect
import os
import sys

import pytest
import torch

aiter = pytest.importorskip("aiter")
pytest.importorskip("flydsl")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="needs gfx950",
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_minimax_m3_mono_k4_mxfp8 as K4T
import test_minimax_m3_mono_moe_a4w4 as Q4
from aiter import dtypes
from aiter.ops.shuffle import shuffle_scale, shuffle_weight

from atom.model_ops.blockscale import quantize_fp8
from atom.models.minimax_m3.mono.config import (
    HIDDEN,
    INTER,
    MOE_SLOTS,
    N_ROUTED,
    TOP_K,
)
from atom.models.minimax_m3.mono.kernels.post_attn import (
    K4_ABI,
    build_post_attn_kernel,
)
from atom.models.minimax_m3.mono.layout import SCRATCH, sym_layout
from atom.mono.runtime.uncached import uncached_bytes

DEV = "cuda"
BF = torch.bfloat16
SWIGLU_LIMIT = 7.0
ROUTE_SCALE = 2.0
SHARED_WEIGHT = 1.0
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)


def bf(x):
    """Round to bf16, back to f32."""
    return x.to(BF).float()


def pow2(code):
    return torch.ldexp(
        torch.ones((), dtype=torch.float64, device=code.device),
        code.to(torch.int32) - 127,
    )


def dq_fp4(b, sc):
    """[R, K / 2] packed e2m1 bytes (low nibble = even k) and [R, K / 32] E8M0 codes -> f64 [R, K]."""
    lut = E2M1.to(b.device)
    v = lut[(b & 7).long()] * torch.where((b & 8) != 0, -1.0, 1.0).double()
    hi = lut[((b >> 4) & 7).long()] * torch.where((b & 128) != 0, -1.0, 1.0).double()
    v = torch.stack([v, hi], -1).view(b.shape[0], -1)
    return (v.view(b.shape[0], -1, 32) * pow2(sc)[..., None]).view(b.shape[0], -1)


def dq_fp8(w, sc):
    """fp8 [R, K] and E8M0 codes [R, K / 32] -> f64 [R, K]."""
    return (w.double().view(w.shape[0], -1, 32) * pow2(sc)[..., None]).view(
        w.shape[0], -1
    )


def mx8(x):
    """The MXFP8 quantization the native linear / mono apply to a bf16 [n, K], dequantized (f64)."""
    xq, xs = quantize_fp8(x.to(BF).contiguous())
    n, k = x.shape
    return (
        xq.double().view(n, k // 32, 32)
        * pow2(xs.view(torch.uint8).view(n, k // 32))[..., None]
    ).view(n, k)


def swiglu(g, u):
    """aiter ``swiglu_mul_batch`` in f32: gate upper-clamped, up two-sided, gate * sigmoid(1.702 gate) * (up + 1)."""
    g = torch.clamp(g, max=SWIGLU_LIMIT)
    u = torch.clamp(u, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    return g * torch.sigmoid(1.702 * g) * (u + 1.0)


class Experts:
    """The logical (unshuffled) MoE weights and their kernel (preshuffled) images."""

    def __init__(self, seed):
        g = torch.Generator(device=DEV).manual_seed(1000 + seed)
        r8 = lambda *s: torch.randint(
            0, 256, s, dtype=torch.uint8, device=DEV, generator=g
        )
        codes = lambda *s: torch.randint(
            117, 121, s, dtype=torch.uint8, device=DEV, generator=g
        )
        E = N_ROUTED
        self.b13, self.c13 = r8(E, 2 * INTER, HIDDEN // 2), codes(
            E, 2 * INTER, HIDDEN // 32
        )
        self.b2, self.c2 = r8(E, HIDDEN, INTER // 2), codes(E, HIDDEN, INTER // 32)
        # the shuffles the loader applies (GGUU, not guinterleave): (16, 16) weight, 2-D scale shuffle
        self.w13 = shuffle_weight(self.b13, layout=(16, 16)).contiguous()
        self.w2 = shuffle_weight(self.b2, layout=(16, 16)).contiguous()
        self.s13 = shuffle_scale(self.c13.view(-1, HIDDEN // 32)).contiguous()
        self.s2 = shuffle_scale(self.c2.view(-1, INTER // 32)).contiguous()
        # the MXFP8 shared expert: fp8 [rows, K] (16, 16)-preshuffled, row-major E8M0 codes
        f8 = lambda *s: (torch.randn(*s, device=DEV, generator=g) * 0.02).to(dtypes.fp8)
        sc = lambda *s: torch.randint(
            125, 128, s, dtype=torch.uint8, device=DEV, generator=g
        )
        self.f13, self.k13 = f8(2 * INTER, HIDDEN), sc(2 * INTER, HIDDEN // 32)
        self.f2, self.k2 = f8(HIDDEN, INTER), sc(HIDDEN, INTER // 32)
        self.w13s = shuffle_weight(self.f13, layout=(16, 16)).contiguous()
        self.w2s = shuffle_weight(self.f2, layout=(16, 16)).contiguous()
        self.d13s, self.d2s = dq_fp8(self.f13, self.k13), dq_fp8(self.f2, self.k2)
        self._dq = {}
        # the layouts: expert 5's gate row 17 / k chunk 3 through the kernel's own addressing
        self._check_layout(5)

    def _check_layout(self, e):
        """fp4_tile_load's addressing (lane l: row l % 16, 32 k from 32 (l // 16)) and scale_index against the shuffles."""
        from atom.mono.device.mx import scale_index

        rg, kc = 1, 3
        for lane in (0, 5, 16, 63):
            dw = rg * (2 * HIDDEN) + kc * 256 + lane * 4
            got = self.w13.view(-1)[
                e * (2 * INTER * HIDDEN // 2)
                + dw * 4 : e * (2 * INTER * HIDDEN // 2)
                + dw * 4
                + 16
            ]
            g4 = lane // 16
            want = self.b13[
                e,
                16 + lane % 16,
                (kc * 128 + g4 * 32) // 2 : (kc * 128 + g4 * 32) // 2 + 16,
            ]
            assert torch.equal(got, want), ("weight layout", lane)
        for r, c in ((0, 0), (17, 5), (1000, 100), (1535, 191)):
            si = scale_index(e * 2 * INTER + r, c, HIDDEN // 32)
            assert self.s13.view(-1)[si] == self.c13[e, r, c], ("scale layout", r, c)

    def dq13(self, e):
        if ("13", e) not in self._dq:
            self._dq[("13", e)] = dq_fp4(self.b13[e], self.c13[e])
        return self._dq[("13", e)]

    def dq2(self, e):
        if ("2", e) not in self._dq:
            self._dq[("2", e)] = dq_fp4(self.b2[e], self.c2[e])
        return self._dq[("2", e)]


_EXPERTS = {}


def experts_of(seed):
    if seed not in _EXPERTS:
        _EXPERTS.clear()
        _EXPERTS[seed] = Experts(seed)
    return _EXPERTS[seed]


def norm(h_in, g_post, eps=1e-6):
    """xn = bf16((h * rstd) * (1 + g)), f32 math, h the bf16 residual."""
    v = h_in.float()
    rstd = torch.rsqrt((v * v).sum(-1, keepdim=True) / HIDDEN + eps)
    return bf((v * rstd) * (g_post.float() + 1.0))


def moe_oracle(
    h_in, gate, bias, g_post, experts, shared, routing, *, act, prod_round, S
):
    """The MoE's ``ar_out`` [S, HIDDEN] f32 (bf16 values). ``act``: the routed experts' activation, a4w4 / a16w4 / a8w4."""
    del gate, bias, shared
    ids, w = routing
    xn = norm(h_in, g_post)
    q = {
        "a16w4": lambda v: v.double(),
        "a8w4": lambda v: mx8(v),
        "a4w4": lambda v: Q4.dq_fp4(v.float().contiguous()).double(),
    }[act]
    xr = q(xn)
    mid = torch.zeros(S, TOP_K, INTER, device=DEV)
    for e in ids.unique().tolist():
        t, s = (ids == e).nonzero(as_tuple=True)
        o = (xr[t] @ experts.dq13(e).T).float()
        mid[t, s] = bf(swiglu(o[:, :INTER], o[:, INTER:]))
    midq = (
        q(mid.view(-1, INTER)).view(S, TOP_K, INTER) if act != "a16w4" else mid.double()
    )
    acc = torch.zeros(S, TOP_K, HIDDEN, device=DEV)
    for e in ids.unique().tolist():
        t, s = (ids == e).nonzero(as_tuple=True)
        acc[t, s] = (midq[t, s] @ experts.dq2(e).T).float()
    slot = acc * w[..., None]
    if prod_round and act != "a8w4":  # gemm2's epilogue per (expert, token)
        slot = bf(slot)
    routed = slot[:, 0]
    for j in range(1, TOP_K):
        routed = routed + slot[:, j]
    if prod_round:
        routed = bf(routed)
    xs = mx8(xn)
    o = (xs @ experts.d13s.T).float()
    if prod_round:
        o = bf(o)
    y = bf(swiglu(o[:, :INTER], o[:, INTER:]))
    sh = (mx8(y) @ experts.d2s.T).float() * SHARED_WEIGHT
    if prod_round:
        sh = bf(sh)
    return bf(routed + sh)


def _route_check(h_in, gate, bias, g_post, ids, w):
    """The kernel's routing against a torch router (bf16 logit, sigmoid, bias for the pick, renormalized, x route_scale)."""
    xn = norm(h_in, g_post)
    sc = torch.sigmoid(bf(xn @ gate.float().T))
    top = (sc + bias).topk(TOP_K, dim=-1).indices
    same_set = (top.sort(-1).values == ids.sort(-1).values).all(-1)
    sel = sc.gather(-1, ids)
    wref = sel * (ROUTE_SCALE / sel.sum(-1, keepdim=True).clamp_min(1e-20))
    return same_set, (w - wref).abs().max().item()


def run_case(S, act="a4w4", prod_round=True, seed=0, **build_kw):
    """One launch of the single-rank layer kernel on random data. Returns (got, oracle, details): ``got`` the kernel's ``ar_out``
    and ``oracle`` the oracle of (moe_a16w4, prod_round) as f32 [S, HIDDEN] cpu tensors; ``details``: rel L2 per token, identical
    fraction, whether the torch router picked the kernel's experts, and ``rel_vs[(a16w4, prod_round)]`` the max rel L2 of
    ``got`` against each of the four oracles (which semantics the kernel implements)."""
    ex = experts_of(seed)
    g = torch.Generator(device=DEV).manual_seed(seed * 7 + S)
    h_in = (torch.randn(S, HIDDEN, device=DEV, generator=g) * 3).to(BF)
    g_post = (torch.randn(HIDDEN, device=DEV, generator=g) * 0.3).to(BF)
    gate = (torch.randn(N_ROUTED, HIDDEN, device=DEV, generator=g) * 0.02).to(BF)
    bias = torch.randn(N_ROUTED, device=DEV, generator=g) * 0.5
    # zero attention: zero o_proj weights, E8M0 code 127 (exact 1.0)
    real = (
        torch.zeros(HIDDEN, K4T.O_K, dtype=torch.uint8),
        torch.full((HIDDEN, K4T.NB), 127, dtype=torch.uint8),
    )
    c = K4T._case(S, real=real)
    z = lambda *shape, dt=torch.uint8: torch.zeros(*shape, dtype=dt, device=DEV)
    h_mid, ar_out = z(S, HIDDEN, dt=BF), z(S, HIDDEN, dt=BF)
    scratch = uncached_bytes(SCRATCH["_bytes"], torch.device(DEV))
    sym = uncached_bytes(sym_layout(1)["_bytes"], torch.device(DEV))
    peers = torch.tensor([sym.data_ptr()], dtype=torch.int64, device=DEV)
    k1 = torch.zeros(12, dtype=torch.int64, device=DEV)
    ids_t = torch.arange(S, dtype=torch.int32, device=DEV)
    flags = {
        "moe_a16w4": act == "a16w4",
        "moe_a4w4": act == "a4w4",
        "prod_round": prod_round,
        "sp_f32": False,
        **build_kw,
    }
    ps = inspect.signature(build_post_attn_kernel).parameters
    launch = build_post_attn_kernel(
        1, 0.088, 1e-6, ROUTE_SCALE, SHARED_WEIGHT, SWIGLU_LIMIT, 1, 1, S, shared_mxfp8=True, attn_mxfp8=True,
        **{k: v for k, v in flags.items() if k in ps},
    )  # fmt: skip
    p = lambda t: t.data_ptr()
    args = K4_ABI.pack(
        {
            "h_in": p(h_in), "q": p(c["q"]), "block_table": p(c["bt"]), "seq_lens": p(c["seq"]),
            "k_cache": p(c["kc"]), "v_cache": p(c["vc"]), "k_scale": p(c["ks"]), "v_scale": p(c["vs"]),
            "w_o": p(c["w"]), "s_o": p(c["code"]), "g_post": p(g_post), "w_gate": p(gate), "bias": p(bias),
            "w13": p(ex.w13), "s13": p(ex.s13), "w2": p(ex.w2), "s2": p(ex.s2),
            "h_mid": p(h_mid), "ar_out": p(ar_out), "scratch": scratch.data_ptr(), "sym": sym.data_ptr(),
            "peers": p(peers), "rank": 0, "layer": 3, "bt_width": 1, "q_len": 1, "tl": 0, "k1_args": p(k1),
            "positions": p(ids_t), "slot_mapping": p(ids_t), "res": p(h_in), "batch_ids": p(ids_t),
            "w13s": p(ex.w13s), "s13s": p(ex.k13), "w2s": p(ex.w2s), "s2s": p(ex.k2),
        }
    )  # fmt: skip
    launch(*args)
    torch.cuda.synchronize()
    off = SCRATCH["sel"]
    words = scratch[off : off + 2 * MOE_SLOTS * S * 8].view(torch.int32)[
        0::2
    ]  # (value, tag) pairs
    sel = words.view(S, 2, MOE_SLOTS)  # token k: ids at pairs 0..4, weights at 5..9
    ids = sel[:, 0, :TOP_K].long()
    w = sel[:, 1, :TOP_K].contiguous().view(torch.float32)
    assert (ids >= 0).all() and (ids < N_ROUTED).all(), ids
    got = ar_out.float()
    oracles = {
        a: moe_oracle(
            h_in,
            gate,
            bias,
            g_post,
            ex,
            None,
            (ids, w),
            act=a,
            prod_round=prod_round,
            S=S,
        )
        for a in ("a4w4", "a16w4", "a8w4")
    }
    want = oracles[act]
    rel = lambda a, b: ((a - b).norm(dim=-1) / b.norm(dim=-1))
    same_set, wdiff = _route_check(h_in, gate, bias, g_post, ids, w)
    details = {
        "rel": rel(got, want).cpu(),
        "identical": (got == want).float().mean().item(),
        "route_match": bool(same_set.all()),
        "route_weight_maxdiff": wdiff,
        "rel_vs": {k: rel(got, v).max().item() for k, v in oracles.items()},
        "rel_vs_a16w4_mean": rel(got, oracles["a16w4"]).mean().item(),
        "ids": ids.cpu(),
    }
    return got.cpu(), want.cpu(), details


@pytest.mark.parametrize("act", ["a4w4", "a16w4", "a8w4"])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 8, 12, 16])
def test_moe_matches_the_oracle(tokens, act):
    got, _want, d = run_case(tokens, act)
    assert torch.isfinite(got).all() and got.abs().max() > 0
    # a4w4 has more rounding flips (a fp4 grid step is 4-8 % of a value): rel L2 vs its own oracle
    med = 2e-3 if act == "a4w4" else 1e-3
    assert d["rel"].median() <= med and d["rel"].max() <= 1e-2, (d["rel"], d["rel_vs"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", default="1,2,3,4,5,8,12,16")
    ap.add_argument("--acts", default="a4w4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mid-flag", type=int, default=0)
    a = ap.parse_args()
    print(
        "S act | rel L2 vs own oracle median / max | identical | router match | max rel vs oracle a4w4 a16w4 a8w4 | mean rel vs a16w4"
    )
    for s in [int(x) for x in a.S.split(",")]:
        for act in a.acts.split(","):
            got, want, d = run_case(s, act, seed=a.seed, mid_flag=a.mid_flag)
            r = d["rel"]
            rv = d["rel_vs"]
            print(
                f"S={s:2d} {act} | {r.median():.2e} / {r.max():.2e} | {d['identical']:.4f} | {d['route_match']} (w {d['route_weight_maxdiff']:.1e}) | "
                f"{rv['a4w4']:.2e} {rv['a16w4']:.2e} {rv['a8w4']:.2e} | {d['rel_vs_a16w4_mean']:.3f}",
                flush=True,
            )
