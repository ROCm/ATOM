# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's MLA prep stages (``ATOM_MONO_MLA_PREP``, P0 .. P5) on one GPU, against
the production kernels.

The stages run at the tail of K2(M - 1), which needs every TP8 rank. Their
arithmetic does not: this runs ``stages.mla_prep``'s task functions in a
harness kernel of K2's grid (256 CTAs of 512 threads, each stage on its K2
CTAs), every mailbox region at K2's layout. A stage alone gets its inputs from
the host and its upstream hand-off pairs pre-published with the launch's tag
(P0: layer M-1's output in FING, as K2's owners push it; P1: PXRDY; P2: P1DONE;
P3: P2DONE; P4: P3DONE); the ``chain`` build runs P0 .. P5 in one launch from
P0's inputs (K2's tail without K2) and checks every stage against the reference
applied to the previous stage's own output. References:

* P0: ``apply_attn_res`` with the per-token FP8 quant folded (the triton AttnRes
  seam: ``prefix + delta``, the mix, input_layernorm, the quant). ``pref`` must
  match bit for bit; x8 / xs within the row-reduction order (one FP8 step);
* P1 / P3: fused_qkv_a_proj / q_b_proj as the exact FP8 products summed in
  float64, ``(sum * xs) * ws`` in fp32 (the FlyDSL a8w8 GEMM sums the same
  products in another order): a few bf16 ulps;
* P2: aiter's ``fused_qk_rmsnorm_group_quant`` per-token kernel
  (``_fuse_rmsnorm_quant``): q8 / qs / kv bit for bit;
* P4: AITER's triton FP8 BMM (the absorb) and aiter's
  ``fused_qk_rope_concat_and_cache_mla``: q_out's rope part bit for bit, its
  nope part within the BMM's summation order: one FP8 step, or, near 0 where
  E4M3's steps are finest, 2^-14 of the dot's sum |a w| (``ABSORB_ACC_TOL``);
  each side's distance to the exact dot is reported. Padded tokens (negative
  slot) keep their q_out rows;
* P5 (P2's kv CTAs in sub-mode 3): the same rope / cache kernel's FP8 KV rows,
  bit for bit; a negative slot writes nothing.

Skips without a GPU (the module collects on CPU);
``monofold_check/prep_harness_compile.py`` compiles every build for gfx950
without one.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("flydsl", reason="the stages are FlyDSL kernels")

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl.expr import const_expr
from flydsl.expr.typing import Int32, Int64

from atom.models.kimi_k3.mono.kernels.attention.kda import COLS, HIDDEN, NA
from atom.models.kimi_k3.mono.kernels.stages import mla_post_plan as pl
from atom.models.kimi_k3.mono.kernels.stages import mla_prep as pr
from atom.mono.device.ops import kernel_symbol, load_ptr64
from atom.mono.device.sync import Mailbox, preg, sreg
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.plan.layout import pair_layout

FP8 = torch.float8_e4m3fn
NB = 4  # layer M's seam blocks (prep_nblocks): 5 sources with the prefix
NS = NB + 1
EPS = 1e-6
HEADS_PAD = 16  # the decode's head padding: q_out's head stride
MAXPOS = 128
SLOTS = 64  # KV-cache rows: 16 blocks of 4 (a paged cache, block size > 1)
BLOCK = 4
XQ_BUDGET = 64 * 1024
LAYER = 1  # the launch's tag: LAYER + 1
# the harness kernel's pointer table (one i64 each), its Int32 arguments apart
PTRS = (
    "fing",
    "scratch",
    "prefix",
    "blocks",
    *(n for n in pr.ARGS if n not in pr.INT32_ARGS),
)
# the stages a build runs: each alone (its inputs and upstream pairs from the
# host), or all of them in one launch
BUILDS = {
    "p0": ("p0",),
    "p1": ("p1",),
    "p2": ("p2",),  # sub-mode 2: q8 / qs / kvn
    "p2p5": ("p2",),  # sub-mode 3: also P5's KV rows
    "p3": ("p3",),
    "p4": ("p4",),
    "chain": ("p0", "p1", "p2", "p3", "p4"),
}
_BUILDS: dict = {}
_STREAM = fx.Stream(None)


def scratch_layout(s: int) -> dict:
    return pair_layout(pr.scratch_pairs(s, NS).items())


def scratch_bytes(s: int) -> int:
    return max(o + n for o, n in scratch_layout(s).values())


def _mode(build: str) -> int:
    return 2 if build == "p2" else 3


def _build(s: int, build: str, neox: bool = True):
    key = (s, build, neox)
    if key in _BUILDS:
        return _BUILDS[key]
    stages = BUILDS[build]
    mode = _mode(build)
    assert s <= pr.MAX_ROWS and 2 * s <= NA and pr.red_fits(s, NS)
    lay = scratch_layout(s)
    slabs = tuple(pl.slabs(s, XQ_BUDGET))
    assert pl.lds_words(s, XQ_BUDGET) * 4 >= s * (pr.QB_K + pl.XPAD)

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        vals: fx.Array[fx.Float32, s * NS * COLS, 16]
        dl: fx.Array[fx.Float32, s * COLS, 16]
        xq: fx.Array[fx.Int32, pl.lds_words(s, XQ_BUDGET), 16]

    name = kernel_symbol(
        "k3_mono_mla_prep_check",
        s=s,
        st=list(BUILDS).index(build),
        nx=int(neox),
        pp=mode,
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def prep(
        tab: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        qout_sr: Int32,
        kvc_sr: Int32,
        layer: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        mb = Mailbox(layer)
        ptr = {n: load_ptr64(tab, i) for i, n in enumerate(PTRS)}
        args = {n: ptr[n] for n in PTRS[2:]}
        args.update(blk_sm=blk_sm, blk_sr=blk_sr, qout_sr=qout_sr, kvc_sr=kvc_sr)
        c = {
            "S": s,
            "tid": tid,
            "lane": tid % 64,
            "wave": tid // 64,
            "put": mb.put,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "vals": lds.vals.ptr,
            "delta_lds": lds.dl.ptr,
            "peer": {"fing_own": preg(ptr["fing"], 0, "fing")},
            "prep_xl": lds.xq.ptr,
            "prep_slabs": slabs,
            "qa_eps": EPS,
            "kva_eps": EPS,
            "prep_mode": mode,
            "rope_neox": neox,
            "rope_maxpos": MAXPOS,
            "args": args,
        }
        for region in pr.REGIONS:
            c[region] = sreg(ptr["scratch"], lay[region][0], region)
        # K2's tail (``layer._body`` after ``k2.final``), stage by stage
        if const_expr("p0" in stages):
            cp = pr.prep_context(c, NS, EPS, EPS)
            if bid < NA:
                pr.p0_seam(cp, bid)
        if const_expr("p1" in stages):  # noqa: SIM102
            if (bid >= pr.P1_CTA0) & (bid < pr.P1_CTA0 + pr.P1_TASKS):
                pr.p1_task(c, bid - pr.P1_CTA0)
        if const_expr("p2" in stages):  # noqa: SIM102
            if bid < 2 * s:
                pr.p2_norm(c, bid)
        if const_expr("p3" in stages):  # noqa: SIM102
            if (bid >= pr.P3_CTA0) & (bid < pr.P3_CTA0 + pr.P3_TASKS):
                pr.p3_task(c, bid - pr.P3_CTA0)
        if const_expr("p4" in stages):  # noqa: SIM102
            if bid >= pr.P4_CTA0:
                pr.p4_task(c, bid - pr.P4_CTA0)

    @flyc.jit
    def launch(
        tab: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        qout_sr: Int32,
        kvc_sr: Int32,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        prep(tab, blk_sm, blk_sr, qout_sr, kvc_sr, layer).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    _BUILDS[key] = launch
    return launch


# ------------------------------------------------------------------ host side
def _inputs(s: int, seed: int = 0) -> dict:
    """Every stage's inputs, and its output buffers (NaN / garbage filled)."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda")

    def bf(*shape, scale=1.0, shift=0.0):
        return (randn(*shape) * scale + shift).to(torch.bfloat16)

    def nan(*shape, dtype=torch.bfloat16):
        return torch.full(shape, float("nan"), device="cuda").to(dtype)

    def garbage(*shape):
        return torch.randint(
            0, 256, shape, generator=g, device="cuda", dtype=torch.uint8
        ).view(FP8)

    d = {
        # P0: layer M's seam
        "prefix": bf(s, HIDDEN),
        # one block more than the seam reads: a strided [:, :NB] view
        "blocks": bf(s, NB + 1, HIDDEN),
        "delta": bf(s, HIDDEN, scale=0.5),  # layer M-1's output (FING)
        "p_nw": bf(HIDDEN, scale=0.1, shift=1.0),
        "p_qk": bf(HIDDEN, scale=0.02),
        "p_onw": bf(HIDDEN, scale=0.1, shift=1.0),
        # P1
        "x8": (randn(s, HIDDEN) * 4).to(FP8),
        "xs": (randn(s).abs() * 0.01 + 1e-3).float(),
        "w_qa_raw": (randn(pr.QKV_A, HIDDEN) * 0.1).to(FP8),
        "w_qa_s": (randn(pr.QKV_A).abs() * 0.01 + 1e-3).float(),
        # P2
        "qkv": bf(s, pr.QKV_A, scale=2.0),
        "qa_nw": bf(pr.QL, scale=0.1, shift=1.0),
        "kva_nw": bf(pr.KVL, scale=0.1, shift=1.0),
        # P3
        "q8": (randn(s, pr.QL) * 4).to(FP8),
        "qs": (randn(s).abs() * 0.01 + 1e-3).float(),
        "w_qb_raw": (randn(pr.QB, pr.QB_K) * 0.1).to(FP8),
        "w_qb_s": (randn(pr.QB).abs() * 0.01 + 1e-3).float(),
        # P4 / P5
        "qb": bf(s, pr.QB),
        "w_k": (randn(pr.HEADS, pr.KVL, pr.QK_NOPE) * 0.5).to(FP8),
        "w_k_s": torch.full((1,), 0.75, device="cuda"),
        "k_sc": torch.full((1,), 0.05, device="cuda"),
        "q_sc": torch.full((1,), 0.05, device="cuda"),
        "kvn": nan(s, pr.KVL),
        "kvc": garbage(SLOTS // BLOCK, BLOCK, pr.ENTRY),
        "qout": torch.zeros(s, HEADS_PAD, pr.ENTRY, device="cuda").to(FP8),
    }
    # distinct slots; from 4 rows the last token is a padded row (slot -1)
    slots = torch.randperm(SLOTS, generator=g, device="cuda")[:s]
    if s >= 4:
        slots[-1] = -1
    d["slots"] = slots.to(torch.int64)
    d["pos"] = torch.randint(
        0, MAXPOS, (s,), generator=g, device="cuda", dtype=torch.int64
    )
    inv_freq = 1.0 / (10000 ** (torch.arange(32, device="cuda").float() / 32))
    ang = torch.arange(MAXPOS, device="cuda").float()[:, None] * inv_freq[None]
    d["cosc"] = ang.cos().to(torch.bfloat16).contiguous()
    d["sinc"] = ang.sin().to(torch.bfloat16).contiguous()
    d["w_qa"] = pl.shuffle16(d["w_qa_raw"].view(torch.uint8)).contiguous()
    d["w_qb"] = pl.shuffle16(d["w_qb_raw"].view(torch.uint8)).contiguous()
    d["pref"] = nan(s, HIDDEN)
    d["fing"] = _fing(d["delta"])
    d["scratch"] = torch.zeros(scratch_bytes(s), dtype=torch.uint8, device="cuda")
    return d


def _fing(out: torch.Tensor) -> torch.Tensor:
    """Layer M-1's output as K2's owners leave it in FING: (bf16 pair, tag)."""
    words = out.contiguous().view(torch.int32).reshape(-1)
    pairs = torch.empty(words.numel(), 2, dtype=torch.int32, device="cuda")
    pairs[:, 0] = words
    pairs[:, 1] = LAYER + 1
    return pairs


def _publish(d: dict, s: int, region: str) -> None:
    """Region ``region``'s pairs published with the launch's tag (a stage's
    upstream hand-off, when the stage runs alone)."""
    o, n = scratch_layout(s)[region]
    d["scratch"][o : o + n].view(torch.int32).view(-1, 2).copy_(
        torch.tensor([1, LAYER + 1], dtype=torch.int32, device="cuda")
    )


# a stage alone: the region its first wait polls
UPSTREAM = {
    "p1": "pxrdy",
    "p2": "p1done",
    "p2p5": "p1done",
    "p3": "p2done",
    "p4": "p3done",
}


def _run(s: int, build: str, d: dict, neox: bool = True) -> None:
    if build in UPSTREAM:
        _publish(d, s, UPSTREAM[build])
    p = torch.Tensor.data_ptr
    tab = torch.tensor([p(d[n]) for n in PTRS], dtype=torch.int64, device="cuda")
    kvrows = d["kvc"].view(-1, pr.ENTRY)
    _run_compiled(
        _build(s, build, neox),
        p(tab),
        d["blocks"].stride(0),
        d["blocks"].stride(1),
        d["qout"].stride(0),
        kvrows.stride(0),
        LAYER,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()


# ------------------------------------------------------------------ references
def _bits(t):
    return t.contiguous().view(torch.uint8)


def _ndiff(a, b) -> int:
    return int((_bits(a) != _bits(b)).sum())


def _ulps(a, b):
    a = a.view(torch.int16).int()
    b = b.view(torch.int16).int()
    a = torch.where(a < 0, -(a & 0x7FFF), a)
    b = torch.where(b < 0, -(b & 0x7FFF), b)
    return (a - b).abs()


def _fp8_steps(a, b):
    """|a - b| in E4M3 codes (sign-magnitude ordered; +0 == -0)."""

    def signed(x):
        x = x.contiguous().view(torch.uint8).int()
        return torch.where(x >= 128, -(x & 0x7F), x)

    return (signed(a) - signed(b)).abs()


def _gemm_ref(x8, xs, w, ws):
    """The per-token a8w8 GEMM: exact FP8 products summed in float64, then the
    epilogue ``(sum * xs) * ws`` in fp32 -> bf16."""
    acc = (x8.double() @ w.double().t()).float()
    return ((acc * xs[:, None]) * ws[None]).to(torch.bfloat16)


def _check_gemm(tag, got, ref, s):
    d = _ulps(got, ref)
    diff = (got.float() - ref.float()).abs()
    differ = int((d > 0).sum())
    print(
        f"s={s}: {tag} {differ} / {d.numel()} bf16 values differ, max {int(d.max())} "
        f"ulp, max |diff| {float(diff.max()):.3e} (GEMM order)"
    )
    assert not got.isnan().any()
    bad = (d > 4) & (diff > ref.float().abs().max() * 2**-8)
    assert (
        not bad.any()
    ), f"{tag}: {int(bad.sum())} values off: {diff[bad][:8].tolist()}"
    assert differ <= max(8, d.numel() // 20), tag


def _check_p0(d, s):
    from atom.model_ops.kimi_k3 import apply_attn_res

    sw = (d["p_nw"].float() * d["p_qk"].float()).contiguous()
    (x8r, xsr), prefr = apply_attn_res(
        d["prefix"].clone(),
        d["blocks"][:, :NB],
        sw,
        EPS,
        d["delta"],
        d["p_onw"],
        EPS,
        None,
        FP8,
    )
    torch.cuda.synchronize()
    xsr = xsr.reshape(-1)
    want = (d["prefix"].float() + d["delta"].float()).to(torch.bfloat16)
    steps = _fp8_steps(d["x8"], x8r)
    rel = ((d["xs"] - xsr).abs() / xsr.abs()).max()
    print(
        f"s={s}: P0 pref {_ndiff(d['pref'], prefr)} (must be 0), x8 "
        f"{int((steps > 0).sum())} / {steps.numel()} differ (max {int(steps.max())} "
        f"step), xs {_ndiff(d['xs'], xsr)} / {s} differ (max rel {float(rel):.2e})"
    )
    assert _ndiff(d["pref"], prefr) == 0 and _ndiff(d["pref"], want) == 0
    assert int(steps.max()) <= 1
    assert int((steps > 0).sum()) <= max(16, steps.numel() // 50)
    assert float(rel) <= 1e-4


def _check_p1(d, s):
    ref = _gemm_ref(d["x8"], d["xs"], d["w_qa_raw"], d["w_qa_s"])
    _check_gemm("P1 qkv", d["qkv"], ref, s)


def _p2_ref(d):
    from aiter import QuantType

    from atom.models.deepseek_v2 import _fuse_rmsnorm_quant

    qkv = d["qkv"]
    (q8e, qse), _, kve, _ = _fuse_rmsnorm_quant(
        qkv[:, : pr.QL], d["qa_nw"], EPS,
        qkv[:, pr.QL : pr.QL + pr.KVL], d["kva_nw"], EPS, None,
        dtype_quant=FP8, shuffle=False, scale_shuffle_padding=False,
        group_size=128, quant_type=QuantType.per_Token.value,
        output_unquantized_inp1=False, transpose_scale=True,
    )  # fmt: skip
    torch.cuda.synchronize()
    return q8e, qse.reshape(-1), kve


def _check_p2(d, s):
    q8e, qse, kve = _p2_ref(d)
    n = (_ndiff(d["q8"], q8e), _ndiff(d["qs"], qse), _ndiff(d["kvn"], kve))
    print(f"s={s}: P2 q8 {n[0]} qs {n[1]} kv {n[2]} (must be 0)")
    assert n == (0, 0, 0)


def _check_p3(d, s):
    ref = _gemm_ref(d["q8"], d["qs"], d["w_qb_raw"], d["w_qb_s"])
    _check_gemm("P3 qb", d["qb"], ref, s)


def _rope_ref(d, neox, kvc):
    """The production absorb + rope + KV-cache write on ``d``'s qb / kvn /
    k_rope: (q_out [s, 16, 576], the BMM's bf16 output [s, 12, 512]), ``kvc``
    written in place."""
    from atom.model_ops.attention_mla import (
        _aiter_triton_fp8_bmm,
        fused_qk_rope_concat_and_cache_mla,
    )

    s = d["qb"].size(0)
    qb = d["qb"].view(s, pr.HEADS, pr.QH)
    ql = _aiter_triton_fp8_bmm(
        qb[..., : pr.QK_NOPE].transpose(0, 1), d["w_k"], d["w_k_s"],
        group_size=128, transpose_bm=True,
    )  # fmt: skip
    q_ref = torch.zeros_like(d["qout"])
    # the kernel dereferences the two scales' pointers on the device
    fused_qk_rope_concat_and_cache_mla(
        ql, qb[..., pr.QK_NOPE :], d["kvn"], d["qkv"][:, pr.QL + pr.KVL :], kvc,
        q_ref[:, : pr.HEADS], d["slots"], d["k_sc"], d["q_sc"],
        d["pos"], d["cosc"], d["sinc"], is_neox=neox, is_nope_first=True,
        compute_all_q_rope=False,
    )  # fmt: skip
    torch.cuda.synchronize()
    return q_ref, ql


def _absorb_exact(d):
    """The absorb with the triton BMM's A prequant (per (token, head) 128-group:
    ``a_scale = max(amax, 1e-10) * f32(1 / 448)``, ``clamp(a * (1 / a_scale))``
    RNE E4M3) and its epilogue (``bf16((dot * a_scale) * b_scale)``), the dot
    exact (float64 of the FP8 products), then the rope kernel's
    ``bf16 * inv_q``: (q_out's nope part pre-quant [s, 12, 512] f32, the dot's
    unrounded value in q_out units, sum |a w| in q_out units)."""
    s = d["qb"].size(0)
    f32 = torch.float32
    a = d["qb"].view(s, pr.HEADS, pr.QH)[..., : pr.QK_NOPE].float()
    m = a.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    asc = m * torch.tensor(1.0 / pr.FP8_MAX, dtype=f32, device=a.device)
    a8 = (a * (torch.ones_like(asc) / asc)).clamp(-pr.FP8_MAX, pr.FP8_MAX).to(FP8)
    w = d["w_k"].double()
    acc = torch.einsum("shk,hnk->shn", a8.double(), w)
    mag = torch.einsum("shk,hnk->shn", a8.double().abs(), w.abs())
    bsc = d["w_k_s"].reshape(()).to(f32)
    inv_q = torch.ones((), dtype=f32, device=a.device) / d["q_sc"].reshape(())
    ql = ((acc.to(f32) * asc) * bsc).to(torch.bfloat16)
    unit = asc.double() * bsc.double() * inv_q.double()
    return ql.float() * inv_q, acc * unit, mag * unit


def _e4m3_step(v):
    """E4M3's step at |v| (subnormal below 2^-6: 2^-9)."""
    e = torch.floor(torch.log2(v.abs().clamp_min(2.0**-6)))
    return torch.exp2(e - 3)


def _check_p5(d, s, neox, kvc0):
    kvr = kvc0.clone()
    _rope_ref(d, neox, kvr)
    n = _ndiff(d["kvc"], kvr)
    live = int((d["slots"] >= 0).sum())
    print(
        f"s={s}: P5 KV cache {n} / {kvr.numel()} bytes differ, {live} rows (must be 0)"
    )
    assert n == 0


# The absorb's accumulation bound (q_out's nope part): 2^-12 of the dot's
# sum |a w|, in q_out units, past one E4M3 step. The FlyDSL absorb (four bf16
# MFMAs over K = 128) and the triton BMM (one FP8 MFMA, K = 128 inside the
# instruction) quantize A alike and sum the same exact FP8 products, but not
# in the same adders: two fp32 chains would agree to ~2^-24 of sum |a w| (no
# q_out step moves), while the ~0.08% one-step differences seen at every s
# match an adder keeping ~14 bits under the largest product. That error is
# absolute: a fraction of an E4M3 step except near 0, where E4M3's steps
# shrink to 2^-9 (subnormals), many binades under the row's scale. The
# standalone P4 test's N(0, 1) qb puts q_out at a std of ~85, so its few
# |q_out| < 2^-5 values move several steps for a dot difference of ~2^-17 of
# sum |a w| (s = 16: one value, 3 steps). The bound leaves 2^5 of headroom
# over that; a real defect (a wrong row, head, column or scale, an unready
# input) is off by ~sum |a w| / sqrt(K), 2^8 times the bound.
ABSORB_ACC_TOL = 2.0**-12


def _p4_bound(gf, rf, mag):
    return _e4m3_step(torch.maximum(gf.abs(), rf.abs())) + ABSORB_ACC_TOL * mag


def _p4_report(rows, d, got, ref, exact, xr, xe, xd, mag, bad, steps, k=8):
    """The worst nope elements (by steps, then |diff| over the bound): position,
    the kernel's / triton's / the exact dot's E4M3 and the pre-quant values (the
    kernel's own is not observable)."""
    gf, rf = got.float(), ref.float()
    diff = (gf - rf).abs()
    tol = _p4_bound(gf, rf, mag)
    key = steps.double() * 1e6 + diff.double() / tol.double()
    idx = torch.topk(key.flatten(), min(k, key.numel())).indices
    lines = []
    for i in idx.tolist():
        r, h, n = (
            i // (pr.HEADS * pr.KVL),
            (i // pr.KVL) % pr.HEADS,
            i % pr.KVL,
        )
        t = int(rows[r])
        lines.append(
            f"  token {t} (slot {int(d['slots'][t])}) head {h} col {n}: "
            f"{int(steps[r, h, n])} step, got {gf[r, h, n].item():.6g} "
            f"ref {rf[r, h, n].item():.6g} exact "
            f"{exact[r, h, n].float().item():.6g} | pre-quant: ref "
            f"{xr[r, h, n].item():.8g} (bf16 ql * inv_q), exact {xe[r, h, n].item():.8g}"
            f" (unrounded {xd[r, h, n].item():.8g}), sum|aw| {mag[r, h, n].item():.6g}"
            f", |diff| {diff[r, h, n].item():.4g} vs bound {tol[r, h, n].item():.4g}"
            f"{' BAD' if bool(bad[r, h, n]) else ''}"
        )
    return "\n".join(lines)


def _check_p4(d, s, neox):
    q_ref, ql = _rope_ref(d, neox, d["kvc"].clone())
    live = d["slots"] >= 0
    rows = live.nonzero().flatten()
    got, ref = d["qout"][live][:, : pr.HEADS], q_ref[live][:, : pr.HEADS]
    rope = _ndiff(got[..., pr.KVL :], ref[..., pr.KVL :])
    got, ref = got[..., : pr.KVL], ref[..., : pr.KVL]
    steps = _fp8_steps(got, ref)
    xe, xd, mag = (v[live] for v in _absorb_exact(d))
    inv_q = torch.ones((), device="cuda") / d["q_sc"].reshape(())
    xr = ql[live].float() * inv_q
    exact = xe.clamp(-pr.FP8_MAX, pr.FP8_MAX).to(FP8)
    se, sr = _fp8_steps(got, exact), _fp8_steps(ref, exact)
    gf, rf = got.float(), ref.float()
    bad = ~((gf - rf).abs() <= _p4_bound(gf, rf, mag))  # NaN: bad
    print(
        f"s={s}: P4 q_out rope {rope} (must be 0), nope {int((steps > 0).sum())} / "
        f"{steps.numel()} differ (max {int(steps.max())} step, absorb order; "
        f"{int((steps > 1).sum())} > 1 step, {int(bad.sum())} over the bound); "
        f"vs the exact dot: kernel {int((se > 0).sum())} (max {int(se.max())}), "
        f"triton {int((sr > 0).sum())} (max {int(sr.max())})"
    )
    if int(steps.max()) > 1 or bool(bad.any()):
        print(_p4_report(rows, d, got, ref, exact, xr, xe, xd, mag, bad, steps))
    assert rope == 0
    assert not bool(bad.any()), "P4 nope over the absorb bound:\n" + _p4_report(
        rows, d, got, ref, exact, xr, xe, xd, mag, bad, steps
    )
    assert int((steps > 0).sum()) <= max(16, steps.numel() // 50)
    # padded rows and the head padding: not written
    assert _ndiff(d["qout"][~live], torch.zeros_like(d["qout"][~live])) == 0
    pad = d["qout"][:, pr.HEADS :]
    assert _ndiff(pad, torch.zeros_like(pad)) == 0


@pytest.fixture(autouse=True)
def _gfx950():
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("the stages emit OCP E4M3 (gfx950's FP8)")


@pytest.mark.parametrize("s", [1, 4, 16])
def test_p0_seam(s):
    d = _inputs(s)
    _run(s, "p0", d)
    _check_p0(d, s)


@pytest.mark.parametrize("s", [1, 4, 16])
def test_p1_qkv_a(s):
    d = _inputs(s)
    d["qkv"].fill_(float("nan"))
    _run(s, "p1", d)
    _check_p1(d, s)


@pytest.mark.parametrize("s", [1, 4, 16])
@pytest.mark.parametrize("build", ["p2", "p2p5"])
def test_p2_norms_and_p5_kv_rows(s, build):
    d = _inputs(s)
    d["q8"].view(torch.uint8).fill_(0x7F)
    d["qs"].fill_(float("nan"))
    kvc0 = d["kvc"].clone()
    _run(s, build, d)
    _check_p2(d, s)
    if build == "p2p5":
        _check_p5(d, s, True, kvc0)
    else:
        assert _ndiff(d["kvc"], kvc0) == 0, "sub-mode 2 wrote the KV cache"


@pytest.mark.parametrize("s", [1, 4, 16])
def test_p3_q_b(s):
    d = _inputs(s)
    d["qb"].fill_(float("nan"))
    _run(s, "p3", d)
    _check_p3(d, s)


@pytest.mark.parametrize("s", [1, 4, 16])
def test_p4_absorb_rope(s):
    d = _inputs(s)
    _run(s, "p4", d)
    _check_p4(d, s, True)


@pytest.mark.parametrize("build", ["p2p5", "p4"])
def test_rope_not_neox(build):
    s = 4
    d = _inputs(s, seed=1)
    kvc0 = d["kvc"].clone()
    _run(s, build, d, neox=False)
    if build == "p4":
        _check_p4(d, s, False)
    else:
        _check_p5(d, s, False, kvc0)


@pytest.mark.parametrize("s", [1, 4, 16])
def test_chain(s):
    """P0 .. P5 in one launch from P0's inputs: every hand-off inside the
    launch, each stage checked on the previous one's output."""
    d = _inputs(s, seed=2)
    for n in ("x8", "q8"):
        d[n].view(torch.uint8).fill_(0x7F)
    for n in ("xs", "qs", "qkv", "qb"):
        d[n].fill_(float("nan"))
    kvc0 = d["kvc"].clone()
    _run(s, "chain", d)
    pairs = d["scratch"].view(torch.int32).view(-1, 2)
    lay = scratch_layout(s)
    for r in ("pxrdy", "p1done", "p2done", "p3done"):
        o, n = lay[r]
        tags = pairs[o // 8 : (o + n) // 8, 1]
        assert bool((tags == LAYER + 1).all()), f"{r}: not every pair published"
    _check_p0(d, s)
    _check_p1(d, s)
    _check_p2(d, s)
    _check_p3(d, s)
    _check_p5(d, s, True, kvc0)
    _check_p4(d, s, True)
