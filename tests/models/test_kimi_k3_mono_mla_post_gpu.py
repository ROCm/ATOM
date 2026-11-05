# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's MLA post stage (``ATOM_MONO_MLA_POST``) on one GPU, against the
production tail.

The stage runs as K2's first stage, which needs every TP8 rank. Its arithmetic
does not: this runs ``mla_post.post_task`` on its own, one CTA a task (96),
each publishing its ``mpdone`` pair and then polling all of them as K2's o_proj
CTAs do, and compares the ``core`` it writes with

* the production v-up: AITER's triton FP8 BMM (``MLAAttention._v_up_proj``'s),
* g_proj as the exact FP8 products summed in fp32 (the production FlyDSL a8w8
  GEMM sums the same products in another order),
* torch's bf16 ``av * sigmoid(gate)``.

The GEMMs' summation order differs, so a few values may differ by an ulp or two
(``ATOM_MONO_MLA_POST`` is documented as not bit-identical); anything more is a
layout or quantization bug.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("flydsl", reason="the stage is a FlyDSL kernel")

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl.expr.typing import Int32, Int64

from atom.models.kimi_k3.mono.kernels.stages import mla_post as mp
from atom.models.kimi_k3.mono.kernels.stages import mla_post_plan as pl
from atom.mono.device.ops import kernel_symbol
from atom.mono.device.sync import Mailbox, sreg
from atom.mono.plan.execution import THREADS, WAVES

FP8 = torch.float8_e4m3fn
HEADS_PAD = 16  # the decode's head padding: the latent is a strided view
LDS_BUDGET = 64 * 1024
_BUILDS: dict = {}
_STREAM = fx.Stream(None)


def _build(s: int):
    if s in _BUILDS:
        return _BUILDS[s]
    slabs = tuple(pl.slabs(s, LDS_BUDGET))

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        xq: fx.Array[fx.Int32, pl.lds_words(s, LDS_BUDGET), 16]

    name = kernel_symbol("k3_mono_mla_post_check", s=s)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def post(
        olat: Int64,
        olat_sr: Int32,
        w_v: Int64,
        w_v_s: Int64,
        xg: Int64,
        xg_s: Int64,
        w_g: Int64,
        w_g_s: Int64,
        core: Int64,
        scratch: Int64,
        layer: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek()
        mb = Mailbox(layer)
        c = {
            "S": s,
            "tid": tid,
            "lane": tid % 64,
            "wave": tid // 64,
            "red": lds.red.ptr,
            "post_xl": lds.xq.ptr,
            "post_slabs": slabs,
            "put": mb.put,
            "poll": mb.poll,
            "mpdone": sreg(scratch, 0, "mpdone"),
            "args": {
                "olat": olat,
                "olat_sr": olat_sr,
                "w_v": w_v,
                "w_v_s": w_v_s,
                "xg": xg,
                "xg_s": xg_s,
                "w_g": w_g,
                "w_g_s": w_g_s,
                "core": core,
            },
        }
        mp.post_task(c, bid)
        # as K2's o_proj CTAs: every task's pair, one round trip
        mp.wait_core(c)

    @flyc.jit
    def launch(
        olat: Int64,
        olat_sr: Int32,
        w_v: Int64,
        w_v_s: Int64,
        xg: Int64,
        xg_s: Int64,
        w_g: Int64,
        w_g_s: Int64,
        core: Int64,
        scratch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        post(
            olat, olat_sr, w_v, w_v_s, xg, xg_s, w_g, w_g_s, core, scratch, layer
        ).launch(grid=(pl.PT,), block=(THREADS,), stream=stream)

    _BUILDS[s] = launch
    return launch


def _inputs(s: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda")

    olat_pad = (randn(s, HEADS_PAD, pl.LORA) * 0.5).to(torch.bfloat16)
    olat = olat_pad[:, : pl.HEADS]
    # a few outliers per group: the per-group amax matters
    olat_pad[:, :, ::97] *= 8
    w_v = (randn(pl.HEADS, pl.VH, pl.LORA) * 0.1).to(FP8)
    w_v_s = torch.full((1,), 0.0123, device="cuda")
    x = (randn(s, pl.HIDDEN) * 4).to(FP8)
    xs = (randn(s).abs() * 0.01 + 1e-3).float()
    w_g = (randn(pl.PROJ, pl.HIDDEN) * 0.1).to(FP8)
    ws = (randn(pl.PROJ).abs() * 0.01 + 1e-3).float()
    return olat, w_v, w_v_s, x, xs, w_g, ws


def _reference(olat, w_v, w_v_s, x, xs, w_g, ws):
    from atom.model_ops.attention_mla import _aiter_triton_fp8_bmm

    s = olat.size(0)
    av = _aiter_triton_fp8_bmm(
        olat.reshape(-1, pl.HEADS, pl.LORA).transpose(0, 1),
        w_v,
        w_v_s,
        group_size=128,
        transpose_bm=True,
    ).reshape(s, pl.PROJ)
    gate = ((x.float() @ w_g.float().t()) * xs[:, None] * ws[None]).to(torch.bfloat16)
    return av, gate, av * torch.sigmoid(gate)


def _ulps(a, b):
    a = a.view(torch.int16).int()
    b = b.view(torch.int16).int()
    a = torch.where(a < 0, -(a & 0x7FFF), a)
    b = torch.where(b < 0, -(b & 0x7FFF), b)
    return (a - b).abs()


def test_preshuffle_matches_aiter():
    """``shuffle16`` (what ``w8_dword`` reads) is AITER's (16, 16) shuffle."""
    from aiter.ops.shuffle import shuffle_weight

    w = (torch.randn(64, 256, device="cuda") * 0.1).to(FP8)
    ref = shuffle_weight(w, layout=(16, 16))
    assert torch.equal(
        pl.shuffle16(w.view(torch.uint8)), ref.view(torch.uint8).reshape(64, 256)
    )


def _run_stage(s, olat, w_v, w_v_s, x, xs, w_g, ws):
    """The stage alone on these inputs: its ``core``, after checking every task
    published its pair with this launch's tag."""
    w_g_shuf = pl.shuffle16(w_g.view(torch.uint8)).contiguous()
    core = torch.full((s, pl.PROJ), float("nan"), device="cuda").to(torch.bfloat16)
    scratch = torch.zeros(pl.PT * 8, dtype=torch.uint8, device="cuda")
    p = torch.Tensor.data_ptr
    _run_compiled(
        _build(s), p(olat), olat.stride(0), p(w_v), p(w_v_s), p(x), p(xs),
        p(w_g_shuf), p(ws), p(core), p(scratch), 1, torch.cuda.current_stream(),
    )  # fmt: skip
    torch.cuda.synchronize()
    pairs = scratch.view(torch.int32).view(pl.PT, 2)
    assert torch.equal(pairs[:, 1].cpu(), torch.full((pl.PT,), 2, dtype=torch.int32))
    return core


@pytest.mark.parametrize("s", [1, 16])
def test_post_stage_ignores_nan_latent_columns(s):
    """The MLA decode can return whole NaN latent columns (seen on live traffic
    once KV blocks are recycled). The production v-up's quant ignores a NaN in
    its amax and clamp, so its output stays finite; the stage must do the same,
    or one NaN column makes every group's scale, hence the whole core, NaN."""
    inputs = _inputs(s, seed=3)
    olat = inputs[0]
    olat[:, :, 7] = float("nan")
    olat[:, :, 300] = float("nan")
    core = _run_stage(s, *inputs)
    _, _, ref = _reference(*inputs)
    assert not ref.isnan().any(), "the production tail no longer absorbs NaN columns"
    assert not core.isnan().any(), f"{int(core.isnan().sum())} NaN values in core"
    d = _ulps(core, ref)
    diff = (core.float() - ref.float()).abs()
    bad = (d > 4) & (diff > ref.float().abs().max() * 2**-8)
    assert not bad.any(), f"{int(bad.sum())} values off: {diff[bad][:8].tolist()}"


@pytest.mark.parametrize("s", [1, 4, 16])
def test_post_stage_matches_the_production_tail(s):
    olat, w_v, w_v_s, x, xs, w_g, ws = _inputs(s)
    w_g_shuf = pl.shuffle16(w_g.view(torch.uint8)).contiguous()
    core = torch.full((s, pl.PROJ), float("nan"), device="cuda").to(torch.bfloat16)
    scratch = torch.zeros(pl.PT * 8, dtype=torch.uint8, device="cuda")
    p = torch.Tensor.data_ptr
    _run_compiled(
        _build(s),
        p(olat),
        olat.stride(0),
        p(w_v),
        p(w_v_s),
        p(x),
        p(xs),
        p(w_g_shuf),
        p(ws),
        p(core),
        p(scratch),
        1,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    # every task published its pair with this launch's tag (layer 1 + 1)
    pairs = scratch.view(torch.int32).view(pl.PT, 2)
    assert torch.equal(pairs[:, 1].cpu(), torch.full((pl.PT,), 2, dtype=torch.int32))
    _, _, ref = _reference(olat, w_v, w_v_s, x, xs, w_g, ws)
    assert not core.isnan().any()
    d = _ulps(core, ref)
    differ = int((d > 0).sum())
    print(
        f"s={s}: {differ} / {d.numel()} bf16 values differ, max {int(d.max())} ulp, "
        f"max |diff| {float((core.float() - ref.float()).abs().max()):.3e}"
    )
    # a few ulps, or (where the product is near zero and an ulp is tiny) a
    # bf16 ulp of the largest value
    diff = (core.float() - ref.float()).abs()
    bad = (d > 4) & (diff > ref.float().abs().max() * 2**-8)
    assert not bad.any(), f"{int(bad.sum())} values off: {diff[bad][:8].tolist()}"
    assert differ <= max(8, d.numel() // 20)
