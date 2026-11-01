# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono's split stage over MXFP4 K / V pages (gfx950), alone: the stage the layer
kernel runs per (token, 256-key partition) -- its ``sp_o`` / ``sp_m`` / ``sp_l`` --
against a torch softmax over the dequantized pages, in the same harness for the
FP8 pages so the two stages are read against one yardstick. A harness kernel runs
only the split stage (the table of pages is handed in, as ``reuse_selection``
leaves it) and every other stage's inputs are absent."""

import math

import pytest
import torch

aiter = pytest.importorskip("aiter")
flyc = pytest.importorskip("flydsl.compiler")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="needs gfx950",
)

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu
from flydsl.expr.typing import Int32, Int64, T

from atom.model_ops.minimax_m3.kv_mxfp4 import PAGE_BYTES, dequantize_pages
from atom.models.minimax_m3.mono.config import HEAD_DIM
from atom.models.minimax_m3.mono.kernels.common import pack_q_e4m3
from atom.models.minimax_m3.mono.kernels.stages.attention import split_defs
from atom.models.minimax_m3.mono.layout import (
    N_SPLIT,
    O_K,
    SCRATCH,
    SCRATCH_BYTES,
    SPLIT_KEYS,
    H,
)
from atom.mono.device.ops import rsrc, traced
from atom.mono.device.sync import Mailbox, sreg
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES

SM_SCALE = HEAD_DIM**-0.5
NEG = -3.4e38
LAYER = 5
TABLE_WORDS = 18  # the two pages, the sparse context length at 16


@traced
def _table_stage(ctx):
    """``reuse_selection``'s stand-in: a task's blk from the table handed in."""
    tid, blk, table = ctx["tid"], ctx["blk"], ctx["table"]

    def load_blocks(tok, t):
        if tid < TABLE_WORDS:
            fx.ptr_store(
                fx.Int32(
                    bo.buffer_load(
                        rsrc(table),
                        (tok * N_SPLIT + t) * TABLE_WORDS + tid,
                        vec_width=1,
                        dtype=T.i32,
                    )
                ),
                blk + tid,
            )
        gpu.barrier()

    return {"reuse_selection": load_blocks}


def _launcher(tokens: int):
    @fx.struct
    class Smem:
        q8: fx.Array[fx.Int32, O_K // 4, 16]
        p8: fx.Array[fx.Int32, H * SPLIT_KEYS // 4, 16]
        opart: fx.Array[fx.Float32, WAVES * O_K, 16]
        hst: fx.Array[fx.Float32, 3 * WAVES * H + WAVES, 16]
        blk: fx.Array[fx.Int32, 32, 16]

    @flyc.kernel(name=f"mono_split_harness_{tokens}", known_block_size=[THREADS, 1, 1])
    def kernel(
        q: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        table: Int64,
        scratch: Int64,
        layer: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        lds = fx.SharedAllocator().allocate(Smem).peek()
        mbox = Mailbox(layer)
        mb_put = mbox.put
        mb_put_bf = mbox.put_bf
        blk = lds.blk.ptr

        def mr(name):
            return sreg(scratch, SCRATCH[name], name)

        def stamp(k):
            pass

        def start(name):
            return bid

        def acquire():
            pass

        ctx = {
            "CM_K1": 0,
            "G": BLOCKS,
            "acquire": acquire,
            "blk": blk,
            "fuse_k1": False,
            "g4": lane // 16,
            "hdone_mb": None,
            "hst": lds.hst.ptr,
            "k1_poll": None,
            "k_cache": k_cache,
            "k_scale": k_scale,
            "kv_mxfp4": True,
            "l16": lane % 16,
            "lane": lane,
            "mb_put": mb_put,
            "mb_put_bf": mb_put_bf,
            "mr": mr,
            "opart": lds.opart.ptr,
            "p8": lds.p8.ptr,
            "q": q,
            "q8": lds.q8.ptr,
            "index_topk": False,
            "sm_scale": SM_SCALE,
            "stamp": stamp,
            "start": start,
            "tid": tid,
            "tokens": tokens,
            "v2i": fx.Vector.make_type(2, fx.Int32),
            "v_cache": v_cache,
            "v_scale": v_scale,
            "wave": wave,
            "table": table,
        }
        ctx.update(_table_stage(ctx))
        split_defs(ctx)["stage_split"]()

    @flyc.jit
    def launch(
        q: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        table: Int64,
        scratch: Int64,
        layer: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(q, k_cache, v_cache, k_scale, v_scale, table, scratch, layer).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch


def _pages(n_pages, seed):
    """Random MXFP4 pages (e8m0 codes around 1)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = torch.randint(
        0, 256, (n_pages, 1, PAGE_BYTES), device="cuda", dtype=torch.uint8, generator=g
    )
    k[..., 8192:] = torch.randint(
        119, 127, (n_pages, 1, PAGE_BYTES - 8192), device="cuda", dtype=torch.uint8,
        generator=g,
    )  # fmt: skip
    return k


def _reference(q, kd, vd, n_ctx, t):
    """Partition t of one token: (o [H, 128], m [H], l [H]) over the keys
    t * 256 .. min(n_ctx, t * 256 + 256) of the sparse context ``kd`` / ``vd``
    ([n_sparse, 128] floats), q already e4m3 (K1's output)."""
    lo, hi = t * SPLIT_KEYS, min(n_ctx, t * SPLIT_KEYS + SPLIT_KEYS)
    if hi <= lo:
        return None
    qf = q.view(H, HEAD_DIM).to(torch.float8_e4m3fn).float()
    s = qf @ kd[lo:hi].T * SM_SCALE
    m = s.max(-1).values
    p = torch.exp(s - m[:, None])
    return (p / p.sum(-1, keepdim=True)) @ vd[lo:hi], m, p.sum(-1)


@pytest.mark.parametrize("n_ctx", [256, 190, 129, 40, 77, 300, 130])
@pytest.mark.parametrize("scale", [1.0, 6.0])
def test_the_split_stage_over_mxfp4_pages(n_ctx, scale):
    _check_split(n_ctx, scale)


def test_a_wrong_page_table_is_caught():
    """The positive control: the stage reads another page than the reference."""
    with pytest.raises(AssertionError):
        _check_split(256, 1.0, table_shift=1)


def _check_split(n_ctx, scale, table_shift=0):
    tokens = 1
    # the sparse context: pages 0 .. ceil(n_ctx / 128) of a pool, 2 slots a partition
    n_slots = 2 * N_SPLIT
    pool = _pages(n_slots, 1)
    pool_v = _pages(n_slots, 2)
    # slot s of the sparse table is page (3 * s + 1) % n_slots: scattered
    order = [(3 * s + 1) % n_slots for s in range(n_slots)]
    # poison the rows past the context in the last slot's page
    last_slot, rows = (n_ctx - 1) // 128, n_ctx % 128
    if rows:
        pg = order[last_slot]
        for pages in (pool, pool_v):
            pages[pg, 0, rows * 64 : 128 * 64] = 255
            pages[pg, 0, 8192 + rows * 4 :] = 255
    kd = dequantize_pages(pool)[:, 0]  # [n_slots, 128, 128]
    vd = dequantize_pages(pool_v)[:, 0]
    seq_k = torch.stack([kd[order[s]] for s in range(n_slots)]).reshape(-1, HEAD_DIM)
    seq_v = torch.stack([vd[order[s]] for s in range(n_slots)]).reshape(-1, HEAD_DIM)

    g = torch.Generator(device="cuda").manual_seed(4)
    # K1's q: e4m3, unit scale, saturating (the aiter writer's q_out convention)
    q = (
        (torch.randn(tokens, H, HEAD_DIM, device="cuda", generator=g) * scale)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    table = torch.zeros(tokens * N_SPLIT, TABLE_WORDS, device="cuda", dtype=torch.int32)
    for t in range(N_SPLIT):
        table[t, 0] = order[(2 * t + table_shift) % n_slots]
        table[t, 1] = order[(2 * t + 1 + table_shift) % n_slots]
        table[t, 16] = n_ctx
    scratch = torch.zeros(SCRATCH_BYTES, device="cuda", dtype=torch.uint8)
    _launcher(tokens)(
        q.data_ptr(), pool.data_ptr(), pool_v.data_ptr(), 0, 0, table.data_ptr(),
        scratch.data_ptr(), LAYER,
    )  # fmt: skip
    torch.cuda.synchronize()

    def region(name, n_words):
        o = SCRATCH[name]
        return scratch[o : o + 4 * n_words].view(torch.int32)

    sp_o = region("sp_o", N_SPLIT * O_K).view(-1, 2)  # (packed pair, tag)
    sp_m = region("sp_m", 2 * N_SPLIT * H).view(-1, 2)
    sp_l = region("sp_l", 2 * N_SPLIT * H).view(-1, 2)
    for t in range(N_SPLIT):
        o_words = sp_o[t * O_K // 2 : (t + 1) * O_K // 2]
        assert (o_words[:, 1] == LAYER + 1).all(), f"partition {t}: sp_o not written"
        got_o = (
            o_words[:, 0].contiguous().view(torch.bfloat16).float().view(H, HEAD_DIM)
        )
        got_m = sp_m[t * H : (t + 1) * H, 0].contiguous().view(torch.float32)
        got_l = sp_l[t * H : (t + 1) * H, 0].contiguous().view(torch.float32)
        assert (sp_m[t * H : (t + 1) * H, 1] == LAYER + 1).all()
        ref = _reference(q[0], seq_k, seq_v, n_ctx, t)
        if ref is None:
            assert (got_o == 0).all() and (got_l == 0).all()
            assert (got_m <= NEG * 0.99).all()
            continue
        want_o, want_m, want_l = ref
        assert torch.isfinite(got_o).all()
        torch.testing.assert_close(got_m, want_m, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(got_l, want_l, rtol=1e-2, atol=1e-5)
        rel = ((got_o - want_o).norm() / want_o.norm().clamp_min(1e-12)).item()
        assert rel < 1e-2, f"partition {t}: O relative error {rel:.3e}"
    assert math.isfinite(rel)


def _q_pack_launcher():
    @flyc.kernel(name="mono_q_e4m3_pack_test", known_block_size=[64, 1, 1])
    def kernel(x: Int64, out: Int64):
        t = fx.thread_idx.x
        v = [
            fx.Float32(bo.buffer_load(rsrc(x), 4 * t + j, vec_width=1, dtype=T.f32))
            for j in range(4)
        ]
        bo.buffer_store(pack_q_e4m3(v), rsrc(out), t)

    @flyc.jit
    def launch(x: Int64, out: Int64, stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        kernel(x, out).launch(grid=(1,), block=(64,), stream=stream)

    return launch


def test_k1_writes_q_as_saturating_e4m3():
    """K1's main-Q store over MXFP4 K / V: in-range values are torch's e4m3 cast,
    anything beyond +-448 (inf too) saturates to +-448, NaN stays NaN."""
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(64, 4, device="cuda", generator=g) * 30
    x[0] = torch.tensor([1000.0, -1000.0, 448.0, -448.0])
    x[1] = torch.tensor([449.0, -3e38, float("inf"), float("-inf")])
    x[2] = torch.tensor([float("nan"), 0.0, -0.0, 1e-9])
    out = torch.zeros(64, dtype=torch.int32, device="cuda")
    _q_pack_launcher()(x.data_ptr(), out.data_ptr())
    torch.cuda.synchronize()
    got = out.view(torch.uint8).view(64, 4).view(torch.float8_e4m3fn).float()
    want = x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    nan = x.isnan()
    assert torch.equal(got.isnan(), nan), "NaN must stay NaN, and only NaN"
    assert torch.equal(got[~nan], want[~nan])
    assert got[0].tolist() == [448.0, -448.0, 448.0, -448.0]
    assert got[1].tolist() == [448.0, -448.0, 448.0, -448.0]
