# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The wide (> 16 rows, ``BIG``) nvfp4 index scorer of ``emit_index_scores``.

CPU: the plan of ``score_big`` (the bulk list, the tail list, the contiguous runs of a SIMD's
two waves, the 6-step decode) scores every (request, block) exactly once; a bulk block holds
no new keys and is visible to every column of its request; a tail block is a ``new_keys_in``
one. GPU (gfx950 and aiter's nvfp4 scorer: ``ATOM_NVFP4_AITER_KERNELS``, as
``test_minimax_m3_mono_index_nvfp4``): the scores of a 64 / 32-row build equal
``score_flydsl_nvfp4``'s bit for bit."""

import os
import random

import pytest

BLOCKS, WAVES = 256, 8
SB, MAXT, TOPK, OLDER_SHARE = 128, 16, 16, 51


def plan(seqs, q, init=1, local=1):
    """The lane-resident plan of ``emit_index_scores`` (BIG, nvfp4), lists instead of lanes."""
    n = len(seqs)
    nb = [(max(s, 1) + SB - 1) // SB for s in seqs]
    sc = [
        (nb[k] - (init + local) if nb[k] > TOPK else 0) if k % q == q - 1 else 0
        for k in range(n)
    ]
    tail_from = [max(max(seqs[k] - MAXT, 0) // SB, init) for k in range(n)]
    n_tail = [max(init + sc[k] - tail_from[k], 0) for k in range(n)]
    n_bulk = [sc[k] - n_tail[k] for k in range(n)]
    return sc, tail_from, n_tail, n_bulk


def decode(u, end, start, first, init):
    tk = 0
    for step in (32, 16, 8, 4, 2, 1):
        tk += step if end[min(tk + step - 1, 63)] <= u else 0
    return tk, (init if first is None else first[tk]) + u - start[tk]


def runs(total):
    """(t0, t1) of every wave: a SIMD pair's run, the older wave's share first."""
    pairs = BLOCKS * (WAVES // 2)
    out = []
    for slot in range(BLOCKS):
        for wave in range(WAVES):
            gp = slot * (WAVES // 2) + wave % (WAVES // 2)
            p0, p1 = gp * total // pairs, (gp + 1) * total // pairs
            mid = p0 + (p1 - p0) * OLDER_SHARE // 100
            out.append((p0, mid) if wave < WAVES // 2 else (mid, p1))
    return out


@pytest.mark.parametrize("rows, q", [(64, 1), (64, 4), (32, 1), (48, 16), (32, 4)])
def test_the_plan_scores_every_block_once_and_splits_bulk_from_tail(rows, q):
    rng = random.Random(rows * 31 + q)
    for trial in range(6):
        reqs = rows // q
        if trial % 3 == 0:
            last = [
                rng.choice([2051, 32768, 131072, 215040, 17 * 128, 16 * 128 + 5])
                for _ in range(reqs)
            ]
        elif trial % 3 == 1:
            last = [rng.randint(1, 300000) for _ in range(reqs)]
        else:
            last = [131072 + rng.randint(-20, 20) for _ in range(reqs)]
        seqs = [max(s - (q - 1 - j), 1) for s in last for j in range(q)]
        sc, tail_from, n_tail, n_bulk = plan(seqs, q)
        pad = lambda v: v + [v[-1]] * (64 - len(v))
        end_b = pad([sum(n_bulk[: k + 1]) for k in range(rows)])
        start_b = [e - c for e, c in zip(end_b, pad(n_bulk))]
        end_t = pad([sum(n_tail[: k + 1]) for k in range(rows)])
        start_t = [e - c for e, c in zip(end_t, pad(n_tail))]
        total_bulk, total = end_b[63], end_b[63] + end_t[63]
        seen = {}
        for t0, t1 in runs(total):
            for u in range(t0, min(t1, total_bulk)):
                tk, b = decode(u, end_b, start_b, None, 1)
                seen.setdefault((tk, b), []).append("bulk")
                assert (b + 1) * SB <= seqs[
                    tk
                ] - q + 1  # visible to the request's first column
                assert (b + 1) * SB <= seqs[tk] - MAXT  # no key inserted this launch
            for u in range(max(t0, total_bulk), t1):
                tk, b = decode(u - total_bulk, end_t, start_t, tail_from, 1)
                seen.setdefault((tk, b), []).append("tail")
                assert (b + 1) * SB > seqs[tk] - MAXT  # ``new_keys_in``
        want = {
            (k, b) for k in range(rows) if k % q == q - 1 for b in range(1, 1 + sc[k])
        }
        assert set(seen) == want
        assert all(len(v) == 1 for v in seen.values())


KERNELS = os.environ.get("ATOM_NVFP4_AITER_KERNELS")


def _gpu_ok():
    torch = pytest.importorskip("torch")
    return (
        torch.cuda.is_available()
        and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    )


@pytest.mark.skipif(
    not KERNELS or not _gpu_ok(),
    reason="needs gfx950 and aiter's nvfp4 scorer (ATOM_NVFP4_AITER_KERNELS)",
)
@pytest.mark.parametrize(
    "rows, q, ctxs",
    [
        (32, 1, [2051, 20000, 70000, 131072]),
        (64, 1, [2051, 33000, 131072, 215040]),
        (64, 4, [2049, 66000, 131072, 300000]),
    ],
)
def test_the_wide_scores_equal_the_aiter_nvfp4_scorer_bit_for_bit(rows, q, ctxs):
    import aiter.ops.flydsl.kernels as pkg
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import torch
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    from flydsl.expr.typing import Int32, Int64

    from atom.models.minimax_m3.mono.config import (
        HEAD_DIM,
        MAX_INDEX_BLOCKS,
        ONE_INDEX_HEAD,
    )
    from atom.models.minimax_m3.mono.kernels import index_score as isc
    from atom.mono.device.ops import CM_DEV, kernel_symbol, launcher, uniform
    from atom.mono.device.sync import Addr, Mailbox
    from atom.mono.plan.execution import BLOCKS as NB
    from atom.mono.plan.execution import THREADS
    from atom.mono.plan.execution import WAVES as NW
    from atom.mono.plan.trace import Space

    pkg.__path__.insert(0, KERNELS)
    from aiter.ops.flydsl.kernels import minimax_m3_index_score as ref

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(3)
    reqs = rows // q
    last = [ctxs[r % len(ctxs)] for r in range(reqs)]
    nblk = [(s + 127) // 128 for s in last]
    total = sum(nblk) + 3
    # pages: e2m1 data random, e4m3 scale bytes positive and finite (0x28..0x3F)
    pages = torch.randint(
        0,
        256,
        (total, isc.NVFP4_PAGE_BYTES),
        generator=g,
        device=dev,
        dtype=torch.int32,
    ).to(torch.uint8)
    pages[:, 8192:] = torch.randint(
        0x28,
        0x40,
        (total, isc.NVFP4_PAGE_BYTES - 8192),
        generator=g,
        device=dev,
        dtype=torch.int32,
    ).to(torch.uint8)
    perm = torch.randperm(total, generator=g, device=dev).to(torch.int32)
    bt = torch.zeros(reqs, max(nblk), dtype=torch.int32, device=dev)
    o = 0
    for r, nb in enumerate(nblk):
        bt[r, :nb] = perm[o : o + nb]
        o += nb
    seq_req = torch.tensor(last, dtype=torch.int32, device=dev)
    seq = torch.tensor(
        [s - (q - 1 - j) for s in last for j in range(q)], dtype=torch.int32, device=dev
    )
    iq = torch.randn(rows, 1, HEAD_DIM, generator=g, device=dev).to(torch.bfloat16)
    iq8 = iq.float().clamp(-448, 448).to(torch.float8_e4m3fn)
    kd = torch.as_strided(pages, (total, 128, 64), (isc.NVFP4_PAGE_BYTES, 64, 1))
    ks = torch.as_strided(
        pages, (total, 128, 8), (isc.NVFP4_PAGE_BYTES, 8, 1), 128 * 64
    )
    sm_scale = HEAD_DIM**-0.5
    want = ref.score_flydsl_nvfp4(
        iq8, kd, ks, bt, seq_req, q, 1, sm_scale, max(nblk), k_round="e4m3"
    )

    scale = isc.index_scale_log2e(sm_scale)
    name = kernel_symbol("m3_index_big_test", s=rows, q=q)

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, NW * 16, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kern(
        iq: Int64,
        cache: Int64,
        bt_: Int64,
        seq_: Int64,
        bt_width: Int32,
        qlen: Int32,
        iscore: Int64,
        flags: Int64,
        layer: Int32,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        mbox = Mailbox(layer)
        lane = tid % 64
        q_frag, q_frag_head = isc.q_frag_from(iq, ONE_INDEX_HEAD, True, CM_DEV)

        def wait_index_q(tk=None):  # every row's flag is preset
            row = (
                fx.min(lane, rows - 1)
                if tk is None
                else tk - (qlen - 1) + fx.min(lane, qlen - 1)
            )
            mbox.poll([(Addr("flags", Space.SCRATCH, flags), row, 1)])

        isc.emit_index_scores(
            fx.block_idx.x, lane, uniform(tid // 64), lds.red.ptr, mbox.put, Addr("iscore", Space.SCRATCH, iscore),
            q_frag, rows, qlen, 1, 1, scale, cache, bt_, seq_, bt_width, None, None, ONE_INDEX_HEAD, q_frag_head, None,
            nvfp4=True, wait_index_q=wait_index_q, stamp=lambda i: None,
        )  # fmt: skip

    current_stream = fx.Stream(None)

    @launcher(name)
    def launch(
        iq: Int64,
        cache: Int64,
        bt_: Int64,
        seq_: Int64,
        bt_width: Int32,
        qlen: Int32,
        iscore: Int64,
        flags: Int64,
        layer: Int32,
        stream: fx.Stream = current_stream,
    ):
        kern(iq, cache, bt_, seq_, bt_width, qlen, iscore, flags, layer).launch(
            grid=(NB,), block=(THREADS,), stream=stream
        )

    layer = 3
    iscore = torch.zeros(2 * rows * MAX_INDEX_BLOCKS, device=dev, dtype=torch.int32)
    flags = torch.zeros(rows, 2, device=dev, dtype=torch.int32)
    flags[:, 0], flags[:, 1] = 1, layer + 1
    bt_rows = bt.repeat_interleave(q, dim=0).contiguous()
    _run_compiled(
        launch,
        iq.data_ptr(),
        pages.data_ptr(),
        bt_rows.data_ptr(),
        seq.data_ptr(),
        bt_rows.shape[1],
        q,
        iscore.data_ptr(),
        flags.data_ptr(),
        layer,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    sc = iscore.view(-1, 2)
    got = sc[:, 0].view(torch.float32).view(rows, MAX_INDEX_BLOCKS)
    tags = sc[:, 1].view(rows, MAX_INDEX_BLOCKS)
    for r in range(rows):
        nb = nblk[r // q]
        if nb <= isc.TOPK_BLOCKS:
            continue  # a context of <= 16 blocks is not scored
        blocks = slice(
            1, nb - 1
        )  # init 1, local 1: the first and the last block are pinned
        assert (tags[r, blocks] == layer + 1).all()
        assert torch.equal(got[r, blocks], want[0, r, blocks])
