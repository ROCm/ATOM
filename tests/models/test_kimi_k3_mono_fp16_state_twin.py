# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""K1 on an fp16 recurrent state pool is its fp32 build plus a rounding of
every state it stores.

An fp16 build (``KdaPreBuild.state16``) widens each state it loads to fp32,
runs the recurrence as the fp32 build does and rounds each state it stores
(``state_loads`` / ``store_states``). So, launched on the same inputs as its
fp32 twin, whose pool holds the fp16 pool widened exactly, everything K1 writes
but the pool must be bit-equal -- the core rows, the prefix, the conv window,
ReplaySSM's records -- and the fp16 pool must be the twin's rounded to nearest.

The cases reach every store: each token's state (a plain decode, a spec
verify) and, under ReplaySSM, the checkpoint a flush folds the records into,
on the chunk tasks and on the head tasks (``head_tasks``). Which slots changed
is checked too, so a build that stored nothing would not pass as one that
stored the right thing.
"""

from dataclasses import dataclass, replace

import pytest
import torch

pytest.importorskip("flydsl", reason="K1 is a FlyDSL kernel")
pytest.importorskip("aiter", reason="K1 launches through aiter's FlyDSL shim")
if (
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
):
    pytest.skip("K1 is gfx950 code", allow_module_level=True)

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.models.kimi_k3.mono.kernels.attention import kda as k1
from atom.mono.runtime.uncached import uncached_bytes

NB = 2  # AttnRes block sources
SL = k1.CONV_W - 1 + 7  # the conv window of a 7-token DSpark deployment
CUDA = {"device": "cuda"}


@dataclass
class Case:
    key: k1.KdaPreBuild
    weights: dict
    written: dict  # every buffer K1 may write but the state pool
    reads: dict  # the slot tables and ReplaySSM cursors
    pool: torch.Tensor  # fp16
    changed: set  # the slots whose state K1 must store


def _case(requests, qlen, cap=0, h_rec=(), flush=(), pad=None, seed=0) -> Case:
    g = torch.Generator(**CUDA).manual_seed(seed)

    def randn(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g, **CUDA) * scale).to(dtype)

    s, f32 = requests * qlen, torch.float32
    slots = requests * qlen + requests + 2
    perm = torch.randperm(slots, generator=g, **CUDA).int()
    table = perm[:s].view(requests, qlen).clone()
    if pad is not None:
        table[pad] = -1  # PAD_SLOT_ID
    accepted = torch.randint(1, qlen + 1, (requests,), generator=g, **CUDA).int()
    reads = {"table": table, "accepted": accepted}
    written = {
        "prefix": randn(s, k1.HIDDEN),
        "delta": randn(s, k1.HIDDEN, scale=0.5),
        "blocks": randn(s, NB, k1.HIDDEN),
        # token-major, as the pool is allocated; K1 reads the strides
        "conv_st": randn(slots, SL, k1.QKV, scale=0.5).transpose(1, 2),
        "core": torch.zeros(s, k1.PROJ, dtype=torch.bfloat16, **CUDA),
    }
    if cap:
        checkpoints = perm[s : s + requests].clone()
        # ReplaySSM holds one slot a request: its checkpoint, first in the row
        table[:, 0], table[:, 1:] = checkpoints, 0
        write_pos = torch.full((slots,), -1, dtype=torch.int32, **CUDA)
        write_pos[checkpoints.long()] = torch.tensor(h_rec, dtype=torch.int32, **CUDA)
        reads |= {"slot_idx": checkpoints, "write_pos": write_pos}
        records = (slots, k1.NH, cap, k1.HD)
        written |= {
            "rbuf_k": randn(*records, scale=0.1),
            "rbuf_u": randn(*records, scale=0.1),
            "rbuf_g": -randn(*records, scale=0.1, dtype=f32).abs(),
        }
        # a flush folds the records into the checkpoint; without one no
        # state is stored at all, only records
        changed = {int(slot) for slot, f in zip(checkpoints.tolist(), flush) if f}
    else:
        changed = {int(slot) for row in table.tolist() for slot in row if row[0] >= 0}
    weights = {
        "ares_nw": 1 + randn(k1.HIDDEN, scale=0.1),
        "ares_qk": randn(k1.HIDDEN, scale=0.05),
        "in_nw": 1 + randn(k1.HIDDEN, scale=0.1),
        "w_in": randn(k1.FUSED, k1.HIDDEN, scale=0.02),
        "w_fb": randn(k1.PROJ, k1.HD, scale=0.05),
        "conv_w": randn(k1.QKV, k1.CONV_W, scale=0.3, dtype=f32),
        "a_log": torch.empty(k1.NH, **CUDA).uniform_(0.0, 2.7, generator=g),
        "dt_bias": randn(k1.PROJ, scale=0.5, dtype=f32),
        "on_w": 1 + randn(k1.HD, scale=0.1, dtype=f32),
    }
    key = k1.KdaPreBuild(
        tokens=s,
        qlen=qlen,
        nblocks=NB,
        delta=True,
        state_len=SL,
        replay_cap=cap,
        replay_window=qlen if cap else 0,
        **k1.gemv_staging(s),
    )
    pool = randn(slots, k1.NH, k1.HD, k1.HD, scale=0.5, dtype=torch.float16)
    return Case(key, weights, written, reads, pool, changed)


def _launch(case: Case, state16: bool) -> dict:
    """``case``'s step, built for an fp16 pool or for its fp32 twin, on fresh
    copies of everything K1 writes."""
    key = replace(case.key, state16=state16)
    out = {name: t.clone() for name, t in case.written.items()}
    out["rstate"] = case.pool.clone() if state16 else case.pool.float()
    rd, w, p = case.reads, case.weights, torch.Tensor.data_ptr
    replay = 0
    if key.replay_cap:
        bufs = {**rd, **out}
        table = torch.tensor([p(bufs[n]) for n in k1.REPLAY_TABLE], **CUDA)
        replay = p(table)
    # zeroed before the launch, as `StepMailboxes.begin_step` zeroes it
    # before each step: no pair carries a live tag
    scratch = uncached_bytes(-(-k1.scratch_bytes(key) // 16) * 16, out["core"].device)
    plain = uncached_bytes(k1.plain_bytes(key), out["core"].device)
    blocks, conv, rstate = out["blocks"], out["conv_st"], out["rstate"]
    args = k1.ABI.pack({
        "prefix": p(out["prefix"]), "delta": p(out["delta"]), "blocks": p(blocks),
        "blk_sm": blocks.stride(0), "blk_sr": blocks.stride(1),
        "ares_nw": p(w["ares_nw"]), "ares_qk": p(w["ares_qk"]),
        "in_nw": p(w["in_nw"]), "w_in": p(w["w_in"]), "w_fb": p(w["w_fb"]),
        "conv_w": p(w["conv_w"]), "conv_st": p(conv),
        "cs_seq": conv.stride(0), "cs_dim": conv.stride(1), "cs_tok": conv.stride(2),
        "a_log": p(w["a_log"]), "dt_bias": p(w["dt_bias"]), "on_w": p(w["on_w"]),
        "rstate": p(rstate), "rs_seq": rstate.stride(0),
        "st_idx": p(rd["table"]), "st_stride": rd["table"].stride(0),
        "num_acc": p(rd["accepted"]), "replay": replay,
        "core_out": p(out["core"]), "x_dbg": 0, "proj_dbg": 0,
        "scratch": p(scratch), "plain": p(plain), "layer": 1, "tl": 0,
    })  # fmt: skip
    _run_compiled(k1.build(key), *args, torch.cuda.current_stream())
    torch.cuda.synchronize()
    return out


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize(
    ("requests", "qlen", "cap", "h_rec", "flush", "pad", "heads"),
    [
        (4, 1, 0, (), (), 2, True),
        (2, 8, 0, (), (), None, False),
        (4, 8, 0, (), (), None, True),
        # cap 24 at an 8-token window flushes past 8 records: none committed
        # (-1, the prefill sentinel), replayed without a flush, flushed
        (2, 8, 24, (4, 16), (False, True), None, False),
        (4, 8, 24, (-1, 4, 12, 16), (False, False, True, True), None, True),
    ],
    ids=[
        "decode",
        "verify-chunks",
        "verify-heads",
        "replayssm-chunks",
        "replayssm-heads",
    ],
)
def test_an_fp16_pool_is_the_fp32_build_rounded(
    requests, qlen, cap, h_rec, flush, pad, heads
):
    case = _case(requests, qlen, cap, h_rec, flush, pad)
    assert k1.head_tasks(case.key.tokens, qlen) is heads
    assert not k1.lds_refusal(case.key)
    fp16, fp32 = _launch(case, state16=True), _launch(case, state16=False)
    for name in case.written:
        assert torch.equal(_bits(fp16[name]), _bits(fp32[name])), name
    assert torch.equal(_bits(fp16["rstate"]), _bits(fp32["rstate"].half()))
    moved = (fp16["rstate"] != case.pool).flatten(1).any(1)
    assert set(moved.nonzero().flatten().tolist()) == case.changed
