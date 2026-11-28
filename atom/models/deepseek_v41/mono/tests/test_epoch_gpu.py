# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""GPU tests of the V4.1 epoch tags (one GPU, no peers): the device tags equal ``epoch_policy``'s, a stale pair (the
previous epoch's) is never taken for the live one -- the poll waits until the live pair lands -- and the epoch begin's
device decision (slim only after a draft pass's record, full on the period and when forced; the epoch word; the draft
runner's record into the target's state). The begin runs at TP 1 over a fake one-rank peer buffer.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl.expr import const_expr
from flydsl.expr.typing import Int32, Int64

from atom.models.deepseek_v41.mono import epoch_begin as ebm
from atom.models.deepseek_v41.mono.kernels.mailbox import (
    Mailbox,
    epoch_of,
    sreg,
)
from atom.mono.device.ops import kernel_symbol, launcher, memrealtime, rsrc, traced
from atom.mono.device.sync import sleep_ticks
from atom.mono.plan.execution import BLOCKS
from atom.mono.runtime.uncached import uncached_bytes

_spec = importlib.util.spec_from_file_location(
    "v41_epoch_policy_gpu", Path(__file__).resolve().parent.parent / "epoch_policy.py"
)
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)

_STREAM = fx.Stream(None)
LAUNCH_TAG = 6  # Mailbox(5): layer + 1
# the producer CTA's wait before it puts the live pair (100 MHz ticks: 2 ms), the poller spinning on the stale one
PUT_DELAY_TICKS = 200_000


def _put_kernel():
    name = kernel_symbol("g_epoch_test_put", v=1)

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def put(buf: Int64, eword: Int64, code: Int32):
        mb = Mailbox(LAUNCH_TAG - 1, epoch=epoch_of(eword))
        a = sreg(buf, 0, "t")
        # every lane stores the same pairs (no branch around the mailbox)
        mb.put(a, 0, fx.Int32(1234))
        mb.put_coded(a, 1, fx.Int32(99), code)

    @launcher(name)
    def launch(buf: Int64, eword: Int64, code: Int32, stream: fx.Stream = _STREAM):
        put(buf, eword, code).launch(grid=(1,), block=(64,), stream=stream)

    return launch


@traced
def _stale_producer(buf, eword, out, code, coded):
    """CTA 1: wait PUT_DELAY_TICKS, record the time, then put the live pair the way a V4.1 producer does
    (``Mailbox.put`` / ``put_coded``: the word and its tag in one 8-byte store)."""
    mb = Mailbox(LAUNCH_TAG - 1, epoch=epoch_of(eword))
    a = sreg(buf, 0, "t")
    sleep_ticks(PUT_DELAY_TICKS)
    t_put = memrealtime()
    bo.buffer_store(fx.Int32(t_put & 0xFFFFFFFF), rsrc(out), 4)
    bo.buffer_store(fx.Int32(t_put >> 32), rsrc(out), 5)
    if const_expr(coded):
        mb.put_coded(a, 0, fx.Int32(222), code)
    else:
        mb.put(a, 0, fx.Int32(222))


@traced
def _stale_poller(buf, eword, out, coded):
    """CTA 0: poll pair 0 (stale at launch), record what it took and when."""
    mb = Mailbox(LAUNCH_TAG - 1, epoch=epoch_of(eword))
    a = sreg(buf, 0, "t")
    if const_expr(coded):
        ((words, got),) = mb.poll_coded([(a, 0, 1)])
    else:
        (words,) = mb.poll([(a, 0, 1)])
        got = fx.Int32(0)
    t_got = memrealtime()
    bo.buffer_store(fx.Int32(words[0]), rsrc(out), 0)
    bo.buffer_store(fx.Int32(got), rsrc(out), 1)
    bo.buffer_store(fx.Int32(t_got & 0xFFFFFFFF), rsrc(out), 2)
    bo.buffer_store(fx.Int32(t_got >> 32), rsrc(out), 3)


def _stale_kernel(coded: bool):
    """One launch, two CTAs: CTA 0 polls, CTA 1 produces (``_stale_poller`` / ``_stale_producer``). The producer is
    not a launch on a second stream: a spinning kernel and a later launch on another stream can share a hardware
    queue, and a memcpy as the producer writes no 8-byte pair at once."""
    name = kernel_symbol("g_epoch_test_stale", coded=int(coded))

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def stale(buf: Int64, eword: Int64, out: Int64, code: Int32):
        if fx.block_idx.x == 1:
            _stale_producer(buf, eword, out, code, coded)
        else:
            _stale_poller(buf, eword, out, coded)

    @launcher(name)
    def launch(
        buf: Int64, eword: Int64, out: Int64, code: Int32, stream: fx.Stream = _STREAM
    ):
        stale(buf, eword, out, code).launch(grid=(2,), block=(64,), stream=stream)

    return launch


def _i32(u):
    return u - (1 << 32) if u >= 1 << 31 else u


@pytest.mark.parametrize("epoch", [1, 2, 300, ep.FULL_EVERY + 7, (1 << 24) + 5])
def test_the_device_tags_are_the_policys(epoch):
    buf = torch.zeros(8, dtype=torch.int32, device="cuda")
    eword = torch.tensor([epoch, 0], dtype=torch.int64, device="cuda")
    code = 0xA7
    _run_compiled(
        _put_kernel(),
        buf.data_ptr(),
        eword.data_ptr(),
        code,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    got = [w & 0xFFFFFFFF for w in buf.tolist()]
    tag = ep.epoch_tag(epoch, LAUNCH_TAG)
    assert got[0] == 1234 and got[1] == tag
    assert got[2] == 99 and got[3] == ep.coded_word(tag, code)


@pytest.mark.parametrize("coded", [False, True])
def test_a_stale_pair_is_not_taken_the_poll_waits_for_the_live_one(coded):
    e = 1000
    stale_tag = ep.epoch_tag(e - 1, LAUNCH_TAG)
    if coded:
        stale_tag = ep.coded_word(stale_tag, 3)
    # the hand-off pair in uncached memory, as the runners' scratch (no stale cache line between the CTAs)
    buf = uncached_bytes(16, torch.device("cuda")).view(torch.int32)
    buf[:2] = torch.tensor([111, _i32(stale_tag)], dtype=torch.int32)
    eword = torch.tensor([e, 0], dtype=torch.int64, device="cuda")
    out = torch.zeros(6, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    _run_compiled(
        _stale_kernel(coded), buf.data_ptr(), eword.data_ptr(), out.data_ptr(), 5,
        torch.cuda.current_stream(),
    )  # fmt: skip
    torch.cuda.synchronize()
    word, got, *ticks = out.tolist()
    t_got, t_put = [
        (lo & 0xFFFFFFFF) | (hi << 32) for lo, hi in (ticks[0:2], ticks[2:4])
    ]
    assert word == 222
    if coded:
        assert got == 5
    # the poller took the pair only after the producer put the live one
    assert t_got >= t_put, (t_got, t_put)


def _peers(peer_n):
    """A one-rank peer buffer: its data, then FENCE_BYTES of epoch slots."""
    data = torch.zeros(peer_n + ebm.FENCE_SLOTS * 8, dtype=torch.uint8, device="cuda")
    addresses = torch.tensor([data.data_ptr()], dtype=torch.int64, device="cuda")
    return SimpleNamespace(bytes=data[:peer_n], addresses=addresses, rank=0, _all=data)


def _state(mb):
    s = mb._state.tolist()
    return {"epoch": s[ebm.EPOCH], "drafted": s[ebm.DRAFTED], "full": s[ebm.TOOK_FULL]}


def test_a_new_runner_begins_full():
    """No draft pass recorded at construction (``DRAFTED`` -1): epoch 1 cannot be slim."""
    assert not ep.slim_allowed(1, -1)


def test_the_epoch_begin_decides_on_the_device():
    peer_n = 4096
    peers = _peers(peer_n)
    scratch = torch.zeros(8192, dtype=torch.uint8, device="cuda")
    queue = torch.zeros(256, dtype=torch.uint8, device="cuda")
    eword = torch.zeros(2, dtype=torch.int64, device="cuda")
    tgt = ebm.EpochMailboxes(peers, scratch, epoch_word=eword)
    extents = (
        peer_n,
        [(scratch.data_ptr(), scratch.nbytes), (queue.data_ptr(), queue.nbytes)],
    )
    q = (queue.data_ptr(), queue.nbytes)

    def fill():
        peers.bytes.fill_(0xAB)
        scratch.fill_(0xCD)
        queue.fill_(0xEF)

    def begin():
        tgt.begin_step(extents, q)
        torch.cuda.synchronize()

    # the first step: nothing recorded -> full; everything zeroed, epoch 1
    fill()
    begin()
    assert _state(tgt) == {"epoch": 1, "drafted": -1, "full": 1}
    assert eword.tolist()[0] == 1
    assert not peers.bytes.any() and not scratch.any() and not queue.any()
    # a draft pass recorded epoch 1 -> slim: the queue words only, epoch 2
    drf_peers = _peers(peer_n)
    drf = ebm.EpochMailboxes(
        drf_peers, torch.zeros(64, dtype=torch.uint8, device="cuda"), always_full=True
    )
    drf.link = tgt
    drf.begin_step()
    torch.cuda.synchronize()
    assert _state(drf)["full"] == 1 and _state(tgt)["drafted"] == 1
    fill()
    begin()
    assert _state(tgt) == {"epoch": 2, "drafted": 1, "full": 0}
    assert eword.tolist()[0] == 2
    assert not queue.any()
    assert (peers.bytes == 0xAB).all() and (scratch == 0xCD).all()
    # no draft pass since -> full again
    fill()
    begin()
    assert _state(tgt)["full"] == 1 and _state(tgt)["epoch"] == 3
    assert not peers.bytes.any()
    # the periodic full beginning: epoch FULL_EVERY is full even after a draft pass
    tgt._state[ebm.EPOCH] = ep.FULL_EVERY - 1
    tgt._state[ebm.DRAFTED] = ep.FULL_EVERY - 1
    peers._all[peer_n:].zero_()  # the fence slots below the new epoch
    fill()
    begin()
    assert _state(tgt)["epoch"] == ep.FULL_EVERY and _state(tgt)["full"] == 1
    # and the next one after a draft record is slim again
    drf.begin_step()
    torch.cuda.synchronize()
    fill()
    begin()
    assert _state(tgt)["epoch"] == ep.FULL_EVERY + 1 and _state(tgt)["full"] == 0
    # forced full (the draft runner's own beginning) whatever the record
    assert _state(drf)["full"] == 1


def test_a_slim_begin_waits_for_every_cta_before_it_advances_the_epoch():
    """Many back-to-back slim beginnings: every one advances the epoch by exactly one (a CTA that read the new epoch
    would take the full path and zero the peer data)."""
    peer_n = 4096
    peers = _peers(peer_n)
    scratch = torch.zeros(4096, dtype=torch.uint8, device="cuda")
    queue = torch.zeros(256, dtype=torch.uint8, device="cuda")
    eword = torch.zeros(2, dtype=torch.int64, device="cuda")
    tgt = ebm.EpochMailboxes(peers, scratch, epoch_word=eword)
    drf = ebm.EpochMailboxes(
        _peers(peer_n),
        torch.zeros(64, dtype=torch.uint8, device="cuda"),
        always_full=True,
    )
    drf.link = tgt
    extents = (peer_n, [(scratch.data_ptr(), scratch.nbytes)])
    q = (queue.data_ptr(), queue.nbytes)
    tgt.begin_step(extents, q)
    peers.bytes.fill_(0xAB)
    for _ in range(200):
        drf.begin_step()
        tgt.begin_step(extents, q)
    torch.cuda.synchronize()
    assert _state(tgt)["epoch"] == 201 and _state(tgt)["full"] == 0
    counts = tgt._state.tolist()
    assert counts[ebm.SLIM_COUNT] == 200 and counts[ebm.FULL_COUNT] == 1
    assert (peers.bytes == 0xAB).all(), "a slim beginning zeroed the peer data"
    assert BLOCKS > 1
