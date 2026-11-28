# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The V4.1 target runner's step beginning with epoch tags (``kernels.mailbox``: a pair's tag carries the step's
epoch), one launch that decides on the device which of two beginnings this step takes:

- full: as ``atom.mono.runtime.step_begin`` -- the peer data and every scratch range zeroed, then the peer fence (each
  rank's epoch into its slot at every rank's buffer tail, its own slots polled until all reach it);
- slim: the epoch advanced and the launches' queue words zeroed (atomic task counters: no tag tells their steps
  apart), nothing else -- no peer zeroing, no fence. The step's pairs carry the new epoch, so a pair an earlier step
  left never passes for one of this step's, and the step's own data dependencies order every rank's first push of it
  after every rank's last read of the previous step's: the draft runner's own full begin (its fence) between two
  target steps, and the LM head's and the draft head's every-rank gathers before it.

The decision, the same on every rank (every rank runs the same launches): slim iff a draft pass began after the
previous target step (``drafted``, a word the draft runner's begin sets to this runner's epoch, equals the epoch this
step follows) and the new epoch is not a multiple of FULL_EVERY (a periodic full begin clears every tag before a
coded pair's 16 epoch bits could repeat, ``kernels.mailbox.CODED_TAG_BITS``). Any other step -- a capture or warm-up
forward, the first after a prefill, one whose draft did not run -- begins full.

The waits are FlyDSL's, as the shared module's (Triton compiled its poll into a loop that could spin for ever).
"""

import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import Int32, Int64

from atom.models.deepseek_v41.mono.epoch_policy import FULL_EVERY
from atom.mono.device.ops import (
    CM_DEV,
    CM_SYS,
    kernel_symbol,
    launcher,
    load_ptr64,
    rsrc,
)
from atom.mono.plan.build_key import key_tuple, source_digest
from atom.mono.plan.execution import BLOCKS, THREADS
from atom.mono.runtime.peer_memory import PeerBuffer
from atom.mono.runtime.step_begin import (
    ARRIVED,
    EPOCH,
    FENCE_SLOTS,
    MAX_SCRATCH,
    MISSING,
    STATE_WORDS,
    ZERO_ALIGN,
    _fence,
    _ld64,
    _wait_arrived,
    _words,
    _zero,
)

_CURRENT_STREAM = fx.Stream(None)

# this runner's state words past the shared layout: the epoch the last draft pass followed (written by the draft
# runner's begin), then the last beginning taken (1 full, 0 slim; a test reads it)
DRAFTED = STATE_WORDS
TOOK_FULL = STATE_WORDS + 1
# the beginnings taken, slim and full (CTA 0 counts them; a test reads them)
SLIM_COUNT = STATE_WORDS + 2
FULL_COUNT = STATE_WORDS + 3
EPOCH_STATE_WORDS = STATE_WORDS + 4


@dataclass(frozen=True)
class EpochBeginBuild:
    tp: int
    debug: bool


@functools.cache
def _build(key: EpochBeginBuild):
    tp, debug = key.tp, key.debug
    assert 1 <= tp <= FENCE_SLOTS
    keyed = key_tuple(
        key,
        source_digest(
            "mono",
            "models/deepseek_v41/mono/epoch_begin.py",
            "models/deepseek_v41/mono/epoch_policy.py",
        ),
    )
    name = kernel_symbol("v41_epoch_begin", tp=tp, debug=int(debug))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def epoch_begin(
        s0: Int64, n0: Int64, s1: Int64, n1: Int64, s2: Int64, n2: Int64,
        s3: Int64, n3: Int64, q0: Int64, qn: Int64, peers: Int64, peer_n: Int64,
        fence_off: Int64, rank: Int32, state: Int64, eword: Int64, link: Int64,
        force_full: Int32,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane0, stride = bid * THREADS + tid, BLOCKS * THREADS
        # this step's epoch, and the decision, from words no CTA of this launch
        # writes before every CTA has flagged (below): every CTA decides alike
        e = _ld64(rsrc(state), EPOCH) + 1
        drafted = _ld64(rsrc(state), DRAFTED)
        slim = (force_full == 0) & (drafted == e - 1) & ((e & (FULL_EVERY - 1)) != 0)
        if slim:
            # the queue words only (atomic task counters: no tag tells their
            # steps apart); then every CTA flags, CTA 0 waits for all of them
            # before it advances the epoch (a late CTA must still read the old
            # one: it decides from it)
            _zero(q0, qn, lane0, stride)
            gpu.barrier()
            if (bid != 0) & (tid == 0):
                bo.buffer_store(
                    fx.Int32(e & 0xFFFFFFFF), rsrc(state), ARRIVED * 2 + bid,
                    cache_modifier=CM_DEV,
                )  # fmt: skip
            if (bid == 0) & (tid < 64):
                _wait_arrived(state, e)
                if tid == 0:
                    bo.buffer_store(_words(e), rsrc(state), EPOCH * 2)
                    bo.buffer_store(_words(fx.Int64(0)), rsrc(state), TOOK_FULL * 2)
                    bo.buffer_store(
                        _words(_ld64(rsrc(state), SLIM_COUNT) + 1),
                        rsrc(state),
                        SLIM_COUNT * 2,
                    )
        else:
            # the peer data first, at system scope: in memory before the epoch
            # announces it to a peer (``atom.mono.runtime.step_begin``'s order)
            _zero(load_ptr64(peers, rank), peer_n, lane0, stride, CM_SYS)
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if (bid != 0) & (tid == 0):
                bo.buffer_store(
                    fx.Int32(e & 0xFFFFFFFF), rsrc(state), ARRIVED * 2 + bid,
                    cache_modifier=CM_DEV,
                )  # fmt: skip
            if (bid == 0) & (tid < 64):
                _wait_arrived(state, e)
                _fence(peers, fence_off, rank, state, e, tp, debug)
                if tid == 0:
                    bo.buffer_store(_words(fx.Int64(1)), rsrc(state), TOOK_FULL * 2)
                    bo.buffer_store(
                        _words(_ld64(rsrc(state), FULL_COUNT) + 1),
                        rsrc(state),
                        FULL_COUNT * 2,
                    )
            _zero(s0, n0, lane0, stride)
            _zero(s1, n1, lane0, stride)
            _zero(s2, n2, lane0, stride)
            _zero(s3, n3, lane0, stride)
        # the epoch word the step's launches read their tags' epoch from
        # (``kernels.mailbox.epoch_of``; none of this launch reads it), and a
        # draft runner's record of the target epoch it followed (``link``: the
        # target's state), CTA 0's
        if (bid == 0) & (tid == 0):
            if eword != 0:
                bo.buffer_store(_words(e), rsrc(eword), 0)
            if link != 0:
                bo.buffer_store(
                    _words(_ld64(rsrc(link), EPOCH)), rsrc(link), DRAFTED * 2
                )

    @launcher(name)
    def launch(
        s0: Int64, n0: Int64, s1: Int64, n1: Int64, s2: Int64, n2: Int64,
        s3: Int64, n3: Int64, q0: Int64, qn: Int64, peers: Int64, peer_n: Int64,
        fence_off: Int64, rank: Int32, state: Int64, eword: Int64, link: Int64,
        force_full: Int32, stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        epoch_begin(
            s0, n0, s1, n1, s2, n2, s3, n3, q0, qn, peers, peer_n, fence_off, rank, state,
            eword, link, force_full,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch


class EpochMailboxes:
    """``StepMailboxes`` for the target runner with epoch tags: ``begin_step(extents, queue)`` launches the epoch
    begin (``queue``: (address, bytes) of the launches' queue words, the one range a slim begin zeroes);
    ``epoch_ptr`` is the word every epoch-tagged launch reads its epoch from; ``drafted_ptr`` the word the draft
    runner's begin writes (``mark_drafted``)."""

    def __init__(
        self,
        peers: PeerBuffer,
        *scratch: torch.Tensor,
        debug: bool = False,
        epoch_word=None,
        always_full: bool = False,
    ) -> None:
        """``epoch_word``: the runner's (an int64 slot right before its queue words) the begin stores each new epoch
        in; ``always_full``: a full beginning every step (the draft runner: its record of the target epoch is the
        target's slim condition, ``link``)."""
        self.peers = peers
        self._scratch = scratch
        self._debug = debug
        self._eword = 0 if epoch_word is None else epoch_word.data_ptr()
        self._always_full = always_full
        self.link = None  # the target runner's EpochMailboxes this (draft) runner's begin records into
        self._state = torch.zeros(
            EPOCH_STATE_WORDS, dtype=torch.int64, device=peers.bytes.device
        )
        # no draft pass recorded yet: the first beginning (epoch 1) is full
        self._state[DRAFTED] = -1

    @property
    def epoch_ptr(self) -> int:
        return self._state.data_ptr() + EPOCH * 8

    @property
    def drafted_ptr(self) -> int:
        return self._state.data_ptr() + DRAFTED * 8

    def begin_step(self, extents=None, queue=None) -> None:
        if extents is None:
            extents = (
                self.peers.bytes.nbytes,
                [(b.data_ptr(), b.nbytes) for b in self._scratch],
            )
        peer_n, ranges = extents
        assert len(ranges) <= MAX_SCRATCH and peer_n <= self.peers.bytes.nbytes
        q0, qn = queue if queue is not None else (0, 0)
        sizes = [peer_n, qn] + [n for _, n in ranges]
        assert all(n % ZERO_ALIGN == 0 for n in sizes), sizes
        args = []
        for at, n in ranges:
            args += [at, n]
        args += [0, 0] * (MAX_SCRATCH - len(ranges))
        _run_compiled(
            _build(EpochBeginBuild(self.peers.addresses.numel(), self._debug)),
            *args, q0, qn,
            self.peers.addresses.data_ptr(), peer_n, self.peers.bytes.nbytes, self.peers.rank,
            self._state.data_ptr(), self._eword,
            0 if self.link is None else self.link._state.data_ptr(), int(self._always_full),
            torch.cuda.current_stream(),
        )  # fmt: skip

    def missing_ranks(self) -> list[int]:
        missing = self._state[MISSING : MISSING + self.peers.addresses.numel()]
        ranks = [r for r, m in enumerate(missing.tolist()) if m]
        missing.zero_()
        return ranks
