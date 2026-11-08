# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A device-round step's pre-target launches as one kernel (the V4.1 mono hi hook's device round, ``device_round``).

Between the draft graph and the target graph a device-round step (``device_round``) ran four eager launches, back to
back, all from ``ModelRunner.prepare_model``: the token inputs' packed publication (``scatter_packed_bytes``: the
sampling rows, ``cu_seqlens_q``, the staged ids, the deferred requests' source rows), ``fill_deferred_decode_ids``
(each carried request's anchor and drafts from the previous step's sampled and draft ids, the graph's padded tail
zeroed), the V4.1 step metadata's packed publication (positions, plans, state slots, block tables) and the device
round's ``_advance_kernel`` (each request's positions from its previous start and the verdict). The three before the
last are recorded instead of launched (``HostArenaReader.take``: the two publications opt in through
``PublicationGroup.read_host_arena``; the runner's decode-round hook in place of ``fill_deferred_decode_ids``) and the advance launches ``_prologue_kernel``: both arenas' scatter programs (each
``scatter_packed_bytes``' (member, 1 KiB block)), every program then arriving on a counter (acquire-release at device
scope: its stores before, across the XCDs' L2s); the last to arrive writes the fill, then the advance (the fill after
the token inputs land, the advance after the positions do, as the four ran). Every write is the one the four made,
from the same sources, so the step's buffers hold the same bytes when the target graph starts. (One program doing all
of it, the first form, paid each member's host-arena round trip in turn: +9..+17 us at 18-24 rows.)

Every launch stays where it was relative to the graphs: before the target graph, after the draft graph, on the same
stream, so what the draft round and the target read is unchanged. The arenas stay fenced: the owner's reuse event
(``_mark_staging_h2d_enqueued``) is recorded after the fused launch, which also flushes anything still recorded there.
Only device-round steps arm the recorder (a served lean verify of the requests that left the verdict); a P/D
consumer's first step, a prefill, a dummy or graph-capture forward and every other step launch as before.

The same reader makes the decode publications zero-copy (``reads``): on a decode step the hook serves, an eager launch
of the two publications (``DECODE_PUBLICATIONS``) of at most ``HOST_ARENA_MAX_BYTES`` reads the pinned arena through
its device address instead of after an H2D copy (the token inputs on the device round's test before the step's routing
decision, ``device_round.may_serve``; the step metadata on the decision itself, ``HiRound.service``); a decode step
ATOM's own path runs (a width the hook does not serve), a prefill's, a graph capture's and every other publication
copy as before.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from atom.utils.h2d import HostArenaReader
from atom.utils.packed_h2d import DeviceAddress
from atom.models.deepseek_v41.mono.hi_gate import log_once

logger = logging.getLogger("atom")

# the publications a served decode step makes (the token inputs, the V4.1 step metadata): the only ones opted into
# host-arena reads (``PublicationGroup.read_host_arena``)
DECODE_PUBLICATIONS = ("token_inputs", "v41_metadata")
# a decode publication at the served widths, block tables included, stays below this (a seatbelt: a step the hook
# does not serve is copied whatever its size)
HOST_ARENA_MAX_BYTES = 1 << 20

BLOCK = (
    1024  # bytes a scatter program (``scatter_packed_bytes``'), ids a pass of the tail
)


@triton.jit
def _scatter_one(arena, destinations, member, block, offs, BLOCK: tl.constexpr):
    """``scatter_packed_bytes``' program (``member``, ``block``): member's header (offset, byte count) at int64 words
    2 m / 2 m + 1 of ``arena``, bytes ``block`` BLOCK .. of it to ``destinations[member]``.
    """
    header = arena.to(tl.pointer_type(tl.int64))
    start = tl.load(header + 2 * member)
    length = tl.load(header + 2 * member + 1)
    destination = tl.load(destinations + member).to(tl.pointer_type(tl.uint8))
    o = block * BLOCK + offs
    tl.store(
        destination + o, tl.load(arena + start + o, o < length, other=0), o < length
    )


@triton.jit(
    do_not_specialize=[
        "members1",
        "blocks1",
        "members2",
        "blocks2",
        "requests",
        "width",
        "advances",
        "rows",
    ]
)
def _prologue_kernel(
    arena1,
    destinations1,
    members1,
    blocks1,
    arena2,
    destinations2,
    members2,
    blocks2,
    arrivals,
    ids_ptr,
    cu_ptr,
    src_ptr,
    prev_ptr,
    draft_ptr,
    draft_row_stride,
    requests,
    width,
    positions_ptr,
    start_ptr,
    verdict_ptr,
    verdict_stride,
    advances,
    rows,
    FILL: tl.constexpr,
    HAS_DRAFT: tl.constexpr,
    ADVANCE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Module docstring: programs ``members x blocks`` of each arena scatter as ``scatter_packed_bytes``' would; each
    program then arrives (``arrivals``, acquire-release at device scope: its stores before, the last arrival's loads
    after), and the last to arrive writes ``_fill_deferred_decode_ids_kernel``'s spans in request order and its zeroed
    tail, then ``_advance_kernel``'s positions, and resets the counter."""
    pid = tl.program_id(0)
    n1 = members1 * blocks1
    n2 = members2 * blocks2
    offs = tl.arange(0, BLOCK)
    if pid < n1:
        _scatter_one(arena1, destinations1, pid // blocks1, pid % blocks1, offs, BLOCK)
    elif pid < n1 + n2:
        q = pid - n1
        _scatter_one(arena2, destinations2, q // blocks2, q % blocks2, offs, BLOCK)
    tl.debug_barrier()
    arrived = tl.atomic_add(arrivals, 1, sem="acq_rel", scope="gpu")
    if arrived == tl.num_programs(0) - 1:
        tl.store(arrivals, 0)
        if FILL:
            col = tl.arange(0, BLOCK_Q)
            for i in range(requests):
                src = tl.load(src_ptr + i, cache_modifier=".cg")
                if src >= 0:  # (NEW_SEQUENCE: staged by the host, left as published)
                    start = tl.load(cu_ptr + i, cache_modifier=".cg")
                    end = tl.load(cu_ptr + i + 1, cache_modifier=".cg")
                    live = col < (end - start)
                    val = tl.load(prev_ptr + src)
                    if HAS_DRAFT:
                        drafts = tl.load(
                            draft_ptr + src * draft_row_stride + (col - 1),
                            mask=live & (col > 0),
                            other=0,
                        )
                        val = tl.where(col == 0, val, drafts)
                    else:
                        val = tl.broadcast_to(val, [BLOCK_Q])
                    tl.store(ids_ptr + start + col, val, mask=live)
            end = tl.load(cu_ptr + requests, cache_modifier=".cg")
            for base in range(0, width, BLOCK):
                c = base + offs
                tl.store(ids_ptr + c, 0, mask=(c >= end) & (c < width))
        if ADVANCE:
            r = tl.arange(0, BLOCK_R)
            for b in range(advances):
                s = (
                    tl.load(start_ptr + b).to(tl.int64)
                    + 1
                    + tl.load(verdict_ptr + b * verdict_stride).to(tl.int64)
                )
                tl.store(
                    positions_ptr + b * rows + r,
                    (s + r).to(positions_ptr.dtype.element_ty),
                    mask=r < rows,
                )


class Prologue(HostArenaReader):
    """The recorder (module docstring): ``armed`` by the device round on its steps; ``take``
    (``HostArenaReader.take``), ``fill`` and ``advance`` record; ``launch`` runs the fused kernel. ``reads``: a served
    decode step's publications read their pinned arenas in place."""

    def __init__(self, device):
        self.armed, self.served = False, False
        self.dummy = torch.zeros(8, dtype=torch.int64, device=device)
        self.arrivals = torch.zeros(1, dtype=torch.int32, device=device)
        self.scatters, self.fills = [], []
        # fused launches; publications past two a step (launched as before)
        self.launches = self.extra = 0

    def arm(self, on: bool, served: bool):
        """Every step, before its publications: a device-round step (``on``) records until its advance; ``served``:
        the step is a decode step the hook may serve (its publications read their pinned arenas, ``reads``; the
        step's routing decision settles it before the step metadata's publication, ``serve``).
        """
        self.armed, self.served = on, served

    def serve(self, served: bool):
        """The step's routing decision (``HiRound.service``): whether the hook serves it."""
        self.served = served

    # ---- the publications (``HostArenaReader``)
    def reads(self, nbytes: int) -> bool:
        """A served decode step's publication of at most ``HOST_ARENA_MAX_BYTES`` reads its pinned arena in place."""
        return self.served and nbytes <= HOST_ARENA_MAX_BYTES

    # ---- recording
    def take(self, copy, arena, largest) -> bool:
        """The publication's deferral: record its scatter (from the host arena at ``arena``), at most two a step
        while armed (a third launches as before)."""
        if not self.armed:
            return False
        if len(self.scatters) == 2:
            self.extra += 1
            return False
        self.scatters.append(
            (arena, copy.destinations, len(copy.members), -(-largest // BLOCK))
        )
        return True

    def fill(self, out, cu, src, prev, draft, *, max_tokens_per_seq, width) -> bool:
        """``fill_deferred_decode_ids``' deferral: record its arguments."""
        bs = int(src.shape[0])
        self.fills.append(
            (out, cu, src, prev, draft, bs, int(width), int(max_tokens_per_seq))
        )
        return True

    def advance(self, positions, start, verdict, rows, requests):
        """The device round's advance: launch the fused kernel with everything recorded."""
        self.launch((positions, start, verdict, int(rows), int(requests)))

    def flush(self):
        """Anything still recorded (the owner's reuse event is next): launch it."""
        if self.scatters or self.fills:
            self.launch(None)
        self.armed = False

    def _reset(self):
        self.scatters, self.fills = [], []

    # ---- the launch
    def launch(self, adv):
        if len(self.fills) > 1:
            raise RuntimeError(
                "V4.1 mono prologue: more than one deferred ids fill in a step"
            )
        sc = self.scatters + [(self.dummy.data_ptr(), self.dummy, 0, 0)] * (
            2 - len(self.scatters)
        )
        grid = max(sc[0][2] * sc[0][3] + sc[1][2] * sc[1][3], 1)
        d = self.dummy
        if self.fills:
            out, cu, src, prev, draft, bs, width, mq = self.fills[0]
            fill = (
                out,
                cu,
                src,
                prev,
                draft if draft is not None else prev,
                draft.stride(0) if draft is not None else 0,
                bs,
                width,
            )
            block_q = triton.next_power_of_2(max(mq, 1))
        else:
            fill, block_q = (d, d, d, d, d, 0, 0, 0), 1
        if adv is not None:
            positions, start, verdict, rows, requests = adv
            av = (positions, start, verdict, verdict.stride(1), requests, rows)
        else:
            av, rows = (d, d, d, 0, 0, 0), 1
        _prologue_kernel[(grid,)](
            DeviceAddress(sc[0][0]),
            sc[0][1],
            sc[0][2],
            sc[0][3],
            DeviceAddress(sc[1][0]),
            sc[1][1],
            sc[1][2],
            sc[1][3],
            self.arrivals,
            *fill,
            *av,
            FILL=bool(self.fills),
            HAS_DRAFT=bool(self.fills) and self.fills[0][4] is not None,
            ADVANCE=adv is not None,
            BLOCK_Q=block_q,
            BLOCK_R=triton.next_power_of_2(max(rows, 2)),
            BLOCK=BLOCK,
        )
        self.launches += 1
        self._reset()
        self.armed = False


def install(runner) -> Prologue:
    """The recorder (the runner opts its ``DECODE_PUBLICATIONS`` into it: ``PublicationGroup.read_host_arena``); the
    runner's ids fill and its reuse-event mark reach it through ``HiRound``."""
    pro = Prologue(runner.device)
    log_once(
        logging.INFO,
        "V4.1 mono hi: device-round steps' pre-target launches as one kernel",
    )
    return pro
