# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Debug-only per-CTA timeline of the TDM GEMV kernels (resident K1, the
replicated front): the low word of the shader cycle counter at each point
(the 100 MHz realtime counter, a message round trip, at entry and end only),
kept in a 128-word LDS record and copied to a global buffer at the CTA's end
(a global store each point would make every barrier's release wait for it).

``enable(address, launches)`` before a build bakes the buffer in: the first
``CTAS`` words are each CTA's launch counter, then ``[launches][CTAS][WORDS]``
stamps, a CTA's launch ``n`` at slot ``n % launches``. Disabled (the default),
every call traces to nothing and the kernels are unchanged. Realtime offsets
differ between XCDs: compare stamps within a CTA, or across CTAs of one XCD.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.mono.device.ops import traced

CTAS = 256
WORDS = 128
LANES = 32
# Points: 0 HW_ID1, 1 entry, 2 first global access of the GEMV, 3 its first
# copies issued; then per slab s, SLAB0 + SLAB_WORDS * s + one of
# A (slab's top barrier passed), T (wave 0's token slab landed),
# W (the waiting weight wave's slab landed), B (every copy landed),
# C (wave 0's WMMAs of the slab issued, before the next top barrier).
HW_ID, ENTRY, GEMV_START, ISSUED = 0, 1, 2, 3
SLAB0, SLAB_WORDS, MAX_SLABS = 8, 5, 16
A, TOKEN, WEIGHT, B, C = range(SLAB_WORDS)
PROJECT_DONE, STAGED, STORES_ISSUED, STORES_DONE, RECURRENCE, END = range(88, 94)
# ENTRY and END are realtime; every other point is the low word of the shader
# cycle counter (no message round trip), with these two to convert them.
CYCLES_ENTRY, CYCLES_END = 94, 95
# K1's whole-head recurrence task (``kda_gfx1250.kda_head``, the CTA's first):
# entered, its projected rows polled, conv done, gates done, recurrence done,
# its output gates polled, core_out stored; a split head's norm exchange done.
HEAD_START, HEAD_POLLED, HEAD_CONV, HEAD_GATES = 96, 97, 98, 99
HEAD_RECUR, HEAD_GATE_POLLED, HEAD_DONE, HEAD_EXCHANGED = 100, 101, 102, 103
# Inside the gates: the gate WMMA issued (wave 0), then every wave past it.
HEAD_GATE_MMA, HEAD_GATE_SYNC = 104, 105
POINTS = {
    "hw_id": HW_ID,
    "entry": ENTRY,
    "gemv_start": GEMV_START,
    "issued": ISSUED,
    "project_done": PROJECT_DONE,
    "staged": STAGED,
    "stores_issued": STORES_ISSUED,
    "stores_done": STORES_DONE,
    "recurrence": RECURRENCE,
    "end": END,
    "head_start": HEAD_START,
    "head_polled": HEAD_POLLED,
    "head_conv": HEAD_CONV,
    "head_gates": HEAD_GATES,
    "head_recur": HEAD_RECUR,
    "head_gate_polled": HEAD_GATE_POLLED,
    "head_done": HEAD_DONE,
    "head_exchanged": HEAD_EXCHANGED,
    "head_gate_mma": HEAD_GATE_MMA,
    "head_gate_sync": HEAD_GATE_SYNC,
}

_config: tuple[int, int] | None = None
_context: dict | None = None


def enable(address: int, launches: int) -> None:
    global _config
    _config = (int(address), int(launches))


def disable() -> None:
    global _config
    _config = None


def key():
    """Part of every timeline build's name and cache key (None: off)."""
    return _config


def suffix() -> str:
    return "" if _config is None else f"_tl{_config[0]:x}"


def buffer_words(launches: int) -> int:
    return CTAS + launches * CTAS * WORDS


def _word(base, index):
    return fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
        base + fx.Int64(index) * 4,
    )


def _store(value, base, index):
    fx.generic_store(_word(base, index), fx.Int32(value))


def _realtime():
    return fx.Int32(
        fx.Int64(
            llvm.inline_asm(
                T.i64,
                [],
                "s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\n"
                " s_wait_kmcnt 0",
                "=s",
                has_side_effects=True,
            )
        )
    )


def _cycles():
    return fx.Int32(
        fx.Int64(
            llvm.inline_asm(
                T.i64, [], "s_get_shader_cycles_u64 $0", "=s", has_side_effects=True
            )
        )
    )


def _hw_id():
    return fx.Int32(
        llvm.inline_asm(
            T.i32,
            [],
            "s_getreg_b32 $0, hwreg(HW_REG_HW_ID1)",
            "=s",
            has_side_effects=True,
        )
    )


@fx.struct
class _Record:
    words: fx.Array[fx.Int32, WORDS, 16]


@traced
def _stamp_on(role, wave_role, record, index):
    if role == wave_role:
        fx.ptr_store(_cycles(), record + index)


@traced
def _slab_on(role, wave_role, record, index, s):
    if (s < MAX_SLABS) & (role == wave_role):
        fx.ptr_store(_cycles(), record + index)


@traced
def _open_on(role, lane, record, entry, cycles):
    if role == 0:
        for k in range(WORDS // LANES):
            fx.ptr_store(fx.Int32(0), record + (lane + LANES * k))
        if lane == 0:
            fx.ptr_store(_hw_id(), record + HW_ID)
            fx.ptr_store(entry, record + ENTRY)
            fx.ptr_store(cycles, record + CYCLES_ENTRY)


@traced
def _close_on(role, lane, record, base, bid, launches):
    if role == 0:
        if lane == 0:
            cycles = _cycles()
            fx.ptr_store(_realtime(), record + END)
            fx.ptr_store(cycles, record + CYCLES_END)
        launch = fx.Int32(fx.ptr_load(_word(base, bid)))
        slot = CTAS + ((launch % launches) * CTAS + bid) * WORDS
        for k in range(WORDS // LANES):
            index = lane + LANES * k
            _store(fx.ptr_load(record + index), base, slot + index)
        if lane == 0:
            _store(launch + 1, base, bid)


def active() -> bool:
    return _context is not None


def open_cta(bid, wave, config, allocator) -> None:
    """At the kernel's entry: this CTA's LDS record, HW_ID1 and the entry
    stamps (wave 0). ``config`` is ``key()`` taken when the kernel was built:
    its body is traced at the first launch, after the build's ``disable()``.
    ``allocator`` is the kernel's (static) SharedAllocator, one a kernel."""
    global _context
    if config is None:
        return
    address, launches = config
    role = fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(fx.Int32(wave))))
    lane = fx.thread_idx.x % LANES
    entry = _realtime()
    cycles = _cycles()
    record = allocator.allocate(_Record).peek().words.ptr
    _context = {
        "base": fx.Int64(address),
        "launches": launches,
        "record": record,
        "role": role,
        "lane": lane,
        "bid": bid,
    }
    _open_on(role, lane, record, entry, cycles)


def stamp(point, wave_role=0) -> None:
    """Point ``point`` (a word index, maybe traced) on wave ``wave_role``."""
    if _context is None:
        return
    _stamp_on(_context["role"], wave_role, _context["record"], point)


def slab(s, which, wave_role=0) -> None:
    if _context is None:
        return
    index = SLAB0 + SLAB_WORDS * s + which
    _slab_on(_context["role"], wave_role, _context["record"], index, s)


def stores_done() -> None:
    """Every wave's stores retired, then the stamp."""
    if _context is None:
        return
    rocdl.s_wait_storecnt(0)
    gpu.barrier()
    stamp(STORES_DONE)


def close_cta() -> None:
    """At the kernel's end (every wave): the END stamps, then wave 0 copies
    the record to the CTA's slot and bumps its launch number."""
    global _context
    if _context is None:
        return
    gpu.barrier()
    c = _context
    _close_on(c["role"], c["lane"], c["record"], c["base"], c["bid"], c["launches"])
    _context = None
