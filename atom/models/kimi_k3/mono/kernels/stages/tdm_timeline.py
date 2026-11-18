# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Debug-only per-CTA timeline of the TDM GEMV kernels (resident K1, the
replicated front): the low word of gfx1250's 100 MHz realtime counter at each
point, stored by one wave straight to a global buffer (stores retire on
storecnt, so the TDM waits never see them).

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
WORDS = 96
# Points: 0 HW_ID1, 1 entry, 2 first global access of the GEMV, 3 its first
# copies issued; then per slab s, SLAB0 + SLAB_WORDS * s + one of
# A (slab's top barrier passed), T (wave 0's token slab landed),
# W (the waiting weight wave's slab landed), B (every copy landed),
# C (wave 0's WMMAs of the slab issued, before the next top barrier).
HW_ID, ENTRY, GEMV_START, ISSUED = 0, 1, 2, 3
SLAB0, SLAB_WORDS, MAX_SLABS = 8, 5, 16
A, TOKEN, WEIGHT, B, C = range(SLAB_WORDS)
PROJECT_DONE, STAGED, STORES_ISSUED, STORES_DONE, RECURRENCE, END = range(88, 94)
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


@traced
def _stamp_on(role, wave_role, base, index):
    if role == wave_role:
        _store(_realtime(), base, index)


@traced
def _slab_on(role, wave_role, base, index, s):
    if (s < MAX_SLABS) & (role == wave_role):
        _store(_realtime(), base, index)


@traced
def _open_on(role, base, slot, entry):
    if role == 0:
        _store(_hw_id(), base, slot + HW_ID)
        _store(entry, base, slot + ENTRY)


@traced
def _close_on(role, base, bid, launch):
    if role == 0:
        _store(fx.Int32(launch) + 1, base, bid)


def active() -> bool:
    return _context is not None


def open_cta(bid, wave) -> None:
    """At the kernel's entry: this CTA's slot, HW_ID1 and the entry stamp
    (wave 0)."""
    global _context
    if _config is None:
        return
    address, launches = _config
    role = fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(fx.Int32(wave))))
    base = fx.Int64(address)
    entry = _realtime()
    launch = fx.Int32(fx.ptr_load(_word(base, bid)))
    slot = CTAS + ((fx.Int32(launch) % launches) * CTAS + bid) * WORDS
    _context = {"base": base, "slot": slot, "role": role, "bid": bid, "n": launch}
    _open_on(role, base, slot, entry)


def stamp(point, wave_role=0) -> None:
    """Point ``point`` (a word index, maybe traced) on wave ``wave_role``."""
    if _context is None:
        return
    _stamp_on(_context["role"], wave_role, _context["base"], _context["slot"] + point)


def slab(s, which, wave_role=0) -> None:
    if _context is None:
        return
    index = _context["slot"] + SLAB0 + SLAB_WORDS * s + which
    _slab_on(_context["role"], wave_role, _context["base"], index, s)


def stores_done() -> None:
    """Every wave's stores retired, then the stamp."""
    if _context is None:
        return
    rocdl.s_wait_storecnt(0)
    gpu.barrier()
    stamp(STORES_DONE)


def close_cta() -> None:
    """At the kernel's end: the END stamp and the CTA's next launch number."""
    global _context
    if _context is None:
        return
    stamp(END)
    _close_on(_context["role"], _context["base"], _context["bid"], _context["n"])
    _context = None
