# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Flag words and device counters of the decode scorer split over four GPUs (``decode_score.py``; TP4, gfx950,
FlyDSL).

No tickets or atomics on a shared word: a workgroup stores its call count into its own flag word on every receiver
once its stores are released, and the waiters spin on the words they need. Call counts live on the device (one per
workgroup slot), so the handshake replays in CUDA graphs; every rank issues the same calls.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T

WORLD = 4
SYS = 0x11  # sc0 | sc1: system-coherent
D = 128
# flag words per rank (uncached int32):
# [SC_FLAGS + p]: source p's fused scorer (decode_score.py);
# [SC_DONE + w]: this rank's fused-scorer workgroup w;
# [SC_IQ + p * SC_MAX_REQS + r]: source p's index-q rows of request r (fused scorer)
SC_FLAGS = 64
SC_DONE = 1024
SC_MAX_WG = 1024
SC_IQ = 2048
SC_MAX_REQS = 256
FLAG_BYTES = 12288
# device counters (int32): [CNT_SC] fused scorer calls
CNT_SC = 128
CNT_WORDS = 512


def _gptr(addr):
    return fx.inttoptr(
        fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4),
        fx.Int64(addr),
    )


def _flag_read(addr):
    """An atomic add of 0 at system scope (never hoisted out of a spin loop)."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _gptr(addr).llvm_ptr,
            fx.Int32(0).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="",
            alignment=4,
        ).result
    )


def _sel4(p, a0, a1, a2, a3):
    return (p == 0).select(a0, (p == 1).select(a1, (p == 2).select(a2, a3)))
