# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 primitives preserving the original tagged mailbox hand-offs."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import T

from atom.mono.device.ops import traced
from atom.mono.device.sync import Mailbox

WAIT_TICKS = 10_000_000  # 100 ms at the validated 100 MHz steady clock.


def scalar_address(address):
    """Restore a wave-uniform mother address at the FFN stage boundary.

    Earlier stages keep addresses in VGPRs to bound their scalar live set.
    Passing those opaque values to a buffer resource makes LLVM emit a
    waterfall and load-counter drain at every access. The FFN's addresses
    are uniform by contract; materialize that scalar value when it is used.
    The side-effecting move keeps it out of the earlier stages' live set.
    """
    scalar = rocdl.readfirstlane(T.i64, fx.Int64(address).ir_value())
    return fx.Int64(llvm.inline_asm(
        T.i64, [scalar], "s_mov_b64 $0, $1", "=s,s", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


def reference_mailbox(layer, diag, regions, failure, *, diagnostic=False):
    """Mother normal polling, with gfx1250's required system-scope encoding.

    Clock, wave-wide pending checks and failure propagation belong to the
    explicitly requested diagnostic variant. Normal polling uses the original
    per-lane pending condition and s_sleep(1), without result masking.
    """
    if diagnostic:
        return Mailbox(
            layer, diag, regions, steady=True, failure=failure, wait_ticks=WAIT_TICKS
        )
    return Mailbox(layer, gfx1250=True)


@traced
def publish_plain(put, addr, index, value, who):
    """Drain each writer wave's stores, join the CTA, then publish a tagged pair.

    gfx1250 separates the load and store counters. The mother's vmcnt drain
    therefore needs a store-counter instruction here; a load-counter wait
    alone cannot fulfill its plain-payload-before-readiness contract.
    """
    llvm.inline_asm(
        None, [], "s_wait_storecnt 0", "", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    gpu.barrier()
    if who:
        put(addr, index, value)
