# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A launch's work queues: an i32 count a queue, zeroed at the step's start,
that CTAs take the next task from by an atomic add -- a CTA its earlier work
delayed takes fewer, where a fixed placement gave every CTA the same count.

A launch's words sit at its tag's offset (``tag_offset``), computed on the
host and handed to the kernel: never ``tag * 4`` inside it. A Kimi-K3 64-row
K2 with 8 AttnRes blocks took its queue word from a register holding a peer
base's high word that way -- the LLVM IR right, the ISA not (2026-10-30) --
and claimed garbage.
"""

import flydsl.expr as fx
from flydsl.expr import gpu
from flydsl.expr.typing import T

from atom.mono.device.ops import traced


def tag_offset(layer: int) -> int:
    """The byte offset of the launch ``layer``'s queue word (its tag's,
    ``Mailbox``: ``layer`` + 1) in a queue's array, a word a tag. A kernel's
    queues each an array of their own: one launch's words packed together
    share a cache line its CTAs' atomics contend for."""
    return (layer + 1) * 4


def word(addr):
    """The i32 queue word at device address ``addr`` (an Int64)."""
    return fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), addr)


def take(queue):
    """The next count of ``queue`` (``word``), one thread's: its old value."""
    return fx.llvm.atomic_add(queue, fx.Int32(1), syncscope="agent")


@traced
def take_shared(queue, slot, tid):
    """``take`` for the whole CTA: thread 0 takes, every thread reads it from
    LDS ``slot`` past a barrier. A barrier first: every thread has read the
    slot's previous take (two takes in a row would let thread 0 overwrite it
    under a slow reader)."""
    gpu.barrier()
    if tid == 0:
        fx.ptr_store(fx.Int32(take(queue)), slot)
    gpu.barrier()
    return fx.ptr_load(slot)
