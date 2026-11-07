# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx950 async buffer -> LDS copies (``buffer_load_dword{,x4} ... lds``) with alias scopes.

A copy writes LDS behind the wave's vector-memory counter; LLVM's waitcnt pass then makes every later LDS access that
may alias it wait for it (``s_waitcnt vmcnt``). Without alias information that is every LDS read: the copies of the
next ring slot would join the reads of the current one (W1). Here the copies and the LDS reads
carry alias scopes (one domain, a scope per LDS area), so a read of another area does not wait for them, and a read
of the copied area waits only for the copies of its scope (the pass tracks LDS DMA stores by scope). Probe:
``experiments/lds_input/probe_waits.py`` (without scopes: ``vmcnt(0)`` before the other slot's ``ds_read``; with them:
none).

Addresses are LDS byte addresses (Int32, wave-uniform for a copy's base): a copy's lane ``l`` writes ``size`` bytes at
``base + size l`` (the hardware's layout: the source offset is per lane, the destination is not), so a caller maps each
lane's destination to its source, and masks the lanes outside the destination area with an ``if``.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rd

from ...plan import trace


def union(*arrays) -> object:
    """The scopes of several one-scope ArrayAttrs as one ArrayAttr."""
    return ir.ArrayAttr.get([a[0] for a in arrays])


def _ptr3(byte):
    return llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), fx.Int32(byte).ir_value()).result


def load(byte, ty, scope=None, noalias=None, volatile=False):
    """An LDS load of type ``ty`` at byte ``byte`` with the given scopes (raw IR value)."""
    kw = {}
    if scope is not None:
        kw["alias_scopes"] = scope
    if noalias is not None:
        kw["noalias_scopes"] = noalias
    align = 16 if isinstance(ty, ir.VectorType) and ty.shape[0] == 4 else 4
    if ty == ir.IntegerType.get_signless(8):
        align = 1
    return llvm.LoadOp(ty, _ptr3(byte), alignment=align, volatile_=volatile or None, **kw).result


def copy(rsrc8, voffset, soffset, lds_byte, size, aux, scope=None, noalias=None, offset=0):
    """This wave's async copy of ``size`` (4 or 16) bytes a lane: buffer ``rsrc8`` (``!llvm.ptr<8>``) at byte
    ``soffset + voffset + offset`` (``voffset`` per lane) to LDS ``lds_byte + offset + size lane`` (``lds_byte``
    uniform; ``offset`` the instruction's immediate, < 4096: copies sharing an LDS base share its M0)."""
    i32 = ir.IntegerType.get_signless(32)
    trace.mem("load", rsrc8, int(aux), size)  # the memory-model record (no IR): a buffer load like buffer_ops'
    kw = {}
    if scope is not None:
        kw["alias_scopes"] = scope
    if noalias is not None:
        kw["noalias_scopes"] = noalias
    _rd.RawPtrBufferLoadLdsOp(
        rsrc8,
        _ptr3(lds_byte),
        fx.Int32(size).ir_value(),
        fx.Int32(voffset).ir_value(),
        fx.Int32(soffset).ir_value(),
        fx.Int32(int(offset)).ir_value(),
        aux=ir.IntegerAttr.get(i32, int(aux)),
        **kw,
    )
