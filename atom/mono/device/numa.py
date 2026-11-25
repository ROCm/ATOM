# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Which copy of a dual region (``atom.mono.plan.layout.dual_layout``) a CTA
reads: the one homed on its own I/O die, a poll's round trip from the other
die's memory being longer. Which die homes which copy, page by page, comes
from the calibration of the scratch it lies in (``atom.mono.runtime.numa``):
a word, 8 bits a page, bit x the copy XCD x reads."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T

from atom.mono.device.sync import dreg, sreg
from atom.mono.plan.layout import NUMA_RUN, dual_places


def xcc_id():
    """This CTA's XCD (``HW_REG_XCC_ID``)."""
    v = llvm.InlineAsmOp(
        T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)", "=s", has_side_effects=True
    )
    return fx.Int32(v.result) & 7


def local_copy(word, page: int, xcc):
    """The word offset of the copy XCD ``xcc`` reads in page ``page`` of a dual
    area whose calibration gave ``word`` (the kernel's ``numa_word`` argument:
    each rank's own, measured at run time, so no build depends on it)."""
    bits = fx.Int32((fx.Int64(word) >> (8 * page)) & 0xFF)
    return ((bits >> xcc) & 1) * (NUMA_RUN // 4)


def region_addrs(scratch, layout, dual: str = "", word=0) -> dict:
    """Each region of ``layout`` (name -> (byte offset, bytes)) in ``scratch``
    as its ``Addr``; one named in ``dual`` (``plan.layout.dual_spec``) kept in
    two copies at its offset there (``dreg``), the CTA reading the copy
    ``word`` gives its XCD -- a page's copy found once for all its regions."""
    places = dual_places(dual)
    xcc = xcc_id() if places else None
    copies = {}
    out = {}
    for region, (off, _) in layout.items():
        if region in places:
            at, page = places[region]
            if page not in copies:
                copies[page] = local_copy(word, page, xcc)
            out[region] = dreg(scratch, at, region, copies[page])
        else:
            out[region] = sreg(scratch, off, region)
    return out
