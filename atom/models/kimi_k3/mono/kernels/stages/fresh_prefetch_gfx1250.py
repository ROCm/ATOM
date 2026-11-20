# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``ATOM_KIMI_MONO_XPF``: a kernel's freshly written input prefetched into
every XCD's L2.

The kernel before wrote it from all XCDs, so it sits in none of the reading
XCDs' L2s and its first reads miss to the MALL. A grid prefetches it once per
XCD: clusters (or single CTAs) go round-robin over the XCDs, so each XCD's
CTAs split the input's 128 B lines between them, one line a thread."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import as_ir_value

from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.mono.device.ops import traced
from atom.utils import envs

_SITES = envs.ATOM_KIMI_MONO_XPF


def on(site: str) -> bool:
    return site in _SITES


_DEFAULT = frozenset({"gather"})


def suffix(site: str) -> str:
    """Kernel-name suffix of ``site``'s prefetch, empty when it is as by
    default."""
    if on(site) == (site in _DEFAULT):
        return ""
    return f"_xpf{site}" if on(site) else f"_noxpf{site}"


def prefetch(
    site: str,
    base,
    nbytes: int,
    bid,
    tid,
    ctas: int,
    threads: int,
    group: int = 1,
) -> None:
    """With ``site`` on: the ``ctas``-CTA grid (dispatched in ``group``-CTA
    clusters) prefetches the 128 B lines of ``nbytes`` from ``base``. Each
    XCD's ``ctas // (XCDS group)`` full rounds of clusters cover every share,
    a partial round repeats some; a grid smaller than one round has each
    cluster prefetch all of it."""
    if const_expr(not on(site)):
        return
    xcds = gemv_tdm.XCDS
    shares = max(ctas // (xcds * group), 1) * group
    lines = (nbytes + 127) // 128
    share = (bid // group // xcds * group + bid % group) % shares
    per_cta = (lines + shares - 1) // shares
    for step in range_constexpr((per_cta + threads - 1) // threads):
        gemv_tdm._prefetch_line(base, share + (tid + step * threads) * shares, lines)


def prefetch_own(site: str, address, lines: int, tid, threads: int) -> None:
    """With ``site`` on: the CTA prefetches ``lines`` 128 B lines, line ``i``
    at ``address(i)`` (an Int64 byte address), one line a thread."""
    if const_expr(not on(site)):
        return
    for step in range_constexpr((lines + threads - 1) // threads):
        line = tid + step * threads
        _prefetch_at(address(fx.min(line, lines - 1)), line < lines)


@traced
def _prefetch_at(address, live):
    if live:
        rocdl_dialect.global_prefetch(
            llvm.IntToPtrOp(
                ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(address))
            ).result
        )
