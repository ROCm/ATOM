# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The routed MoE's per-step work split, defined once: which CTA / wave runs which UG unit, and where the shared
expert's units go. Called by the kernel on traced Int32 / trace-time Python ints and by the CPU test on Python ints
(the same code). Only non-negative operands reach ``//``."""

WIDE_NCH = 3  # >= 3 shared token chunks (> 32 rows): the wide schedule
SH_EARLY = True  # wide: the shared expert's UG runs in the FFN front's idle CTAs (moe.ug_shared_early)
EARLY_NCH_MAX = 3  # the front's idle CTAs (160 past its 96) hold 3 chunks' units x 2 CTAs; wider steps (front 160-192
# CTAs, 4-6 chunks) run the shared UG on dedicated CTAs


def sh_early(nch):
    """The shared expert's UG in the front's idle CTAs (wide batches up to 48 rows)."""
    return SH_EARLY and WIDE_NCH <= nch <= EARLY_NCH_MAX


def sh8_upc(nch):
    """Shared units a dedicated CTA: 3 wide (its bytes ~ a routed CTA's: the UG is bandwidth-shared, the CUs with more
    bytes finish last), else 1."""
    return 3 if nch >= WIDE_NCH else 1


def sh8_dnov(nch):
    """Narrow batches: the shared down on the dedicated CTAs (wide: after the handoff on all CTAs)."""
    return nch < WIDE_NCH


def dedicated_ctas(nch, nug):
    """CTAs that run only the shared expert's UG units (none when they run early in the front)."""
    nshu = nch * nug
    return 0 if sh_early(nch) else (nshu + sh8_upc(nch) - 1) // sh8_upc(nch)


def striped(nch):
    """Wide batches stripe the routed units over the CTAs (unit (it * 8 + w) * CR + b): a hot expert's 16-token units
    (twice the B cache lines a K step) land one a CU instead of all on 2-3 CUs. Narrow batches keep contiguous blocks:
    a CTA's units there mostly share an expert and its B rows."""
    return nch >= WIDE_NCH


def block_lo(b, U, CR):
    """First routed unit of CTA b's contiguous block (narrow batches); the block of CTA b is [lo(b), lo(b + 1))."""
    return (b * U) // CR


def unit_iters(stripe, U, CR, b, w, u1, wu):
    """Routed units of wave w of CTA b (may be negative: none)."""
    return (U - (w * CR + b) + 8 * CR - 1) // (8 * CR) if stripe else (u1 - wu + 7) // 8


def unit_of(stripe, it, CR, b, w, wu):
    """Routed unit `it` of wave w of CTA b (wu = its block's first unit + w, narrow batches)."""
    return (it * 8 + w) * CR + b if stripe else wu + it * 8
