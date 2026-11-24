# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The V4.1 mono scratch regions kept in two copies, one on each I/O die
(``atom.mono.device.numa``): every hand-off a step's critical path polls on
one GPU -- the slice's layer input, its sums of squares and mix partials, the q latent and q, the
attention's scores, output and wo_a output, the router's logits and routes,
the MoE's MID (ug to down) and its picks' and shared expert's down outputs,
an index layer's normed row (NB) and query-done flags, and the flags the
layer kernel hands over by (X8F among them). Each is only put and polled
(``Mailbox``), never read or written as plain words, so its copies are the
mailbox's alone (MID is plain words only off ``RouteShape.down_y``: DP
attention or expert parallel, which ``V41MonoDecodeRunner._numa`` builds
single).

The kernels sharing a scratch (K1 then K2, or the layer kernel) take one table
(``numa_table``) from all their layouts, so a region two of them name (the
seam's LIN / PMIX) has one place. A build without it (``numa`` empty) keeps
every region single."""

from atom.mono.plan.layout import NUMA_PAGE, NUMA_RUN, dual_layout, dual_spec

DUAL_REGIONS = (
    "lin", "lssq", "pmix", "qkv", "qx8", "qx8s", "qp", "sdone", "gdone", "krdy",
    "as", "xo", "xos", "x8b", "x8bs",
    "logit", "route", "x8rdy", "xrdy", "mid", "mids", "dy", "sdo",
    "x8f", "nb", "iqdone",
)  # fmt: skip


def numa_table(layouts, scratch_bytes: int) -> tuple[str, int]:
    """The ``DUAL_REGIONS`` that any of ``layouts`` (name -> (byte offset,
    bytes)) holds, in a dual area from the first page boundary at or past
    ``scratch_bytes``. -> (the table as a build key's ``numa``
    (``dual_spec``); the scratch's bytes with the area)."""
    # a region the step's builds size apart (an indexing layer's IQDONE a
    # group a flag, a plain one's one) gets the widest
    sizes = {}
    for layout in layouts:
        for name, (_, nbytes) in layout.items():
            if name in DUAL_REGIONS:
                sizes[name] = max(sizes.get(name, 0), nbytes)
    start = -(-scratch_bytes // NUMA_PAGE) * NUMA_PAGE
    # a region whose two copies outgrow a page stays single: a page's run
    # parity is its own, a CTA's copy one per region
    fits = [
        (name, sizes[name])
        for name in DUAL_REGIONS
        if name in sizes and 2 * -(-sizes[name] // NUMA_RUN) * NUMA_RUN <= NUMA_PAGE
    ]
    table, area = dual_layout(fits, start)
    return dual_spec(table), start + area
