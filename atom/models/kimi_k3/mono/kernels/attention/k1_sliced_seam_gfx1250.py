# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Skeleton: the column-sliced AttnRes seam inside the resident gfx1250 K1
(docs/kimi_k3_k1_sliced_seam.md). Not wired into ``build_resident_k1`` yet.

Every K1 cluster (``TDM_CLUSTER`` CTAs, the launch's own) is a seam home: in
the projection's ``prologue``, with the first weight slabs in flight, its CTAs
mix their token's column slices with cluster-scope exchanges only
(``attn_res_sliced_gfx1250``'s pass), store x and publish one device-scope
ready pair a CTA. The TDM projection's token-copy wave then polls, right
before the first slab that reads a rank's columns, that rank's pairs of every
home holding a token, and copies x at device scope: no acquire fence."""

import flydsl.expr as fx
from flydsl.expr import range_constexpr

from atom.models.kimi_k3.mono.kernels.attention import kda_gfx1250 as gfx_backend
from atom.models.kimi_k3.mono.kernels.attention.k1_sliced_seam_layout import K1SeamPlan
from atom.mono.device.ops import traced
from atom.mono.plan.architecture import GFX1250


@traced
def publish_slice(plan: K1SeamPlan, x_ready, rank, home, generation, tid):
    """This CTA's slice of its home's x is stored: drain, rendezvous, publish."""
    gfx_backend.mailbox_publish_device(
        x_ready, plan.ready_index(rank, home), fx.Int32(0), generation, tid == 0
    )


@traced
def gate_slab(plan: K1SeamPlan, x_ready, slab, tokens: int, generation, lane):
    """On the wave about to copy token slab ``slab`` (a trace-time int, or a
    runtime index in the rolled slab loop): if it is the first slab reading
    some rank's columns, wait until every home holding a token has published
    that rank's slice. A lane a home, every poll of a lane in flight at once."""
    homes = plan.geometry.clusters(tokens)
    rounds = -(-homes // GFX1250.wave_size)
    for first, rank in plan.polls():
        if isinstance(slab, int) and slab != first:
            continue
        if fx.Int32(slab) == fx.Int32(first):
            gfx_backend.mailbox_poll_many_device(
                [
                    (
                        x_ready,
                        plan.ready_index(
                            rank,
                            fx.min(lane + step * GFX1250.wave_size, homes - 1),
                        ),
                    )
                    for step in range_constexpr(rounds)
                ],
                generation,
            )


def seam_pass(plan: K1SeamPlan, context):
    """One token pass of ``attn_res_sliced_gfx1250.build_attn_res_sliced``'s
    kernel body on K1's operands: ``context`` carries rank, home, the
    ``smem.red`` LDS, the exchange records' scratch, ``ares_nw * ares_qk`` as
    the score weight, ``in_nw`` as the output gain, prefix (+ delta, written
    back in place), ``write_idx``'s block row, and x's scratch row.

    To be extracted from that kernel (its loads, partials, cluster exchange,
    gate, mix and output norm) so both emit the same code."""
    raise NotImplementedError("extract the pass from attn_res_sliced_gfx1250")
