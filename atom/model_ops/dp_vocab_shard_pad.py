# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Row padding that keeps a DP-vocab-sharded LM head bit-identical.

A replicated head runs ``[rows, dim] @ [dim, V]`` on every DP rank. Sharding
the vocab over the DP group runs ``[dp * rows, dim] @ [dim, V / dp]`` instead:
the same products, but a different GEMM shape, for which the GEMM backend may
pick a kernel that sums ``K`` in another order -- the logits then differ by an
ulp, and a near-tie argmax can flip. Zero rows appended to ``M`` (and dropped
from the output) steer the backend to a kernel whose ``K`` order is the
replicated one's; which pad does that depends on the backend's version and
tuning, so it is never tabulated here.

``shard_gemm_pad`` measures it instead, once per shape, the first time the
shape runs eagerly: every candidate pad is run on random rows and compared
bitwise with the replicated GEMM, then the DP group takes the smallest pad
that is exact on every rank. A ``K`` order is a property of the kernel, not
of the data, so a pad exact on random rows is exact on any rows. A shape no
candidate makes exact gets ``None``, and the caller keeps the replicated head
for it: the sharded head never runs on a shape it was not proven exact on.
"""

from __future__ import annotations

import logging

import torch

from atom.distributed.group_agreement import group_min_each

logger = logging.getLogger("atom")

# Multiples of 8 up to 128, then a few coarser steps. Measured on MI355X
# (dp=8, rows = B * 5, dim 5120, V 129280), every B in 1..51 is exact at one
# of these: 33 of them unpadded, the largest pad 192.
PAD_CANDIDATES = tuple(range(0, 129, 8)) + (192, 256, 384, 512)

# (world, rows, V, dim, dtype, device) -> pad, or None for "keep replicated".
_PADS: dict[tuple, int | None] = {}


def shard_gemm_pad(head, rows: int, rank: int, world: int, cpu_group) -> int | None:
    """The pad that makes ``head``'s vocab-sharded GEMM over ``world * rows``
    gathered rows bit-identical to its replicated ``[rows, V]`` GEMM, or None.

    Every rank of ``cpu_group`` must call this at the same point with the
    same ``rows``: a first call per shape runs one collective. Under a stream
    capture an unmeasured shape answers None without caching, so the capture
    records the replicated head and the next eager pass measures it.
    """
    weight = head.weight
    key = (world, rows, *weight.shape, weight.dtype, weight.device)
    if key in _PADS:
        return _PADS[key]
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None
    exact = _exact_pads(head, rows, rank, world)
    agreed = group_min_each([int(ok) for ok in exact], cpu_group)
    pad = next((p for p, ok in zip(PAD_CANDIDATES, agreed) if ok), None)
    _PADS[key] = pad
    if pad is None:
        logger.warning(
            "[DP-DRAFT-ARGMAX] no row pad makes the DP-sharded draft head "
            "bit-identical at rows=%d (dp=%d); that shape keeps the "
            "replicated head",
            rows,
            world,
        )
    return pad


@torch.inference_mode()
def _exact_pads(head, rows: int, rank: int, world: int) -> list[bool]:
    """For each candidate pad, whether this rank's vocab slice of the sharded
    GEMM equals the replicated GEMM bitwise on random rows."""
    weight = head.weight
    vocab, dim = weight.shape
    vshard = vocab // world
    gen = torch.Generator(device=weight.device)
    gen.manual_seed(0)
    x = torch.randn(
        world * rows, dim, generator=gen, device=weight.device, dtype=torch.float32
    ).to(weight.dtype)
    cols = slice(rank * vshard, (rank + 1) * vshard)
    # What each DP rank's replicated head computes for its own rows.
    ref = torch.cat(
        [head.local_logits(x[s * rows : (s + 1) * rows])[:, cols] for s in range(world)]
    )
    exact = []
    for pad in PAD_CANDIDATES:
        xp = torch.cat([x, x.new_zeros(pad, dim)]) if pad else x
        got = head._vocab_shard_logits(xp, rank, world)[: world * rows]
        exact.append(torch.equal(got, ref))
    return exact
