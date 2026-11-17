# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure-Python plan of the column-sliced AttnRes seam inside the resident
gfx1250 K1 (docs/kimi_k3_k1_sliced_seam.md): which cluster CTA owns which
columns of x, which slice's readiness the TDM projection polls before which
token slab, and where the launch's ready mailboxes live. No FlyDSL."""

from __future__ import annotations

from dataclasses import dataclass

from atom.models.kimi_k3.mono.kernels.stages import attn_res_sliced_layout as sliced
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import TDM_CLUSTER, TDM_SLAB

HIDDEN = sliced.HIDDEN
SLABS = HIDDEN // TDM_SLAB


@dataclass(frozen=True)
class K1SeamPlan:
    """The seam sliced over K1's own launch clusters: ``cluster`` consecutive
    CTAs a token, rank ``bid % cluster`` the columns
    ``[rank columns, (rank + 1) columns)``, home ``bid // cluster`` the token."""

    blocks: int
    cluster: int = TDM_CLUSTER
    slab: int = TDM_SLAB

    def __post_init__(self):
        if self.blocks % self.cluster:
            raise ValueError(
                f"{self.blocks} CTAs are no whole {self.cluster}-CTA clusters"
            )
        if HIDDEN % self.slab:
            raise ValueError(f"no {self.slab}-column slabs of the hidden row")
        if self.homes > self.geometry.max_clusters:
            raise ValueError("more K1 clusters than the sliced seam has homes")

    @property
    def geometry(self) -> sliced.SlicedGeometry:
        return sliced.SlicedGeometry(self.cluster)

    @property
    def homes(self) -> int:
        """K1's clusters, every one a seam home."""
        return self.blocks // self.cluster

    @property
    def slabs(self) -> int:
        return HIDDEN // self.slab

    def max_tokens(self) -> int:
        """Tokens one seam pass holds: a token a home a thread group."""
        return self.homes * self.geometry.tokens_a_pass

    def slab_ranks(self, slab: int) -> range:
        """The ranks whose columns slab ``slab`` reads."""
        columns = self.geometry.columns
        first = slab * self.slab
        return range(first // columns, (first + self.slab - 1) // columns + 1)

    def first_slab(self, rank: int) -> int:
        """The first slab reading rank ``rank``'s columns: its ready words are
        polled right before that slab's token copy."""
        return rank * self.geometry.columns // self.slab

    def polls(self) -> tuple[tuple[int, int], ...]:
        """``(slab, rank)``, slab order: before slab's token copy, poll rank's
        ready words of every home holding a token."""
        return tuple((self.first_slab(rank), rank) for rank in range(self.cluster))

    def ready_index(self, rank: int, home: int) -> int:
        """Mailbox pair (payload, generation) home ``home``'s rank ``rank``
        publishes once its slice of x is stored."""
        return rank * self.homes + home

    @property
    def ready_pairs(self) -> int:
        return self.cluster * self.homes

    def scratch_bytes(self, count: int, tokens: int, gram: bool = False) -> int:
        """Exchange records (cluster scope) then the ready mailboxes (device
        scope, two words a pair), 16-byte aligned."""
        records = sliced.scratch_bytes(count, tokens, self.cluster, gram)
        return -(-records // 16) * 16 + self.ready_pairs * 8

    def lds_words(self, candidates: int, gram: bool = False) -> int:
        """The seam's LDS: wave partials, a pass's scores, squares, cross sums
        and the mix's wave squares; it lives in K1's ``smem.red``, outside the
        role union the TDM copies fill during the projection prologue."""
        passing = self.geometry.tokens_a_pass
        crosses = len(sliced.pairs(candidates)) if gram else 0
        return (
            sliced.WAVES * sliced.sums(candidates, gram)
            + 2 * passing * candidates
            + passing * max(crosses, 1)
            + sliced.WAVES
        )
