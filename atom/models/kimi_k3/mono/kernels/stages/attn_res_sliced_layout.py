# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure-Python plan of the column-sliced gfx1250 AttnRes mix
(``attn_res_sliced_gfx1250``): how a cluster's CTAs slice the hidden row, how
a CTA's threads hold its tokens, and what each token's exchange record holds.
No FlyDSL, so every coordinate can be checked on a CPU-only host."""

from __future__ import annotations

from dataclasses import dataclass

from atom.models.kimi_k3.mono.kernels.attention.layout import HIDDEN
from atom.mono.plan.architecture import GFX1250

THREADS = 256
WAVES = THREADS // GFX1250.wave_size
ELEMENTS = 8
# a candidate's two sums a lane of a wave: up to 16 candidates
MAX_CANDIDATES = GFX1250.wave_size // 2
# gfx1250's CUs: at most one cluster's CTA a CU
MAX_CTAS = 256
CLUSTERS = (4, 8, 16)
MAX_ADDENDS = 2


def sums(candidates: int, gram: bool) -> int:
    """A slice's sums of a token: each candidate's square and score dot, and
    with ``gram`` each pair's cross product."""
    return 2 * candidates + (candidates * (candidates - 1) // 2 if gram else 0)


def pairs(candidates: int) -> list[tuple[int, int]]:
    """The Gram cross products' order in a record, after the squares and dots."""
    return [(c, d) for c in range(candidates) for d in range(c + 1, candidates)]


@dataclass(frozen=True)
class SlicedGeometry:
    """A ``cluster``-CTA slicing of the hidden row."""

    cluster: int

    def __post_init__(self):
        if self.cluster not in CLUSTERS or HIDDEN % (self.cluster * ELEMENTS):
            raise ValueError(f"no {self.cluster}-CTA slicing of the hidden row")

    @property
    def columns(self) -> int:
        return HIDDEN // self.cluster

    @property
    def lanes(self) -> int:
        """Threads a token's slice keeps busy, ``ELEMENTS`` columns each."""
        return self.columns // ELEMENTS

    @property
    def token_threads(self) -> int:
        """A token's thread group: its busy threads, whole waves."""
        return -(-self.lanes // GFX1250.wave_size) * GFX1250.wave_size

    @property
    def token_waves(self) -> int:
        return self.token_threads // GFX1250.wave_size

    @property
    def tokens_a_pass(self) -> int:
        return THREADS // self.token_threads

    @property
    def max_clusters(self) -> int:
        return MAX_CTAS // self.cluster

    def record_words(self, candidates: int, gram: bool = False) -> int:
        """A token's scratch, slices innermost: each slice's candidate sums
        (with ``gram``, then the candidates' cross products), and without it
        each slice's sum of squares of the mix."""
        return (sums(candidates, gram) + (not gram)) * self.cluster

    def clusters(self, tokens: int) -> int:
        return min(tokens, self.max_clusters)

    def ctas(self, tokens: int) -> int:
        return self.clusters(tokens) * self.cluster

    def passes(self, tokens: int, home: int) -> int:
        """Cluster ``home``'s passes: its tokens ``home + i clusters``, a
        thread group each, ``tokens_a_pass`` at once."""
        clusters = self.clusters(tokens)
        mine = (tokens - home + clusters - 1) // clusters
        return (mine + self.tokens_a_pass - 1) // self.tokens_a_pass

    def token(self, tokens: int, home: int, step: int, local: int) -> int:
        """The token thread group ``local`` of cluster ``home`` holds in pass
        ``step``; ``tokens`` or past: none."""
        return home + (step * self.tokens_a_pass + local) * self.clusters(tokens)

    def column(self, rank: int, slot: int) -> int:
        """The first of the ``ELEMENTS`` columns thread ``slot`` of a token
        group holds on cluster rank ``rank``; idle slots repeat the last."""
        return rank * self.columns + min(slot, self.lanes - 1) * ELEMENTS


def check(count: int, addends: int, out_norm: bool = True, gram: bool = False) -> None:
    if not 1 <= count < MAX_CANDIDATES:
        raise ValueError(f"the AttnRes mix holds 1 to {MAX_CANDIDATES - 1} blocks")
    if addends not in range(MAX_ADDENDS + 1):
        raise ValueError(f"the AttnRes mix folds up to {MAX_ADDENDS} addends")
    if gram and not out_norm:
        raise ValueError("the Gram form is the output norm's; no out_norm, no Gram")


def scratch_bytes(count: int, tokens: int, cluster: int, gram: bool = False) -> int:
    return SlicedGeometry(cluster).record_words(count + 1, gram) * tokens * 4
