# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure-Python geometry of the gfx1250 MLA absorb (``mla_absorb_gfx1250``):
which CTA projects which ``q_b_proj`` rows, which ``W_UK`` rows it absorbs,
which CTA writes a token's KV-cache entry, and the launch's buffers. No
FlyDSL, so every coordinate is checked on a CPU-only host.

The absorb takes the MLA decode branch of ``MLAAttention.forward_impl`` from
q_b_proj's FP8 input to the attention's ``q_out`` and the paged KV cache:
``q = q_b_proj(q_fp8)`` (``[rows, heads, 192]``: 128 nope then 64 rope), the
nope part absorbed through ``W_UK`` (``[heads, 512, 128]`` FP8, one
per-tensor scale) into ``ql_nope [rows, heads, 512]``, and
``fused_qk_rope_concat_and_cache_mla``'s ``q_out [rows, out_heads, 576]`` and
KV-cache entry ``[kv 512 | k_rope 64]``. A cluster of four CTAs owns two
heads: CTA ``r`` projects rows ``32 r .. 32 r + 32`` of each head's nope part
and rows ``16 r .. 16 r + 16`` of its rope part (three 16-row tasks a head),
the cluster swaps the projected rows through ``q`` at SE scope, and CTA
``r`` then absorbs ``W_UK`` rows ``128 r .. 128 r + 128`` of each head and
writes q_out's rope part of every fourth token. The CTAs past the heads'
write one token's KV-cache entry each, and zero q_out's padding heads.
"""

from __future__ import annotations

from dataclasses import dataclass

from atom.models.kimi_k3.mono.config import MAX_ROWS
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import ROWS, TDM_CLUSTER
from atom.mono.plan.architecture import GFX1250

Q_LORA = 1536
NOPE = 128
ROPE = 64
HEAD = NOPE + ROPE
KV_LORA = 512
ENTRY = KV_LORA + ROPE
THREADS = 256
WAVES = THREADS // GFX1250.wave_size
HEADS_PER_CLUSTER = 2
# A CTA's q_b tasks of a head: two nope tasks, one rope task
NOPE_TASKS = NOPE // (TDM_CLUSTER * ROWS)
ROPE_TASKS = ROPE // (TDM_CLUSTER * ROWS)
HEAD_TASKS = NOPE_TASKS + ROPE_TASKS
assert NOPE_TASKS == 2 and ROPE_TASKS == 1
# q_b's K split over a head's four waves (12 K32 steps, three 128-wide
# scale groups each)
K_QUARTERS = WAVES // HEADS_PER_CLUSTER
K_QUARTER = Q_LORA // K_QUARTERS
QUANT_GROUP = 128
GROUPS = Q_LORA // QUANT_GROUP
QUARTER_GROUPS = K_QUARTER // QUANT_GROUP
# W_UK rows a CTA absorbs of a head, and a wave's (two tasks)
UK_ROWS = KV_LORA // TDM_CLUSTER
UK_WAVE_ROWS = UK_ROWS // K_QUARTERS
UK_TASKS = UK_WAVE_ROWS // ROWS
# E4M3's largest finite value: the nope part saturates to it
FP8_MAX = 448.0
VEC = 8


def clusters(heads: int) -> int:
    if heads <= 0 or heads % HEADS_PER_CLUSTER:
        raise ValueError(f"{heads} heads: the absorb takes heads in pairs")
    return heads // HEADS_PER_CLUSTER


def head_ctas(heads: int) -> int:
    return clusters(heads) * TDM_CLUSTER


def grid(heads: int) -> int:
    """The heads' CTAs, then one a token row for the KV cache (MAX_ROWS)."""
    return head_ctas(heads) + MAX_ROWS


def cta_heads(cta: int, heads: int) -> tuple[int, int, int]:
    """``(first head, second head, rank)`` of a head CTA."""
    if not 0 <= cta < head_ctas(heads):
        raise ValueError(f"CTA {cta} projects no head")
    cluster, rank = divmod(cta, TDM_CLUSTER)
    return HEADS_PER_CLUSTER * cluster, HEADS_PER_CLUSTER * cluster + 1, rank


def q_b_tasks(head: int, rank: int) -> list[int]:
    """The first q_b_proj weight row of each of CTA ``rank``'s three tasks of
    ``head``: two nope, one rope."""
    base = head * HEAD
    nope = [base + (NOPE_TASKS * rank + i) * ROWS for i in range(NOPE_TASKS)]
    return nope + [base + NOPE + rank * ROWS]


def uk_rows(head: int, rank: int, wave: int) -> range:
    """The ``W_UK[head]`` rows wave ``wave`` (of the four of that head) of CTA
    ``rank`` absorbs."""
    first = rank * UK_ROWS + wave * UK_WAVE_ROWS
    return range(first, first + UK_WAVE_ROWS)


def wave_job(wave: int) -> tuple[int, int]:
    """``(head of the pair, K quarter)`` a wave projects: the q_b task triple
    of a head over one K quarter; past the exchange the same wave absorbs
    two W_UK tasks of that head (``uk_rows``)."""
    if not 0 <= wave < WAVES:
        raise ValueError(f"wave {wave}")
    return divmod(wave, K_QUARTERS)


def rope_tokens(rank: int, rows: int) -> range:
    """The tokens whose q_out rope part CTA ``rank`` writes, both heads."""
    return range(rank, rows, TDM_CLUSTER)


@dataclass(frozen=True)
class AbsorbPlan:
    """The absorb of one layer shape: ``heads`` projected and absorbed (the
    DCP group's under QREP, else this rank's), q_out's ``out_heads``
    (``>= heads``; the rest zero) and whether every token's q_out is
    written (``compute_all``: DCP) or a token without a KV slot's is zero."""

    rows: int
    heads: int
    out_heads: int
    compute_all: bool

    def __post_init__(self):
        if not 1 <= self.rows <= MAX_ROWS:
            raise ValueError(
                f"the MLA absorb takes 1..{MAX_ROWS} rows, got {self.rows}"
            )
        clusters(self.heads)
        if self.out_heads < self.heads:
            raise ValueError(f"{self.out_heads} q_out heads < {self.heads}")

    @property
    def tiles(self) -> int:
        return -(-self.rows // ROWS)

    @property
    def q_rows(self) -> int:
        """q_b_proj's output rows."""
        return self.heads * HEAD

    @property
    def q_out_row(self) -> int:
        """q_out's bytes a token."""
        return self.out_heads * ENTRY


def lds_words(rows: int) -> int:
    """A CTA's LDS: its six q_b tasks' FP8 weights, then each wave's K-quarter
    sums (a v8 a lane a task a tile)."""
    tiles = -(-rows // ROWS)
    weights = HEADS_PER_CLUSTER * HEAD_TASKS * ROWS * Q_LORA // 4
    sums = WAVES * HEAD_TASKS * tiles * GFX1250.wave_size * 8
    return weights + sums


def q_words(rows: int, heads: int) -> int:
    """The q exchange buffer (``[rows, heads, 192]`` BF16) in i32 words: q_b's
    rounded output, as production's ``q``."""
    return rows * heads * HEAD // 2
