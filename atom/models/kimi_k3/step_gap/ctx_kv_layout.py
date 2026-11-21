# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure-Python geometry of the draft's context KV projection
(``ctx_kv_kernels``): which CTA projects which rows of which layer's
``fused_qkv_a_proj``, and where its sums go. No FlyDSL.

A draft layer writes its context rows from ``fused_qkv_a_proj``'s kv_a and rope
rows only (576 of the 2112; production computes all and drops the q_a rows).
The GEMV is the draft front's (``mla_draft_front_layout``): a cluster of four
CTAs owns a 64-row group of the preshuffled weight, at split-K 4 CTA ``r``
chaining every 16-row task over K quarter ``r``, at split-K 1 task ``r`` over
all of K. A layer's nine groups -- the front's clusters 24..31 (kv_a) and 32
(rope) -- are nine clusters here, the layers one after another.
"""

from __future__ import annotations

from atom.models.kimi_k3.mono.config import MAX_ROWS
from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_layout as front
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import ROWS, TDM_CLUSTER

HIDDEN = 7168
KV_LORA = front.KV_LORA
ROPE = front.ROPE
ENTRY = KV_LORA + ROPE  # a context row as the cache write reads it: kv_a | rope
FIRST_CLUSTER = front.Q_CLUSTERS  # the front's first kv_a cluster
CLUSTERS = front.A_CLUSTERS - front.Q_CLUSTERS  # kv_a's eight and the rope's one
LAYER_CTAS = CLUSTERS * TDM_CLUSTER
MAX_LAYERS = 8
KSPLITS = front.KSPLITS
assert (
    CLUSTERS * front.GROUP_ROWS == ENTRY
    and FIRST_CLUSTER * front.GROUP_ROWS == front.Q_LORA
)


def grid(layers: int) -> int:
    return layers * LAYER_CTAS


def cta_job(cta: int, layers: int) -> tuple[int, int, int]:
    """``(layer, the front's cluster, rank in the cluster)`` of CTA ``cta``."""
    if not 0 <= cta < grid(layers):
        raise ValueError(f"CTA {cta} outside the {layers}-layer grid")
    cluster, rank = divmod(cta, TDM_CLUSTER)
    return cluster // CLUSTERS, FIRST_CLUSTER + cluster % CLUSTERS, rank


def column(cluster: int, task: int, half: int) -> int:
    """First of the eight context-row columns a lane half ``half`` of 16-row
    task ``task`` of the front's cluster ``cluster`` stores."""
    return (cluster - FIRST_CLUSTER) * front.GROUP_ROWS + task * ROWS + half * 8


def out_word(layer: int, rows: int, token: int, col: int) -> int:
    """i32 word of ``out`` ([layers, rows, ENTRY] BF16) holding columns
    ``col``, ``col + 1`` of ``token``'s row of ``layer``."""
    return ((layer * rows + token) * ENTRY + col) // 2


def out_bytes(layers: int, rows: int) -> int:
    return layers * rows * ENTRY * 2


def partial_bytes(layers: int) -> int:
    """The split-K exchange: every CTA's three foreign tasks' f32 sums at up to
    MAX_ROWS rows (a v8 a lane a tile), as the front's."""
    tiles = -(-MAX_ROWS // ROWS)
    return grid(layers) * front.TASKS * tiles * front.LANES * 8 * 4


def refusal(rows: int, ksplit: int, layers: int) -> str:
    """Why there is no build for this shape, or ''."""
    if not 1 <= rows <= MAX_ROWS:
        return f"{rows} context rows (1..{MAX_ROWS})"
    if ksplit not in KSPLITS:
        return f"split-K {ksplit}: the projection mirrors {KSPLITS}"
    if not 1 <= layers <= MAX_LAYERS:
        return f"{layers} draft layers (1..{MAX_LAYERS})"
    return ""
