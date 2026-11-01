# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A DSA indexer's shape and its per-row decisions over the ranks that split a
row's keys, one definition for the mono kernel (traced Int32) and its CPU test
(Python ints): which local keys a rank holds, where a key's index row lives,
whether a row needs scoring at all.

A rank's keys are runs of ``interleave`` global positions, every ``W``-th run:
local key ``j`` of rank ``r`` is global ``((j // I) W + r) I + j % I``
(``key_pos``). Under DCP (interleave 1, ``atom.distributed.dcp_layout``) a
rank stores only its own keys, local key j at its local index j, and a block
table entry names a rank-local block of ``block_rows`` rows. Without DCP every
rank stores every key at its global position and the ranks split the scoring
by 128-key runs (``REPLICA_INTERLEAVE``: a score unit's keys stay one run).
"""

from dataclasses import dataclass

from atom.mono.plan.arith import sel
from atom.mono.plan.execution import BLOCKS, WAVES

TILE = 16  # the preshuffled layout's and the MFMA's 16 x 16 tile
SCALE_BYTES = 4  # a key's fp32 scale
# a score unit: a row's local keys 16 a wave over every wave, whatever the
# cache's block size (each wave reads its 16 from the block they sit in)
SCORE_KEYS = TILE * WAVES
# the runs the ranks split a replicated cache's keys by: one score unit
REPLICA_INTERLEAVE = SCORE_KEYS
# the index RoPE the kernel rotates: dims 0 .. 63, a q lane's pair or a k
# lane and its partner lane (``kernel.stage_k`` / ``iquant_task``)
ROPE_DIM = 64


@dataclass(frozen=True)
class IndexerShape:
    """What a DSA indexer is, as the kernel reads it: the layer's numbers and
    the storage of its operands. Every field is in the kernel's name."""

    heads: int  # index_n_heads
    head_dim: int  # index_head_dim
    topk: int  # index_topk
    q_lora: int  # q_lora_rank: wq_b's K
    wq_dtype: str  # wq_b's storage: "bf16", or what it is instead
    kv_dtype: str  # the index cache: "fp8" (+ an fp32 scale a key) or "fp4"
    block_rows: int  # index rows a cache block (kv_cache_block_size)
    # index_rope_dim: 0, a NoPE indexer (the NoPE writer's numerics, nope.py);
    # else the fused RoPE writer's (``indexer_qk_rope_quant_and_cache``)
    rope_dim: int
    rope_neox: bool = False  # the rotation's pairs: (d, d + half), else (2i, 2i + 1)

    @property
    def qrows(self) -> int:
        """wq_b's output rows."""
        return self.heads * self.head_dim

    @property
    def row_bytes(self) -> int:
        """A cache row: head_dim fp8 and an fp32 scale, padded to 16 B
        (``aiter_mla.aligned_index_cache_dim``)."""
        return -(-(self.head_dim + SCALE_BYTES) // 16) * 16

    @property
    def block_bytes(self) -> int:
        return self.block_rows * self.row_bytes

    @property
    def weights_scale(self) -> float:
        """The head weights' factor, the original's ``_weights_scale``."""
        return self.head_dim**-0.5 * self.heads**-0.5

    @property
    def tag(self) -> str:
        return (
            f"h{self.heads}_d{self.head_dim}_topk{self.topk}_ql{self.q_lora}"
            f"_wq{self.wq_dtype}_kv{self.kv_dtype}_blk{self.block_rows}"
            f"_rope{self.rope_dim}{'n' if self.rope_neox else ''}"
        )


# Functions of a shape, not its methods: a method called on a local inside a
# traced ``if`` makes FlyDSL carry that local through the branch as state, and
# a shape is no IR value.
def short(shape: IndexerShape, ctx):
    """A row whose every key is selected (``ctx <= topk``): nothing to score,
    every rank emits all of its keys in local order."""
    return ctx <= shape.topk


def cache_byte(shape: IndexerShape, row, col):
    """The byte of key ``row`` (in its block), dim ``col``: the preshuffled
    16 x 16 MFMA layout the gluon scorer reads."""
    return (
        (row // TILE) * TILE * shape.head_dim
        + (col // TILE) * TILE * TILE
        + (row % TILE) * TILE
        + col % TILE
    )


def scale_byte(shape: IndexerShape, row):
    """A block's fp32 scales follow its fp8 rows."""
    return shape.block_rows * shape.head_dim + SCALE_BYTES * row


def unsupported(shape: IndexerShape) -> str:
    """Why the kernel cannot run an indexer of this shape, or ''. Each is a
    piece of the kernel written for one value, not a measured limit."""
    if shape.wq_dtype != "bf16":
        return f"wq_b {shape.wq_dtype}: the q GEMV reads bf16"
    if shape.kv_dtype != "fp8":
        return f"{shape.kv_dtype} index cache: the scorer reads FP8 keys"
    if shape.rope_dim not in (0, ROPE_DIM):
        return f"index RoPE {shape.rope_dim}: the rotation is written for {ROPE_DIM}"
    if shape.head_dim != 128:
        return f"head_dim {shape.head_dim}: a key is 2 dims a lane of one wave"
    if shape.heads != 32:
        return f"{shape.heads} heads: the head sum is gluon's 32-head order"
    if shape.qrows != TILE * BLOCKS:
        return f"wq_b {shape.qrows} rows: the q GEMV is one 16-row task a CTA"
    if shape.block_rows % TILE:
        return f"{shape.block_rows}-row blocks: a score wave reads 16 keys of a block"
    if shape.topk % 4:
        return f"top-{shape.topk}: a rank's winners are copied 4 slots a 16 B load"
    return ""


def local_len(ctx, rank, world, interleave=1):
    """How many of a row's ``ctx`` global keys rank ``rank`` holds."""
    span = world * interleave
    rest = ctx % span - rank * interleave
    return ctx // span * interleave + sel(
        rest > interleave, interleave, sel(rest > 0, rest, 0)
    )


def key_pos(j, rank, world, interleave):
    """Local key ``j`` of rank ``rank``'s global position."""
    return ((j // interleave) * world + rank) * interleave + j % interleave
