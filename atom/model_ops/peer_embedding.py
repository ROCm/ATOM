# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A vocab-parallel embedding read where each row lies: one launch.

``VocabParallelEmbedding`` at TP > 1 looks up every token on every rank in
its own vocab shard (zeros outside it) and all-reduces the rows. Each rank's
shard mapped into every rank (``PeerView``, HIP IPC) lets a rank copy a
token's row straight from its owner's shard instead: the masked lookup and
the all-reduce become one gather. The rows are the original's: the sum adds
only zeros to the owner's row (an id no rank owns reads zeros, as there).
"""

import weakref

import torch
import triton
import triton.language as tl

from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.lifecycle import closed_with
from atom.mono.runtime.peer_memory import PeerView

BLOCK = 1024  # columns a program a round


@triton.jit
def _peer_embedding_kernel(
    shards,
    ids,
    out,
    rows_per_rank,
    ranks,
    width,
    fill,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Row ``program_id(0)``: the row of its token -- ``ids[t // width]`` for
    the first of every ``width`` rows, ``fill`` for the rest (``width`` 1:
    ``ids[t]``) -- from its owner rank's shard (``shards``: every rank's shard
    address, int64), zeros where no rank owns the id."""
    t = tl.program_id(0).to(tl.int64)
    token = tl.load(ids + t // width).to(tl.int64)
    token = tl.where(t % width == 0, token, fill)
    owner = token // rows_per_rank
    owned = (token >= 0) & (owner < ranks)
    owner = tl.where(owned, owner, 0)
    local = tl.where(owned, token - owner * rows_per_rank, 0)
    shard = tl.load(shards + owner).to(tl.pointer_type(tl.bfloat16))
    cols = tl.arange(0, BLOCK)
    for c0 in tl.static_range(0, DIM, BLOCK):
        live = c0 + cols < DIM
        row = tl.load(shard + local * DIM + c0 + cols, mask=live & owned, other=0.0)
        tl.store(out + t * DIM + c0 + cols, row, mask=live)


class PeerEmbedding:
    """``embed`` (a ``VocabParallelEmbedding`` of bf16 rows, ``tp_size`` > 1)
    read from every rank's shard, bound over its TP ``group`` (collective).
    Holds the remote mappings until :meth:`close`."""

    def __init__(self, embed, group) -> None:
        weight = embed.weight
        if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
            raise ValueError(
                f"peer embedding needs contiguous bf16, got {weight.dtype}"
            )
        self.rows_per_rank, self.dim = weight.shape
        self.ranks = embed.tp_size
        self.view = PeerView(
            weight.data_ptr, embed.tp_rank, self.ranks, group, "peer embedding"
        )

    def __call__(
        self, ids: torch.Tensor, out: torch.Tensor, width: int = 1, fill: int = 0
    ) -> torch.Tensor:
        """``ids`` [tokens] -> ``out`` [tokens, dim] (bf16), their rows; with
        ``width`` > 1, ``out`` [tokens x width, dim] blocks: each token's row,
        then ``width`` - 1 rows of token ``fill`` (a DSpark draft block: the
        anchor, then the noise token)."""
        rows = ids.numel() * width
        if out.shape != (rows, self.dim) or not out.is_contiguous():
            raise ValueError(f"peer embedding output {tuple(out.shape)}")
        if rows:
            _peer_embedding_kernel[(rows,)](
                self.view.addresses,
                ids,
                out,
                self.rows_per_rank,
                self.ranks,
                width,
                fill,
                DIM=self.dim,
                BLOCK=BLOCK,
            )
        return out

    def close(self) -> None:
        self.view.close()


# each embedding's one ``PeerEmbedding`` (``shared_peer_embedding``)
_SHARED: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def shared_peer_embedding(embed, group) -> PeerEmbedding:
    """``embed``'s one ``PeerEmbedding`` over ``group`` (collective where it
    is made): every reader of the same embedding (a DSpark draft shares its
    target's) shares one set of remote mappings, closed with ``embed``. The
    mappings are ``group``'s: a reader over another group is refused."""
    bound = _SHARED.get(embed)
    if bound is None:
        bound = _SHARED[embed] = (group, PeerEmbedding(embed, group))
        closed_with(embed, bound[1])
    if bound[0] is not group:
        raise MonoUnsupported("the embedding is mapped over another group")
    return bound[1]
