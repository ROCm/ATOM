# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3's token embedding from a whole table on every TP rank
(``ATOM_KIMI_MONO_EMBED_LOCAL``): no all-reduce, no cross-rank wait.

Production shards the table over the vocabulary: a rank looks an id up in its
shard and writes +0.0 for an id it does not hold (``masked_embedding``), and a
TP all-reduce adds the ranks' rows. So every output row is one real row plus
+0.0 from the other ranks, summed in fp32 in rank order by aiter's all-reduce
kernels and rounded to BF16 once. Those kernels run with f32 denormals flushed
(FLOAT_DENORM_MODE_32 0 in their kernel descriptors, as aiter's other HIP
kernels), so the sum leaves a value as it is except: -0.0 becomes +0.0, a BF16
denormal becomes +0.0, a signalling NaN becomes quiet. ``canonicalize_`` does
that to the table once, in bits; a lookup in the canonical table is then
production's row. (``tools/kimi_k3_step_gap_embed_check.py``: all 65536 BF16
patterns through a HIP kernel's add in that float mode, and a whole vocabulary
against the shards' masked lookups summed in rank order.)

The table is assembled once by ``install``, at the decode installer (weights
loaded, before the first warmup, outside any capture): the ranks all-gather
their shards over the TP group's device group, a few thousand rows a
collective, each checks its own shard came back, and one host collective
agrees every rank's verdict and the tables' checksum. Any refusal, on any rank,
leaves every rank on production's path. 2.35 GB a rank for Kimi-K3
([163840, 7168] BF16); the shard stays loaded.

The lookup is production's own masked-embedding kernel over the full id range:
an id outside the vocabulary (the -1 placeholders of deferred output) reads
+0.0, as production's sum of eight masked rows does. One launch, where
production runs two.
"""

from __future__ import annotations

import logging

import torch

from atom.model_ops.embed_head import VocabParallelEmbedding, masked_embedding
from atom.models.kimi_k3.step_gap.embed_table import (
    canonicalize_,
    checksum,
    group_min,
)
from atom.utils.decorators import mark_trace

logger = logging.getLogger("atom")

# rows of a shard a collective moves (29 MB a rank at 7168 wide)
GATHER_ROWS = 2048


class K3LocalEmbedding(VocabParallelEmbedding):
    """``VocabParallelEmbedding`` that looks ids up in ``local_table`` (the
    whole canonical table) once ``install`` bound one; until then, and on any
    refusal, production's masked lookup and all-reduce. ``install`` gives the
    loaded model's embedding this class in place."""

    def __init__(self, num_embeddings: int, embedding_dim: int, prefix: str = ""):
        super().__init__(num_embeddings, embedding_dim, prefix)
        # a plain attribute: not a parameter, not in the state dict
        self.local_table = None

    @mark_trace
    def forward(self, x: torch.Tensor):
        table = self.local_table
        if table is None:
            return super().forward(x)
        return masked_embedding(x, table, 0, self.num_embeddings)


def local_refusal(embedding) -> str | None:
    """Why this rank cannot hold the whole table, or None (voted on)."""
    weight = embedding.weight
    rows = embedding.num_embeddings_per_partition
    if weight.dtype != torch.bfloat16:
        return f"a {weight.dtype} table"
    if weight.dim() != 2 or weight.shape[0] != rows or not weight.is_contiguous():
        return f"a shard of {tuple(weight.shape)}"
    if rows * embedding.tp_size != embedding.num_embeddings:
        return "a vocabulary the ranks do not share evenly"
    if weight.device.type != "cuda":
        return f"a shard on {weight.device}"
    need = embedding.num_embeddings * weight.shape[1] * weight.element_size()
    free = torch.cuda.mem_get_info(weight.device)[0]
    if free < need + need // 4:
        return f"{free >> 20} MiB free for a {need >> 20} MiB table"
    return None


def gather_table(embedding, device_group) -> torch.Tensor:
    """Every rank's shard, in rank order: the whole table. Collective."""
    import torch.distributed as dist

    weight = embedding.weight.data
    tp, (rows, dim) = embedding.tp_size, weight.shape
    table = torch.empty((tp, rows, dim), dtype=weight.dtype, device=weight.device)
    for start in range(0, rows, GATHER_ROWS):
        stop = min(rows, start + GATHER_ROWS)
        piece = torch.empty(
            (tp, stop - start, dim), dtype=weight.dtype, device=weight.device
        )
        dist.all_gather_into_tensor(piece, weight[start:stop], group=device_group)
        table[:, start:stop].copy_(piece)
    return table


def bind(embedding, tp_group) -> bool:
    """Assemble, check and bind the whole table on every rank of
    ``tp_group``, or on none. Collective: every rank calls it at the same
    point, outside a capture."""
    try:
        why = local_refusal(embedding)
    except Exception as error:  # noqa: BLE001 -- this rank's refusal, voted below
        why = f"{type(error).__name__}: {error}"
    if why is not None:
        logger.warning("Kimi-K3 local embedding: this rank refuses (%s)", why)
    if not group_min((int(why is None),), tp_group.cpu_group)[0]:
        logger.info("Kimi-K3 local embedding off: a TP rank refused")
        return False
    table, ok, digest = None, True, 0
    try:
        table = gather_table(embedding, tp_group.device_group)
        own = table[embedding.tp_rank].view(torch.int16)
        if not torch.equal(own, embedding.weight.data.view(torch.int16)):
            raise RuntimeError("this rank's shard did not come back from the gather")
        table = table.view(embedding.num_embeddings, -1)
        canonicalize_(table)
        digest = checksum(table)
    except Exception as error:  # noqa: BLE001 -- any failure is this rank's refusal
        ok = False
        logger.warning("Kimi-K3 local embedding: this rank's table failed (%s)", error)
    # every rank's verdict, and that the tables are one table
    every_ok, low, neg_high = group_min((int(ok), digest, -digest), tp_group.cpu_group)
    if not every_ok or low != -neg_high:
        logger.info(
            "Kimi-K3 local embedding off: %s",
            "the ranks' tables differ" if every_ok else "a TP rank's table failed",
        )
        del table
        return False
    embedding.local_table = table
    logger.info(
        "Kimi-K3 local embedding bound: %d x %d on every TP rank (%d MiB)",
        table.shape[0],
        table.shape[1],
        table.numel() * table.element_size() >> 20,
    )
    return True


def install(model) -> bool:
    """Bind the target's embedding (the draft shares the module) on a plain
    TP deployment; True when lookups are local. The model's
    ``VocabParallelEmbedding`` becomes a ``K3LocalEmbedding`` in place (the
    same object, parameters and state dict: only its lookup differs, and only
    once a table is bound), so no model file names the class. Every refusal
    here is one all ranks take alike."""
    from aiter.dist.parallel_state import get_pp_group, get_tp_group

    inner = getattr(getattr(model, "language_model", model), "model", None)
    embedding = getattr(inner, "embed_tokens", None)
    if type(embedding) not in (VocabParallelEmbedding, K3LocalEmbedding):
        logger.info(
            "Kimi-K3 local embedding off: the model's embedding is a %s",
            type(embedding).__name__,
        )
        return False
    if getattr(embedding, "local_table", None) is not None:
        return True
    if embedding.tp_size <= 1:
        logger.info("Kimi-K3 local embedding off: one rank holds the table")
        return False
    if get_pp_group().world_size > 1:
        logger.info("Kimi-K3 local embedding off: pipeline parallelism")
        return False
    if type(embedding) is VocabParallelEmbedding:
        embedding.__class__ = K3LocalEmbedding
        embedding.local_table = None
    return bind(embedding, get_tp_group())
