# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The token embedding's shards, read in place across the TP group (the V4.1 mono hi hook).

V4.1's embedding (``VocabParallelEmbedding``) holds a vocab shard a rank: ATOM's lookup is the masked shard lookup
and an all-reduce of the partials, one of which is the row and the rest zeros (~14 us of all-reduce, on the round's
critical path, twice a c1 round: the target's step ids and the DSpark draft's anchors). The hook's begin-step and its
block draft's begin kernel instead read each id's row directly from the rank that holds it, through that shard's
address on this GPU (the shard moved into IPC-exportable memory, exported once at load, opened on the others): the same
row, no collective, no copy of the table. ATOM's own forwards (prefill) keep their lookup.

``check`` refuses an embedding it cannot read so before anything is moved (the install's agreed check,
``hi_runner.check``); ``install`` exchanges the addresses and checks the read on every rank (sampled ids, each shard's
edges, out-of-range ids: the rows ATOM's lookup gives), refusing on any difference on every rank alike, with the
shard moved back into ordinary device memory.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from atom.models.deepseek_v41.mono.hi_bridge import _shard_row
from atom.models.deepseek_v41.mono.hi_gate import log_once
from atom.mono.runtime.consensus import MonoUnsupported

logger = logging.getLogger("atom")


@triton.jit
def _read_rows_kernel(
    shards, ids, out, rows, per, HIDDEN: tl.constexpr, BLOCK: tl.constexpr
):
    """Program (i, c): row ``ids[i]``, block c, as the begin kernels read it (``hi_bridge._shard_row``)."""
    i = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    src, hit = _shard_row(shards, tl.load(ids + i).to(tl.int64), rows, per, HIDDEN)
    tl.store(
        out + i * HIDDEN + d,
        tl.load(src + d, mask=hit & (d < HIDDEN), other=0.0),
        mask=d < HIDDEN,
    )


def check(model):
    """The model's one sharded token embedding (ATOM's ``VocabParallelEmbedding``), its shard bf16 and contiguous;
    raises ``MonoUnsupported`` otherwise. Rank-local, nothing moved."""
    from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding

    embeds = [
        m
        for m in model.modules()
        if isinstance(m, VocabParallelEmbedding) and not isinstance(m, ParallelLMHead)
    ]
    if len(embeds) != 1:
        raise MonoUnsupported(
            f"V4.1 mono hi embedding: {len(embeds)} sharded token embeddings, expected one"
        )
    shard = embeds[0].weight.data
    if shard.dtype != torch.bfloat16 or not shard.is_contiguous():
        raise MonoUnsupported(
            f"V4.1 mono hi embedding: shard {shard.dtype}, contiguous {shard.is_contiguous()}"
        )
    return embeds[0]


def install(model, group) -> SimpleNamespace:
    """The model's (one) sharded token embedding's shards on every rank of the TP group (``group``: its CPU group)
    -> ``shards`` (int64 [tp] on this GPU: each rank's shard address), ``per`` (rows a shard), ``rows`` (the
    vocabulary), ``hidden``. Collective; ``check`` passed on every rank. A read that differs from ATOM's lookup on
    any rank raises ``MonoUnsupported`` on every rank, the shard back in ordinary device memory (the uncached block it
    was moved into stays allocated: uncached memory is never freed, ``atom.mono.runtime.uncached``).
    """
    import torch.distributed as dist

    from atom.mono.runtime.hip_ipc import (
        get_allocation_base,
        get_ipc_handle,
        open_ipc_handle,
    )
    from atom.model_ops.embed_head import VocabParallelEmbedding

    emb = check(model)
    shard = emb.weight.data
    tp, rank, per = emb.tp_size, emb.tp_rank, emb.num_embeddings_per_partition
    if (
        tp > 1
    ):  # the shard in IPC-exportable memory (torch's allocator may not export: expandable segments)
        from atom.mono.runtime.uncached import uncached_bytes

        own = uncached_bytes(shard.numel() * shard.element_size(), shard.device).view(
            torch.bfloat16
        )
        own = own[: shard.numel()].view(shard.shape)
        own.copy_(shard)
        emb.weight.data = own  # (the module's shard: ATOM's own lookups read it; the old allocation freed)
        shard = own
    addrs = [shard.data_ptr()] * tp
    if tp > 1:
        base = get_allocation_base(shard.data_ptr())
        mine = (get_ipc_handle(base), shard.data_ptr() - base)
        every = [None] * tp
        dist.all_gather_object(every, mine, group=group)
        for q, (handle, offset) in enumerate(every):
            if q != rank:
                addrs[q] = open_ipc_handle(handle) + offset
    shards = torch.tensor(addrs, dtype=torch.int64, device=shard.device)
    out = SimpleNamespace(
        shards=shards, per=per, rows=emb.num_embeddings, hidden=shard.shape[1]
    )
    # the check: the same ids on every rank (a seeded draw), the shards' edges, out-of-range ids
    g = torch.Generator(device=shard.device).manual_seed(0x5EED)
    edges = [q * per + d for q in range(tp) for d in (0, 1, per - 2, per - 1)]
    ids = torch.cat([
        torch.randint(0, emb.num_embeddings, (4096,), device=shard.device, generator=g),
        torch.tensor(edges + [-1, emb.num_embeddings], device=shard.device),
    ])  # fmt: skip
    want = VocabParallelEmbedding.forward(emb, ids)
    got = torch.empty_like(want)
    _read_rows_kernel[(ids.numel(), triton.cdiv(out.hidden, 1024))](
        shards, ids, got, out.rows, per, HIDDEN=out.hidden, BLOCK=1024
    )
    ok = torch.tensor([int(torch.equal(want, got))], dtype=torch.int32, device="cpu")
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    if not int(ok):
        if tp > 1:  # ATOM's own lookups back on ordinary memory
            emb.weight.data = shard.clone()
        raise MonoUnsupported(
            "V4.1 mono hi embedding: the in-place shard read differs from ATOM's lookup"
        )
    log_once(
        logging.INFO,
        "V4.1 mono hi: the token embedding read in place from %d ranks' shards, checked on %d ids",
        tp,
        ids.numel(),
    )
    return out
