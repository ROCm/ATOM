# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 index-cache sharding under indexer CP (``dcp_config.index_shard``).

Indexer CP has rank r score logical blocks ``j % P == r`` at decode
(`atom.distributed.indexer_cp`). With this on, rank r also *stores* only those
blocks' index rows:

* The block allocator hands logical block j a block id with
  ``id % P == j % P`` (`BlockPool` lanes, `BlockManager._fresh_block`), so a
  block id alone says who owns its index: rank ``id % P``, at local page
  ``id // P`` of index planes ``P`` times smaller than the K/V's.
* Writes: `owned_index_slots` maps the step's slot mapping onto the local
  planes, -1 (skip) for rows another rank keeps.
* Decode reads: the CP scorers address ``base + page * stride(0)`` and only ever
  read owned pages; `page_view` is the plane under that addressing.
* Prefill reads: the TP scorer reads every block of its requests, so
  `plan_index_gather` / `gather_index_pages` assemble the batch's pages from
  their owners first (one all-gather a scoring layer) and hand the scorer a
  table into the gathered pages.

Torch and numpy only, so the CPU tests reach it without AITER.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Tokens a logical index block (and a scheduler block, which sharding requires
# to be the same: `config.index_shard_unsupported_reason`) holds.
INDEX_BLOCK_TOKENS = 128


def index_shard_world(config) -> int:
    """Ranks the index cache is sharded over (the indexer-CP group, which is the
    TP group), or 1 when it is replicated."""
    return int(config.tensor_parallel_size) if config.dcp_config.index_shard else 1


def owned_index_slots(
    slot_mapping: torch.Tensor,
    rank: int,
    world: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Where this rank writes each token's index row: slot ``(p // world) * 128
    + t`` of its local planes for a token at slot ``p * 128 + t`` of a block it
    owns (``p % world == rank``), -1 for every other token (padding included).

    ``out`` (int64, at least as long): a persistent buffer whose pointer a
    decode graph captured; the result is its leading slice.
    """
    page = torch.div(slot_mapping, INDEX_BLOCK_TOKENS, rounding_mode="floor")
    local = torch.div(page, world, rounding_mode="floor") * INDEX_BLOCK_TOKENS
    local += slot_mapping.remainder(INDEX_BLOCK_TOKENS)
    owned = (slot_mapping >= 0) & (page.remainder(world) == rank)
    slots = torch.where(owned, local, -1)
    if out is None:
        return slots
    result = out[: slot_mapping.numel()]
    result.copy_(slots)
    return result


def page_view(plane: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """``plane`` (this rank's ``[pages / world, *row]`` local index plane) as a
    tensor indexed by block id: block ``world * q + rank`` reads local page q.

    The CP scorers read page p at ``data_ptr + p * stride(0)``; with page stride
    ``s / world`` and the base moved back ``rank * s / world`` that address is
    the local page's, ``plane_base + (p // world) * s``, for every owned p. The
    pages between owned ones overlap their neighbours and are never read.
    Ends exactly at the plane's last byte; starts ``rank * s / world`` bytes
    before it, which must still be in the plane's storage (the pool puts the
    index planes after the K/V).
    """
    if world == 1:
        return plane
    step, rest = divmod(plane.stride(0), world)
    if rest:
        raise ValueError(
            f"an index page of {plane.stride(0)} elements does not split into "
            f"{world} block strides"
        )
    offset = plane.storage_offset() - rank * step
    if offset < 0:
        raise ValueError("the index plane starts too close to its storage's start")
    return plane.as_strided(
        (world * (plane.shape[0] - 1) + rank + 1, *plane.shape[1:]),
        (step, *plane.stride()[1:]),
        offset,
    )


@dataclass(frozen=True)
class IndexGather:
    """A prefill batch's index pages, sent by their owners.

    ``send_pages`` [slots] int64: this rank's local pages in send order, request
    after request, its blocks ``rank, rank + world, ...`` of each, padded to
    ``ceil(blocks / world)`` a request (the padding sends local page 0, which no
    table entry reads). ``score_table`` [requests, max blocks] int32: logical
    block j of request b sits at gathered page ``(j % world) * slots + start_b
    + j // world``.
    """

    send_pages: torch.Tensor
    score_table: torch.Tensor
    world: int


def plan_index_gather(
    block_tables,
    context_lens: np.ndarray,
    rank: int,
    world: int,
    device: torch.device | str,
) -> IndexGather:
    """The gather of a prefill whose request b holds ``context_lens[b]`` tokens
    in ``block_tables[b]`` (host lists of block ids). Requests without a table
    row (a padding request) send local page 0.

    Raises if a block id breaks the allocator's rule ``id % world == j %
    world``: the scorer would read another block's rows.
    """
    lens = np.asarray(context_lens, dtype=np.int64)
    blocks = -(-lens // INDEX_BLOCK_TOKENS)
    per_request = -(-blocks // world)
    starts = np.cumsum(per_request) - per_request
    slots = int(per_request.sum())
    send = np.zeros(slots, dtype=np.int64)
    table = np.zeros((len(lens), max(1, int(blocks.max(initial=0)))), dtype=np.int32)
    for b, (count, start) in enumerate(zip(blocks.tolist(), starts.tolist())):
        row = np.asarray(
            block_tables[b][:count] if b < len(block_tables) else (), dtype=np.int64
        )
        positions = np.arange(len(row))
        wrong = np.flatnonzero(row % world != positions % world)
        if wrong.size:
            j = int(wrong[0])
            raise RuntimeError(
                f"index shard: request {b} holds block {int(row[j])} at logical "
                f"block {j}; the allocator must give logical block j a block id "
                f"== j mod {world}"
            )
        owned = row[rank::world] // world
        send[start : start + len(owned)] = owned
        j = np.arange(count)
        table[b, :count] = (j % world) * slots + start + j // world
    return IndexGather(
        send_pages=_upload(send, device),
        score_table=_upload(table, device),
        world=world,
    )


def _upload(array: np.ndarray, device) -> torch.Tensor:
    """``array`` on ``device`` without stalling the host: a pageable copy would
    wait for the GPU work already queued (the previous step's)."""
    tensor = torch.from_numpy(array)
    if torch.device(device).type == "cpu":
        return tensor
    return tensor.pin_memory().to(device, non_blocking=True)


def pack_index_pages(
    gather: IndexGather, payload: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """This rank's send buffer: ``[slots, payload page | scale page]`` bytes."""
    split = payload[0].numel()
    send = torch.empty(
        (gather.send_pages.numel(), split + scale[0].numel()),
        dtype=torch.uint8,
        device=payload.device,
    )
    send[:, :split] = payload.view(payload.shape[0], -1)[gather.send_pages]
    send[:, split:] = scale.view(scale.shape[0], -1)[gather.send_pages]
    return send


def gathered_planes(
    gathered: torch.Tensor, payload: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The all-gathered send buffers as a payload and a scale plane, one page
    a row of ``gathered`` (the page stride is the row's bytes, which both
    prefill scorers take)."""
    split = payload[0].numel()
    return (
        gathered[:, :split].view(-1, *payload.shape[1:]),
        gathered[:, split:].view(-1, *scale.shape[1:]),
    )


def gather_index_pages(
    gather: IndexGather, payload: torch.Tensor, scale: torch.Tensor, group
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every rank's owned pages of the batch, as the scorer's ``(payload,
    scale)`` planes indexed by ``gather.score_table`` (``group``: the indexer-CP
    group, i.e. TP)."""
    send = pack_index_pages(gather, payload, scale)
    gathered = send.new_empty((gather.world * send.shape[0], send.shape[1]))
    torch.distributed.all_gather_into_tensor(gathered, send, group=group.device_group)
    return gathered_planes(gathered, payload, scale)
