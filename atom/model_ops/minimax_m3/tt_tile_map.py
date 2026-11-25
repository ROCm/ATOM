# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The token-tiled prefill's row and tile maps (`tt_prefill`) in one Triton launch."""

import torch
import triton
import triton.language as tl

# rows / tiles a program owns, and requests a pass of its loop takes
_BLOCK = 256
_REQ_BLOCK = 64


@triton.jit(do_not_specialize=["nreq", "total_q", "ntiles"])
def _tile_map_kernel(
    cu_ptr,  # int32 [nreq + 1] query starts
    prefix_ptr,  # int32 [nreq] keys before each request's chunk
    req_ptr,  # out int32 [total_q]
    kv_len_ptr,  # out int32 [total_q]
    t0_ptr,  # out int32 [ntiles]
    n_ptr,  # out int32 [ntiles]
    nreq,
    total_q,
    ntiles,
    TQ: tl.constexpr,
    BLOCK: tl.constexpr,
    REQ_BLOCK: tl.constexpr,
):
    """`tile_map`'s maps: a program owns BLOCK rows and BLOCK tiles and walks the requests
    REQ_BLOCK at a time, carrying the tile count of the requests before."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cnt = tl.zeros((BLOCK,), tl.int32)  # requests that end at or before the row
    nb = tl.zeros((BLOCK,), tl.int32)  # requests whose tiles end at or before the tile
    prev = tl.zeros((BLOCK,), tl.int32)  # tiles before the tile's request
    carry = tl.sum(tl.zeros((REQ_BLOCK,), tl.int32), axis=0)
    for r0 in range(0, nreq, REQ_BLOCK):
        b = r0 + tl.arange(0, REQ_BLOCK)
        live = b < nreq
        lo = tl.load(cu_ptr + b, mask=live, other=0)
        hi = tl.load(cu_ptr + b + 1, mask=live, other=0)
        ended = (hi[None, :] <= offs[:, None]) & live[None, :]
        cnt += tl.sum(ended.to(tl.int32), axis=1)
        tiles = tl.where(live, (hi - lo + TQ - 1) // TQ, 0)
        ends = tl.cumsum(tiles, axis=0) + carry
        hit = (ends[None, :] <= offs[:, None]) & live[None, :]
        nb += tl.sum(hit.to(tl.int32), axis=1)
        prev = tl.maximum(prev, tl.max(tl.where(hit, ends[None, :], 0), axis=1))
        carry += tl.sum(tiles, axis=0)
    rmask = offs < total_q
    req = tl.minimum(cnt, nreq - 1)
    row0 = tl.load(cu_ptr + req, mask=rmask, other=0)
    kv_len = tl.load(prefix_ptr + req, mask=rmask, other=0) + (offs - row0) + 1
    tl.store(req_ptr + offs, req, mask=rmask)
    tl.store(kv_len_ptr + offs, kv_len, mask=rmask)
    tmask = offs < ntiles
    tile_live = nb < nreq
    bc = tl.minimum(nb, nreq - 1)
    start = tl.load(cu_ptr + bc, mask=tmask, other=0)
    length = tl.load(cu_ptr + bc + 1, mask=tmask, other=0) - start
    local = offs - prev
    t0 = tl.where(tile_live, start + local * TQ, 0)
    n = tl.where(tile_live, tl.minimum(tl.maximum(length - local * TQ, 0), TQ), 0)
    tl.store(t0_ptr + offs, t0, mask=tmask)
    tl.store(n_ptr + offs, n, mask=tmask)


def tile_map(cu: torch.Tensor, prefix: torch.Tensor, total_q: int, tq: int):
    """(row request, row causal key count, tile starts, tile lengths, tile count) of a
    step's `total_q` rows: tiles of at most `tq` rows of one request, padded with empty
    tiles to a count known on the host (at most one partial tile a request)."""
    nreq = prefix.shape[0]
    ntiles = (total_q + tq - 1) // tq + nreq
    i32 = {"dtype": torch.int32, "device": cu.device}
    req, kv_lens = torch.empty(total_q, **i32), torch.empty(total_q, **i32)
    t0, n = torch.empty(ntiles, **i32), torch.empty(ntiles, **i32)
    _tile_map_kernel[(triton.cdiv(max(total_q, ntiles), _BLOCK),)](
        cu,
        prefix,
        req,
        kv_lens,
        t0,
        n,
        nreq,
        total_q,
        ntiles,
        TQ=tq,
        BLOCK=_BLOCK,
        REQ_BLOCK=_REQ_BLOCK,
    )
    return req, kv_lens, t0, n, ntiles
