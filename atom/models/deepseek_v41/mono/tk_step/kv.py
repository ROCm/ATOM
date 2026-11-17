# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# The planes and their arithmetic port TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) ``dsv41_rocm_convert.py``
# (``mono.hi.pd.convert``); what is read of ATOM's caches follows lirzhang/atom-tilert-pd's
# ``kv_transfer/disaggregation/tilert/codec.py`` (ATOM V4.1 BF16 PAGE/STATE -> TileRT wire), here without the wire.
"""One request's KV in ATOM's V4.1 caches (BF16 KV mode, FP8 index cache) as the decode planes a ``TkRank`` imports
(``import_planes``): the same planes ``pd.convert`` makes of TileRT's wire.

What it reads (``RequestKV``, ``from_cache``), for the request in STATE slot ``slot`` with PAGE blocks ``blocks``:

- main KV of the sources 2 / 8 / 14 (ratio 2: 64 rows a 128-token block) and 20 (ratio 1: 128 rows): bf16 rows
  (``cache.pages.view("main_<s>")[0]``), the first ``n // ratio`` through the block table -> E2M1 + E4M3 per 16
  (``convert.to_main``);
- index keys: FP8 8-row tiles (128 E4M3 bytes a row interleaved by 16, then 8 fp32 scales;
  ``cache.index_planes[s][0]``), dequantized -> MXFP4 + UE8M0 per 32 (``mxfp4``: TileRT's index arithmetic);
- the 43 window rings: bf16 ``[layers, 133, 512]`` at ``pos % 133`` (``cache.state.view("window")[:, slot]``) ->
  positions ``[n - 128, n)`` at ``pos % 256``, E4M3 + UE8M0 per 32 (``convert.to_window``);
- the compressor rings of the ratio-2 sources: ``(kv, score)`` float32 ``[7, 512]`` at ``pos % 7``
  (``cache.compress_state(s)``) -> position ``n - 1``'s row when ``n`` is odd (``convert.state_plane``);
- Engram history: TileRT keeps the last 16 prompt tokens' compressed ids (``hist``); ATOM's cursor
  (``cache.cursor[slot]``: ``[end, 3 ids]``) keeps 3. ``hist`` takes the prompt's tail ids when the caller has them,
  else the cursor's 3 ids (the rest 0; that the step reads no more than the last 3 is not established: GPU check).

Exactness: main and window are TileRT's own quantizers over ATOM's bf16 rows; the index goes through ATOM's FP8 cache
(not the bf16 keys TileRT quantizes), as the P-side codec does. ATOM's quantized KV modes (fp4 / fp8 main, FP4 index
planes) are not read here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..hi.pd import convert as C
from . import BLOCK  # ATOM's KV block

KV_SOURCES = {2: 2, 8: 2, 14: 2, 20: 1}
INDEX_DIM, INDEX_ROW = 128, 132  # FP8 index: 128 E4M3 bytes + one fp32 scale a row
INDEX_TILE = 8


@dataclass
class RequestKV:
    """One request's view of ATOM's caches (module docstring); every tensor on one device."""

    main: dict  # source -> bf16 [pages, 128 // ratio, 512]
    index: dict  # source -> uint8 [pages, (128 // ratio) * 132]
    window: torch.Tensor  # bf16 [layers, ring, 512], this slot's
    compress: dict  # ratio-2 source -> (kv, score) float32 [ring, 512], this slot's
    blocks: torch.Tensor  # int64 [>= ceil(n / 128)], the request's PAGE blocks
    cursor: torch.Tensor | None = (
        None  # int64 [4]: end position, 3 compressed ids (oldest first, -1 none)
    )


def from_cache(cache, slot: int, blocks) -> RequestKV:
    """``RequestKV`` of the request in STATE slot ``slot`` over ``blocks`` (``batch.state_slots_committed[i]``,
    ``batch.block_tables[i]``), from ATOM's ``PagedAttentionCache``; refuses another geometry.
    """
    g = cache.geometry
    if (
        g.kv_dtype != "bf16"
        or g.index_fp4
        or g.index_block_rows != INDEX_TILE
        or g.head_dim != 512
        or g.index_dim != INDEX_DIM
        or tuple(g.owners) != tuple(KV_SOURCES.items())
        or g.layers != C.NUM_WIN
        or g.window_size != C.WINDOW
        or g.block_size != BLOCK
    ):
        raise ValueError(f"TileRT-path step: unsupported ATOM cache geometry {g}")
    dev = cache.backing.device
    return RequestKV(
        main={s: cache.pages.view(f"main_{s}")[0] for s in KV_SOURCES},
        index={s: cache.index_planes[s][0].flatten(1) for s in KV_SOURCES},
        window=cache.state.view("window")[:, slot],
        compress={
            s: tuple(t[slot] for t in cache.compress_state(s))
            for s, r in KV_SOURCES.items()
            if r == 2
        },
        blocks=torch.as_tensor(blocks, dtype=torch.int64, device=dev),
        cursor=cache.cursor[slot],
    )


def gather_rows(pages: torch.Tensor, blocks: torch.Tensor, count: int) -> torch.Tensor:
    """The first ``count`` rows of a request, its blocks' pages in order (``pages``: ``[pages, rows a page, ...]``)."""
    per = pages.shape[1]
    sel = pages.index_select(0, blocks[: -(-count // per)].to(pages.device))
    return sel.reshape(-1, *pages.shape[2:])[:count]


def index_fp8_rows(
    pages: torch.Tensor, blocks: torch.Tensor, count: int
) -> torch.Tensor:
    """float32 [count, 128]: the first ``count`` FP8 index keys (a page: tiles of 8 rows, their bytes interleaved by
    16 -- byte ``(d // 16) * 128 + row * 16 + d % 16`` -- then the tile's 8 fp32 scales).
    """
    per = pages.shape[1] // INDEX_ROW
    raw = gather_rows(
        pages.view(pages.shape[0], per, INDEX_ROW), blocks, -(-count // per) * per
    )
    tiles = raw.reshape(-1, INDEX_TILE * INDEX_ROW)
    data = tiles[:, : INDEX_TILE * INDEX_DIM].reshape(
        -1, INDEX_DIM // 16, INDEX_TILE, 16
    )
    vals = data.permute(0, 2, 1, 3).reshape(-1, INDEX_DIM)
    scales = (
        tiles[:, INDEX_TILE * INDEX_DIM :].contiguous().view(torch.float32).reshape(-1)
    )
    return (vals.view(torch.float8_e4m3fn).float() * scales[:, None])[:count]


def mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """TileRT's index records: E2M1 pairs (even element low; uint8 [rows, 64]) and a UE8M0 scale per 32 dims
    (uint8 [rows, 4]): amax / 6 rounded up to a power of two."""
    g = x.float().reshape(-1, INDEX_DIM // 32, 32)
    s = C._pow2_ceil(g.abs().amax(-1).clamp_min(0.0001) / 6)
    codes = C._fp4_code((g / s[..., None]).clamp(-6, 6)).reshape(-1, INDEX_DIM)
    return codes[:, 0::2] | codes[:, 1::2] << 4, s.to(torch.float8_e8m0fnu).view(
        torch.uint8
    )


def window_ring(rows: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A layer's TileRT ring (E4M3 [256, 512], uint8 [256, 16]) from ATOM's ring ``rows`` (bf16 [ring, 512] at
    ``pos % ring``): positions ``[n - 128, n)`` at ``pos % 256``."""
    kv = torch.zeros(C.WIN_RING, C.HEAD_DIM, dtype=torch.uint8, device=rows.device)
    sc = torch.zeros(
        C.WIN_RING, C.HEAD_DIM // 32, dtype=torch.uint8, device=rows.device
    )
    lo = max(0, n - C.WINDOW)
    if n > lo:
        if n - lo > rows.shape[0]:
            raise ValueError(f"window ring of {rows.shape[0]} rows < {n - lo}")
        pos = torch.arange(lo, n, device=rows.device)
        q, s = C.to_window(rows.index_select(0, pos % rows.shape[0]).float())
        kv[pos % C.WIN_RING], sc[pos % C.WIN_RING] = q, s
    return kv.view(torch.float8_e4m3fn), sc


def hist(
    n: int, tail_ids=None, token_map=None, cursor=None, device=None
) -> torch.Tensor:
    """int32 [16]: the prompt's last compressed ids at ``pos % 16`` -- from ``tail_ids`` (raw token ids, mapped by
    ``token_map``, as the wire's tail), else from ATOM's cursor (its 3 ids, already compressed).
    """
    if tail_ids is not None:
        return C.hist_plane(
            tail_ids, n, torch.as_tensor(token_map, dtype=torch.int64), device
        )
    h = torch.zeros(C.HIST, dtype=torch.int32, device=device)
    if cursor is None:
        return h
    cur = cursor.tolist()
    if cur[0] != n:
        raise ValueError(f"ATOM cursor at {cur[0]}, request at {n}")
    for j, cid in enumerate(cur[1:]):
        p = n - (len(cur) - 1) + j
        if p >= 0 and cid >= 0:
            h[p % C.HIST] = cid
    return h


@torch.inference_mode()
def request_planes(kv: RequestKV, n: int, tail_ids=None, token_map=None) -> dict:
    """The planes of the request's first ``n`` positions (``TkRank.import_planes``)."""
    dev = kv.window.device
    planes: dict = {"main_kv": {}, "index_k": {}, "state": {}}
    for s, ratio in KV_SOURCES.items():
        ne = n // ratio
        planes["main_kv"][s] = C.to_main(gather_rows(kv.main[s], kv.blocks, ne).float())
        planes["index_k"][s] = mxfp4(index_fp8_rows(kv.index[s], kv.blocks, ne))
        if ratio == 2:
            ring = (
                torch.cat(kv.compress[s], -1) if n % 2 else None
            )  # [ring, kv | score]
            planes["state"][s] = C.state_plane(ring, n, dev)
    planes["win"] = [window_ring(kv.window[i], n) for i in range(kv.window.shape[0])]
    planes["hist"] = hist(n, tail_ids, token_map, kv.cursor, dev)
    return planes
