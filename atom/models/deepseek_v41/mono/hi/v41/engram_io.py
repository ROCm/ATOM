# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1's Engram (layers 1 and 14) in the composed layer: the host inputs of ``recipe.engram_kv_blocks``
(the token-only half, an earlier launch) and of the gate half (``layer.window_layer_blocks(engram=True)``).

- ``hash_words``: a layer's ``<name>.hash`` from ``NgramHashState``'s constants (``consts``: token map, multipliers,
  primes, offsets, pad id, table rows).
- ``pack_kv``: a kv block's args (wkv's column share, the hash words); the table shard is ATOM's host one.
- ``hist``: a request's compressed ids by position (the prefill's), -1 for a dead (image) token.
"""

from __future__ import annotations

import torch

from ..blocks.base import Channel, Fmt
from ..blocks.gfx950.engram import H_WORDS, HC, HIDDEN, Engram, pack_hash

LAYERS = (1, 14)


def hash_words(consts: dict, layer: int) -> torch.Tensor:
    """``<name>.hash`` of ``layer`` (``pack_hash``)."""
    li = consts["layer_ids"].index(layer)
    return pack_hash(consts["multipliers"][li], consts["primes"][li], consts["offsets"][li],
                     consts["num_embeddings"][li], consts["pad_id"])  # fmt: skip


def hist(
    tokens: torch.Tensor, ctx: int, tmap: torch.Tensor, hist_len: int
) -> torch.Tensor:
    """int32 [1, hist_len]: the compressed ids of positions < ``ctx`` (the prefill's); later slots poisoned (-7: a
    step only reads positions before its own rows, and writes its rows')."""
    h = torch.full((1, hist_len), -7, dtype=torch.int32)
    h[0, :ctx] = tmap.to(torch.int32)[tokens[:ctx].long()]
    return h


def pack_kv(name: str, layer: int, w: dict, consts: dict | None, tp: int = 1, rank: int = 0,
            device="cuda") -> dict[str, torch.Tensor]:  # fmt: skip
    """Block ``name``'s args (``recipe.engram_kv_blocks``) for ``layer``: wkv's column share and the hash words from
    ``w`` (``engram.wkv.{weight,scale}``) and ``consts`` (None: the hash words are unread, zeros), not the table
    shard (ATOM's host one)."""
    words = (
        hash_words(consts, layer)
        if consts is not None
        else torch.zeros(H_WORDS, dtype=torch.int64)
    )
    blk = Engram(name, Channel("ra", HC * HIDDEN, Fmt.BF16), "ids", "pos", "hist", "tmap", 0, mode="kv",
                 kv_arg=f"{name}.kv")  # fmt: skip
    return blk.pack(
        {"wkv": w["engram.wkv.weight"], "wkv_s": w["engram.wkv.scale"], "hash": words},
        tp,
        rank,
        device,
    )
