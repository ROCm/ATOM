# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in capture of real sparse-prefill operands for offline kernel work.

``ATOM_M3_PREFILL_DUMP=<dir>`` saves, on TP rank 0, the operands of the NVFP4 sparse
prefill (index q, main q, the request's index / KV pages, the selection) of the layers in
``ATOM_M3_PREFILL_DUMP_LAYERS`` (default ``0,15,30,45,59``) for chunks of at least
``ATOM_M3_PREFILL_DUMP_MIN_Q`` (2048) rows onto a prefix of at least
``ATOM_M3_PREFILL_DUMP_MIN_P`` (16384) tokens, up to ``ATOM_M3_PREFILL_DUMP_MAX`` (4)
chunks. Synchronizes the device; never enable it for a timed run.
"""

import os

import torch

_DIR = os.environ.get("ATOM_M3_PREFILL_DUMP")
_LAYERS = {
    int(x) for x in os.environ.get("ATOM_M3_PREFILL_DUMP_LAYERS", "0,15,30,45,59").split(",")
}
_MIN_Q = int(os.environ.get("ATOM_M3_PREFILL_DUMP_MIN_Q", "2048"))
_MIN_P = int(os.environ.get("ATOM_M3_PREFILL_DUMP_MIN_P", "16384"))
_MAX = int(os.environ.get("ATOM_M3_PREFILL_DUMP_MAX", "4"))
_EVERY = int(os.environ.get("ATOM_M3_PREFILL_DUMP_EVERY", "1"))
_state = {"chunks": 0, "seen": 0, "key": None, "take": False}


def enabled() -> bool:
    return _DIR is not None


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def maybe_dump(layer, *, index_q, q, index_cache, kv_cache, topk_idx, block_table,
               cu_seqlens_q, seq_lens, prefix_lens, k_scale, v_scale):
    if _DIR is None or layer not in _LAYERS or _rank() != 0:
        return
    total_q = q.shape[0]
    if total_q < _MIN_Q:
        return
    prefix = prefix_lens.cpu()
    if int(prefix.max()) < _MIN_P:
        return
    key = (total_q, tuple(prefix.tolist()))
    if _state["key"] != key:
        _state["key"] = key
        _state["seen"] += 1
        _state["take"] = (
            _state["chunks"] < _MAX and (_state["seen"] - 1) % _EVERY == 0
        )
        if _state["take"]:
            _state["chunks"] += 1
    if not _state["take"]:
        return
    seq = seq_lens.cpu()
    nreq = cu_seqlens_q.shape[0] - 1
    mb = [-(-int(s) // 128) for s in seq[:nreq]]
    bt = block_table[:nreq].cpu()
    pages = torch.cat([bt[b, : mb[b]] for b in range(nreq)]).long().to(index_cache.device)
    os.makedirs(_DIR, exist_ok=True)
    path = os.path.join(_DIR, f"c{_state['chunks']}_q{total_q}_L{layer}.pt")
    torch.save(
        {
            "layer": layer,
            "index_q": index_q.cpu(),
            "q": q.cpu(),
            "index_pages": index_cache[pages].cpu(),
            "kv_pages": kv_cache[pages].cpu(),
            "page_ids": pages.cpu(),
            "topk_idx": topk_idx.cpu(),
            "block_table": bt,
            "mb": mb,
            "cu_seqlens_q": cu_seqlens_q.cpu(),
            "seq_lens": seq[:nreq],
            "prefix_lens": prefix[:nreq],
            "k_scale": k_scale.cpu(),
            "v_scale": v_scale.cpu(),
        },
        path,
    )
