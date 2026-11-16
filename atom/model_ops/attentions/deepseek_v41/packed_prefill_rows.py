# SPDX-License-Identifier: MIT
"""Prefill attention over planar FP4 / FP8 sources in one launch a layer.

`packed_attention.packed_prefill` decodes each 32-query tile's selected rows
into BF16 scratch and attends tile by tile: four launches a tile and a layer,
host-bound past a few thousand tokens (a 28K-token prompt: ~67K launches). A
prefill's queries share their history, so this path decodes every distinct
row the batch's CSR names once -- main pool rows and window ring rows alike --
into a BF16 shadow, remaps the CSR onto it and runs the same BF16 attention
kernel once over all queries. The shadow holds the same BF16 values the tiled
path stages (`gather_prefix_rows`), so the two differ only in the attention
kernel's split / reduction order.

`ATOM_DSV41_PACKED_PREFILL`: "auto" (the default: "rows" from AUTO_ROWS_MIN
query tokens, "tile256" below -- measured on gfx1250 TP4: the remap's scans
cost more than the wider tiles' launches on ~2K-token prompts and less from
~8K), "rows" (this path), "tile256" (the tiled path with 256-query tiles and
the tiles' indptr taken once), "tile32" (the original tiled path).

(The kernels are imported where they are called: the remap is importable and testable without a GPU.)
"""

import os
import weakref

import torch

MODES = ("auto", "rows", "tile256", "tile32")
AUTO_ROWS_MIN = 4096


def prefill_mode(tokens=0):
    mode = os.environ.get("ATOM_DSV41_PACKED_PREFILL", "auto")
    if mode not in MODES:
        raise ValueError(f"ATOM_DSV41_PACKED_PREFILL={mode!r} is not one of {MODES}")
    if mode == "auto":
        return "rows" if tokens >= AUTO_ROWS_MIN else "tile256"
    return mode


def distinct_rows(indices, indptr, main_rows, window_rows):
    """`(tagged, local)`: the distinct tagged rows `indices[:indptr[-1]]` names
    (`row << 1 | is_window`, main rows first, each source ascending) and each
    entry's position among them (int32; entries past `indptr[-1]` map to a
    named row (main row 0), never read). One scan over the two sources' rows
    and one host sync: the distinct count sizes the shadow."""
    n = indices.numel()
    valid = torch.arange(n, device=indices.device) < indptr[-1].to(torch.int64)
    window = (indices & 1) != 0
    key = torch.where(window, main_rows + (indices >> 1), indices >> 1)
    key = torch.where(valid, key, torch.zeros_like(key))
    mark = torch.zeros(main_rows + window_rows, dtype=torch.int32, device=indices.device)
    mark[key] = 1
    position = torch.cumsum(mark, 0, dtype=torch.int32) - 1
    local = position[key]
    tagged = torch.empty(int(position[-1]) + 1, dtype=torch.int64, device=indices.device)
    # each slot's tag from its key, so every entry naming it writes the same value (a padding entry's key 0 is
    # window row 0 when the layer has no main rows: writing main row 0's tag there read an empty source)
    tagged[local.long()] = torch.where(key >= main_rows, ((key - main_rows) << 1) | 1, key << 1)
    return tagged, local


# The remap a CSR plane gets, kept while the plane lives: an index group's
# layers share one plane (`attention_indices`), so its rows are remapped and
# counted (the host sync) once a group, not once a layer.
_REMAPS = {}   # id(plane base) -> (weakref to it, signature, remap); entries go with their plane


def _remap(indices, indptr, main_rows, window_rows):
    base = indices if indices._base is None else indices._base
    sig = (indices.data_ptr(), indices.numel(), indptr.data_ptr(), main_rows, window_rows)
    hit = _REMAPS.get(id(base))
    if hit is not None and hit[0]() is base and hit[1] == sig:
        return hit[2]
    remap = distinct_rows(indices, indptr, main_rows, window_rows)
    key = id(base)
    # (keyed by id, not by the tensor: a WeakKeyDictionary compares a hit's tensors with ==)
    _REMAPS[key] = (weakref.ref(base, lambda _, k=key: _REMAPS.pop(k, None)), sig, remap)
    return remap


def rows_prefill(q, pool, indices, indptr, kv, extend, extend_indptr, sink, scale, out):
    """One decode of the batch's distinct rows, one attention launch."""
    from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src

    from .packed_rows import gather_prefix_rows

    (main, _), (window, _) = pool
    tagged, local = _remap(indices, indptr, main.shape[0], window.shape[0])
    shadow = torch.empty((tagged.numel(), q.shape[-1]), device=q.device, dtype=q.dtype)
    ptr = torch.tensor([0, tagged.numel()], dtype=torch.int32, device=q.device)
    gather_prefix_rows(pool, tagged, ptr, shadow, 0, 1)
    result = sparse_attn_v4_paged_2src(
        q, shadow, local, indptr, kv, extend, extend_indptr, sink, scale, out=out
    )
    if result.data_ptr() != out.data_ptr():
        out.copy_(result)
    return out


def tiled_prefill(q, pool, indices, indptr, kv, extend, extend_indptr, sink, scale, out, query_tile):
    """`packed_prefill`'s tiles, `query_tile` queries each, every tile's local
    indptr taken in one subtraction up front."""
    from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src

    from .packed_rows import gather_prefix_rows

    tokens, _, dim = q.shape
    capacity = min(query_tile, tokens) * (indices.numel() // tokens)
    scratch = torch.empty((capacity, dim), device=q.device, dtype=q.dtype)
    local_indices = torch.arange(capacity, device=q.device, dtype=torch.int32)
    starts = torch.arange(0, tokens, query_tile, device=q.device)
    # [tiles, query_tile + 1]: tile b's indptr less its first entry (clamped past the last query)
    window = (starts[:, None] + torch.arange(query_tile + 1, device=q.device)).clamp(max=tokens)
    local_ptrs = indptr[window] - indptr[starts][:, None]
    for i, begin in enumerate(range(0, tokens, query_tile)):
        end = min(begin + query_tile, tokens)
        gather_prefix_rows(pool, indices, indptr, scratch, begin, end)
        result = sparse_attn_v4_paged_2src(
            q[begin:end],
            scratch,
            local_indices,
            local_ptrs[i, : end - begin + 1],
            kv,
            extend,
            extend_indptr[begin : end + 1],
            sink,
            scale,
            out=out[begin:end],
        )
        if result.data_ptr() != out[begin:end].data_ptr():
            out[begin:end].copy_(result)
    return out
