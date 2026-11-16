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

With the batch's pages (`batch_pages`, from the step's host spans and block
table) the shadow is laid out by page: every page each request's context spans
(rows_per_page rows each) and each request's window ring, and one pass maps the
CSR's rows onto it (`_page_rows`) -- no scan over the pool and no host sync.
Without them, the distinct rows are found by a mark-and-scan over both sources
(`distinct_rows`, one host sync).

`ATOM_DSV41_PACKED_PREFILL`: "auto" (the default: "rows" with the batch's pages;
without them "rows" from AUTO_ROWS_MIN query tokens and "tile256" below -- the
scan costs more than the wider tiles' launches on ~2K-token prompts), "rows"
(this path), "tile256" (the tiled path with 256-query tiles and the tiles'
indptr taken once), "tile32" (the original tiled path).

(The kernels are imported where they are called: the remap is importable and testable without a GPU.)
"""

import os
import weakref
from dataclasses import dataclass

import torch

MODES = ("auto", "rows", "tile256", "tile32")
AUTO_ROWS_MIN = 4096


def prefill_mode(tokens=0, pages=False):
    mode = os.environ.get("ATOM_DSV41_PACKED_PREFILL", "auto")
    if mode not in MODES:
        raise ValueError(f"ATOM_DSV41_PACKED_PREFILL={mode!r} is not one of {MODES}")
    if mode == "auto":
        return "rows" if pages or tokens >= AUTO_ROWS_MIN else "tile256"
    return mode


@dataclass(frozen=True, eq=False)
class BatchPages:
    """A prefill's requests, in batch order: `tables` the step's block table
    (`[bs, cols]` int32) and `slot_ids` its slots (`[bs]`), both on the device;
    `columns` the pages each request's context spans and `slots` its STATE slot
    (host); `rows_per_page` main rows a page at the layer's ratio (0: a
    window-only layer), `ring` ring slots."""

    tables: torch.Tensor
    slot_ids: torch.Tensor
    columns: tuple
    slots: tuple
    rows_per_page: int
    ring: int
    layouts: dict   # the step's `_page_layout`s (kept on the step: a block table's content is the step's)

    def key(self):
        return (self.tables.data_ptr(), self.slot_ids.data_ptr(), self.columns, self.slots,
                self.rows_per_page, self.ring)


def batch_pages(step, ratio, geometry):
    """`BatchPages` of `step` for a layer at `ratio` (host spans only: no sync)."""
    return BatchPages(
        step.block_tables,
        step.slots,
        tuple(-(-r.end // geometry.block_size) for r in step.requests),
        tuple(r.slot for r in step.requests),
        geometry.rows_per_page(ratio) if ratio else 0,
        geometry.ring_slots,
        step.__dict__.setdefault("_packed_prefill_layouts", {}),
    )


def _remap_rows_torch(indices, indptr, inv_page, inv_slot, window0, rpp, ring):
    n = indices.numel()
    valid = torch.arange(n, device=indices.device) < indptr[-1].to(torch.int64)
    row = indices >> 1
    window = (indices & 1) != 0
    main = inv_page[torch.where(valid & ~window, row // max(rpp, 1), 0)].long() * rpp + row % max(rpp, 1)
    win = window0 + inv_slot[torch.where(valid & window, row // ring, 0)].long() * ring + row % ring
    return torch.where(valid, torch.where(window, win, main), 0).clamp(min=0).to(torch.int32)


def _page_layout(pages, main_rows, window_rows, dev):
    """The shadow's rows by page (`tagged`) and the inverse maps a CSR row is
    placed through, from device slices of the block table (no host copy)."""
    key = (pages.key(), main_rows, window_rows)
    hit = pages.layouts.get(key)
    if hit is not None:
        return hit
    rpp, ring, bs = pages.rows_per_page, pages.ring, len(pages.slots)
    parts = [pages.tables[b, :n] for b, n in enumerate(pages.columns) if n] if rpp else []
    page_ids = torch.cat(parts).long() if parts else torch.zeros(0, dtype=torch.int64, device=dev)
    nmain = page_ids.numel() * rpp
    tagged = torch.empty(nmain + bs * ring, dtype=torch.int64, device=dev)
    inv_page = torch.zeros(max(main_rows // max(rpp, 1), 1), dtype=torch.int32, device=dev)
    if nmain:
        inv_page[page_ids] = torch.arange(page_ids.numel(), dtype=torch.int32, device=dev)
        tagged[:nmain] = ((page_ids[:, None] * rpp + torch.arange(rpp, device=dev)) << 1).flatten()
    slots = pages.slot_ids[:bs].long()
    inv_slot = torch.zeros(max(window_rows // ring, 1), dtype=torch.int32, device=dev)
    inv_slot[slots] = torch.arange(bs, dtype=torch.int32, device=dev)
    tagged[nmain:] = (((slots[:, None] * ring + torch.arange(ring, device=dev)) << 1) | 1).flatten()
    pages.layouts[key] = (tagged, inv_page, inv_slot, nmain)
    return pages.layouts[key]


def _page_rows(indices, indptr, main_rows, window_rows, pages):
    """`(tagged, local)` as `distinct_rows`'s, the shadow laid out by page:
    each request's context pages (`rows_per_page` rows each), then each request's
    ring (`_page_layout`, once a step and ratio), then one pass over the CSR;
    nothing waits for the GPU."""
    dev, rpp, ring = indices.device, pages.rows_per_page, pages.ring
    tagged, inv_page, inv_slot, nmain = _page_layout(pages, main_rows, window_rows, dev)
    if dev.type == "cuda" and indices.numel():
        import triton

        local = torch.empty(indices.numel(), dtype=torch.int32, device=dev)
        _remap_rows[(triton.cdiv(indices.numel(), 1024),)](
            indices, indptr, indptr.numel() - 1, local, inv_page, inv_slot, indices.numel(),
            nmain, max(rpp, 1), ring, BLOCK=1024,
        )
    else:
        local = _remap_rows_torch(indices, indptr, inv_page, inv_slot, nmain, rpp, ring)
    return tagged, local


def _remap_rows_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def remap(indices, indptr, last, local, inv_page, inv_slot, n, window0, RPP: tl.constexpr,
              RING: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        valid = m & (i < tl.load(indptr + last).to(tl.int64))
        t = tl.load(indices + i, valid, other=0)
        row = t >> 1
        window = (t & 1) != 0
        main = tl.load(inv_page + row // RPP, valid & ~window, other=0).to(tl.int64) * RPP + row % RPP
        win = window0 + tl.load(inv_slot + row // RING, valid & window, other=0).to(tl.int64) * RING + row % RING
        loc = tl.where(window, win, main)
        tl.store(local + i, tl.maximum(tl.where(valid, loc, 0), 0).to(tl.int32), m)

    return remap


class _Lazy:
    def __init__(self, make):
        self.make, self.fn = make, None

    def __getitem__(self, grid):
        if self.fn is None:
            self.fn = self.make()
        return self.fn[grid]


_remap_rows = _Lazy(_remap_rows_kernel)


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


def _remap(indices, indptr, main_rows, window_rows, pages=None):
    base = indices if indices._base is None else indices._base
    psig = None if pages is None else pages.key()
    sig = (indices.data_ptr(), indices.numel(), indptr.data_ptr(), main_rows, window_rows, psig)
    hit = _REMAPS.get(id(base))
    if hit is not None and hit[0]() is base and hit[1] == sig:
        return hit[2]
    if pages is not None and pages.slots:
        remap = _page_rows(indices, indptr, main_rows, window_rows, pages)
    else:
        remap = distinct_rows(indices, indptr, main_rows, window_rows)
    key = id(base)
    # (keyed by id, not by the tensor: a WeakKeyDictionary compares a hit's tensors with ==)
    _REMAPS[key] = (weakref.ref(base, lambda _, k=key: _REMAPS.pop(k, None)), sig, remap)
    return remap


def rows_prefill(q, pool, indices, indptr, kv, extend, extend_indptr, sink, scale, out, pages=None):
    """One decode of the batch's rows (its pages' with `pages`, else the
    distinct ones the CSR names), one attention launch."""
    from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src

    from .packed_rows import gather_prefix_rows

    (main, _), (window, _) = pool
    tagged, local = _remap(indices, indptr, main.shape[0], window.shape[0], pages)
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
