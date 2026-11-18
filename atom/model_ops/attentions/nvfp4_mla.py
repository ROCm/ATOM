# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""NVFP4 MLA cache and decode on top of AITER PR 7987.

The producers and the decode shader live in aiter (`mla_nvfp4_quantize_q`,
`mla_nvfp4_update_cache`, `mla_decode_nvfp4_fwd`). This module owns the page64
buffers and the BF16 tail, and turns ATOM's slots into that ABI.

Where this follows TensorRT-LLM, and where it does not:

- Both quantize a short tile immediately (valid < 16) and keep the original
  BF16 of that tile so the next token recomputes the scale. Neither path
  dequantizes FP4 to do that.
- Both use the same fixed gains: Q 6.72, KV 89.6, P 2688.
- NVIDIA's page holds 128 tokens. This shader's page holds 64, so the
  scheduler block size has to be 64 and one block is one physical page.
- NVIDIA dots a 640-wide query (RoPE is repeated, residual on the K side).
  The shader takes absorbed BF16 ``[B, 128, 576]`` and packs the RoPE residual
  into ``q_aux``. The score omits the Q-residual times K-residual term.
- Both keep FP4 as the only resident KV. NVIDIA attends prefill in FP8 from a
  per-forward scratch; here prefill attends to the forward's own BF16 K/V
  (the MHA path) and this module only publishes the latent into FP4 pages.
  Neither supports a prefill that reads cached context (prefix hits, chunked
  prefill), since that history exists only in FP4.
- The shader is one query, 128 heads, one KV head. MTP and other head counts
  are not dispatched.
- Both index the BF16 tail by request, not by batch row: NVIDIA maps
  ``request_ids`` to manager-owned HP pages each step; here ``seq.id`` maps to
  a tail slot, freed when the engine finishes the request. NVIDIA also keeps
  rewind slots for rejected drafts; this ring has none.
"""

from __future__ import annotations

import numpy as np
import torch

from atom.model_ops.attentions.mla_kv_pool import (
    NVFP4_AUX_BYTES,
    NVFP4_NOPE_BYTES,
    NVFP4_PAGE_TOKENS,
)
from atom.model_ops.attentions.nvfp4_hp_ring import Nvfp4HpRing

NVFP4_PAGE = NVFP4_PAGE_TOKENS
NVFP4_HEADS = 128


def nvfp4_split_count(batch: int, row_budget: int, max_splits: int) -> int:
    """Page ranges per request for a decode of ``batch`` rows.

    Static for a given batch width, so a captured graph and its replays agree.
    The shader walks one row's pages in one workgroup; splitting gives a long
    context more workgroups when the batch is too small to fill the GPU.
    """
    if batch <= 0:
        return 1
    return max(1, min(max_splits, row_budget // batch))


def split_page_csr(indptr, last, splits: int, out_indptr, out_last) -> None:
    """Cut each request of a page CSR into ``splits`` consecutive page ranges.

    ``indptr [B+1]`` and ``last [B]`` describe whole requests. Writes
    ``out_indptr [B*splits+1]`` and ``out_last [B*splits]`` for rows
    ``b*splits + s``. The ranges of a request tile its pages in order, so the
    split rows read the same ``indices`` as the whole ones. A range without
    pages has length 0; only the range holding a request's last page keeps its
    ``last``, the others are full pages. Device ops only, no host reads.
    """
    batch = last.shape[0]
    pages = (indptr[1:] - indptr[:-1]).long()
    s = torch.arange(splits, device=indptr.device)
    start = (s[None, :] * pages[:, None]) // splits
    end = ((s[None, :] + 1) * pages[:, None]) // splits
    rows = batch * splits
    out_indptr[:rows].copy_((indptr[:-1].long()[:, None] + start).reshape(-1))
    out_indptr[rows : rows + 1].copy_(indptr[batch : batch + 1])
    full = torch.full_like(start, NVFP4_PAGE)
    lens = torch.where(end == pages[:, None], last.long()[:, None], full)
    out_last[:rows].copy_(
        torch.where(end > start, lens, torch.zeros_like(lens)).reshape(-1)
    )


def merge_splits(out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """Combine per-range attention ``out [B, S, H, D]`` by ``lse [B, S, H]``.

    Each range's output is normalized over its own keys; weighting by
    ``exp(lse)`` restores the softmax over all of them. An empty range has
    O = 0 and LSE = -inf and drops out; a request with no keys stays 0.
    """
    peak = lse.amax(dim=1, keepdim=True)
    peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    weight = torch.exp(lse - peak)
    total = weight.sum(dim=1, keepdim=True)
    weight = torch.where(total > 0, weight / total, torch.zeros_like(weight))
    return (out * weight.unsqueeze(-1).to(out.dtype)).sum(dim=1).to(out.dtype)


NVFP4_KV_GAIN = 89.6
# E2M1 midpoints and the step to the next grid value (0 .5 1 1.5 2 3 4 6).
# A tie goes to the even code: down at the first set, up at the second.
_E2M1_TIE_DOWN = ((0.25, 0.5), (1.25, 0.5), (2.5, 1.0), (5.0, 2.0))
_E2M1_TIE_UP = ((0.75, 0.5), (1.75, 0.5), (3.5, 1.0))


def _fp4_blocks(x: torch.Tensor, ideal: bool) -> torch.Tensor:
    """Quantize then dequantize FP32 ``x`` in blocks along the last axis.

    Matches the AITER producer's NumPy oracle bit for bit: E4M3 scale of
    ``amax / 6`` times the KV gain, clamped to [2**-9, 448] and rounded to
    nearest even; E2M1 codes of ``x`` over ``amax / 6`` when ``ideal`` (the
    NoPE C path), over the rounded scale otherwise (RoPE and its residual).
    No host-built tensors, so it can run inside a CUDA graph capture. Divisors
    are tensors: a GPU tensor divided by a Python scalar multiplies by its
    reciprocal, which is not correctly rounded.
    """
    amax = x.abs().amax(dim=-1, keepdim=True)
    div = torch.where(amax == 0, torch.ones_like(amax), amax / torch.full_like(amax, 6))
    scale = (div * NVFP4_KV_GAIN).clamp(1 / 512, 448).to(torch.float8_e4m3fn).float()
    gain = torch.full_like(scale, NVFP4_KV_GAIN)
    if not ideal:
        div = scale / gain
    v = x.abs() / div
    q = torch.zeros_like(v)
    for t, step in _E2M1_TIE_DOWN:
        q += (v > t) * step
    for t, step in _E2M1_TIE_UP:
        q += (v >= t) * step
    return torch.where(torch.signbit(x), -q, q) * scale / gain


def nvfp4_roundtrip(tiles: torch.Tensor, rank: int = 512) -> torch.Tensor:
    """What ``mla_nvfp4_update_cache`` stores for ``tiles [N, 16, D]``,
    dequantized to BF16. Rows past a tile's valid count must be zero.

    NoPE shares one scale per 16-token x 16-channel block; RoPE is per token
    in 16-channel blocks plus an independently quantized residual.
    """
    x = tiles.float()
    n, rows, dim = x.shape
    nope = x[..., :rank].reshape(n, rows, rank // 16, 16).transpose(1, 2)
    nope = _fp4_blocks(nope.reshape(n, rank // 16, rows * 16), ideal=True)
    nope = nope.reshape(n, rank // 16, rows, 16).transpose(1, 2).reshape(n, rows, rank)
    rope = x[..., rank:].reshape(n, rows, -1, 16)
    base = _fp4_blocks(rope, ideal=False)
    rope = base + _fp4_blocks(rope - base, ideal=False)
    return torch.cat((nope, rope.reshape(n, rows, dim - rank)), dim=-1).to(
        torch.bfloat16
    )


class Nvfp4MlaCache:
    """FP4 pages plus the BF16 tail, for every MLA layer.

    In the engine the pages are the paged pool's two planes, passed in as
    ``kv_nope`` / ``kv_aux`` ``[layers, pages, bytes]``. Without them the
    cache allocates its own, which is what the unit tests use.
    """

    def __init__(
        self,
        layers: int,
        pages: int,
        max_reqs: int,
        head_dim: int,
        *,
        device: torch.device | str,
        kv_nope: torch.Tensor | None = None,
        kv_aux: torch.Tensor | None = None,
        emulate_no_ring: bool = False,
        triton_ring: int = 0,
    ):
        if layers < 1 or pages < 1 or max_reqs < 1:
            raise ValueError(
                f"NVFP4 cache needs positive layers, pages and requests, "
                f"got layers={layers} pages={pages} reqs={max_reqs}."
            )
        self.layers = layers
        self.pages = pages
        self.head_dim = head_dim
        self.device = torch.device(device)
        if kv_nope is None:
            kv_nope = torch.empty(
                (layers, pages, NVFP4_NOPE_BYTES), dtype=torch.uint8, device=self.device
            )
        if kv_aux is None:
            kv_aux = torch.empty(
                (layers, pages, NVFP4_AUX_BYTES), dtype=torch.uint8, device=self.device
            )
        for name, plane, width in (
            ("kv_nope", kv_nope, NVFP4_NOPE_BYTES),
            ("kv_aux", kv_aux, NVFP4_AUX_BYTES),
        ):
            if plane.dtype != torch.uint8 or tuple(plane.shape) != (
                layers,
                pages,
                width,
            ):
                raise ValueError(
                    f"NVFP4 {name} must be uint8 [{layers}, {pages}, {width}], "
                    f"got {plane.dtype} {tuple(plane.shape)}."
                )
            # The shader reads one layer as a contiguous [pages, width] plane.
            if not plane[0].is_contiguous():
                raise ValueError(f"NVFP4 {name} layer slice must be contiguous.")
        self.kv_nope = kv_nope
        self.kv_aux = kv_aux
        # ATOM_MLA_NVFP4_EMULATE_NO_RING: the tail holds the FP4 round trip of
        # each written tile instead of the original BF16.
        self.emulate_no_ring = emulate_no_ring
        # Decode cache write (ATOM_MLA_NVFP4_TRITON_RING): 2 `ring_append_quantize`,
        # 1 `ring_append_gather` + update_cache, 0 torch ops + update_cache.
        self.triton_ring = triton_ring if self.device.type == "cuda" else 0
        # One row per live request, plus `pad_row`: padding rows of a batch
        # write their throwaway token there instead of into a request's tail.
        self.pad_row = max_reqs
        self.ring = Nvfp4HpRing(
            max_reqs + 1, layers, head_dim, rewind=0, device=self.device
        )
        # seq.id -> ring row. Kept for the whole request: a request that is
        # rescheduled to another batch row keeps reading its own open tile.
        self._slot_of: dict[int, int] = {}
        self._free_slots = list(range(max_reqs - 1, -1, -1))
        self._params = None
        self._params_key = None

    def bind_rows(self, req_ids, pad_to: int | None = None) -> np.ndarray:
        """Ring row of each scheduled request, in batch-row order, on the host.

        A request seen for the first time takes a free row. Its first write is
        at a tile start (position 0, or a block-aligned prefix hit), which
        discards whatever the previous owner left in that row. Rows from
        ``len(req_ids)`` to ``pad_to`` are batch padding and get ``pad_row``.
        """
        n = len(req_ids)
        width = n if pad_to is None else pad_to
        if width < n:
            raise ValueError(f"pad_to={pad_to} is below {n} scheduled requests.")
        rows = np.full(width, self.pad_row, dtype=np.int32)
        for i, rid in enumerate(req_ids):
            rid = int(rid)
            slot = self._slot_of.get(rid)
            if slot is None:
                if not self._free_slots:
                    raise RuntimeError(
                        f"NVFP4 tail ring is full ({len(self._slot_of)} live "
                        "requests). Raise the ring size to max_num_seqs."
                    )
                slot = self._free_slots.pop()
                self._slot_of[rid] = slot
            rows[i] = slot
        return rows

    def plan_prefill(self, rows, starts, lengths, block_tables) -> dict:
        """Where one prefill step's latent goes, planned on the host once.

        Request ``i`` forwards ``lengths[i]`` tokens from position
        ``starts[i]`` (a tile start), laid out back to back in the step's flat
        token axis, paged by ``block_tables[i]``. Returns numpy arrays:

        - ``gather [M*16]``: token index of every row of every tile. Rows past
          a tile's ``valid`` repeat its last token; the producer ignores them.
        - ``pages``, ``tile_ids``, ``valid`` ``[M]``: one update per tile.
        - ``ring_rows``, ``ring_slots``, ``ring_tokens`` ``[K]``: the BF16 of
          each request's short last tile, for the next token's requantize.
        - ``short_tiles [S]``: the tile index of each short last tile, and
          ``ring_short``, ``ring_lanes`` ``[K]``: each ring token's entry in it
          and row in that tile (for ``emulate_no_ring``).
        """
        gather, pages, tile_ids, valid = [], [], [], []
        ring_rows, ring_slots, ring_tokens = [], [], []
        short_tiles, ring_short, ring_lanes = [], [], []
        lane = np.arange(16)
        token0 = 0
        tiles = 0
        for row, start, n, table in zip(rows, starts, lengths, block_tables):
            start, n = int(start), int(n)
            if n == 0:
                continue
            if start % 16:
                raise ValueError(
                    f"NVFP4 prefill starts at position {start}, not a tile start."
                )
            offsets = np.arange(0, n, 16)
            positions = start + offsets
            idx = token0 + offsets[:, None] + lane[None, :]
            gather.append(np.minimum(idx, token0 + n - 1).reshape(-1))
            pages.append(np.asarray(table, dtype=np.int64)[positions // NVFP4_PAGE])
            tile_ids.append((positions % NVFP4_PAGE) // 16)
            valid.append(np.minimum(16, n - offsets))
            short = n % 16
            if short:
                first = n - short
                ring_rows.append(np.full(short, row))
                ring_slots.append((start + first + lane[:short]) % self.ring.ring_size)
                ring_tokens.append(token0 + first + lane[:short])
                ring_short.append(np.full(short, len(short_tiles)))
                ring_lanes.append(lane[:short])
                short_tiles.append(np.array([tiles + len(offsets) - 1]))
            tiles += len(offsets)
            token0 += n

        def cat(parts):
            return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

        return {
            "gather": cat(gather),
            "pages": cat(pages),
            "tile_ids": cat(tile_ids),
            "valid": cat(valid),
            "ring_rows": cat(ring_rows),
            "ring_slots": cat(ring_slots),
            "ring_tokens": cat(ring_tokens),
            "short_tiles": cat(short_tiles),
            "ring_short": cat(ring_short),
            "ring_lanes": cat(ring_lanes),
        }

    def upload_plan(self, plan: dict) -> dict:
        """The plan on the device, in the dtypes the writers take."""
        i32 = ("pages", "tile_ids", "valid")
        return {
            name: torch.from_numpy(np.ascontiguousarray(arr)).to(
                device=self.device, dtype=torch.int32 if name in i32 else torch.long
            )
            for name, arr in plan.items()
        }

    def publish_prefill(self, layer: int, kv: torch.Tensor, plan: dict) -> None:
        """Write a whole prefill step's tiles with one producer launch.

        ``kv`` is the step's flat ``[T, head_dim]`` latent after RoPE; ``plan``
        comes from `upload_plan`. No host reads.
        """
        if plan["pages"].numel():
            tiles = kv[plan["gather"]].view(-1, 16, self.head_dim)
            self._write(layer, tiles, plan["pages"], plan["tile_ids"], plan["valid"])
        if not plan["ring_rows"].numel():
            return
        if self.emulate_no_ring:
            short = tiles[plan["short_tiles"]]
            lane = torch.arange(16, device=short.device)
            live = lane[None, :] < plan["valid"][plan["short_tiles"]][:, None]
            short = nvfp4_roundtrip(short.masked_fill(~live[:, :, None], 0))
            values = short[plan["ring_short"], plan["ring_lanes"]]
        else:
            values = kv[plan["ring_tokens"]]
        self.ring.buf[plan["ring_rows"], layer, plan["ring_slots"]] = values.to(
            self.ring.buf.dtype
        )

    def release(self, req_ids) -> None:
        """Return the rows of finished requests. Unknown ids are ignored."""
        for rid in req_ids:
            slot = self._slot_of.pop(int(rid), None)
            if slot is not None:
                self._free_slots.append(slot)

    def params(self, sm_scale: float) -> torch.Tensor:
        key = (float(sm_scale), self.pages)
        if self._params is None or self._params_key != key:
            from aiter.ops.mla_nvfp4 import prepare_mla_nvfp4_params

            self._params = prepare_mla_nvfp4_params(
                device=self.device,
                physical_pages=self.pages,
                sm_scale=float(sm_scale),
            )
            self._params_key = key
        return self._params

    def publish_tokens(
        self,
        layer: int,
        req: int,
        kv: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """Quantize one request's new BF16 rows into their pages.

        ``kv`` is ``[T, head_dim]`` after RoPE. ``positions`` and ``slots``
        are ``[T]``. A slot is ``block_id * 64 + offset`` with block size 64,
        so ``slot // 64`` is the physical page. Tokens are written one 16-token
        tile at a time: the ring holds 16 rows, and a second tile would
        overwrite the first.
        """
        if kv.shape[0] == 0:
            return
        positions = positions.to(device=self.device, dtype=torch.long)
        slots = slots.to(device=self.device, dtype=torch.long)
        kv = kv.to(device=self.device, dtype=torch.bfloat16)
        keep = slots >= 0
        if not bool(keep.all()):
            positions, slots, kv = positions[keep], slots[keep], kv[keep]
        if positions.numel() == 0:
            return
        tiles, valids, pages, tile_ids = [], [], [], []
        start = 0
        pos_list = positions.tolist()
        slot_list = slots.tolist()
        while start < len(pos_list):
            tile_limit = (pos_list[start] // 16) * 16 + 16
            end = start + 1
            while end < len(pos_list) and pos_list[end] < tile_limit:
                end += 1
            req_ids = torch.full(
                (end - start,), req, device=self.device, dtype=torch.long
            )
            gathered, valid, _, tile_id = self.ring.append_tokens(
                layer,
                req_ids,
                positions[start:end],
                kv[start:end],
                NVFP4_PAGE,
            )
            last = end - start - 1
            if self.emulate_no_ring:
                self.ring.store_tiles(
                    layer,
                    req_ids[last:],
                    positions[end - 1 : end],
                    nvfp4_roundtrip(gathered[last:]),
                )
            tiles.append(gathered[last])
            valids.append(valid[last])
            pages.append(slot_list[end - 1] // NVFP4_PAGE)
            tile_ids.append(tile_id[last])
            start = end
        self._write(
            layer,
            torch.stack(tiles),
            torch.tensor(pages, dtype=torch.int32, device=self.device),
            torch.stack(tile_ids),
            torch.stack(valids),
        )

    def publish_decode(
        self,
        layer: int,
        req_ids: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """One new token per row, padding rows included. One producer launch.

        Graph-safe: no host reads. A padding row has slot -1, so its page is -1
        and the AITER producer skips the update; its BF16 goes to ``pad_row``.
        """
        if self.triton_ring == 2 and not self.emulate_no_ring:
            from atom.model_ops.attentions.nvfp4_ring_triton import ring_append_quantize

            ring_append_quantize(
                self.ring.buf[:, layer],
                req_ids,
                positions,
                kv.to(self.ring.buf.dtype),
                slots,
                self.kv_nope[layer],
                self.kv_aux[layer],
            )
            return
        if self.triton_ring:
            from atom.model_ops.attentions.nvfp4_ring_triton import ring_append_gather

            tiles, valid, tile_ids, pages = ring_append_gather(
                self.ring.buf[:, layer],
                req_ids,
                positions,
                kv.to(self.ring.buf.dtype),
                slots,
                NVFP4_PAGE,
            )
        else:
            tiles, valid, _, tile_ids = self.ring.append_tokens(
                layer,
                req_ids,
                positions,
                kv,
                NVFP4_PAGE,
                check=False,
            )
            pages = torch.div(
                slots.to(dtype=torch.long), NVFP4_PAGE, rounding_mode="floor"
            ).to(torch.int32)
        self._write(layer, tiles, pages, tile_ids, valid)
        if self.emulate_no_ring:
            self.ring.store_tiles(layer, req_ids, positions, nvfp4_roundtrip(tiles))

    def decode(
        self,
        layer: int,
        q: torch.Tensor,
        csr: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        sm_scale: float,
        splits: int = 1,
    ) -> torch.Tensor:
        """BF16 ``q`` is ``[B, 128, 576]``. Returns ``[B, 128, 512]`` BF16.

        ``csr`` is the step's ``(indptr[R+1], indices, last[R])`` with
        ``R = B * splits`` rows (`split_page_csr`), built once per step by the
        metadata builder and shared by every layer. With ``splits > 1`` each
        request's Q is repeated over its rows and the rows merged by LSE.
        """
        from aiter.ops.mla_nvfp4 import mla_decode_nvfp4_fwd, mla_nvfp4_quantize_q

        if q.shape[1] != NVFP4_HEADS or q.shape[-1] != self.head_dim:
            raise RuntimeError(
                f"NVFP4 decode wants Q [B, {NVFP4_HEADS}, {self.head_dim}], "
                f"got {tuple(q.shape)}."
            )
        batch = q.shape[0]
        rows = batch * splits
        indptr, indices, last = csr
        if indptr.shape[0] != rows + 1 or last.shape[0] != rows:
            raise RuntimeError(
                f"NVFP4 CSR is for {last.shape[0]} rows, the query has "
                f"{batch} x {splits} splits."
            )
        q_nope, q_aux = mla_nvfp4_quantize_q(q)
        if splits > 1:
            q_nope = q_nope.repeat_interleave(splits, dim=0)
            q_aux = q_aux.repeat_interleave(splits, dim=0)
        out = torch.empty(
            rows, NVFP4_HEADS, 512, dtype=torch.bfloat16, device=self.device
        )
        lse = torch.empty(rows, NVFP4_HEADS, dtype=torch.float32, device=self.device)
        mla_decode_nvfp4_fwd(
            q_nope,
            q_aux,
            self.kv_nope[layer],
            self.kv_aux[layer],
            out,
            indptr,
            indices,
            last,
            params=self.params(sm_scale),
            lse=lse,
            return_lse=False,
        )
        if splits == 1:
            return out
        return merge_splits(
            out.view(batch, splits, NVFP4_HEADS, 512),
            lse.view(batch, splits, NVFP4_HEADS),
        )

    def _write(self, layer, tiles, pages, tile_ids, valid) -> None:
        from aiter.ops.mla_nvfp4 import mla_nvfp4_update_cache

        mla_nvfp4_update_cache(
            tiles.contiguous(),
            pages.contiguous(),
            tile_ids.contiguous(),
            valid.contiguous(),
            self.kv_nope[layer],
            self.kv_aux[layer],
        )


def _as_bf16(x: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    if x.dtype == torch.bfloat16 and scale is None:
        return x
    y = x.float()
    if scale is not None:
        y = y * scale.to(dtype=y.dtype, device=y.device)
    return y.to(torch.bfloat16)


def _latent(
    k_nope: torch.Tensor, k_rope: torch.Tensor, rank: int, rope: int
) -> torch.Tensor:
    return torch.cat(
        (k_nope.reshape(-1, rank), k_rope.reshape(-1, rope)), dim=-1
    ).contiguous()


def publish_prefill_kv(attn, k_nope, k_rope, positions, attn_metadata) -> None:
    """Copy an already-roped prefill latent into the FP4 pages."""
    cache = getattr(attn, "nvfp4", None)
    if cache is None or positions is None:
        return
    plan = getattr(attn_metadata, "nvfp4_prefill_plan", None)
    if plan is not None:
        kv = _latent(k_nope, k_rope, attn.kv_lora_rank, attn.qk_rope_head_dim)
        cache.publish_prefill(attn.nvfp4_layer, kv, plan)
        return
    # No step plan (a request resuming mid-tile): one request at a time.
    tail_rows = getattr(attn_metadata, "nvfp4_tail_rows_host", None)
    if tail_rows is None:
        raise RuntimeError(
            "NVFP4 MLA prefill needs nvfp4_tail_rows_host; the MLA metadata "
            "builder binds it per step."
        )
    kv = _latent(k_nope, k_rope, attn.kv_lora_rank, attn.qk_rope_head_dim)
    cu = attn_metadata.cu_seqlens_q
    slots = attn_metadata.slot_mapping.reshape(-1)
    for row, tail_row in enumerate(tail_rows):
        start, end = int(cu[row]), int(cu[row + 1])
        if end <= start:
            continue
        cache.publish_tokens(
            attn.nvfp4_layer,
            tail_row,
            kv[start:end],
            positions[start:end],
            slots[start:end],
        )


def _tail_rows(attn_metadata) -> torch.Tensor:
    rows = getattr(attn_metadata, "nvfp4_tail_slots", None)
    if rows is None:
        raise RuntimeError(
            "NVFP4 MLA needs nvfp4_tail_slots in the attention metadata; "
            "the MLA metadata builder binds them per step."
        )
    return rows


def try_nvfp4_decode(
    attn, q_nope, q_rope, k_nope, k_rope, positions, attn_metadata
) -> torch.Tensor | None:
    """Decode one token per row. ``None`` leaves the FP8 path in place.

    ``q_rope`` and ``k_rope`` are still pre-RoPE. They are rotated on copies,
    so the FP8 writer still sees the original tensors when this returns None.

    Every row of the batch runs, padding included, so the work depends only
    on the batch width and can be captured in a CUDA graph. Padding rows have
    slot -1 (no cache write), ``pad_row`` in the tail, and length 0 in the
    CSR (O = 0).
    """
    cache = getattr(attn, "nvfp4", None)
    if cache is None:
        return None
    if attn_metadata.max_seqlen_q != 1:
        raise RuntimeError(
            "NVFP4 MLA decode takes one query token. "
            f"max_seqlen_q={attn_metadata.max_seqlen_q}."
        )
    if attn.num_heads != NVFP4_HEADS or attn.num_kv_heads != 1:
        raise RuntimeError(
            f"NVFP4 MLA is wired for {NVFP4_HEADS} query heads and 1 KV head, "
            f"got heads={attn.num_heads} kv_heads={attn.num_kv_heads}."
        )
    rows = q_nope.shape[0]
    if rows == 0:
        return attn._v_up_proj_and_o_proj(
            torch.zeros(
                (0, NVFP4_HEADS, attn.kv_lora_rank),
                dtype=torch.bfloat16,
                device=q_nope.device,
            )
        )
    tail_rows = _tail_rows(attn_metadata)
    csr = getattr(attn_metadata, "nvfp4_csr", None)
    if tail_rows.shape[0] < rows or csr is None:
        raise RuntimeError(
            f"NVFP4 MLA metadata covers {tail_rows.shape[0]} rows "
            f"(csr {'missing' if csr is None else 'present'}); the batch has {rows}."
        )
    positions = positions[:rows]
    q_pe = q_rope.clone()
    k_pe = k_rope[:rows].clone()
    attn.rotary_emb(positions, q_pe, k_pe)
    q = torch.cat(
        (_as_bf16(q_nope, attn._q_scale if q_nope.element_size() == 1 else None), q_pe),
        dim=-1,
    ).contiguous()
    kv = _latent(k_nope[:rows], k_pe, attn.kv_lora_rank, attn.qk_rope_head_dim)
    slots = attn_metadata.slot_mapping.reshape(-1)[:rows]
    cache.publish_decode(attn.nvfp4_layer, tail_rows[:rows], kv, positions, slots)
    splits = getattr(attn_metadata, "nvfp4_splits", 1)
    split_rows = rows * splits
    indptr, indices, last = csr
    out = cache.decode(
        attn.nvfp4_layer,
        q,
        (indptr[: split_rows + 1], indices, last[:split_rows]),
        attn.scale,
        splits=splits,
    )
    return attn._v_up_proj_and_o_proj(out)
