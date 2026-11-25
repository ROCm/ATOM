# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""NVFP4 MLA cache and decode on top of AITER's NVFP4 MLA (PR 8200).

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


def aiter_aux_n16() -> bool:
    """Whether AITER's cache writer, and so its decoder, uses the N16 AUX
    layout (PR 8201 on), the only layout the fused decode writers produce;
    prefill writes through ``mla_nvfp4_update_cache``'s default. Once N32 is
    retired the writer has no ``aux_layout`` at all and documents its N16
    layout (PR 8200); its first N16-only release is also the one whose rebase
    threshold defaults to 6. The N32-only writer (PR 7987) has neither."""
    import inspect

    try:
        from aiter.ops.mla_nvfp4 import mla_nvfp4_update_cache, prepare_mla_nvfp4_params
    except ImportError:
        return False
    param = inspect.signature(mla_nvfp4_update_cache).parameters.get("aux_layout")
    if param is not None:
        return param.default == "n16"
    if "N16" in (mla_nvfp4_update_cache.__doc__ or ""):
        return True
    thr = inspect.signature(prepare_mla_nvfp4_params).parameters.get("rebase_threshold")
    return thr is not None and thr.default == 6.0


class Nvfp4MlaCache:
    """FP4 pages plus the BF16 tail, for every MLA layer.

    In the engine the pages are the paged pool's two planes, passed in as
    ``kv_nope`` / ``kv_aux`` ``[layers, pages, bytes]``, and ``ring`` is the
    tail the metadata builder allocated before KV sizing. Without them the
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
        ring: Nvfp4HpRing | None = None,
        fused: bool = True,
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
        # On the GPU, decode quantizes the NoPE Q in the K-up GEMM and does the
        # RoPE, the rest of Q and the cache write in one launch
        # (`try_nvfp4_decode`). Off, or on the CPU: torch ops and the AITER
        # producers.
        self.fused = fused and self.device.type == "cuda"
        if self.fused and not aiter_aux_n16():
            raise RuntimeError(
                "NVFP4 MLA decode writes the N16 AUX layout; this AITER's NVFP4 "
                "MLA uses N32 (PR 7987). Use an AITER with PR 8200."
            )
        # One row per live request, plus `pad_row`: padding rows of a batch
        # write their throwaway token there instead of into a request's tail.
        self.pad_row = max_reqs
        if ring is None:
            ring = Nvfp4HpRing(max_reqs + 1, layers, head_dim, device=self.device)
        if (ring.num_reqs, ring.layers, ring.head_dim) != (
            max_reqs + 1,
            layers,
            head_dim,
        ):
            raise ValueError(
                f"NVFP4 tail must be {max_reqs + 1} rows x {layers} layers x "
                f"{head_dim}, got {ring.num_reqs} x {ring.layers} x {ring.head_dim}."
            )
        self.ring = ring
        if self.fused:
            from atom.model_ops.attentions.nvfp4_q_triton import warm

            warm(self.device)
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
        """
        gather, pages, tile_ids, valid = [], [], [], []
        ring_rows, ring_slots, ring_tokens = [], [], []
        lane = np.arange(16)
        token0 = 0
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
        if plan["ring_rows"].numel():
            self.ring.buf[plan["ring_rows"], layer, plan["ring_slots"]] = kv[
                plan["ring_tokens"]
            ].to(self.ring.buf.dtype)

    def publish_prefill_split(
        self, layer: int, k_nope: torch.Tensor, k_rope: torch.Tensor, plan: dict
    ) -> None:
        """`publish_prefill` from the split latent, ``k_nope [T, 512]`` and
        ``k_rope [T, 64]`` after RoPE, without concatenating it: one Triton
        launch gathers, quantizes and writes the tiles (`prefill_quantize`)."""
        if plan["pages"].numel():
            from atom.model_ops.attentions.nvfp4_ring_triton import prefill_quantize

            prefill_quantize(
                k_nope,
                k_rope,
                plan["gather"],
                plan["pages"],
                plan["tile_ids"],
                plan["valid"],
                self.kv_nope[layer],
                self.kv_aux[layer],
            )
        if plan["ring_rows"].numel():
            tokens = plan["ring_tokens"]
            self.ring.buf[plan["ring_rows"], layer, plan["ring_slots"]] = torch.cat(
                (k_nope[tokens], k_rope[tokens]), dim=-1
            ).to(self.ring.buf.dtype)

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
        if self.fused:
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

    def prep_decode(
        self,
        layer,
        q_nope,
        q_rope,
        req_ids,
        k_nope,
        k_rope,
        positions,
        slots,
        cos=None,
        sin=None,
        q_planes=None,
    ):
        """Quantize Q and write this step's K into the tail and pages in one
        launch (`decode_prep`). RoPE is applied in the kernel when the cos/sin
        caches are given, else both RoPE inputs are already roped. Returns the
        quantized ``(q_nope, q_aux)`` for `decode`. Graph-safe."""
        from atom.model_ops.attentions.nvfp4_q_triton import decode_prep

        return decode_prep(
            q_nope,
            q_rope,
            self.ring.buf[:, layer],
            req_ids,
            k_nope,
            k_rope,
            slots,
            self.kv_nope[layer],
            self.kv_aux[layer],
            positions,
            cos,
            sin,
            q_planes=q_planes,
        )

    def decode(
        self,
        layer: int,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        csr: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        sm_scale: float,
    ) -> torch.Tensor:
        """Returns ``[B, 128, 512]`` BF16. ``q`` is BF16 ``[B, 128, 576]``, or
        already quantized as ``(q_nope [B, 128, 256], q_aux [B, 128, 128])``.

        ``csr`` is the step's ``(indptr[B+1], indices, last[B])``, built once
        per step by the metadata builder and shared by every layer.
        """
        from aiter.ops.mla_nvfp4 import mla_decode_nvfp4_fwd, mla_nvfp4_quantize_q

        if isinstance(q, tuple):
            q_nope, q_aux = q
            want = ((NVFP4_HEADS, 256), (NVFP4_HEADS, 128))
            got = (tuple(q_nope.shape[1:]), tuple(q_aux.shape[1:]))
        else:
            want = ((NVFP4_HEADS, self.head_dim),)
            got = (tuple(q.shape[1:]),)
        if got != want:
            raise RuntimeError(f"NVFP4 decode wants Q {want}, got {got}.")
        batch = (q[0] if isinstance(q, tuple) else q).shape[0]
        indptr, indices, last = csr
        if indptr.shape[0] != batch + 1 or last.shape[0] != batch:
            raise RuntimeError(
                f"NVFP4 CSR is for {last.shape[0]} rows, the query has {batch}."
            )
        if not isinstance(q, tuple):
            q_nope, q_aux = mla_nvfp4_quantize_q(q)
        out = torch.empty(
            batch, NVFP4_HEADS, 512, dtype=torch.bfloat16, device=self.device
        )
        lse = torch.empty(batch, NVFP4_HEADS, dtype=torch.float32, device=self.device)
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
        return out

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
        if cache.fused:
            cache.publish_prefill_split(
                attn.nvfp4_layer,
                k_nope.reshape(-1, attn.kv_lora_rank),
                k_rope.reshape(-1, attn.qk_rope_head_dim),
                plan,
            )
            return
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


def _one_launch_ok(cache, q_nope, k_nope) -> bool:
    """Whether Q quantization and the cache write can share one launch."""
    return (
        cache.fused
        and q_nope.dtype == torch.bfloat16
        and k_nope.dtype == torch.bfloat16
    )


def _fused_rope_ok(cache, rope, q_nope, k_nope, k_rope) -> bool:
    """Whether that launch can also apply the RoPE: the interleaved,
    front-half BF16-cache kind its kernels reproduce."""
    cos = getattr(rope, "cos_cache", None)
    return (
        _one_launch_ok(cache, q_nope, k_nope)
        and not getattr(rope, "is_neox_style", True)
        and cos is not None
        and cos.dtype == torch.bfloat16
        and cos.shape[-1] * 2 == k_rope.shape[-1]
        and k_rope.dtype == torch.bfloat16
    )


def kup_fusable(attn) -> bool:
    """Whether this decode should skip the BF16 K-up GEMM and hand
    `try_nvfp4_decode` the un-absorbed ``q_nope [B, H, 128]`` plus ``W_K``:
    the fused NVFP4 decode is on and the K-up is the BF16 absorb."""
    cache = getattr(attn, "nvfp4", None)
    return cache is not None and cache.fused and attn.W_K_scale is None


def try_nvfp4_decode(
    attn, q_nope, q_rope, k_nope, k_rope, positions, attn_metadata, w_k=None
) -> torch.Tensor | None:
    """Decode one token per row. ``None`` leaves the FP8 path in place.

    ``q_rope`` and ``k_rope`` are still pre-RoPE. They are rotated on copies,
    so the FP8 writer still sees the original tensors when this returns None.

    With ``w_k`` (`kup_fusable`), ``q_nope`` is the un-absorbed ``[B, H, 128]``
    and the K-up GEMM writes the NVFP4 NoPE Q directly (`kup_quantize_q_nope`).

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
    slots = attn_metadata.slot_mapping.reshape(-1)[:rows]
    q_nope = _as_bf16(q_nope, attn._q_scale if q_nope.element_size() == 1 else None)
    rope = attn.rotary_emb
    k_nope_rows = k_nope[:rows].reshape(rows, attn.kv_lora_rank)
    q_planes = None
    if w_k is not None:
        from atom.model_ops.attentions.nvfp4_q_triton import kup_quantize_q_nope

        q_planes = kup_quantize_q_nope(q_nope[:rows], w_k)
    q_in = None if q_planes is not None else q_nope
    if _fused_rope_ok(cache, rope, q_nope, k_nope, k_rope):
        # One launch: RoPE, Q quantization, tail append and the FP4 write.
        q = cache.prep_decode(
            attn.nvfp4_layer,
            q_in,
            q_rope,
            tail_rows[:rows],
            k_nope_rows,
            k_rope[:rows].reshape(rows, attn.qk_rope_head_dim),
            positions,
            slots,
            rope.cos_cache,
            rope.sin_cache,
            q_planes=q_planes,
        )
    elif _one_launch_ok(cache, q_nope, k_nope):
        # The model's RoPE on copies, then one launch for Q and K (no concat).
        q_pe = q_rope.clone()
        k_pe = k_rope[:rows].clone()
        rope(positions, q_pe, k_pe)
        q = cache.prep_decode(
            attn.nvfp4_layer,
            q_in,
            q_pe,
            tail_rows[:rows],
            k_nope_rows,
            k_pe.reshape(rows, attn.qk_rope_head_dim),
            positions,
            slots,
            q_planes=q_planes,
        )
    elif q_planes is not None:
        raise RuntimeError("NVFP4 fused K-up needs the one-launch decode path.")
    else:
        q_pe = q_rope.clone()
        k_pe = k_rope[:rows].clone()
        rope(positions, q_pe, k_pe)
        if cache.fused:
            from atom.model_ops.attentions.nvfp4_q_triton import quantize_q

            q = quantize_q(q_nope, q_pe)
        else:
            q = torch.cat((q_nope, q_pe), dim=-1).contiguous()
        kv = _latent(k_nope[:rows], k_pe, attn.kv_lora_rank, attn.qk_rope_head_dim)
        cache.publish_decode(attn.nvfp4_layer, tail_rows[:rows], kv, positions, slots)
    indptr, indices, last = csr
    out = cache.decode(
        attn.nvfp4_layer, q, (indptr[: rows + 1], indices, last[:rows]), attn.scale
    )
    return attn._v_up_proj_and_o_proj(out)
