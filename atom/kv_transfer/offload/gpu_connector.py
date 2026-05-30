# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""AITER-layout-aware byte codec between ATOM's paged GPU KV cache and a flat
pinned host buffer (an LMCache ``MemoryObj``'s ``uint8`` tensor).

Why a byte codec instead of an LMCache ``GPUConnectorInterface`` subclass:
LMCache's ``engine.store/retrieve`` GPU path only emits token-major formats
(``KV_2LTD`` etc.) via ``normalize_kv_and_discover_format``, which rejects
AITER's swizzled K layout ``(nb, H, D//x, bs, x)`` and strided V ``(nb, H, D, bs)``.
We therefore bypass that path: we store **opaque per-block bytes** (byte-identical
round-trip — the attention kernel reads back its own layout) and drive LMCache only
as a storage tier (``StorageManager`` + ``ChunkedTokenDatabase``).

A whole *block* of any per-layer cache tensor (``t[block_id]``) is contiguous, so a
block's KV is a set of contiguous byte slices: per layer K, V, and (fp8) k_scale,
v_scale. The flat per-block layout in the host buffer is::

    [ L0.K | L0.V | L0.kS | L0.vS | L1.K | L1.V | ... ]   (only present segments)

which is self-consistent for store and load (we never reinterpret it).
"""

from __future__ import annotations

import torch


class ATOMKVByteCodec:
    """Per-block byte mover between paged GPU KV tensors and a flat host buffer."""

    def __init__(self, kv_caches: dict) -> None:
        """``kv_caches``: ordered ``{layer_name: KVCacheTensor}`` from
        ``register_kv_caches``. We flatten every movable per-layer tensor (K, V,
        and fp8 scales when present) into one ordered segment list. Each segment
        is a GPU tensor shaped ``[num_blocks, ...]``; segment[block_id] is a
        contiguous block slice we copy as raw bytes."""
        self._segments: list[torch.Tensor] = []
        for _name, kvt in kv_caches.items():
            for t in (
                getattr(kvt, "k_cache", None),
                getattr(kvt, "v_cache", None),
                getattr(kvt, "k_scale", None),
                getattr(kvt, "v_scale", None),
            ):
                if t is not None and isinstance(t, torch.Tensor) and t.numel() > 0:
                    self._segments.append(t)

        if not self._segments:
            raise ValueError("ATOMKVByteCodec: no movable KV tensors registered")

        # Bytes for one block of each segment (block is dim 0).
        self._seg_block_bytes: list[int] = [
            int(t[0].numel()) * t.element_size() for t in self._segments
        ]
        # Byte offset of each segment within one block's flat record.
        self._seg_off: list[int] = []
        acc = 0
        for nb in self._seg_block_bytes:
            self._seg_off.append(acc)
            acc += nb
        self.bytes_per_block: int = acc
        self.num_blocks: int = int(self._segments[0].shape[0])

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _block_bytes_view(seg: torch.Tensor, block_id: int) -> torch.Tensor:
        """Flat ``uint8`` view of one contiguous block slice (no copy)."""
        blk = seg[block_id]
        if not blk.is_contiguous():
            # Block slices of the paged cache are contiguous in practice; guard
            # anyway. A non-contiguous block would break in-place H2D, so fail loud.
            raise RuntimeError("ATOMKVByteCodec: block slice not contiguous")
        return blk.reshape(-1).view(torch.uint8)

    # -- public API -------------------------------------------------------
    def gpu_to_host(
        self,
        host_buf: torch.Tensor,
        block_ids: list[int],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """D2H: gather ``block_ids`` from the paged GPU cache into the flat
        pinned ``host_buf`` (uint8, length == len(block_ids) * bytes_per_block)."""
        ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
        with ctx:
            for i, bid in enumerate(block_ids):
                base = i * self.bytes_per_block
                for seg, off, nb in zip(
                    self._segments, self._seg_off, self._seg_block_bytes
                ):
                    src = self._block_bytes_view(seg, bid)
                    host_buf[base + off : base + off + nb].copy_(src, non_blocking=True)

    def host_to_gpu(
        self,
        host_buf: torch.Tensor,
        block_ids: list[int],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """H2D: scatter the flat pinned ``host_buf`` back into the paged GPU
        cache at ``block_ids`` (in-place into the real KV tensors)."""
        ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
        with ctx:
            for i, bid in enumerate(block_ids):
                base = i * self.bytes_per_block
                for seg, off, nb in zip(
                    self._segments, self._seg_off, self._seg_block_bytes
                ):
                    dst = self._block_bytes_view(seg, bid)
                    dst.copy_(host_buf[base + off : base + off + nb], non_blocking=True)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
