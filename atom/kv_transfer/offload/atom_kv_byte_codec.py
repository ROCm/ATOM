# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""AITER-layout-aware byte codec between ATOM's paged GPU KV cache and flat
``uint8`` staging buffers.

Why a byte codec instead of an LMCache ``GPUConnectorInterface`` subclass:
LMCache's ``engine.store/retrieve`` GPU path only emits token-major formats
(``KV_2LTD`` etc.) via ``normalize_kv_and_discover_format``, which only accepts the
clean NHD/HND family and rejects ATOM's **x-packed, head-major** K layout
``(nb, H, D//x, bs, x)`` and strided V ``(nb, H, D, bs)`` (``x = 16 // elem``; verified
``atom/model_ops/attentions/aiter_attention.py:488-502``). NB: this is a *persistent
HBM storage layout*, NOT the transient LDS bank-conflict "swizzle"; we call it "swizzle"
only as loose shorthand. It is also specific to this ATOM aiter path — stock vLLM's aiter
FA backend (``rocm_aiter_fa``) uses the clean token-major ``(2,nb,bs,H,D)`` LMCache handles.
We therefore bypass that path: we store **opaque per-block bytes** (byte-identical
round-trip — the attention kernel reads back its own layout) and drive LMCache only
as a storage tier (``StorageManager`` + ``ChunkedTokenDatabase``).

A whole *block* of any per-layer cache tensor (``t[block_id]``) is contiguous, so a
block's KV is a set of contiguous byte slices: per layer K, V, and (fp8) k_scale,
v_scale. The canonical staging layout for one chunk is segment-major::

    [ all L0.K blocks | all L0.V blocks | all L0.kS blocks | ... ]

and batched transfers concatenate those per-chunk buffers for LMCache MemoryObjs.
The production path requires Triton fused chunk-major staging.
"""

from __future__ import annotations

import logging
import operator

import torch

logger = logging.getLogger("atom")


class ATOMKVByteCodec:
    """Per-block byte mover between paged GPU KV tensors and flat buffers."""

    def __init__(self, kv_caches: dict, num_blocks: int | None = None) -> None:
        """``kv_caches``: ordered ``{layer_name: KVCacheTensor}`` from
        ``register_kv_caches``. We flatten every movable per-layer tensor (K, V,
        and fp8 scales when present) into one ordered segment list.

        Each segment is a contiguous GPU tensor whose first ``num_blocks``
        equal slices are the per-physical-block payloads we copy as raw bytes.
        Two layouts must both work:

        * **Standard MHA/GQA** — block-major ``[num_blocks, ...]`` (e.g. ATOM's
          x-packed K ``(nb, H, D//x, bs, x)`` and strided V), so dim 0 IS the
          block count.
        * **MLA** (DeepSeek R1/V3, Kimi) — a single 576-dim latent cache viewed
          token-major as ``(num_blocks * block_size, 1, 576)`` with no separate
          V/scale tensors, so dim 0 is the *token* count.

        Because the contiguous byte layout is identical (block ``b`` always
        starts at ``b * bytes_per_physical_block``), we don't branch on layout:
        we take ``num_blocks`` explicitly and derive each segment's per-block
        byte stride as ``segment_bytes / num_blocks``. ``num_blocks`` falls back
        to ``segment.shape[0]`` (the block-major assumption) when not supplied,
        preserving the original non-MLA behaviour."""
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

        first = self._segments[0]
        self._device = first.device
        self.num_blocks: int = (
            int(num_blocks) if num_blocks is not None else int(first.shape[0])
        )
        if self.num_blocks <= 0:
            raise ValueError(
                f"ATOMKVByteCodec: num_blocks must be > 0, got {self.num_blocks}"
            )
        for seg in self._segments:
            if seg.device != self._device:
                raise ValueError(
                    "ATOMKVByteCodec: all KV tensors must be on the same device"
                )
            if not seg.is_contiguous():
                raise ValueError(
                    "ATOMKVByteCodec: KV tensors must be contiguous for byte copy"
                )
            if seg.numel() % self.num_blocks != 0:
                raise ValueError(
                    "ATOMKVByteCodec: KV tensor size "
                    f"{seg.numel()} not divisible by num_blocks={self.num_blocks} "
                    f"(shape={tuple(seg.shape)})"
                )

        # Bytes for one physical block of each segment. Works for both
        # block-major (numel = num_blocks * per_block) and token-major MLA
        # (numel = num_blocks * block_size * per_token) because both reduce to
        # the same contiguous per-block stride.
        self._seg_block_bytes: list[int] = [
            (int(t.numel()) // self.num_blocks) * t.element_size()
            for t in self._segments
        ]
        self.bytes_per_block: int = sum(self._seg_block_bytes)
        self._fused_kv_staging = None
        if self._device.type == "cuda":
            try:
                from atom.kv_transfer.offload import triton_kv_staging

                self._fused_kv_staging = triton_kv_staging
            except Exception:
                logger.warning(
                    "ATOMKVByteCodec: Triton KV staging unavailable; "
                    "fused chunk-major staging unavailable",
                    exc_info=True,
                )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def has_fused_chunk_major_staging(self) -> bool:
        return self._fused_kv_staging is not None

    # -- helpers ----------------------------------------------------------
    def _device_ctx(self):
        if self._device.type == "cuda":
            return torch.cuda.device(self._device)
        return _nullctx()

    def _normalize_block_ids(self, block_ids: list[int]) -> list[int]:
        try:
            normalized = [operator.index(bid) for bid in block_ids]
        except TypeError as exc:
            raise ValueError("ATOMKVByteCodec: block_ids must be integers") from exc
        if not normalized:
            return normalized
        min_bid = min(normalized)
        max_bid = max(normalized)
        if min_bid < 0 or max_bid >= self.num_blocks:
            raise ValueError(
                "ATOMKVByteCodec: block id out of range "
                f"[0, {self.num_blocks}); min={min_bid} max={max_bid}"
            )
        return normalized

    def _normalize_block_id_groups(
        self,
        block_id_groups: list[list[int]],
        *,
        reject_repeated: bool,
    ) -> tuple[list[list[int]], list[int], list[int]]:
        groups = [
            self._normalize_block_ids(list(block_ids)) for block_ids in block_id_groups
        ]
        flat = [bid for block_ids in groups for bid in block_ids]
        if reject_repeated and len(set(flat)) != len(flat):
            raise ValueError("ATOMKVByteCodec: duplicate block ids are not supported")
        return groups, flat, [len(block_ids) for block_ids in groups]

    def _validate_device_buf(self, device_buf: torch.Tensor, nblocks: int) -> None:
        if device_buf.dtype != torch.uint8:
            raise TypeError("ATOMKVByteCodec: device_buf must be a uint8 tensor")
        if device_buf.device != self._device:
            raise TypeError(
                "ATOMKVByteCodec: device_buf must be on the KV cache device "
                f"{self._device}, got {device_buf.device}"
            )
        required = int(nblocks) * self.bytes_per_block
        if int(device_buf.numel()) < required:
            raise ValueError(
                "ATOMKVByteCodec: device_buf is too small "
                f"for {nblocks} blocks; need {required} bytes, "
                f"got {int(device_buf.numel())}"
            )

    # -- public API -------------------------------------------------------
    def gpu_to_chunk_major_device_buffer(
        self,
        device_buf: torch.Tensor,
        block_id_groups: list[list[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Gather ATOM KV blocks into a chunk-major device staging buffer.

        Layout is MemoryObj-compatible:
        ``[chunk0: seg0 blocks | seg1 blocks | ...][chunk1: ...]``.
        Fused Triton staging is required.
        """
        _, flat_block_ids, chunk_block_counts = self._normalize_block_id_groups(
            block_id_groups,
            reject_repeated=True,
        )
        self._validate_device_buf(device_buf, len(flat_block_ids))
        if not flat_block_ids:
            return
        if self._fused_kv_staging is None:
            raise RuntimeError(
                "ATOMKVByteCodec requires Triton fused chunk-major staging"
            )
        with self._device_ctx():
            stream_ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
            with stream_ctx:
                self._fused_kv_staging.fused_pack_chunk_major(
                    self._segments,
                    self._seg_block_bytes,
                    chunk_block_counts,
                    flat_block_ids,
                    device_buf,
                )

    def chunk_major_device_buffer_to_gpu(
        self,
        device_buf: torch.Tensor,
        block_id_groups: list[list[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Scatter a chunk-major device staging buffer into ATOM KV blocks."""
        _, flat_block_ids, chunk_block_counts = self._normalize_block_id_groups(
            block_id_groups,
            reject_repeated=True,
        )
        self._validate_device_buf(device_buf, len(flat_block_ids))
        if not flat_block_ids:
            return
        if self._fused_kv_staging is None:
            raise RuntimeError(
                "ATOMKVByteCodec requires Triton fused chunk-major staging"
            )
        with self._device_ctx():
            stream_ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
            with stream_ctx:
                self._fused_kv_staging.fused_unpack_chunk_major(
                    device_buf,
                    self._segments,
                    self._seg_block_bytes,
                    chunk_block_counts,
                    flat_block_ids,
                )


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
