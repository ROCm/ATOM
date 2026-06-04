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
"""

from __future__ import annotations

import logging
import operator

import torch

logger = logging.getLogger("atom")


class ATOMKVByteCodec:
    """Per-block byte mover between paged GPU KV tensors and flat buffers."""

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

        first = self._segments[0]
        self.num_blocks: int = int(first.shape[0])
        self._device = first.device
        for seg in self._segments:
            if seg.device != self._device:
                raise ValueError(
                    "ATOMKVByteCodec: all KV tensors must be on the same device"
                )
            if int(seg.shape[0]) != self.num_blocks:
                raise ValueError(
                    "ATOMKVByteCodec: all KV tensors must have the same block count"
                )

        # Bytes for one block of each segment (block is dim 0).
        self._seg_block_bytes: list[int] = [
            int(t[0].numel()) * t.element_size() for t in self._segments
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
                    "fused chunk-major staging disabled",
                    exc_info=True,
                )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def has_fused_chunk_major_staging(self) -> bool:
        return self._fused_kv_staging is not None

    # -- helpers ----------------------------------------------------------
    def _segment_bases(self, nblocks: int) -> list[int]:
        bases = []
        acc = 0
        for nb in self._seg_block_bytes:
            bases.append(acc)
            acc += nb * nblocks
        return bases

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

    @staticmethod
    def _segment_bytes_matrix(seg: torch.Tensor) -> torch.Tensor:
        if not seg.is_contiguous():
            raise RuntimeError("ATOMKVByteCodec: segment tensor not contiguous")
        return seg.reshape(seg.shape[0], -1).view(torch.uint8)

    # -- public API -------------------------------------------------------
    def gpu_to_device_buffer(
        self,
        device_buf: torch.Tensor,
        block_ids: list[int],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Gather ATOM KV blocks into a flat device staging buffer.

        The staging layout is always segment-major:
        ``[seg0 blocks | seg1 blocks | ...]``. This is the layout consumed by
        the ATOM LMCache connector before it copies the bytes to a
        ``MemoryObj``.
        """
        block_ids = self._normalize_block_ids(block_ids)
        self._validate_device_buf(device_buf, len(block_ids))
        if not block_ids:
            return
        with self._device_ctx():
            stream_ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
            with stream_ctx:
                idx = torch.tensor(block_ids, dtype=torch.long, device=self._device)
                bases = self._segment_bases(len(block_ids))
                for seg, base, nb in zip(self._segments, bases, self._seg_block_bytes):
                    mat = self._segment_bytes_matrix(seg)
                    dst = device_buf[base : base + len(block_ids) * nb].reshape(
                        len(block_ids), nb
                    )
                    torch.index_select(mat, 0, idx, out=dst)

    def device_buffer_to_gpu(
        self,
        device_buf: torch.Tensor,
        block_ids: list[int],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Scatter a segment-major device staging buffer into ATOM KV blocks."""
        block_ids = self._normalize_block_ids(block_ids)
        self._validate_device_buf(device_buf, len(block_ids))
        if not block_ids:
            return
        with self._device_ctx():
            stream_ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
            with stream_ctx:
                idx = torch.tensor(block_ids, dtype=torch.long, device=self._device)
                bases = self._segment_bases(len(block_ids))
                for seg, base, nb in zip(self._segments, bases, self._seg_block_bytes):
                    mat = self._segment_bytes_matrix(seg)
                    src = device_buf[base : base + len(block_ids) * nb].reshape(
                        len(block_ids), nb
                    )
                    mat.index_copy_(0, idx, src)

    def gpu_to_chunk_major_device_buffer(
        self,
        device_buf: torch.Tensor,
        block_id_groups: list[list[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Gather ATOM KV blocks into a chunk-major device staging buffer.

        Layout is MemoryObj-compatible:
        ``[chunk0: seg0 blocks | seg1 blocks | ...][chunk1: ...]``.
        Fused Triton staging is used when available; otherwise this method
        provides a reference implementation for tests.
        """
        groups, flat_block_ids, chunk_block_counts = self._normalize_block_id_groups(
            block_id_groups,
            reject_repeated=True,
        )
        self._validate_device_buf(device_buf, len(flat_block_ids))
        if not flat_block_ids:
            return
        with self._device_ctx():
            stream_ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
            with stream_ctx:
                if self._fused_kv_staging is not None:
                    self._fused_kv_staging.fused_pack_chunk_major(
                        self._segments,
                        self._seg_block_bytes,
                        chunk_block_counts,
                        flat_block_ids,
                        device_buf,
                    )
                    return

                offset = 0
                for block_ids in groups:
                    nblocks = len(block_ids)
                    chunk_nbytes = nblocks * self.bytes_per_block
                    self.gpu_to_device_buffer(
                        device_buf[offset : offset + chunk_nbytes],
                        block_ids,
                        stream=stream,
                    )
                    offset += chunk_nbytes

    def chunk_major_device_buffer_to_gpu(
        self,
        device_buf: torch.Tensor,
        block_id_groups: list[list[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Scatter a chunk-major device staging buffer into ATOM KV blocks."""
        groups, flat_block_ids, chunk_block_counts = self._normalize_block_id_groups(
            block_id_groups,
            reject_repeated=True,
        )
        self._validate_device_buf(device_buf, len(flat_block_ids))
        if not flat_block_ids:
            return
        with self._device_ctx():
            stream_ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
            with stream_ctx:
                if self._fused_kv_staging is not None:
                    self._fused_kv_staging.fused_unpack_chunk_major(
                        device_buf,
                        self._segments,
                        self._seg_block_bytes,
                        chunk_block_counts,
                        flat_block_ids,
                    )
                    return

                offset = 0
                for block_ids in groups:
                    nblocks = len(block_ids)
                    chunk_nbytes = nblocks * self.bytes_per_block
                    self.device_buffer_to_gpu(
                        device_buf[offset : offset + chunk_nbytes],
                        block_ids,
                        stream=stream,
                    )
                    offset += chunk_nbytes


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
