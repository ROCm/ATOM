# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""AITER-layout-aware byte codec between ATOM's paged GPU KV cache and a flat
pinned host buffer (an LMCache ``MemoryObj``'s ``uint8`` tensor).

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
v_scale. The flat per-block layout in the host buffer is::

    [ L0.K | L0.V | L0.kS | L0.vS | L1.K | L1.V | ... ]   (only present segments)

which is self-consistent for store and load (we never reinterpret it).
"""

from __future__ import annotations

import logging
import operator
import os
import threading

import torch

logger = logging.getLogger("atom")


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
        # Byte offset of each segment within one block's flat record.
        self._seg_off: list[int] = []
        acc = 0
        for nb in self._seg_block_bytes:
            self._seg_off.append(acc)
            acc += nb
        self.bytes_per_block: int = acc
        self.layout = os.environ.get("OFFLOAD_CODEC_LAYOUT", "block").lower()
        if self.layout not in ("block", "segment", "segment_indexed"):
            self.layout = "block"
        self._tls = threading.local()
        self._native_stitch = None
        self._native_split = None
        if (
            self.layout == "segment_indexed"
            and os.environ.get("OFFLOAD_NATIVE_STITCH", "0").lower()
            not in ("0", "false", "no", "off")
        ):
            try:
                from atom.kv_transfer.offload import native_stitch

                native_stitch.load_extension()
                self._native_stitch = native_stitch.stitch_chunk_buffers
                self._native_split = native_stitch.split_request_buffer
            except Exception:
                logger.warning(
                    "ATOMKVByteCodec: native stitch unavailable; using torch stitch",
                    exc_info=True,
                )

    @property
    def segments_per_block(self) -> int:
        return len(self._segments)

    @property
    def device(self) -> torch.device:
        return self._device

    def copy_calls_for_blocks(self, nblocks: int) -> int:
        return int(nblocks) * len(self._segments)

    def copy_calls_for_block_ids(self, block_ids: list[int]) -> int:
        if self.layout == "block":
            return self.copy_calls_for_blocks(len(block_ids))
        if self.layout == "segment_indexed":
            return len(self._segments) * 2
        return len(self._segments) * len(list(self._contiguous_runs(block_ids)))

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

    @staticmethod
    def _blocks_bytes_view(
        seg: torch.Tensor,
        block_id: int,
        nblocks: int,
    ) -> torch.Tensor:
        """Flat ``uint8`` view of a contiguous block range (no copy)."""
        blk = seg[block_id : block_id + nblocks]
        if not blk.is_contiguous():
            raise RuntimeError("ATOMKVByteCodec: block range not contiguous")
        return blk.reshape(-1).view(torch.uint8)

    @staticmethod
    def _contiguous_runs(block_ids: list[int]):
        """Yield ``(logical_start, physical_start, run_len)`` for increasing
        physical block-id runs in logical order."""
        if not block_ids:
            return
        logical_start = 0
        physical_start = block_ids[0]
        prev = block_ids[0]
        run_len = 1
        for logical_idx, bid in enumerate(block_ids[1:], start=1):
            if bid == prev + 1:
                prev = bid
                run_len += 1
                continue
            yield logical_start, physical_start, run_len
            logical_start = logical_idx
            physical_start = bid
            prev = bid
            run_len = 1
        yield logical_start, physical_start, run_len

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

    def _validate_host_buf(self, host_buf: torch.Tensor, nblocks: int) -> None:
        if host_buf.dtype != torch.uint8:
            raise TypeError("ATOMKVByteCodec: host_buf must be a uint8 tensor")
        required = int(nblocks) * self.bytes_per_block
        if int(host_buf.numel()) < required:
            raise ValueError(
                "ATOMKVByteCodec: host_buf is too small "
                f"for {nblocks} blocks; need {required} bytes, "
                f"got {int(host_buf.numel())}"
            )

    def stitch_chunk_buffers(
        self,
        dst: torch.Tensor,
        chunk_buffers: list[torch.Tensor],
        chunk_block_counts: list[int],
    ) -> None:
        """CPU-side stitch from per-LMCache-chunk segment-major buffers into one
        request-level segment-major buffer.

        Each stored chunk is laid out as ``[seg0 chunk_blocks | seg1 ...]``. A
        single request-level indexed H2D scatter expects
        ``[seg0 all_blocks | seg1 all_blocks | ...]``.
        """
        if self._native_stitch is not None:
            self._native_stitch(
                dst,
                chunk_buffers,
                chunk_block_counts,
                self._seg_block_bytes,
            )
            return
        total_blocks = sum(chunk_block_counts)
        dst_bases = self._segment_bases(total_blocks)
        src_bases_by_chunk = [
            self._segment_bases(nblocks) for nblocks in chunk_block_counts
        ]
        for seg_idx, (dst_base, nb) in enumerate(
            zip(dst_bases, self._seg_block_bytes)
        ):
            parts = [
                src[
                    bases[seg_idx] : bases[seg_idx] + nblocks * nb
                ]
                for src, bases, nblocks in zip(
                    chunk_buffers, src_bases_by_chunk, chunk_block_counts
                )
            ]
            torch.cat(
                parts,
                out=dst[dst_base : dst_base + total_blocks * nb],
            )

    def split_request_buffer(
        self,
        src: torch.Tensor,
        chunk_buffers: list[torch.Tensor],
        chunk_block_counts: list[int],
    ) -> None:
        """CPU-side inverse of :meth:`stitch_chunk_buffers`.

        ``src`` is one request-level segment-major buffer
        ``[seg0 all_blocks | seg1 all_blocks | ...]``. Each destination chunk
        receives its own segment-major slice
        ``[seg0 chunk_blocks | seg1 chunk_blocks | ...]`` for LMCache storage.
        """
        if self._native_split is not None:
            self._native_split(
                src,
                chunk_buffers,
                chunk_block_counts,
                self._seg_block_bytes,
            )
            return
        total_blocks = sum(chunk_block_counts)
        src_bases = self._segment_bases(total_blocks)
        dst_bases_by_chunk = [
            self._segment_bases(nblocks) for nblocks in chunk_block_counts
        ]
        for seg_idx, (src_base, nb) in enumerate(
            zip(src_bases, self._seg_block_bytes)
        ):
            logical_block_start = 0
            for dst, bases, nblocks in zip(
                chunk_buffers, dst_bases_by_chunk, chunk_block_counts
            ):
                nbytes = nblocks * nb
                if nbytes:
                    dst[
                        bases[seg_idx] : bases[seg_idx] + nbytes
                    ].copy_(
                        src[
                            src_base
                            + logical_block_start * nb : src_base
                            + logical_block_start * nb
                            + nbytes
                        ]
                    )
                logical_block_start += nblocks

    def _tmp_bytes(self, seg: torch.Tensor, nblocks: int) -> torch.Tensor:
        elems = int(seg[0].numel()) * seg.element_size()
        key = (str(seg.device), "uint8", elems, int(nblocks))
        cache = getattr(self._tls, "tmp", None)
        if cache is None:
            cache = {}
            self._tls.tmp = cache
        tmp = cache.get(key)
        if tmp is None:
            tmp = torch.empty((nblocks, elems), dtype=torch.uint8, device=seg.device)
            cache[key] = tmp
        return tmp

    @staticmethod
    def _segment_bytes_matrix(seg: torch.Tensor) -> torch.Tensor:
        if not seg.is_contiguous():
            raise RuntimeError("ATOMKVByteCodec: segment tensor not contiguous")
        return seg.reshape(seg.shape[0], -1).view(torch.uint8)

    # -- public API -------------------------------------------------------
    def gpu_to_host(
        self,
        host_buf: torch.Tensor,
        block_ids: list[int],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """D2H: gather ``block_ids`` from the paged GPU cache into the flat
        pinned ``host_buf`` (uint8, length == len(block_ids) * bytes_per_block)."""
        block_ids = self._normalize_block_ids(block_ids)
        self._validate_host_buf(host_buf, len(block_ids))
        if not block_ids:
            return
        with self._device_ctx():
            stream_ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
            with stream_ctx:
                if self.layout == "segment_indexed":
                    idx = torch.tensor(
                        block_ids, dtype=torch.long, device=self._device
                    )
                    bases = self._segment_bases(len(block_ids))
                    for seg, base, nb in zip(
                        self._segments, bases, self._seg_block_bytes
                    ):
                        mat = self._segment_bytes_matrix(seg)
                        tmp = self._tmp_bytes(seg, len(block_ids))
                        torch.index_select(mat, 0, idx, out=tmp)
                        host_buf[base : base + len(block_ids) * nb].copy_(
                            tmp.reshape(-1), non_blocking=True
                        )
                    return

                if self.layout == "segment":
                    bases = self._segment_bases(len(block_ids))
                    runs = list(self._contiguous_runs(block_ids))
                    for seg, base, nb in zip(
                        self._segments, bases, self._seg_block_bytes
                    ):
                        for logical_start, physical_start, run_len in runs:
                            src = self._blocks_bytes_view(seg, physical_start, run_len)
                            dst = base + logical_start * nb
                            host_buf[dst : dst + run_len * nb].copy_(
                                src, non_blocking=True
                            )
                    return

                for i, bid in enumerate(block_ids):
                    base = i * self.bytes_per_block
                    for seg, off, nb in zip(
                        self._segments, self._seg_off, self._seg_block_bytes
                    ):
                        src = self._block_bytes_view(seg, bid)
                        host_buf[base + off : base + off + nb].copy_(
                            src, non_blocking=True
                        )

    def host_to_gpu(
        self,
        host_buf: torch.Tensor,
        block_ids: list[int],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """H2D: scatter the flat pinned ``host_buf`` back into the paged GPU
        cache at ``block_ids`` (in-place into the real KV tensors)."""
        block_ids = self._normalize_block_ids(block_ids)
        self._validate_host_buf(host_buf, len(block_ids))
        if not block_ids:
            return
        with self._device_ctx():
            stream_ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
            with stream_ctx:
                if self.layout == "segment_indexed":
                    idx = torch.tensor(
                        block_ids, dtype=torch.long, device=self._device
                    )
                    bases = self._segment_bases(len(block_ids))
                    for seg, base, nb in zip(
                        self._segments, bases, self._seg_block_bytes
                    ):
                        mat = self._segment_bytes_matrix(seg)
                        tmp = self._tmp_bytes(seg, len(block_ids))
                        tmp.copy_(
                            host_buf[base : base + len(block_ids) * nb].reshape_as(tmp),
                            non_blocking=True,
                        )
                        mat.index_copy_(0, idx, tmp)
                    return

                if self.layout == "segment":
                    bases = self._segment_bases(len(block_ids))
                    runs = list(self._contiguous_runs(block_ids))
                    for seg, base, nb in zip(
                        self._segments, bases, self._seg_block_bytes
                    ):
                        for logical_start, physical_start, run_len in runs:
                            dst = self._blocks_bytes_view(seg, physical_start, run_len)
                            src = base + logical_start * nb
                            dst.copy_(
                                host_buf[src : src + run_len * nb],
                                non_blocking=True,
                            )
                    return

                for i, bid in enumerate(block_ids):
                    base = i * self.bytes_per_block
                    for seg, off, nb in zip(
                        self._segments, self._seg_off, self._seg_block_bytes
                    ):
                        dst = self._block_bytes_view(seg, bid)
                        dst.copy_(
                            host_buf[base + off : base + off + nb],
                            non_blocking=True,
                        )


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
