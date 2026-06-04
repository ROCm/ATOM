# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM LMCache raw-byte connector for offload.

This module lets ATOM use LMCache ``CacheEngine.store()`` /
``CacheEngine.retrieve()`` without adopting LMCache's vLLM token-major KV
layout. LMCache still owns chunking, keys, lookup pins, and storage-manager
orchestration. ATOM owns how a token range maps to AITER KV-cache blocks and
how those blocks are packed as opaque bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any

import torch

from atom.kv_transfer.offload.atom_kv_byte_codec import ATOMKVByteCodec
from atom.kv_transfer.offload.atom_lmcache_staging import (
    _StagingSlot,
    _ThreadTransferState,
    _env_flag,
    _env_int,
    _env_optional_int,
)


def _cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


@dataclass(frozen=True)
class _TransferChunk:
    memory_obj: Any
    block_ids: list[int]
    tensor: torch.Tensor
    nbytes: int


@dataclass(frozen=True)
class _TransferGroup:
    chunks: list[_TransferChunk]
    nbytes: int


class ATOMLMCacheGPUConnector:
    """LMCache GPUConnectorInterface for ATOM's opaque KV-block byte layout."""

    def __init__(
        self,
        codec: ATOMKVByteCodec,
        block_size: int,
        *,
        chunk_size: int | None = None,
    ) -> None:
        self.codec = codec
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("ATOM LMCache connector: block_size must be > 0")
        self.chunk_size = int(chunk_size if chunk_size is not None else block_size)
        if self.chunk_size <= 0:
            raise ValueError("ATOM LMCache connector: chunk_size must be > 0")
        if self.chunk_size % self.block_size != 0:
            raise ValueError(
                "LMCache chunk size must be divisible by ATOM KV block size: "
                f"chunk_size={self.chunk_size}, block_size={self.block_size}"
            )
        self._blocks_per_lmcache_chunk = self.chunk_size // self.block_size
        self._gpu_staging_chunk_bytes = (
            self._blocks_per_lmcache_chunk * self.codec.bytes_per_block
        )
        if self._gpu_staging_chunk_bytes <= 0:
            raise ValueError(
                "ATOM LMCache connector: GPU staging chunk bytes must be > 0"
            )
        self.device = torch.device(codec.device)
        self._tls = threading.local()
        requested_group_chunks = _env_int("OFFLOAD_GPU_STAGING_CHUNKS", 2)
        max_staging_bytes = _env_optional_int("OFFLOAD_GPU_STAGING_MAX_BYTES")
        if max_staging_bytes is not None:
            if max_staging_bytes < self._gpu_staging_chunk_bytes:
                raise ValueError(
                    "OFFLOAD_GPU_STAGING_MAX_BYTES must be at least one "
                    "LMCache chunk: "
                    f"max_bytes={max_staging_bytes}, "
                    f"chunk_bytes={self._gpu_staging_chunk_bytes}"
                )
            requested_group_chunks = min(
                requested_group_chunks,
                max_staging_bytes // self._gpu_staging_chunk_bytes,
            )
        self._staging_group_chunks = max(1, int(requested_group_chunks))
        self._gpu_staging_capacity_bytes = (
            self._staging_group_chunks * self._gpu_staging_chunk_bytes
        )
        self._staging_slots = _env_int("OFFLOAD_GPU_STAGING_SLOTS", 1)
        self._release_gpu_staging_after_transfer = _env_flag(
            "OFFLOAD_RELEASE_GPU_STAGING_AFTER_TRANSFER"
        )

    @property
    def staging_slots(self) -> int:
        return self._staging_slots

    @property
    def gpu_staging_chunk_bytes(self) -> int:
        return self._gpu_staging_chunk_bytes

    @property
    def gpu_staging_group_chunks(self) -> int:
        return self._staging_group_chunks

    @property
    def gpu_staging_capacity_bytes(self) -> int:
        return self._gpu_staging_capacity_bytes

    @property
    def release_gpu_staging_after_transfer(self) -> bool:
        return self._release_gpu_staging_after_transfer

    def _set_last_transfer_stats(
        self,
        *,
        chunks: int = 0,
        max_chunk_bytes: int = 0,
        groups: int = 0,
        max_group_bytes: int = 0,
        total_bytes: int = 0,
        pack_ms: float = 0.0,
        copy_ms: float = 0.0,
        sync_ms: float = 0.0,
        transfer_ms: float = 0.0,
    ) -> None:
        effective_gbps = 0.0
        if transfer_ms > 0 and total_bytes > 0:
            effective_gbps = total_bytes / (transfer_ms * 1_000_000.0)
        self._tls.last_transfer_stats = {
            "chunks": int(chunks),
            "max_chunk_bytes": int(max_chunk_bytes),
            "groups": int(groups),
            "max_group_bytes": int(max_group_bytes),
            "total_bytes": int(total_bytes),
            "gpu_staging_chunk_bytes": self._gpu_staging_chunk_bytes,
            "gpu_staging_group_chunks": self._staging_group_chunks,
            "gpu_staging_capacity_bytes": self._gpu_staging_capacity_bytes,
            "gpu_staging_slots": self._staging_slots,
            "pack_ms": float(pack_ms),
            "copy_ms": float(copy_ms),
            "sync_ms": float(sync_ms),
            "transfer_ms": float(transfer_ms),
            "effective_gbps": float(effective_gbps),
        }

    def reset_transfer_stats(self) -> None:
        self._set_last_transfer_stats()

    def last_transfer_stats(self) -> dict[str, int | float]:
        return dict(getattr(self._tls, "last_transfer_stats", {}))

    def _use_cuda(self) -> bool:
        return self.device.type == "cuda"

    def _thread_state(self) -> _ThreadTransferState:
        states = getattr(self._tls, "states", None)
        if states is None:
            states = {}
            self._tls.states = states
        key = str(self.device)
        state = states.get(key)
        if state is None:
            state = _ThreadTransferState(
                self.device,
                self._use_cuda(),
                self._staging_slots,
            )
            states[key] = state
        return state

    def _ensure_slot(self, slot: _StagingSlot, nbytes: int) -> torch.Tensor:
        nbytes = int(nbytes)
        if nbytes > self._gpu_staging_capacity_bytes:
            raise RuntimeError(
                "ATOM LMCache connector internal error: transfer group exceeds "
                "bounded GPU staging capacity: "
                f"nbytes={nbytes}, capacity={self._gpu_staging_capacity_bytes}"
            )
        if (
            slot.tensor is None
            or int(slot.tensor.numel()) != self._gpu_staging_capacity_bytes
        ):
            slot.tensor = torch.empty(
                (self._gpu_staging_capacity_bytes,),
                dtype=torch.uint8,
                device=self.device,
            )
            slot.free_event_valid = False
        return slot.tensor[:nbytes]

    def _next_slot(self, state: _ThreadTransferState) -> _StagingSlot:
        slot = state.slots[state.next_slot % len(state.slots)]
        state.next_slot += 1
        return slot

    def _release_slot_if_requested(self, slot: _StagingSlot) -> None:
        if not self._release_gpu_staging_after_transfer:
            return
        slot.tensor = None
        slot.free_event_valid = False

    def _assert_fused_chunk_major_available(self) -> None:
        if self._use_cuda() and self.codec.has_fused_chunk_major_staging:
            return
        raise RuntimeError(
            "ATOM LMCache connector requires Triton fused chunk-major staging; "
            "ensure KV tensors are on CUDA/HIP and the Triton staging kernel "
            "loads successfully"
        )

    def _memory_tensor(self, memory_obj: Any, nbytes: int) -> torch.Tensor:
        tensor = getattr(memory_obj, "tensor", None)
        if tensor is None and hasattr(memory_obj, "get_tensor"):
            tensor = memory_obj.get_tensor(0)
        if tensor is None:
            raise RuntimeError("ATOM LMCache connector: invalid MemoryObj tensor")
        if tensor.dtype != torch.uint8:
            raise TypeError(
                "ATOM LMCache connector: MemoryObj tensor must be uint8, "
                f"got {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise RuntimeError(
                "ATOM LMCache connector: MemoryObj tensor not contiguous"
            )
        flat = tensor.reshape(-1)
        if int(flat.numel()) < int(nbytes):
            raise ValueError(
                "ATOM LMCache connector: MemoryObj tensor is too small "
                f"for {nbytes} bytes; got {int(flat.numel())}"
            )
        return flat[: int(nbytes)]

    def _range_block_ids(
        self,
        all_block_ids: list[int],
        start: int,
        end: int,
    ) -> list[int]:
        start = int(start)
        end = int(end)
        if start < 0 or end < start:
            raise ValueError(
                f"invalid LMCache token range for ATOM KV blocks: {start}:{end}"
            )
        if start % self.block_size != 0:
            raise ValueError(
                "LMCache chunk start must be ATOM block-aligned: "
                f"start={start}, block_size={self.block_size}"
            )
        start_block = start // self.block_size
        end_block = _cdiv(end, self.block_size)
        if end_block > len(all_block_ids):
            raise ValueError(
                "LMCache token range exceeds ATOM block table: "
                f"range={start}:{end}, needed_blocks={end_block}, "
                f"available_blocks={len(all_block_ids)}"
            )
        return list(all_block_ids[start_block:end_block])

    def _ranges_to_block_ids(
        self,
        starts: list[int],
        ends: list[int],
        **kwargs,
    ) -> list[list[int]]:
        block_ids = kwargs.get("block_ids")
        if block_ids is None:
            raise ValueError("ATOM LMCache connector requires block_ids")
        all_block_ids = [int(bid) for bid in block_ids]
        return [
            self._range_block_ids(all_block_ids, start, end)
            for start, end in zip(starts, ends, strict=True)
        ]

    def _iter_transfer_chunks(
        self,
        memory_objs: list[Any],
        block_id_groups: list[list[int]],
    ) -> list[_TransferChunk]:
        chunks: list[_TransferChunk] = []
        for memory_obj, block_ids in zip(memory_objs, block_id_groups, strict=True):
            block_count = len(block_ids)
            if block_count == 0:
                continue
            nbytes = block_count * self.codec.bytes_per_block
            if nbytes > self._gpu_staging_chunk_bytes:
                raise ValueError(
                    "ATOM LMCache connector: single MemoryObj exceeds bounded "
                    "GPU staging chunk capacity; caller must pass LMCache "
                    "chunk-sized ranges: "
                    f"nbytes={nbytes}, capacity={self._gpu_staging_chunk_bytes}, "
                    f"blocks={block_count}, max_blocks="
                    f"{self._blocks_per_lmcache_chunk}, chunk_size="
                    f"{self.chunk_size}, block_size={self.block_size}"
                )
            chunks.append(
                _TransferChunk(
                    memory_obj=memory_obj,
                    block_ids=block_ids,
                    tensor=self._memory_tensor(memory_obj, nbytes),
                    nbytes=nbytes,
                )
            )
        return chunks

    def _iter_transfer_groups(
        self,
        chunks: list[_TransferChunk],
    ) -> list[_TransferGroup]:
        groups: list[_TransferGroup] = []
        current: list[_TransferChunk] = []
        current_bytes = 0
        for chunk in chunks:
            would_exceed_count = len(current) >= self._staging_group_chunks
            would_exceed_bytes = (
                current_bytes + chunk.nbytes > self._gpu_staging_capacity_bytes
            )
            if current and (would_exceed_count or would_exceed_bytes):
                groups.append(_TransferGroup(chunks=current, nbytes=current_bytes))
                current = []
                current_bytes = 0
            current.append(chunk)
            current_bytes += chunk.nbytes
        if current:
            groups.append(_TransferGroup(chunks=current, nbytes=current_bytes))
        return groups

    def _record_transfer_stats(
        self,
        chunks: list[_TransferChunk],
        groups: list[_TransferGroup] | None = None,
        *,
        pack_ms: float = 0.0,
        copy_ms: float = 0.0,
        sync_ms: float = 0.0,
        transfer_ms: float = 0.0,
    ) -> None:
        if groups is None:
            groups = []
        total_bytes = sum(chunk.nbytes for chunk in chunks)
        self._set_last_transfer_stats(
            chunks=len(chunks),
            max_chunk_bytes=max((chunk.nbytes for chunk in chunks), default=0),
            groups=len(groups),
            max_group_bytes=max((group.nbytes for group in groups), default=0),
            total_bytes=total_bytes,
            pack_ms=pack_ms,
            copy_ms=copy_ms,
            sync_ms=sync_ms,
            transfer_ms=transfer_ms,
        )

    @staticmethod
    def _group_block_ids(group: _TransferGroup) -> list[list[int]]:
        return [chunk.block_ids for chunk in group.chunks]

    @staticmethod
    def _slice_to_memory_objs(group: _TransferGroup, src_buf: torch.Tensor) -> None:
        offset = 0
        for chunk in group.chunks:
            chunk.tensor.copy_(
                src_buf[offset : offset + chunk.nbytes],
                non_blocking=chunk.tensor.device.type != "cpu",
            )
            offset += chunk.nbytes

    @staticmethod
    def _memory_objs_to_slice(group: _TransferGroup, dst_buf: torch.Tensor) -> None:
        offset = 0
        for chunk in group.chunks:
            dst_buf[offset : offset + chunk.nbytes].copy_(
                chunk.tensor,
                non_blocking=chunk.tensor.device.type != "cpu",
            )
            offset += chunk.nbytes

    @staticmethod
    def _remember_slot(used_slots: list[_StagingSlot], slot: _StagingSlot) -> None:
        if not any(existing is slot for existing in used_slots):
            used_slots.append(slot)

    def _release_slots_if_requested(self, used_slots: list[_StagingSlot]) -> None:
        if not self._release_gpu_staging_after_transfer:
            return
        for slot in used_slots:
            self._release_slot_if_requested(slot)

    def from_gpu(self, memory_obj: Any, start: int, end: int, **kwargs) -> None:
        self.batched_from_gpu([memory_obj], [start], [end], **kwargs)

    def to_gpu(self, memory_obj: Any, start: int, end: int, **kwargs) -> None:
        self.batched_to_gpu([memory_obj], [start], [end], **kwargs)

    def batched_from_gpu(
        self,
        memory_objs: list[Any],
        starts: list[int],
        ends: list[int],
        **kwargs,
    ) -> None:
        """Pack ATOM KV blocks to LMCache MemoryObjs via bounded staging."""
        if not (len(memory_objs) == len(starts) == len(ends)):
            raise ValueError("memory_objs, starts, and ends must have equal length")
        block_id_groups = self._ranges_to_block_ids(starts, ends, **kwargs)
        if not memory_objs:
            self._set_last_transfer_stats()
            return

        state = self._thread_state()
        chunks = self._iter_transfer_chunks(memory_objs, block_id_groups)
        groups = self._iter_transfer_groups(chunks)
        self._record_transfer_stats(chunks, groups)
        if not chunks:
            return

        self._assert_fused_chunk_major_available()
        used_slots: list[_StagingSlot] = []
        pack_ms = 0.0
        copy_ms = 0.0
        sync_ms = 0.0
        t_total0 = time.perf_counter()
        try:
            for group in groups:
                slot = self._next_slot(state)
                self._remember_slot(used_slots, slot)
                device_buf = self._ensure_slot(slot, group.nbytes)
                if slot.free_event_valid:
                    state.pack_stream.wait_event(slot.free_event)
                t0 = time.perf_counter()
                with state.stream_ctx(state.pack_stream):
                    self.codec.gpu_to_chunk_major_device_buffer(
                        device_buf,
                        self._group_block_ids(group),
                        stream=state.pack_stream,
                    )
                pack_ms += (time.perf_counter() - t0) * 1000
                slot.ready_event.record(state.pack_stream)
                state.copy_stream.wait_event(slot.ready_event)
                t0 = time.perf_counter()
                with state.stream_ctx(state.copy_stream):
                    self._slice_to_memory_objs(group, device_buf)
                copy_ms += (time.perf_counter() - t0) * 1000
                slot.free_event.record(state.copy_stream)
                slot.free_event_valid = True
            t0 = time.perf_counter()
            state.copy_stream.synchronize()
            sync_ms += (time.perf_counter() - t0) * 1000
        except Exception:
            for slot in used_slots:
                slot.free_event_valid = False
            raise
        finally:
            self._release_slots_if_requested(used_slots)
        self._record_transfer_stats(
            chunks,
            groups,
            pack_ms=pack_ms,
            copy_ms=copy_ms,
            sync_ms=sync_ms,
            transfer_ms=(time.perf_counter() - t_total0) * 1000,
        )

    def batched_to_gpu(
        self,
        memory_objs: list[Any] | None = None,
        starts: list[int] | None = None,
        ends: list[int] | None = None,
        **kwargs,
    ) -> None:
        """Load LMCache MemoryObjs back into ATOM KV blocks via bounded staging."""
        if memory_objs is None or starts is None or ends is None:
            raise ValueError("memory_objs, starts, and ends are required")
        if not (len(memory_objs) == len(starts) == len(ends)):
            raise ValueError("memory_objs, starts, and ends must have equal length")
        block_id_groups = self._ranges_to_block_ids(starts, ends, **kwargs)
        if not memory_objs:
            self._set_last_transfer_stats()
            return

        state = self._thread_state()
        chunks = self._iter_transfer_chunks(memory_objs, block_id_groups)
        groups = self._iter_transfer_groups(chunks)
        self._record_transfer_stats(chunks, groups)
        if not chunks:
            return

        self._assert_fused_chunk_major_available()
        used_slots: list[_StagingSlot] = []
        copy_ms = 0.0
        pack_ms = 0.0
        sync_ms = 0.0
        t_total0 = time.perf_counter()
        try:
            for group in groups:
                slot = self._next_slot(state)
                self._remember_slot(used_slots, slot)
                device_buf = self._ensure_slot(slot, group.nbytes)
                if slot.free_event_valid:
                    state.copy_stream.wait_event(slot.free_event)
                t0 = time.perf_counter()
                with state.stream_ctx(state.copy_stream):
                    self._memory_objs_to_slice(group, device_buf)
                copy_ms += (time.perf_counter() - t0) * 1000
                slot.ready_event.record(state.copy_stream)
                state.pack_stream.wait_event(slot.ready_event)
                t0 = time.perf_counter()
                with state.stream_ctx(state.pack_stream):
                    self.codec.chunk_major_device_buffer_to_gpu(
                        device_buf,
                        self._group_block_ids(group),
                        stream=state.pack_stream,
                    )
                pack_ms += (time.perf_counter() - t0) * 1000
                slot.free_event.record(state.pack_stream)
                slot.free_event_valid = True
            t0 = time.perf_counter()
            state.pack_stream.synchronize()
            sync_ms += (time.perf_counter() - t0) * 1000
        except Exception:
            for slot in used_slots:
                slot.free_event_valid = False
            raise
        finally:
            self._release_slots_if_requested(used_slots)
        self._record_transfer_stats(
            chunks,
            groups,
            pack_ms=pack_ms,
            copy_ms=copy_ms,
            sync_ms=sync_ms,
            transfer_ms=(time.perf_counter() - t_total0) * 1000,
        )
