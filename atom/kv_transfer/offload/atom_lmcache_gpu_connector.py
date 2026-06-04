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
from typing import Any, Callable

import torch

from atom.kv_transfer.offload.atom_kv_byte_codec import ATOMKVByteCodec
from atom.kv_transfer.offload.atom_lmcache_staging import (
    _StagingBuffer,
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


@dataclass(frozen=True)
class _PipelineStage:
    """One leg of the two-stage staging pipeline.

    ``stream`` is the CUDA stream the work is issued on; ``run(group,
    device_buf)`` does the work.
    """

    stream: Any
    run: Callable[[_TransferGroup, torch.Tensor], None]


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
        requested_buffer_chunks = _env_int("OFFLOAD_GPU_STAGING_CHUNKS", 2)
        max_staging_bytes = _env_optional_int("OFFLOAD_GPU_STAGING_MAX_BYTES")
        if max_staging_bytes is not None:
            if max_staging_bytes < self._gpu_staging_chunk_bytes:
                raise ValueError(
                    "OFFLOAD_GPU_STAGING_MAX_BYTES must be at least one "
                    "LMCache chunk: "
                    f"max_bytes={max_staging_bytes}, "
                    f"chunk_bytes={self._gpu_staging_chunk_bytes}"
                )
            requested_buffer_chunks = min(
                requested_buffer_chunks,
                max_staging_bytes // self._gpu_staging_chunk_bytes,
            )
        self._staging_buffer_chunks = max(1, int(requested_buffer_chunks))
        self._gpu_staging_buffer_bytes = (
            self._staging_buffer_chunks * self._gpu_staging_chunk_bytes
        )
        self._release_gpu_staging_after_transfer = _env_flag(
            "OFFLOAD_RELEASE_GPU_STAGING_AFTER_TRANSFER"
        )

    @property
    def gpu_staging_chunk_bytes(self) -> int:
        return self._gpu_staging_chunk_bytes

    @property
    def gpu_staging_buffer_chunks(self) -> int:
        return self._staging_buffer_chunks

    @property
    def gpu_staging_buffer_bytes(self) -> int:
        return self._gpu_staging_buffer_bytes

    @property
    def release_gpu_staging_after_transfer(self) -> bool:
        return self._release_gpu_staging_after_transfer

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
            )
            states[key] = state
        return state

    def _ensure_staging_buffer(
        self,
        staging_buffer: _StagingBuffer,
        nbytes: int,
    ) -> torch.Tensor:
        nbytes = int(nbytes)
        if nbytes > self._gpu_staging_buffer_bytes:
            raise RuntimeError(
                "ATOM LMCache connector internal error: transfer group exceeds "
                "bounded GPU staging buffer: "
                f"nbytes={nbytes}, capacity={self._gpu_staging_buffer_bytes}"
            )
        if (
            staging_buffer.tensor is None
            or int(staging_buffer.tensor.numel()) != self._gpu_staging_buffer_bytes
        ):
            staging_buffer.tensor = torch.empty(
                (self._gpu_staging_buffer_bytes,),
                dtype=torch.uint8,
                device=self.device,
            )
            staging_buffer.free_event_valid = False
        return staging_buffer.tensor[:nbytes]

    def _release_staging_buffer_if_requested(
        self,
        staging_buffer: _StagingBuffer,
    ) -> None:
        if not self._release_gpu_staging_after_transfer:
            return
        staging_buffer.tensor = None
        staging_buffer.free_event_valid = False

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
            would_exceed_count = len(current) >= self._staging_buffer_chunks
            would_exceed_bytes = (
                current_bytes + chunk.nbytes > self._gpu_staging_buffer_bytes
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

    def _prepare_transfer(
        self,
        memory_objs: list[Any] | None,
        starts: list[int] | None,
        ends: list[int] | None,
        **kwargs,
    ) -> tuple[_ThreadTransferState, list[_TransferGroup]] | None:
        """Validate inputs and build the chunk/group transfer plan."""
        if memory_objs is None or starts is None or ends is None:
            raise ValueError("memory_objs, starts, and ends are required")
        if not (len(memory_objs) == len(starts) == len(ends)):
            raise ValueError("memory_objs, starts, and ends must have equal length")
        block_id_groups = self._ranges_to_block_ids(starts, ends, **kwargs)
        if not memory_objs:
            return None
        state = self._thread_state()
        chunks = self._iter_transfer_chunks(memory_objs, block_id_groups)
        if not chunks:
            return None
        return state, self._iter_transfer_groups(chunks)

    def _run_staged_pipeline(
        self,
        state: _ThreadTransferState,
        groups: list[_TransferGroup],
        stage_a: _PipelineStage,
        stage_b: _PipelineStage,
    ) -> None:
        """Drive an event-synced two-stage staging pipeline.

        Each group flows ``stage_a`` -> ``stage_b`` on their respective streams,
        handed off via the staging buffer's ready event; the free event gates a
        later group's reuse of the same buffer. ``stage_b``'s stream produces
        the observable result, so it is the one synchronized at the end.
        """
        self._assert_fused_chunk_major_available()
        staging_buffer = state.staging_buffer
        used_buffer = False
        try:
            for group in groups:
                device_buf = self._ensure_staging_buffer(staging_buffer, group.nbytes)
                used_buffer = True
                if staging_buffer.free_event_valid:
                    stage_a.stream.wait_event(staging_buffer.free_event)
                with state.stream_ctx(stage_a.stream):
                    stage_a.run(group, device_buf)
                staging_buffer.ready_event.record(stage_a.stream)
                stage_b.stream.wait_event(staging_buffer.ready_event)
                with state.stream_ctx(stage_b.stream):
                    stage_b.run(group, device_buf)
                staging_buffer.free_event.record(stage_b.stream)
                staging_buffer.free_event_valid = True
            stage_b.stream.synchronize()
        except Exception:
            staging_buffer.free_event_valid = False
            raise
        finally:
            if used_buffer:
                self._release_staging_buffer_if_requested(staging_buffer)

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
        prepared = self._prepare_transfer(memory_objs, starts, ends, **kwargs)
        if prepared is None:
            return
        state, groups = prepared
        self._run_staged_pipeline(
            state,
            groups,
            stage_a=_PipelineStage(
                state.pack_stream,
                lambda group, buf: self.codec.gpu_to_chunk_major_device_buffer(
                    buf, self._group_block_ids(group), stream=state.pack_stream
                ),
            ),
            stage_b=_PipelineStage(
                state.copy_stream,
                lambda group, buf: self._slice_to_memory_objs(group, buf),
            ),
        )

    def batched_to_gpu(
        self,
        memory_objs: list[Any] | None = None,
        starts: list[int] | None = None,
        ends: list[int] | None = None,
        **kwargs,
    ) -> None:
        """Load LMCache MemoryObjs back into ATOM KV blocks via bounded staging."""
        prepared = self._prepare_transfer(memory_objs, starts, ends, **kwargs)
        if prepared is None:
            return
        state, groups = prepared
        self._run_staged_pipeline(
            state,
            groups,
            stage_a=_PipelineStage(
                state.copy_stream,
                lambda group, buf: self._memory_objs_to_slice(group, buf),
            ),
            stage_b=_PipelineStage(
                state.pack_stream,
                lambda group, buf: self.codec.chunk_major_device_buffer_to_gpu(
                    buf, self._group_block_ids(group), stream=state.pack_stream
                ),
            ),
        )
