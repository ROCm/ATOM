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

import logging
import threading
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from atom.kv_transfer.offload.atom_lmcache_staging import (
    _env_flag,
    _env_int,
    _env_optional_int,
    _PipelineStage,
    _StagingBuffer,
    _ThreadTransferState,
    memory_object_as_uint8,
    run_staged_pipeline,
)

logger = logging.getLogger("atom")


class BlockByteCodec(Protocol):
    """Block-byte staging contract shared by dense and PAGE codecs."""

    @property
    def device(self) -> torch.device: ...

    @property
    def num_blocks(self) -> int: ...

    @property
    def bytes_per_block(self) -> int: ...

    @property
    def has_fused_chunk_major_staging(self) -> bool: ...

    def gpu_to_chunk_major_device_buffer(
        self,
        device_buf: torch.Tensor,
        block_id_groups: list[list[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None: ...

    def chunk_major_device_buffer_to_gpu(
        self,
        device_buf: torch.Tensor,
        block_id_groups: list[list[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None: ...


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


class BlockGPUConnector:
    """LMCache GPUConnectorInterface for ATOM's opaque KV-block byte layout."""

    def __init__(
        self,
        codec: BlockByteCodec,
        block_size: int,
        *,
        chunk_size: int | None = None,
        virtual_block_size: int | None = None,
    ) -> None:
        self.codec = codec
        self.physical_block_size = int(block_size)
        if self.physical_block_size <= 0:
            raise ValueError("ATOM LMCache connector: block_size must be > 0")
        self.virtual_block_size = int(
            virtual_block_size
            if virtual_block_size is not None
            else self.physical_block_size
        )
        if self.virtual_block_size <= 0:
            raise ValueError("ATOM LMCache connector: virtual_block_size must be > 0")
        if self.virtual_block_size % self.physical_block_size:
            raise ValueError(
                "virtual DCP block size must be divisible by physical block size: "
                f"virtual={self.virtual_block_size}, "
                f"physical={self.physical_block_size}"
            )
        # ``block_size`` remains the token-to-block mapping grid for internal
        # helpers. Under DCP one scheduler block ID covers one virtual global
        # block while the codec still moves one rank-local physical page.
        self.block_size = self.virtual_block_size
        self.chunk_size = int(
            chunk_size if chunk_size is not None else self.virtual_block_size
        )
        if self.chunk_size <= 0:
            raise ValueError("ATOM LMCache connector: chunk_size must be > 0")
        if self.chunk_size % self.block_size != 0:
            raise ValueError(
                "LMCache chunk size must be divisible by the ATOM virtual block "
                f"size: chunk_size={self.chunk_size}, "
                f"virtual_block_size={self.virtual_block_size}"
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
        # A tensor is held here only when stream synchronization failed, so
        # freeing it could race still-running GPU work. Each failed pipeline
        # recovery adds at most one tensor; healthy devices fence successfully
        # and never grow this list. On an uncertain device the list is
        # intentionally unbounded for connector lifetime: correctness takes
        # priority over reclaiming potentially live GPU storage.
        self._quarantined_staging_tensors: list[torch.Tensor] = []
        self._quarantined_staging_lock = threading.Lock()
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

    @staticmethod
    def _fence_pipeline_streams(
        stage_a: _PipelineStage,
        stage_b: _PipelineStage,
    ) -> bool:
        """Attempt both stream fences; return whether completion is confirmed."""
        all_fenced = True
        for stage_name, stream in (
            ("stage_a", stage_a.stream),
            ("stage_b", stage_b.stream),
        ):
            try:
                stream.synchronize()
            except Exception:
                all_fenced = False
                logger.exception(
                    "ATOM LMCache connector: %s stream fence failed after "
                    "pipeline exception; staging will be quarantined",
                    stage_name,
                )
        return all_fenced

    def _quarantine_staging_buffer(self, staging_buffer: _StagingBuffer) -> None:
        """Detach and retain a buffer whose GPU completion is uncertain."""
        tensor = staging_buffer.tensor
        if tensor is not None:
            with self._quarantined_staging_lock:
                self._quarantined_staging_tensors.append(tensor)
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
                    tensor=memory_object_as_uint8(memory_obj, nbytes),
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
        the observable result, so it is the one synchronized at the end. An
        exception fences both streams; an unconfirmed fence permanently
        quarantines that tensor and forces fresh allocation on the next call.
        """
        self._assert_fused_chunk_major_available()

        def recover(staging_buffer, failed_a, failed_b) -> bool:
            if self._fence_pipeline_streams(failed_a, failed_b):
                return True
            self._quarantine_staging_buffer(staging_buffer)
            return False

        run_staged_pipeline(
            state,
            groups,
            stage_a=stage_a,
            stage_b=stage_b,
            ensure_buffer=self._ensure_staging_buffer,
            group_nbytes=lambda group: group.nbytes,
            release_buffer=self._release_staging_buffer_if_requested,
            recover_buffer=recover,
        )

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
