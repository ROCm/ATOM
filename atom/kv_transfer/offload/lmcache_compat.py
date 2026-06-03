# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache-compatible raw-byte connector for ATOM offload.

This module lets ATOM use LMCache ``CacheEngine.store()`` /
``CacheEngine.retrieve()`` without adopting LMCache's vLLM token-major KV
layout. LMCache still owns chunking, keys, lookup pins, and storage-manager
orchestration. ATOM owns how a token range maps to AITER KV-cache blocks and
how those blocks are packed as opaque bytes.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

from atom.kv_transfer.offload.gpu_connector import ATOMKVByteCodec


def _cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


class ATOMRawBytesLMCacheMetadata:
    """Proxy around ``LMCacheMetadata`` with ATOM raw-byte allocation shapes."""

    def __init__(
        self,
        base_metadata: Any,
        *,
        atom_block_size: int,
        bytes_per_block: int,
    ) -> None:
        self._atom_base_metadata = base_metadata
        self.__dict__.update(vars(base_metadata))
        self.atom_block_size = int(atom_block_size)
        self.atom_bytes_per_block = int(bytes_per_block)
        chunk_size = int(getattr(base_metadata, "chunk_size"))
        if self.atom_block_size <= 0:
            raise ValueError("ATOM raw-byte metadata: atom_block_size must be > 0")
        if self.atom_bytes_per_block <= 0:
            raise ValueError("ATOM raw-byte metadata: bytes_per_block must be > 0")
        if chunk_size % self.atom_block_size != 0:
            raise ValueError(
                "LMCache chunk size must be divisible by ATOM KV block size: "
                f"chunk_size={chunk_size}, block_size={self.atom_block_size}"
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._atom_base_metadata, name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ATOMRawBytesLMCacheMetadata):
            return (
                self._atom_base_metadata == other._atom_base_metadata
                and self.atom_block_size == other.atom_block_size
                and self.atom_bytes_per_block == other.atom_bytes_per_block
            )
        return False

    def is_first_rank(self) -> bool:
        return self._atom_base_metadata.is_first_rank()

    def get_dtypes(self) -> list[torch.dtype]:
        return [torch.uint8]

    def get_shapes(self, num_tokens: int | None = None) -> list[torch.Size]:
        if num_tokens is None:
            num_tokens = int(self.chunk_size)
        nblocks = _cdiv(int(num_tokens), self.atom_block_size)
        return [torch.Size((nblocks * self.atom_bytes_per_block,))]

    def get_num_groups(self) -> int:
        return 1


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


class _StagingSlot:
    def __init__(self, use_cuda: bool) -> None:
        self.tensor: torch.Tensor | None = None
        self.ready_event = None
        self.free_event = None
        self.free_event_valid = False
        if use_cuda:
            self.ready_event = torch.cuda.Event(blocking=False)
            self.free_event = torch.cuda.Event(blocking=False)


class _ThreadTransferState:
    def __init__(self, device: torch.device, use_cuda: bool) -> None:
        self.device = device
        self.use_cuda = use_cuda
        self.pack_stream = None
        self.copy_stream = None
        self.next_slot = 0
        self.host_tmp: torch.Tensor | None = None
        if use_cuda:
            with torch.cuda.device(device):
                self.pack_stream = torch.cuda.Stream()
                self.copy_stream = torch.cuda.Stream()
                self.slots = [_StagingSlot(use_cuda), _StagingSlot(use_cuda)]
        else:
            self.slots = [_StagingSlot(use_cuda), _StagingSlot(use_cuda)]

    def stream_ctx(self, stream):
        if stream is None:
            return _NullCtx()
        return torch.cuda.stream(stream)


class ATOMLMCacheGPUConnector:
    """LMCache GPUConnectorInterface for ATOM's opaque KV-block byte layout."""

    def __init__(self, codec: ATOMKVByteCodec, block_size: int) -> None:
        self.codec = codec
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("ATOM LMCache connector: block_size must be > 0")
        self.device = torch.device(codec.device)
        self._tls = threading.local()

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
            state = _ThreadTransferState(self.device, self._use_cuda())
            states[key] = state
        return state

    def _ensure_slot(self, slot: _StagingSlot, nbytes: int) -> torch.Tensor:
        nbytes = int(nbytes)
        if slot.tensor is None or int(slot.tensor.numel()) < nbytes:
            slot.tensor = torch.empty(
                (nbytes,),
                dtype=torch.uint8,
                device=self.device,
            )
            slot.free_event_valid = False
        return slot.tensor[:nbytes]

    def _next_slot(self, state: _ThreadTransferState) -> _StagingSlot:
        slot = state.slots[state.next_slot % len(state.slots)]
        state.next_slot += 1
        return slot

    def _ensure_host_tmp(
        self,
        state: _ThreadTransferState,
        nbytes: int,
    ) -> torch.Tensor:
        nbytes = int(nbytes)
        if state.host_tmp is None or int(state.host_tmp.numel()) < nbytes:
            if state.use_cuda:
                try:
                    state.host_tmp = torch.empty(
                        (nbytes,),
                        dtype=torch.uint8,
                        pin_memory=True,
                    )
                except RuntimeError:
                    state.host_tmp = torch.empty((nbytes,), dtype=torch.uint8)
            else:
                state.host_tmp = torch.empty((nbytes,), dtype=torch.uint8)
        return state.host_tmp[:nbytes]

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
            raise RuntimeError("ATOM LMCache connector: MemoryObj tensor not contiguous")
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
        """Pack ATOM KV blocks to LMCache MemoryObjs via double GPU staging."""
        if not (len(memory_objs) == len(starts) == len(ends)):
            raise ValueError("memory_objs, starts, and ends must have equal length")
        block_id_groups = self._ranges_to_block_ids(starts, ends, **kwargs)
        if not memory_objs:
            return

        state = self._thread_state()
        use_cuda = state.use_cuda
        chunk_block_counts = [len(block_ids) for block_ids in block_id_groups]
        all_block_ids = [
            block_id for block_ids in block_id_groups for block_id in block_ids
        ]
        total_nbytes = len(all_block_ids) * self.codec.bytes_per_block
        if total_nbytes == 0:
            return

        slot = self._next_slot(state)
        device_buf = self._ensure_slot(slot, total_nbytes)
        host_buf = self._ensure_host_tmp(state, total_nbytes)
        dst_tensors = [
            self._memory_tensor(
                memory_obj,
                block_count * self.codec.bytes_per_block,
            )
            for memory_obj, block_count in zip(
                memory_objs,
                chunk_block_counts,
                strict=True,
            )
        ]

        if use_cuda:
            if slot.free_event_valid:
                state.pack_stream.wait_event(slot.free_event)
            with state.stream_ctx(state.pack_stream):
                self.codec.gpu_to_device_buffer(
                    device_buf,
                    all_block_ids,
                    stream=state.pack_stream,
                )
            slot.ready_event.record(state.pack_stream)
            state.copy_stream.wait_event(slot.ready_event)
            with state.stream_ctx(state.copy_stream):
                host_buf.copy_(device_buf, non_blocking=True)
            slot.free_event.record(state.copy_stream)
            slot.free_event_valid = True
            state.copy_stream.synchronize()
        else:
            self.codec.gpu_to_device_buffer(device_buf, all_block_ids)
            host_buf.copy_(device_buf, non_blocking=False)

        self.codec.split_request_buffer(host_buf, dst_tensors, chunk_block_counts)

    def batched_to_gpu(
        self,
        memory_objs: list[Any] | None = None,
        starts: list[int] | None = None,
        ends: list[int] | None = None,
        **kwargs,
    ) -> None:
        """Load LMCache MemoryObjs back into ATOM KV blocks via double staging."""
        if memory_objs is None or starts is None or ends is None:
            raise ValueError("memory_objs, starts, and ends are required")
        if not (len(memory_objs) == len(starts) == len(ends)):
            raise ValueError("memory_objs, starts, and ends must have equal length")
        block_id_groups = self._ranges_to_block_ids(starts, ends, **kwargs)
        if not memory_objs:
            return

        state = self._thread_state()
        use_cuda = state.use_cuda
        chunk_block_counts = [len(block_ids) for block_ids in block_id_groups]
        all_block_ids = [
            block_id for block_ids in block_id_groups for block_id in block_ids
        ]
        total_nbytes = len(all_block_ids) * self.codec.bytes_per_block
        if total_nbytes == 0:
            return

        slot = self._next_slot(state)
        device_buf = self._ensure_slot(slot, total_nbytes)
        host_buf = self._ensure_host_tmp(state, total_nbytes)
        src_tensors = [
            self._memory_tensor(
                memory_obj,
                block_count * self.codec.bytes_per_block,
            )
            for memory_obj, block_count in zip(
                memory_objs,
                chunk_block_counts,
                strict=True,
            )
        ]
        self.codec.stitch_chunk_buffers(host_buf, src_tensors, chunk_block_counts)

        if use_cuda:
            if slot.free_event_valid:
                state.copy_stream.wait_event(slot.free_event)
            with state.stream_ctx(state.copy_stream):
                device_buf.copy_(host_buf, non_blocking=True)
            slot.ready_event.record(state.copy_stream)
            state.pack_stream.wait_event(slot.ready_event)
            with state.stream_ctx(state.pack_stream):
                self.codec.device_buffer_to_gpu(
                    device_buf,
                    all_block_ids,
                    stream=state.pack_stream,
                )
            slot.free_event.record(state.pack_stream)
            slot.free_event_valid = True
            state.pack_stream.synchronize()
        else:
            device_buf.copy_(host_buf, non_blocking=False)
            self.codec.device_buffer_to_gpu(device_buf, all_block_ids)
