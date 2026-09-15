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
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from atom.kv_transfer.disaggregation.types import SaveOperationId, SaveSourceGroupId
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


# Diagnostic only. When off, ``_run_staged_pipeline`` issues exactly the calls
# it always did and ``last_transfer_stats`` reports nothing, so the shipping
# path carries none of this. When on, it decomposes a staged transfer into GPU
# busy time per stage, the wall-clock span the GPU spent on the transfer, and
# the host time spent issuing it -- the three numbers that separate "the GPU is
# genuinely busy", "the GPU is stalled between stages", and "the host never got
# around to issuing the work".
_TRANSFER_PROFILE = _env_flag("OFFLOAD_TRANSFER_PROFILE")

# Experimental. The staging pipeline owns one buffer, so group n+1's pack
# already waits on the free event group n's copy records and the two streams
# never overlap on the GPU -- measured offline, span == pack + copy to within
# 1%. What the second stream does still cost is a cross-stream event handoff
# per group, and in-server that handoff is where the time goes: 2.28 ms of GPU
# work inside a 16.86 ms span. Running both legs on one stream makes the
# handoff a no-op of in-order execution and gives up an overlap that does not
# exist. Correctness is unaffected -- one stream is strictly more ordered than
# two, and the terminal synchronize is unchanged.
_SINGLE_STREAM = _env_flag("OFFLOAD_SINGLE_STREAM")


class _TransferProfile:
    """Per-thread transfer accounting spanning one LMCache store/retrieve.

    LMCache calls ``reset_transfer_stats`` before ``store()`` and reads the
    totals after, and a store issues several ``batched_from_gpu`` calls, so the
    fields accumulate across calls rather than describing the last one. Save
    workers run concurrently on the same connector, which is why this lives in
    the same thread-local as the transfer streams it measures.
    """

    __slots__ = (
        "calls",
        "chunks",
        "copy_ms",
        "from_gpu_ms",
        "gpu_span_ms",
        "groups",
        "issue_copy_ms",
        "issue_pack_ms",
        "max_chunk_bytes",
        "max_group_bytes",
        "pack_ms",
        "prepare_ms",
        "total_bytes",
        "wall_ms",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.groups = 0
        self.chunks = 0
        self.total_bytes = 0
        self.max_chunk_bytes = 0
        self.max_group_bytes = 0
        self.pack_ms = 0.0
        self.copy_ms = 0.0
        self.gpu_span_ms = 0.0
        self.prepare_ms = 0.0
        self.from_gpu_ms = 0.0
        self.issue_pack_ms = 0.0
        self.issue_copy_ms = 0.0
        self.wall_ms = 0.0


class _TimedStage:
    """Wraps one pipeline leg with CUDA events and host issue timing.

    The event pair brackets ``run`` alone on that leg's own stream, so it
    measures the stage's GPU busy time and excludes the cross-stream waits the
    pipeline inserts around it -- the difference between the two is exactly the
    stall being hunted. ``elapsed_time`` is deliberately not queried here: it
    would block, and the caller can read every event for free once
    ``run_staged_pipeline`` has synchronized.

    Read ``pack_ms`` and ``copy_ms``; do not build an argument on
    ``issue_*_ms`` or ``gpu_span_ms`` while a model is co-resident. Both
    ``record`` calls sit inside the region ``issue_ms`` times, and
    ``Event.record`` releases the GIL, so under contention it costs about one
    interpreter switch interval -- the same order as the number it is
    reporting. Measured on gfx950 against two non-yielding spinners: the two
    records alone cost 20.4 ms where the launch they bracket costs 5.2 ms, and
    0.005 ms with the GIL idle. Pre-allocating the events does not help
    (15.3 ms); the object allocation is free and the record is the cost. The
    stage busy times survive this because they are GPU timestamps taken when
    the packet executes, but the host issue time and the span it inflates are
    the instrument as much as the code under it.
    """

    __slots__ = ("_stage", "events", "issue_ms")

    def __init__(self, stage: _PipelineStage) -> None:
        self._stage = stage
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.issue_ms = 0.0

    def as_stage(self) -> _PipelineStage:
        return _PipelineStage(self._stage.stream, self._run)

    def _run(self, group: Any, buf: torch.Tensor) -> None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record(self._stage.stream)
        self._stage.run(group, buf)
        end.record(self._stage.stream)
        self.issue_ms += (time.perf_counter() - t0) * 1000.0
        self.events.append((start, end))

    def busy_ms(self) -> float:
        return sum(start.elapsed_time(end) for start, end in self.events)


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
    start: int
    end: int
    block_ids: list[int]
    tensor: torch.Tensor
    nbytes: int


@dataclass(frozen=True)
class _TransferGroup:
    chunks: list[_TransferChunk]
    nbytes: int


class _SourceSafeEventPoller:
    """Poll staging-group CUDA events without synchronizing the save thread."""

    def __init__(
        self,
        device: torch.device,
        callback: Callable[[SaveSourceGroupId], None],
    ) -> None:
        self._device = device
        self._callback = callback
        self._pending = deque()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._closed = False

    def submit(self, event, identities: tuple[SaveSourceGroupId, ...]) -> None:
        with self._condition:
            if self._closed:
                return
            self._pending.append((event, identities))
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="offload-source-safe",
                    daemon=True,
                )
                self._thread.start()
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if not self._pending:
                    return
                event, identities = self._pending[0]
            try:
                with torch.cuda.device(self._device):
                    ready = bool(event.query())
            except Exception:
                # An unqueryable event is not evidence of source safety. Leave
                # release to the scheduler's conservative abandon timeout.
                logger.exception(
                    "ATOM LMCache connector: source-safe event query failed"
                )
                with self._condition:
                    if self._pending and self._pending[0][0] is event:
                        self._pending.popleft()
                continue
            if not ready:
                time.sleep(0.001)
                continue
            with self._condition:
                if self._pending and self._pending[0][0] is event:
                    self._pending.popleft()
            for identity in identities:
                try:
                    self._callback(identity)
                except Exception:
                    logger.exception(
                        "ATOM LMCache connector: source-safe callback failed for %s",
                        identity,
                    )

    def close(self) -> None:
        """Drain ready events after save workers stop, then retire the poller."""

        with self._condition:
            self._closed = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)


class BlockGPUConnector:
    """LMCache GPUConnectorInterface for ATOM's opaque KV-block byte layout."""

    def __init__(
        self,
        codec: BlockByteCodec,
        block_size: int,
        *,
        chunk_size: int | None = None,
        virtual_block_size: int | None = None,
        source_safe_callback: Callable[[SaveSourceGroupId], None] | None = None,
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
        self._source_safe_callback = source_safe_callback
        self._source_safe_poller = (
            _SourceSafeEventPoller(self.device, source_safe_callback)
            if source_safe_callback is not None and self._use_cuda()
            else None
        )
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

    def close(self) -> None:
        if self._source_safe_poller is not None:
            self._source_safe_poller.close()

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

    def _profile_state(self) -> _TransferProfile:
        profile = getattr(self._tls, "profile", None)
        if profile is None:
            profile = _TransferProfile()
            self._tls.profile = profile
        return profile

    def reset_transfer_stats(self) -> None:
        """Start a new accounting window. Called before each LMCache store."""
        if _TRANSFER_PROFILE:
            self._profile_state().reset()

    def last_transfer_stats(self) -> dict[str, int | float]:
        """Totals since the last reset, in the shape the offload profiler logs.

        ``sync_ms`` is host time inside the pipeline that was not spent issuing
        work -- the terminal stream synchronize plus per-group bookkeeping --
        so ``pack_ms + copy_ms`` against ``gpu_span_ms`` separates GPU busy from
        GPU stalled, and ``gpu_span_ms`` against ``transfer_ms`` separates the
        GPU from the host.
        """
        if not _TRANSFER_PROFILE:
            return {}
        acc = self._profile_state()
        issue_ms = acc.issue_pack_ms + acc.issue_copy_ms
        seconds = acc.wall_ms / 1000.0
        return {
            "calls": acc.calls,
            "groups": acc.groups,
            "chunks": acc.chunks,
            "total_bytes": acc.total_bytes,
            "max_chunk_bytes": acc.max_chunk_bytes,
            "max_group_bytes": acc.max_group_bytes,
            "gpu_staging_chunk_bytes": self._gpu_staging_chunk_bytes,
            "gpu_staging_buffer_chunks": self._staging_buffer_chunks,
            "gpu_staging_buffer_bytes": self._gpu_staging_buffer_bytes,
            "pack_ms": acc.pack_ms,
            "copy_ms": acc.copy_ms,
            "gpu_span_ms": acc.gpu_span_ms,
            "prepare_ms": acc.prepare_ms,
            "from_gpu_ms": acc.from_gpu_ms,
            "issue_pack_ms": acc.issue_pack_ms,
            "issue_copy_ms": acc.issue_copy_ms,
            "sync_ms": max(0.0, acc.wall_ms - issue_ms),
            "transfer_ms": acc.wall_ms,
            "effective_gbps": (acc.total_bytes / seconds / 1e9 if seconds > 0 else 0.0),
        }

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
        starts: list[int],
        ends: list[int],
    ) -> list[_TransferChunk]:
        chunks: list[_TransferChunk] = []
        for memory_obj, block_ids, start, end in zip(
            memory_objs, block_id_groups, starts, ends, strict=True
        ):
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
                    start=int(start),
                    end=int(end),
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
        # A CPU destination is not a reason to block: LMCache hands out slices
        # of a driver-pinned block, so the copy can be a real async DMA, and
        # run_staged_pipeline already fences this stream before batched_from_gpu
        # returns -- nothing reads the host buffers before then. Blocking here
        # cost the caller a full transfer's wait per chunk while holding the
        # GIL; asynchronous it is 0.003 ms to issue instead of 0.066 ms, and it
        # lets the next group's pack overlap this group's copy. Torch falls back
        # to a synchronous copy when the destination is genuinely pageable, so
        # the flag is safe even if the allocator changes.
        offset = 0
        for chunk in group.chunks:
            chunk.tensor.copy_(
                src_buf[offset : offset + chunk.nbytes],
                non_blocking=True,
            )
            offset += chunk.nbytes

    @staticmethod
    def _memory_objs_to_slice(group: _TransferGroup, dst_buf: torch.Tensor) -> None:
        # Symmetric to _slice_to_memory_objs, but fenced transitively rather
        # than directly: this H2D is stage_a, the unpack waits on ready_event,
        # and run_staged_pipeline synchronizes the unpack stream before
        # batched_to_gpu returns -- so every copy out of these pinned host
        # buffers has landed before LMCache can recycle the MemoryObjs.
        offset = 0
        for chunk in group.chunks:
            dst_buf[offset : offset + chunk.nbytes].copy_(
                chunk.tensor,
                non_blocking=True,
            )
            offset += chunk.nbytes

    def _prepare_transfer(
        self,
        memory_objs: list[Any] | None,
        starts: list[int] | None,
        ends: list[int] | None,
        *,
        tail_to_head: bool = False,
        **kwargs,
    ) -> tuple[_ThreadTransferState, list[_TransferGroup]] | None:
        """Validate inputs and build the chunk/group transfer plan.

        LMCache gives the GPU connector parallel ``MemoryObj``/range lists.
        Sorting the assembled chunk records keeps each object attached to its
        exact token range and block IDs. Save staging is deliberately
        tail-to-head so the highest source blocks become reusable first; load
        staging retains LMCache's ordinary head-to-tail order.

        This does not mutate LMCache's input lists. In the current LMCache API,
        ``CacheEngine.store`` owns the separate key list and later passes it,
        together with the original ``memory_objs`` list, to ``batched_put``.
        The connector cannot safely reorder that opaque backend batch without
        also receiving the keys.
        """
        if memory_objs is None or starts is None or ends is None:
            raise ValueError("memory_objs, starts, and ends are required")
        if not (len(memory_objs) == len(starts) == len(ends)):
            raise ValueError("memory_objs, starts, and ends must have equal length")
        block_id_groups = self._ranges_to_block_ids(starts, ends, **kwargs)
        if not memory_objs:
            return None
        state = self._thread_state()
        chunks = self._iter_transfer_chunks(memory_objs, block_id_groups, starts, ends)
        if not chunks:
            return None
        if tail_to_head:
            chunks.sort(key=lambda chunk: (chunk.start, chunk.end), reverse=True)
        return state, self._iter_transfer_groups(chunks)

    @contextmanager
    def track_save_source(self, operation: SaveOperationId | None) -> Iterator[None]:
        """Associate this thread's next LMCache store with an exact save.

        LMCache forwards ``req_id`` to the GPU connector but not ATOM's save
        generation.  The store runs synchronously on this same executor thread,
        so a thread-local context carries the exact identity without changing
        LMCache's public API or encoding generations into user request IDs.
        """

        previous = getattr(self._tls, "save_operation", None)
        self._tls.save_operation = operation
        try:
            yield
        finally:
            self._tls.save_operation = previous

    def _source_safe_identities(
        self, group: _TransferGroup
    ) -> tuple[SaveSourceGroupId, ...]:
        callback = self._source_safe_callback
        operation = getattr(self._tls, "save_operation", None)
        if callback is None or not isinstance(operation, SaveOperationId):
            return ()
        return tuple(
            SaveSourceGroupId(
                save_operation=operation,
                ranges=((chunk.start, chunk.end),),
            )
            for chunk in group.chunks
        )

    def _queue_source_safe_group(self, group: _TransferGroup, stream) -> None:
        """Fence one staging group and publish its exact chunk identities.

        A fresh event is required because the pipeline's reusable ``free_event``
        is recorded again for the next group. The poller queries it from a
        daemon thread, so the save thread never host-synchronizes per chunk or
        per group and can continue into LMCache's storage-manager put.
        """

        identities = self._source_safe_identities(group)
        if not identities:
            return
        if stream is None:
            for identity in identities:
                self._source_safe_callback(identity)
            return
        event = torch.cuda.Event(blocking=False)
        event.record(stream)
        assert self._source_safe_poller is not None
        self._source_safe_poller.submit(event, identities)

    def _run_staged_pipeline(
        self,
        state: _ThreadTransferState,
        groups: list[_TransferGroup],
        stage_a: _PipelineStage,
        stage_b: _PipelineStage,
        stage_b_enqueued: Callable[[_TransferGroup, Any], None] | None = None,
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

        kwargs = {
            "ensure_buffer": self._ensure_staging_buffer,
            "group_nbytes": lambda group: group.nbytes,
            "release_buffer": self._release_staging_buffer_if_requested,
            "recover_buffer": recover,
            "stage_b_enqueued": stage_b_enqueued,
        }
        if not (_TRANSFER_PROFILE and self._use_cuda()):
            run_staged_pipeline(
                state, groups, stage_a=stage_a, stage_b=stage_b, **kwargs
            )
            return

        timed_a = _TimedStage(stage_a)
        timed_b = _TimedStage(stage_b)
        started = time.perf_counter()
        run_staged_pipeline(
            state,
            groups,
            stage_a=timed_a.as_stage(),
            stage_b=timed_b.as_stage(),
            **kwargs,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        # Every event has completed by now and none of these reads can block:
        # run_staged_pipeline synchronized stage_b's stream, and each stage_a
        # end event precedes the ready event that same stream waited on. A
        # failed pipeline raises above this point and is simply not counted --
        # the totals describe transfers that ran, which is what a bandwidth
        # attribution needs.
        acc = self._profile_state()
        acc.calls += 1
        acc.groups += len(groups)
        acc.wall_ms += wall_ms
        for group in groups:
            acc.chunks += len(group.chunks)
            acc.total_bytes += group.nbytes
            acc.max_group_bytes = max(acc.max_group_bytes, group.nbytes)
            for chunk in group.chunks:
                acc.max_chunk_bytes = max(acc.max_chunk_bytes, chunk.nbytes)
        acc.pack_ms += timed_a.busy_ms()
        acc.copy_ms += timed_b.busy_ms()
        acc.issue_pack_ms += timed_a.issue_ms
        acc.issue_copy_ms += timed_b.issue_ms
        if timed_a.events and timed_b.events:
            # First work issued to last work retired, across both streams.
            # Device timestamps share one clock, so this is meaningful even
            # though the two events sit on different streams.
            acc.gpu_span_ms += timed_a.events[0][0].elapsed_time(timed_b.events[-1][1])

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
        """Pack ATOM KV blocks tail-to-head into LMCache MemoryObjs."""
        t0 = time.perf_counter() if _TRANSFER_PROFILE else 0.0
        prepared = self._prepare_transfer(
            memory_objs,
            starts,
            ends,
            tail_to_head=True,
            **kwargs,
        )
        if _TRANSFER_PROFILE:
            self._profile_state().prepare_ms += (time.perf_counter() - t0) * 1000.0
        if prepared is None:
            return
        state, groups = prepared
        copy_stream = state.pack_stream if _SINGLE_STREAM else state.copy_stream
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
                copy_stream,
                lambda group, buf: self._slice_to_memory_objs(group, buf),
            ),
            stage_b_enqueued=self._queue_source_safe_group,
        )
        if _TRANSFER_PROFILE:
            self._profile_state().from_gpu_ms += (time.perf_counter() - t0) * 1000.0

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
        copy_stream = state.pack_stream if _SINGLE_STREAM else state.copy_stream
        self._run_staged_pipeline(
            state,
            groups,
            stage_a=_PipelineStage(
                copy_stream,
                lambda group, buf: self._memory_objs_to_slice(group, buf),
            ),
            stage_b=_PipelineStage(
                state.pack_stream,
                lambda group, buf: self.codec.chunk_major_device_buffer_to_gpu(
                    buf, self._group_block_ids(group), stream=state.pack_stream
                ),
            ),
        )
