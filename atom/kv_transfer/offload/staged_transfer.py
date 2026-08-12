# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Bounded GPU staging buffer, D2H/H2D, and the producer event.

Shared by the KV offload tier (ATOMLMCacheGPUConnector) and the state offload
tier. Neither tier needs the other's orchestration logic; both need a bounded
device buffer, a copy stream, and an event the save worker synchronizes on.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Env helpers (follow the offload module's local pattern, not envs.py)
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _env_optional_int(name: str, *, min_value: int = 1) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class _StagingBuffer:
    """Device buffer plus the two events that gate the pipeline hand-off.

    ``use_cuda`` creates the events here rather than letting a caller attach
    them afterwards: the producer ``ready_event`` is what commit 7427e05e added
    to fix KV corruption on reload, and a buffer that silently carries ``None``
    events would make that protocol a no-op. Correct by construction.
    """

    def __init__(self, use_cuda: bool = False) -> None:
        self.tensor: torch.Tensor | None = None
        self.ready_event = None
        self.free_event = None
        self.free_event_valid = False
        if use_cuda:
            self.ready_event = torch.cuda.Event(blocking=False)
            self.free_event = torch.cuda.Event(blocking=False)


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


class _ThreadTransferState:
    def __init__(
        self,
        device: torch.device,
        use_cuda: bool,
    ) -> None:
        self.device = device
        self.pack_stream = None
        self.copy_stream = None
        if use_cuda:
            with torch.cuda.device(device):
                self.pack_stream = torch.cuda.Stream()
                self.copy_stream = torch.cuda.Stream()
        self.staging_buffer = _StagingBuffer(use_cuda)

    def stream_ctx(self, stream):
        if stream is None:
            return _NullCtx()
        return torch.cuda.stream(stream)


@dataclass(frozen=True)
class _PipelineStage:
    """One leg of the two-stage staging pipeline.

    ``stream`` is the CUDA stream the work is issued on; ``run(group,
    device_buf)`` does the work.
    """

    stream: Any
    run: Callable[..., None]


# ---------------------------------------------------------------------------
# StagedTransfer
# ---------------------------------------------------------------------------


class StagedTransfer:
    """Bounded GPU staging buffer, D2H/H2D, and the producer event.

    The half of the LMCache GPU connector that is not about chunks. KV and
    state both need a bounded device buffer, a copy stream, and an event the
    save worker synchronizes on; neither needs the other's orchestration. The
    chunk layer stays in `ATOMLMCacheGPUConnector` because it is genuinely
    KV-specific: `_iter_transfer_chunks` zips MemoryObjs against block-id
    groups with `strict=True` and sizes each from a startup per-block
    constant, so a single object of a different size breaks both invariants.
    State is not a member of that loop.

    The producer `cuda.Event` recorded on the RPC thread and `synchronize()`d
    on the save worker is load-bearing — it is what commit 7427e05e added to
    fix KV corruption on reload. Do not drop it from either caller.
    """

    def __init__(
        self,
        device: torch.device,
        staging_buffer_bytes: int,
        *,
        release_after_transfer: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self._staging_buffer_bytes = int(staging_buffer_bytes)
        self._release_after_transfer = release_after_transfer
        self._tls = threading.local()

    @property
    def staging_buffer_bytes(self) -> int:
        return self._staging_buffer_bytes

    def _use_cuda(self) -> bool:
        return self.device.type == "cuda"

    def thread_state(self) -> _ThreadTransferState:
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

    def ensure_buffer(
        self,
        staging_buffer: _StagingBuffer,
        nbytes: int,
    ) -> torch.Tensor:
        nbytes = int(nbytes)
        if nbytes > self._staging_buffer_bytes:
            raise RuntimeError(
                "ATOM LMCache connector internal error: transfer group exceeds "
                "bounded GPU staging buffer: "
                f"nbytes={nbytes}, capacity={self._staging_buffer_bytes}"
            )
        if (
            staging_buffer.tensor is None
            or int(staging_buffer.tensor.numel()) != self._staging_buffer_bytes
        ):
            staging_buffer.tensor = torch.empty(
                (self._staging_buffer_bytes,),
                dtype=torch.uint8,
                device=self.device,
            )
            staging_buffer.free_event_valid = False
        return staging_buffer.tensor[:nbytes]

    def release_buffer_if_requested(
        self,
        staging_buffer: _StagingBuffer,
    ) -> None:
        if not self._release_after_transfer:
            return
        staging_buffer.tensor = None
        staging_buffer.free_event_valid = False

    def memory_tensor(self, memory_obj: Any, nbytes: int) -> torch.Tensor:
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

    # -- whole-entry transfer (state tier) --------------------------------

    @staticmethod
    def _segment_block_bytes(segments: list[torch.Tensor]) -> list[int]:
        return [int(seg.numel()) * seg.element_size() for seg in segments]

    def _device_ctx(self):
        if self._use_cuda():
            return torch.cuda.device(self.device)
        return _NullCtx()

    def pack(self, segments: list[torch.Tensor], dst: Any) -> None:
        """Gather `segments` into one contiguous object via the Triton packer.

        The existing kernel needs no modification: it is already a fully
        parameterized gather driven by segment_ptrs[] + segment_block_bytes[] +
        block_ids[]. State passes its own views as the segments with
        block_ids=[0] and chunk_block_counts=[1] -- a single "chunk" of one
        "block", which is what a whole-entry snapshot is. Per-segment sizes may
        differ (GDN's k-views and v-views do); `_build_meta` sums them into one
        `bytes_per_block`.

        `storage_manager.allocate` normally hands back *host* memory, and the
        packer requires a CUDA uint8 contiguous destination, so the general
        path packs into the bounded GPU staging buffer and D2H's from there.
        The producer event around that copy is the same one commit 7427e05e
        added to fix KV corruption on reload: whoever reads the MemoryObj next
        must not see a copy that is still in flight.
        """
        from atom.kv_transfer.offload.triton_kv_staging import fused_pack_chunk_major

        segments = list(segments)
        seg_bytes = self._segment_block_bytes(segments)
        nbytes = sum(seg_bytes)
        dst_tensor = self.memory_tensor(dst, nbytes)
        with self._device_ctx():
            state = self.thread_state()
            staging_buffer = state.staging_buffer
            if dst_tensor.is_cuda and dst_tensor.device == self.device:
                with state.stream_ctx(state.pack_stream):
                    fused_pack_chunk_major(segments, seg_bytes, [1], [0], dst_tensor)
                if state.pack_stream is not None:
                    state.pack_stream.synchronize()
                return
            device_buf = self.ensure_buffer(staging_buffer, nbytes)
            try:
                with state.stream_ctx(state.pack_stream):
                    fused_pack_chunk_major(segments, seg_bytes, [1], [0], device_buf)
                self._handoff(state, state.pack_stream, state.copy_stream)
                with state.stream_ctx(state.copy_stream):
                    dst_tensor.copy_(
                        device_buf,
                        non_blocking=dst_tensor.device.type != "cpu",
                    )
                self._finish(state, state.copy_stream)
            except Exception:
                staging_buffer.free_event_valid = False
                raise
            finally:
                self.release_buffer_if_requested(staging_buffer)

    def unpack(self, src: Any, segments: list[torch.Tensor]) -> None:
        """Scatter one packed object back over `segments` -- `pack`'s mirror.

        Same staging hop in reverse: H2D into the bounded device buffer, then
        the Triton unpack kernel writes the segments. The kernel's stream is
        the one that produces the observable result, so it is the one waited
        on before the segments are handed back to their owner.
        """
        from atom.kv_transfer.offload.triton_kv_staging import fused_unpack_chunk_major

        segments = list(segments)
        seg_bytes = self._segment_block_bytes(segments)
        nbytes = sum(seg_bytes)
        src_tensor = self.memory_tensor(src, nbytes)
        with self._device_ctx():
            state = self.thread_state()
            staging_buffer = state.staging_buffer
            if src_tensor.is_cuda and src_tensor.device == self.device:
                with state.stream_ctx(state.pack_stream):
                    fused_unpack_chunk_major(src_tensor, segments, seg_bytes, [1], [0])
                if state.pack_stream is not None:
                    state.pack_stream.synchronize()
                return
            device_buf = self.ensure_buffer(staging_buffer, nbytes)
            try:
                with state.stream_ctx(state.copy_stream):
                    device_buf.copy_(
                        src_tensor,
                        non_blocking=src_tensor.device.type != "cpu",
                    )
                self._handoff(state, state.copy_stream, state.pack_stream)
                with state.stream_ctx(state.pack_stream):
                    fused_unpack_chunk_major(device_buf, segments, seg_bytes, [1], [0])
                self._finish(state, state.pack_stream)
            except Exception:
                staging_buffer.free_event_valid = False
                raise
            finally:
                self.release_buffer_if_requested(staging_buffer)

    @staticmethod
    def _handoff(state: _ThreadTransferState, producer, consumer) -> None:
        """Record the producer event and make the consumer stream wait on it."""
        state.staging_buffer.ready_event.record(producer)
        consumer.wait_event(state.staging_buffer.ready_event)

    @staticmethod
    def _finish(state: _ThreadTransferState, producer) -> None:
        """Publish the result: the buffer is free again once `producer` drains,
        and the caller may not observe the bytes until it has."""
        state.staging_buffer.free_event.record(producer)
        state.staging_buffer.free_event_valid = True
        producer.synchronize()

    def run_pipeline(
        self,
        state: _ThreadTransferState,
        groups: list,
        stage_a: _PipelineStage,
        stage_b: _PipelineStage,
    ) -> None:
        """Drive an event-synced two-stage staging pipeline.

        Each group flows ``stage_a`` -> ``stage_b`` on their respective streams,
        handed off via the staging buffer's ready event; the free event gates a
        later group's reuse of the same buffer. ``stage_b``'s stream produces
        the observable result, so it is the one synchronized at the end.
        """
        staging_buffer = state.staging_buffer
        used_buffer = False
        try:
            for group in groups:
                device_buf = self.ensure_buffer(staging_buffer, group.nbytes)
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
                self.release_buffer_if_requested(staging_buffer)
