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
