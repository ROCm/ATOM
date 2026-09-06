# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Staging-buffer helpers for the ATOM LMCache GPU connector."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch


def memory_object_as_uint8(memory_obj: Any, nbytes: int) -> torch.Tensor:
    """The leading `nbytes` of a LMCache MemoryObj as a flat uint8 view.

    Shared by every ATOM offload connector (dense block, K3 staging, state
    tier): each transfers raw bytes, so all validate the MemoryObj the same
    way. Kept here -- the connectors' common aiter-free dependency -- so the
    check lives once instead of byte-identically in each.
    """
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


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


class _StagingBuffer:
    def __init__(self, use_cuda: bool) -> None:
        self.tensor: torch.Tensor | None = None
        self.ready_event = None
        self.free_event = None
        self.free_event_valid = False
        if use_cuda:
            self.ready_event = torch.cuda.Event(blocking=False)
            self.free_event = torch.cuda.Event(blocking=False)


def _env_flag(name: str, default: str = "0") -> bool:
    # Stripped, and empty reads as off: `VAR=` is how a shell script clears a
    # flag inline, and a bare membership test reads the empty string as ON --
    # the opposite of what the operator wrote. `VAR="off "` did the same.
    raw = os.environ.get(name, default).strip().lower()
    return bool(raw) and raw not in ("0", "false", "no", "off")


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
    """One leg of the event-synchronized staging pipeline."""

    stream: Any
    run: Callable[[Any, torch.Tensor], None]


def run_staged_pipeline(
    state: _ThreadTransferState,
    groups: Sequence[Any],
    *,
    stage_a: _PipelineStage,
    stage_b: _PipelineStage,
    ensure_buffer: Callable[[_StagingBuffer, int], torch.Tensor],
    group_nbytes: Callable[[Any], int],
    release_buffer: Callable[[_StagingBuffer], None] | None = None,
    recover_buffer: (
        Callable[[_StagingBuffer, _PipelineStage, _PipelineStage], bool] | None
    ) = None,
) -> None:
    """Drive a two-stream staging pipeline with explicit recovery ownership.

    ``recover_buffer`` runs after any stage error and returns whether both
    streams were fenced successfully. An unfenced buffer is never released;
    the caller can quarantine it from the recovery callback.
    """

    staging_buffer = state.staging_buffer
    used_buffer = False
    buffer_safe_to_release = True
    try:
        for group in groups:
            if staging_buffer.free_event_valid:
                stage_a.stream.wait_event(staging_buffer.free_event)
            with state.stream_ctx(stage_a.stream):
                # ``ensure_buffer`` may allocate device storage. Allocate it on
                # the first consumer stream so a caching allocator cannot
                # reuse storage whose previous default-stream user is still
                # running and then let this side stream overwrite it early.
                device_buf = ensure_buffer(staging_buffer, group_nbytes(group))
                used_buffer = True
                stage_a.run(group, device_buf)
            staging_buffer.ready_event.record(stage_a.stream)
            stage_b.stream.wait_event(staging_buffer.ready_event)
            with state.stream_ctx(stage_b.stream):
                stage_b.run(group, device_buf)
            staging_buffer.free_event.record(stage_b.stream)
            staging_buffer.free_event_valid = True
        stage_b.stream.synchronize()
    except Exception:
        buffer_safe_to_release = bool(
            recover_buffer is not None
            and recover_buffer(staging_buffer, stage_a, stage_b)
        )
        if buffer_safe_to_release:
            staging_buffer.free_event_valid = False
        raise
    finally:
        if used_buffer and buffer_safe_to_release and release_buffer is not None:
            release_buffer(staging_buffer)
