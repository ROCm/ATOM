# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Staging-buffer helpers + the shared two-stream copy pipeline.

Both offload GPU connectors (chunked MHA/MLA and the DSV4 offload unit) move
bytes through a bounded GPU staging buffer with the SAME event-synced two-stream
ping-pong: stage A (pack/gather) on one stream, stage B (D2H / H2D) on another,
handed off via the buffer's ready event, with a later group's reuse gated by the
free event. :func:`run_staged_pipeline` is that shared driver; each connector
supplies its own ``ensure_buffer`` sizing and per-group stage work.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Sequence

import torch


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
    """One leg of the two-stage staging pipeline: work issued on ``stream``."""

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
) -> None:
    """Drive an event-synced two-stage staging pipeline over ``groups``.

    Each group flows ``stage_a`` -> ``stage_b`` on their respective streams,
    handed off via the staging buffer's ready event; the free event gates a later
    group's reuse of the same buffer. ``stage_b``'s stream produces the
    observable result, so it is synchronized at the end. Shared by the chunked
    MHA/MLA connector and the DSV4 offload-unit connector.
    """
    staging_buffer = state.staging_buffer
    used_buffer = False
    try:
        for group in groups:
            device_buf = ensure_buffer(staging_buffer, group_nbytes(group))
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
        if used_buffer and release_buffer is not None:
            release_buffer(staging_buffer)
