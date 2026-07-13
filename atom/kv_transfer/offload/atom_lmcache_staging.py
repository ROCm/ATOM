# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Staging-buffer helpers for the ATOM LMCache GPU connector."""

from __future__ import annotations

import os

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
