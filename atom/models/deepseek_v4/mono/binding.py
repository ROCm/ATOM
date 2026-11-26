# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: shared target/draft binding extraction for ATOM, 2026-10-22.
"""Shared target/draft input views and captured-storage ownership."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.mono.runtime.consensus import MonoUnsupported


@dataclass(frozen=True)
class LayerStep:
    """Caller-owned KV/state views; target and draft keep distinct write owners."""

    positions: torch.Tensor
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    dest_rows: torch.Tensor
    indices: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    block_tables: torch.Tensor | None = None
    env_rows: int | None = None
    state: dict | None = None


def input_identity(value):
    """Capture addresses/layout, not mutable token, slot or epoch contents."""
    if isinstance(value, torch.Tensor):
        return (
            value.data_ptr(),
            value.shape,
            value.stride(),
            value.dtype,
            value.device,
        )
    if isinstance(value, dict):
        return tuple((key, input_identity(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(input_identity(item) for item in value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise MonoUnsupported(f"unsupported lane graph input {type(value).__name__}")


def tensor_owners(value):
    """Retain tensors even when their containing state dict is mutated."""
    owners = {}

    def collect(item):
        if isinstance(item, torch.Tensor):
            owners[id(item)] = item
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                collect(child)

    collect(value)
    return tuple(owners.values())
