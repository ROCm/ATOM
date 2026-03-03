# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc.
#
# A tiny, graph-friendly marker op for debugging/graph inspection.
# It is an identity at runtime, but it shows up in FX/graph dumps.

# from __future__ import annotations

import torch

from aiter.jit.utils.torch_guard import torch_compile_guard

_GRAPH_MARKER_ENABLED: bool = False


def set_graph_marker_enabled(enabled: bool) -> None:
    """Enable/disable graph markers globally (per-process)."""
    global _GRAPH_MARKER_ENABLED
    _GRAPH_MARKER_ENABLED = bool(enabled)


def is_graph_marker_enabled() -> bool:
    return _GRAPH_MARKER_ENABLED


def _graph_marker_impl(x: torch.Tensor) -> torch.Tensor:
    # Runtime behavior: identity.
    # Keep this side-effect free to avoid graph breaks.
    return x


def _graph_marker_fake(x: torch.Tensor, name: str) -> torch.Tensor:
    # FakeTensor / meta behavior: identity with preserved shape/stride/dtype.
    return x


@torch_compile_guard(gen_fake=_graph_marker_fake)
def graph_marker(x: torch.Tensor, name: str) -> torch.Tensor:
    """Insert a no-op marker node into the compiled/traced graph.

    The marker `name` is embedded as a constant in the graph dump so you can
    grep it in `computation_graph.py` / generated wrapper files.
    """
    # When disabled, return early so the marker does not even appear in the
    # traced/compiled graph.
    if not _GRAPH_MARKER_ENABLED:
        return x
    return _graph_marker_impl(x)


