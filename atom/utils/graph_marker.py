# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
#
# A graph marker for profiling compiled code without copying activations.

import torch

_GRAPH_MARKER_ENABLED: bool = False


def set_graph_marker_enabled(enabled: bool) -> None:
    """Enable/disable graph markers globally (per-process)."""
    global _GRAPH_MARKER_ENABLED
    _GRAPH_MARKER_ENABLED = bool(enabled)


def is_graph_marker_enabled() -> bool:
    return _GRAPH_MARKER_ENABLED


@torch.library.custom_op("aiter::graph_marker", mutates_args=("x",))
def _graph_marker(x: torch.Tensor, name: str) -> None:
    # An in-place barrier keeps the marker ordered with operations on x and
    # prevents dead-code elimination. It does not actually change any data.
    # Return nothing: returning x from a custom op would introduce an output
    # alias that functionalization cannot represent with this schema.
    pass


@_graph_marker.register_fake
def _graph_marker_fake(x: torch.Tensor, name: str) -> None:
    pass


def graph_marker(x: torch.Tensor, name: str) -> torch.Tensor:
    """Mark a tensor's position in the graph, preserving its identity and layout.

    The compiler sees an in-place barrier with no aliased output. Returning the
    original tensor here keeps its existing views and lets Inductor reuse its
    storage, instead of cloning an opaque custom op's returned alias.
    """
    if _GRAPH_MARKER_ENABLED:
        _graph_marker(x, name)
    return x
