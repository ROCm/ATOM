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
    return x


def _graph_marker_fake(x: torch.Tensor, name: str) -> torch.Tensor:
    return x


@torch_compile_guard(gen_fake=_graph_marker_fake)
def graph_marker(x: torch.Tensor, name: str) -> torch.Tensor:
    return _graph_marker_impl(x)


