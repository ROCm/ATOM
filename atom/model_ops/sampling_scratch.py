"""Fixed device scratch for the lossless sampling kernels.

The sampled draft (`dspark_markov_sample`) and the ratio verify
(`rejection_sampler`) each need a handful of small fp32/int32 tensors per call,
sized by the batch. Allocated once at the largest batch and carved into
per-call views, they exist before the KV cache is sized -- so the profiling run
sees them like any other buffer -- and no step allocates.
"""

from collections.abc import Sequence
from math import prod

import torch


class SamplingScratch:
    """One fp32 and one int32 flat buffer, handed out as contiguous views."""

    def __init__(self, f32_elems: int, i32_elems: int, device):
        self.f32 = torch.empty(f32_elems, dtype=torch.float32, device=device)
        self.i32 = torch.empty(i32_elems, dtype=torch.int32, device=device)


def take(
    f32_shapes: Sequence[tuple[int, ...]],
    i32_shapes: Sequence[tuple[int, ...]],
    device,
    f32: torch.Tensor | None = None,
    i32: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Views of the given shapes, laid out back to back in ``f32`` / ``i32``;
    freshly allocated tensors when no buffer is given (tests, one-off calls)."""
    return (
        _carve(f32_shapes, torch.float32, device, f32),
        _carve(i32_shapes, torch.int32, device, i32),
    )


def _carve(shapes, dtype, device, flat):
    if flat is None:
        return [torch.empty(shape, dtype=dtype, device=device) for shape in shapes]
    sizes = [prod(shape) for shape in shapes]
    if sum(sizes) > flat.numel():
        raise ValueError(
            f"sampling scratch holds {flat.numel()} {dtype} elements, this call "
            f"needs {sum(sizes)}; it was sized for a smaller batch"
        )
    views, start = [], 0
    for shape, size in zip(shapes, sizes):
        views.append(flat[start : start + size].view(shape))
        start += size
    return views
