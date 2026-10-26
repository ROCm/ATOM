"""Fixed device scratch for the lossless sampling kernels.

The sampled draft (`dspark_markov_sample`) and the ratio verify
(`rejection_sampler`) each need a handful of small fp32/int32 tensors per call,
sized by the batch. Allocated once at the largest batch and carved into
per-call views, they exist before the KV cache is sized -- so the profiling run
sees them like any other buffer -- and no step allocates.

Every view starts on a 16-byte boundary. Triton specializes a kernel on
whether each pointer argument is 16-byte aligned, so views packed back to back
-- at offsets that move with the batch -- would compile the same kernel once
per alignment pattern the batch sizes happen to produce, each the first time
a batch of that pattern arrives.
"""

from collections.abc import Sequence
from math import prod

import torch

# Each view's first element is a multiple of this many 4-byte elements.
_ALIGN_ELEMS = 16 // 4


def _padded(size: int) -> int:
    return -(-size // _ALIGN_ELEMS) * _ALIGN_ELEMS


class SamplingScratch:
    """One fp32 and one int32 flat buffer, handed out as contiguous views."""

    def __init__(self, f32_elems: int, i32_elems: int, device):
        self.f32 = torch.empty(f32_elems, dtype=torch.float32, device=device)
        self.i32 = torch.empty(i32_elems, dtype=torch.int32, device=device)

    @classmethod
    def for_shapes(
        cls,
        f32_shapes: Sequence[tuple[int, ...]],
        i32_shapes: Sequence[tuple[int, ...]],
        device,
    ) -> "SamplingScratch":
        """Room for `take` to carve views of these shapes, or of any smaller
        ones."""
        return cls(
            sum(_padded(prod(x)) for x in f32_shapes),
            sum(_padded(prod(x)) for x in i32_shapes),
            device,
        )


def take(
    f32_shapes: Sequence[tuple[int, ...]],
    i32_shapes: Sequence[tuple[int, ...]],
    device,
    f32: torch.Tensor | None = None,
    i32: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Views of the given shapes, laid out in order in ``f32`` / ``i32``, each
    from the next 16-byte boundary; freshly allocated tensors when no buffer is
    given (tests, one-off calls)."""
    return (
        _carve(f32_shapes, torch.float32, device, f32),
        _carve(i32_shapes, torch.int32, device, i32),
    )


def _carve(shapes, dtype, device, flat):
    if flat is None:
        return [torch.empty(shape, dtype=dtype, device=device) for shape in shapes]
    sizes = [prod(shape) for shape in shapes]
    need = sum(_padded(size) for size in sizes)
    if need > flat.numel():
        raise ValueError(
            f"sampling scratch holds {flat.numel()} {dtype} elements, this call "
            f"needs {need}; it was sized for a smaller batch"
        )
    views, start = [], 0
    for shape, size in zip(shapes, sizes):
        views.append(flat[start : start + size].view(shape))
        start += _padded(size)
    return views
