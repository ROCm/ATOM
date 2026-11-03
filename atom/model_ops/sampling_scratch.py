"""Fixed device scratch for the lossless sampling kernels.

The sampled draft and the ratio verify carve their small per-call fp32/int32
tensors from buffers allocated once at the largest batch, before the KV cache
is sized, so no step allocates. Views start on 256-byte boundaries
(`entry_arena.plan_regions`): Triton specializes on 16-byte pointer alignment,
so offsets that move with the batch would recompile mid-serve.
"""

from collections.abc import Sequence
from math import prod

import torch

from atom.model_ops.attentions.pool_layout.entry_arena import carve, plan_regions

# Both buffers hold 4-byte elements.
_ELEM_BYTES = 4


def _region_bytes(shapes: Sequence[tuple[int, ...]]) -> list[int]:
    return [prod(shape) * _ELEM_BYTES for shape in shapes]


def _planned_elems(shapes: Sequence[tuple[int, ...]]) -> int:
    """Elements a buffer needs to hold views of these shapes."""
    return plan_regions(_region_bytes(shapes))[1] // _ELEM_BYTES


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
        return cls(_planned_elems(f32_shapes), _planned_elems(i32_shapes), device)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.f32, self.i32))


def take(
    f32_shapes: Sequence[tuple[int, ...]],
    i32_shapes: Sequence[tuple[int, ...]],
    f32: torch.Tensor,
    i32: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Views of the given shapes, laid out in order in ``f32`` / ``i32`` by
    `plan_regions`."""
    return (
        _carve(f32_shapes, torch.float32, f32),
        _carve(i32_shapes, torch.int32, i32),
    )


def _carve(shapes, dtype, flat):
    need = _planned_elems(shapes)
    if need > flat.numel():
        raise ValueError(
            f"sampling scratch holds {flat.numel()} {dtype} elements, this call "
            f"needs {need}; it was sized for a smaller batch"
        )
    regions = carve(flat.view(torch.uint8), _region_bytes(shapes))
    return [region.view(dtype).view(shape) for region, shape in zip(regions, shapes)]
