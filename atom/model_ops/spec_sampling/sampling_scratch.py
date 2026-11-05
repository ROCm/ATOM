"""Fixed device scratch for the lossless sampling kernels.

The sampled draft and the ratio verify carve their small per-call tensors
from one byte buffer allocated at the largest batch, before the KV cache is
sized, so no step allocates. Each consumer names its regions in a NamedTuple
of `Region`s and gets the same NamedTuple of views back. The bytes are placed
by the cache pools' own carver (`entry_arena.plan_regions` / `carve`), so
every view starts on its `SCRATCH_ALIGN` boundary: Triton specializes on
16-byte pointer alignment, so offsets that move with the batch would
recompile mid-serve.
"""

from math import prod
from typing import NamedTuple, TypeVar

import torch

from atom.model_ops.attentions.pool_layout.entry_arena import carve as carve_bytes
from atom.model_ops.attentions.pool_layout.entry_arena import plan_regions

#: Byte alignment of every view: `plan_regions`', not chosen here.
SCRATCH_ALIGN = plan_regions([1])[1]

Spec = TypeVar("Spec", bound=tuple)


class Region(NamedTuple):
    """One view's shape and dtype."""

    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.dtype.itemsize


def _regions(spec) -> list[Region]:
    """The spec's regions in field order; a field may itself be a spec."""
    out = []
    for field in spec:
        out.extend([field] if isinstance(field, Region) else _regions(field))
    return out


def scratch_bytes(spec) -> int:
    """Bytes a buffer needs to hold this spec's views."""
    return plan_regions([r.nbytes for r in _regions(spec)])[1]


def scratch_buffer(spec, device) -> torch.Tensor:
    """A byte buffer with room to `carve` this spec, or any spec of smaller
    regions."""
    return torch.empty(scratch_bytes(spec), dtype=torch.uint8, device=device)


def carve(buf: torch.Tensor, spec: Spec) -> Spec:
    """``spec`` with each `Region` replaced by its view of ``buf``, laid out in
    field order (`entry_arena.carve`, which refuses a buffer too small)."""
    parts = iter(carve_bytes(buf, [r.nbytes for r in _regions(spec)]))

    def build(node):
        fields = []
        for field in node:
            if isinstance(field, Region):
                fields.append(next(parts).view(field.dtype).view(field.shape))
            else:
                fields.append(build(field))
        return type(node)(*fields)

    return build(spec)
