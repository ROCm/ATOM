"""Fixed device scratch for the lossless sampling kernels.

The sampled draft and the ratio verify carve their small per-call tensors
from one byte buffer allocated at the largest batch, before the KV cache is
sized, so no step allocates. Each consumer names its regions in a NamedTuple
of `Region`s and gets the same NamedTuple of views back. Every view starts on
a `SCRATCH_ALIGN` boundary this module owns: Triton specializes on 16-byte
pointer alignment, so offsets that move with the batch would recompile
mid-serve.
"""

from math import prod
from typing import NamedTuple, TypeVar

import torch

#: Byte alignment of every view.
SCRATCH_ALIGN = 256

Spec = TypeVar("Spec", bound=tuple)


class Region(NamedTuple):
    """One view's shape and dtype."""

    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * torch.empty((), dtype=self.dtype).element_size()


def _regions(spec) -> list[Region]:
    """The spec's regions in field order; a field may itself be a spec."""
    out = []
    for field in spec:
        out.extend([field] if isinstance(field, Region) else _regions(field))
    return out


def _align(nbytes: int) -> int:
    return -(-nbytes // SCRATCH_ALIGN) * SCRATCH_ALIGN


def scratch_bytes(spec) -> int:
    """Bytes a buffer needs to hold this spec's views."""
    return sum(_align(r.nbytes) for r in _regions(spec))


class SamplingScratch:
    """One flat byte buffer, handed out as named, aligned views."""

    def __init__(self, nbytes: int, device):
        self.buf = torch.empty(nbytes, dtype=torch.uint8, device=device)

    @classmethod
    def for_spec(cls, spec, device) -> "SamplingScratch":
        """Room to carve this spec, or any spec of smaller regions."""
        return cls(scratch_bytes(spec), device)

    def nbytes(self) -> int:
        return self.buf.numel()

    def carve(self, spec: Spec) -> Spec:
        return carve(self.buf, spec)


def carve(buf: torch.Tensor, spec: Spec) -> Spec:
    """``spec`` with each `Region` replaced by its view of ``buf``, laid out in
    field order."""
    need = scratch_bytes(spec)
    if need > buf.numel():
        raise ValueError(
            f"sampling scratch holds {buf.numel()} bytes, this call needs "
            f"{need}; it was sized for a smaller batch"
        )
    offset = 0

    def build(node):
        nonlocal offset
        fields = []
        for field in node:
            if isinstance(field, Region):
                view = buf[offset : offset + field.nbytes].view(field.dtype)
                fields.append(view.view(field.shape))
                offset += _align(field.nbytes)
            else:
                fields.append(build(field))
        return type(node)(*fields)

    return build(spec)
