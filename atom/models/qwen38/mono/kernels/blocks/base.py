# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The block interface: one reusable piece of a layer (a linear, an HC, an attention mixer) producing ``Stage``s.

    class MyBlock(Block):
        def stages(self, sc: Scenario) -> list[Stage]: ...   # tasks, regions written, args, emit
        def pack(self, layer) -> dict[str, torch.Tensor]: ...  # its pointer-table tensors
        def weight_specs(self) -> tuple[WeightSpec, ...]: ...  # the layout each weight arg is read in

Blocks talk to each other only through **channels** (``Channel``: per-row tensors handed over as tagged (32-bit
word, tag) pairs in scratch; the producer declares the regions, consumers poll them with
``device.channels``) and **args** (plain device memory named in the pointer table: the
layer's input / output, weights, caches, states). Names are global within a layer program.

Rows: a step has ``sc.rows = batch * spec`` rows; row ``r`` is request ``r // spec`` at offset ``r % spec``. Every
channel is laid out row-major over all rows. Formats (one pair = 8 bytes): ``BF16`` (two bf16 a pair), ``F32`` (one
f32 a pair), ``I32`` (one int32 a pair: integer payloads never take an f32 round trip), ``MXFP8`` (two regions:
``<name>.q`` with four E4M3 bytes a pair, ``<name>.s`` with one E8M0 scale code a pair, group ``g`` of row ``r`` at
pair ``r * width / 32 + g``).
"""

from dataclasses import dataclass
from enum import Enum

import torch

from atom.models.qwen38.mono.kernels.program import Region, Scenario, Stage


class Fmt(Enum):
    BF16 = "bf16"
    F32 = "f32"
    I32 = "i32"
    MXFP8 = "mxfp8"


@dataclass(frozen=True)
class Channel:
    name: str
    width: int  # elements per row
    fmt: Fmt

    def pairs_per_row(self) -> int:
        per = {
            Fmt.BF16: self.width // 2,
            Fmt.F32: self.width,
            Fmt.I32: self.width,
            Fmt.MXFP8: self.width // 4,
        }
        return per[self.fmt]

    def regions(self, rows: int, writers: tuple[str, ...]) -> tuple[Region, ...]:
        """The scratch regions holding this channel for ``rows`` rows, put by ``writers``."""
        if self.fmt is Fmt.MXFP8:
            assert self.width % 32 == 0, self
            words = (
                (f"{self.name}.q", rows * self.width // 4),
                (f"{self.name}.s", rows * self.width // 32),
            )
        else:
            words = ((self.name, rows * self.pairs_per_row()),)
        return tuple(Region(n, w, writers) for n, w in words)


class Block:
    """Base class of blocks. Subclasses set ``name`` and implement ``stages`` (and ``pack`` when they read
    weights)."""

    name: str = "block"

    def stages(self, sc: Scenario) -> list[Stage]:
        raise NotImplementedError

    def pack(self, layer) -> dict[str, torch.Tensor]:
        """Device tensors this block's args need, from a ``QuantizedLayer``."""
        return {}

    def weight_specs(self) -> tuple:
        """``weights.source.WeightSpec`` of every weight arg ``pack`` fills: the layout the emit
        code reads it in, checked when a ``WeightSource`` is bound (``weights.source.bind``).
        """
        return ()
