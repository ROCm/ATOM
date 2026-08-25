# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where every byte of a hybrid model's cache lives, in one flat pool.

A hybrid model asks for two things that are not measured in the same unit.
Paged KV grows with **context length** — 100k tokens of K3 needs ~780 blocks
across its 24 MLA layers. A recurrent state is a fixed-size summary, so the 69
KDA layers need exactly **one slot regardless of prompt length**. Sizing them
as two tensors makes the split a startup constant: neither can ever give space
back to the other, and the unused half of whichever guessed high is unusable.

This module states the layout as one flat byte pool carved into equal
**superblocks**, each large enough to hold either kind whole::

    super_bytes = ceil(state_bytes / block_bytes) * block_bytes

A superblock is then read one of two ways, chosen by the allocator:

    kind=KV     ->  `blocks_per_super` pages, each holding ALL MLA layers
    kind=STATE  ->  one slot, holding ALL KDA layers

The layer axis is **inside** a superblock under both readings, which is what
lets one allocation serve either. It is the same move `v4_pool_geometry` makes
for DeepSeek-V4, one level up: V4 folds the layer into a row index and shares a
row space; this folds it into a byte offset and shares a byte range. The
difference matters because V4's scheme needs a row to mean the same thing in
both planes, and KDA cannot promise that — `_state_dtypes` gives kimi_linear an
fp32 v side where the block path is bf16. Bytes carry no such requirement.

At K3 TP8 (block_size 128, fp8 KV)::

    block_bytes =  1,769,472 B =  1.69 MiB   (24 MLA layers x 128 x 576)
    state_bytes = 56,171,520 B = 53.57 MiB   (69 KDA layers x 814,080)
    super_bytes = 56,623,104 B = 54.00 MiB   blocks_per_super = 32

The 0.80% a STATE superblock leaves unused is the whole cost of the scheme, and
it does not grow with anything. `blocks_per_super` is NOT a constant — at
block_size 64 it is 64, at bf16 KV it is 16 — so it is computed here and
nowhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Superblock kinds. A superblock is UNTYPED until its first allocation
#: decides how it will be read; releasing its last live block returns it.
UNTYPED = "untyped"
KV = "kv"
STATE = "state"


@dataclass(frozen=True)
class LayerField:
    """One layer's slice of a STATE superblock.

    `offset` is bytes from the superblock start; `shape` and `itemsize`
    describe what a view over it looks like. Kept as bytes rather than an
    element index so the fp32 and bf16 halves can sit in one range without
    either dictating the other's stride.
    """

    offset: int
    shape: tuple[int, ...]
    itemsize: int

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.itemsize


class SuperblockGeometry:
    """Sole owner of the arithmetic: byte offsets, counts, and layer fields.

    Allocation, view binding, and the checkpoint path all take their offsets
    from here rather than each re-deriving them. Mirrors
    `UnifiedPoolGeometry`'s contract for V4, which states the same rule and is
    the reason that layout survived a boundary that moves.

    Nothing here allocates. It answers where things go; `BlockPool` and
    `StateSlotPool` decide what goes there.
    """

    def __init__(
        self,
        block_bytes: int,
        state_fields: list[LayerField],
        num_supers: int,
        align: int = 256,
    ) -> None:
        if block_bytes < 1:
            raise ValueError(f"block_bytes must be >= 1, got {block_bytes}")
        if num_supers < 0:
            raise ValueError(f"num_supers must be >= 0, got {num_supers}")
        self.block_bytes = block_bytes
        self.state_fields = list(state_fields)
        self.num_supers = num_supers
        self.align = align

        # The last field's end, not the sum of sizes: `plan_state_fields` pads
        # between fields for alignment, and that padding is part of the state.
        self.state_bytes = max((f.offset + f.nbytes for f in state_fields), default=0)
        self.blocks_per_super = max(1, math.ceil(self.state_bytes / block_bytes))
        self.super_bytes = self.blocks_per_super * block_bytes
        if self.super_bytes < self.state_bytes:
            raise ValueError(
                f"superblock of {self.super_bytes} B cannot hold a "
                f"{self.state_bytes} B state"
            )
        # Every field's alignment rests on the superblock start, so it must be
        # a multiple of the widest one. `block_bytes` is a product of layer
        # count, block size and entry width, so this holds for every real
        # config; the check is here because a synthetic one could break it.
        if self.super_bytes % align:
            raise ValueError(
                f"superblock of {self.super_bytes} B is not {align} B aligned; "
                f"a layer view would be misaligned in every superblock but the first"
            )

    # ---------------------------- capacity --------------------------------- #
    @property
    def num_blocks(self) -> int:
        """Blocks the pool addresses if every superblock is read as KV."""
        return self.num_supers * self.blocks_per_super

    @property
    def total_bytes(self) -> int:
        return self.num_supers * self.super_bytes

    @property
    def state_waste_bytes(self) -> int:
        """Bytes a STATE superblock leaves unused. Does not scale with anything."""
        return self.super_bytes - self.state_bytes

    # ----------------------------- mapping --------------------------------- #
    def super_of_block(self, block_id: int) -> int:
        return block_id // self.blocks_per_super

    def block_span(self, block_id: int) -> tuple[int, int]:
        """`[start, stop)` bytes a KV block occupies, all MLA layers."""
        start = block_id * self.block_bytes
        return start, start + self.block_bytes

    def slot_span(self, slot: int) -> tuple[int, int]:
        """`[start, stop)` bytes a slot occupies, all KDA layers.

        One range rather than one per layer: a slot is the unit a checkpoint
        copies. Today's `mamba_k_cache[layer][slot]` is layer-major, so a
        request's state is `num_layers` disjoint strided pieces and a
        checkpoint is that many copies. Here it is one memcpy.
        """
        start = slot * self.super_bytes
        return start, start + self.super_bytes

    def state_field_offset(self, slot: int, layer_index: int) -> tuple[int, int]:
        """`(byte_offset, itemsize)` of one layer's state inside one slot."""
        field = self.state_fields[layer_index]
        return slot * self.super_bytes + field.offset, field.itemsize

    # ------------------------------ views ---------------------------------- #
    def state_view_params(self, layer_index: int) -> tuple[int, int, tuple[int, ...]]:
        """`(storage_offset_elems, slot_stride_elems, shape)` for a layer view.

        The view is slot-major — `(num_supers, *shape)` with a slot stride of a
        whole superblock — so it indexes exactly as `mamba_v_cache[layer]` does
        today and every caller's `state[indices]` keeps working unchanged.

        The stride is a whole superblock rather than one state, which means the
        view is NOT contiguous. Kernels that read their slot stride from the
        tensor handle this; kernels that recompute it from the shape do not.
        See `assert_reads_tensor_stride`.
        """
        field = self.state_fields[layer_index]
        if field.offset % field.itemsize:
            raise ValueError(
                f"layer {layer_index} at byte {field.offset} is misaligned for "
                f"a {field.itemsize}-byte dtype"
            )
        if self.super_bytes % field.itemsize:
            raise ValueError(
                f"superblock of {self.super_bytes} B does not divide by "
                f"{field.itemsize}; the slot stride would not be a whole number"
            )
        return (
            field.offset // field.itemsize,
            self.super_bytes // field.itemsize,
            field.shape,
        )

    def uniform_layer_stride(self, first: int, per_layer: int) -> int:
        """Bytes between one layer's field and the next layer's, or -1.

        A layer-major view over the pool -- `(num_layers, num_slots, *shape)`
        in one `as_strided`, aliasing rather than copying -- needs the step
        from layer L to layer L+1 to be the same for every L. That holds when
        `plan_state_fields` pads every layer identically, which it does when
        each layer's fields are the same shapes in the same order. Returns -1
        when they are not, so the caller can fall back to per-layer views
        instead of silently addressing the wrong bytes.

        `first` is the index of layer 0's field, `per_layer` how many fields a
        layer owns.
        """
        fields = self.state_fields
        if per_layer < 1 or len(fields) < first + 2 * per_layer:
            return -1
        step = fields[first + per_layer].offset - fields[first].offset
        for index in range(first, len(fields) - per_layer, per_layer):
            if fields[index + per_layer].offset - fields[index].offset != step:
                return -1
        return step

    def describe(self) -> str:
        return (
            f"superblock {self.super_bytes:,} B "
            f"= {self.blocks_per_super} x {self.block_bytes:,} B block; "
            f"state {self.state_bytes:,} B leaves "
            f"{self.state_waste_bytes:,} B "
            f"({100.0 * self.state_waste_bytes / self.super_bytes:.2f}%) unused"
        )


def assert_reads_tensor_stride(fn: object) -> None:
    """Refuse a KDA kernel that recomputes its slot stride from the shape.

    A unified pool hands the state out as a view whose slot stride is a whole
    superblock. Two implementations of the same kernel differ on whether that
    survives:

      ATOM  `fla_ops/fused_sigmoid_gating.py` reads
            `stride_init_state_token = initial_state.stride(0)` -- correct on
            any view.
      aiter `_triton_kernels/.../fused_sigmoid_gating_recurrent.py` computes
            `h0_source + idx * HV * K * V` -- correct only when the state is
            densely packed, and SILENTLY WRONG otherwise. No bounds error, no
            NaN: it reads a neighbouring slot's bytes.

    Phase 0.1 measured the ATOM path bit-exact on a strided view. Nothing in
    the type system stops a later change from swapping in the aiter one, so
    this states the requirement where it will be seen.
    """
    module = getattr(fn, "__module__", "") or ""
    if "aiter" in module and "atom" not in module:
        raise RuntimeError(
            f"{module}.{getattr(fn, '__name__', fn)} derives its state slot "
            "stride from the tensor shape, which is wrong for the unified "
            "pool's strided view. Use atom.model_ops.fla_ops."
            "fused_sigmoid_gating, which reads initial_state.stride(0)."
        )


def plan_state_fields(
    per_layer: list[tuple[tuple[int, ...], int]],
    align: int = 256,
) -> list[LayerField]:
    """Lay out each layer's state back to back inside a superblock.

    `per_layer` is `(shape, itemsize)` per layer, in the order the model's
    layers are indexed. Each field is aligned up so its own dtype's stride is
    whole — the fp32 v side and bf16 k side of a KDA layer sit in one range,
    and only the field start has to land on a boundary.
    """
    fields: list[LayerField] = []
    cursor = 0
    for shape, itemsize in per_layer:
        step = math.lcm(itemsize, align) if align else itemsize
        cursor = -(-cursor // step) * step
        fields.append(LayerField(offset=cursor, shape=shape, itemsize=itemsize))
        cursor += math.prod(shape) * itemsize
    return fields
