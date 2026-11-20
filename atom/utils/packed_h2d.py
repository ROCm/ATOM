# SPDX-License-Identifier: MIT
"""Pack metadata into one pinned arena, upload once, scatter to stable addresses."""

import numpy as np
import torch
import triton
import triton.language as tl

from atom.utils.h2d import gather_rows


@triton.jit
def scatter_packed_words(arena, destinations, wide, WORDS: tl.constexpr):
    """Scatter each packed member (header: int64 arena offset, byte count) to
    its destination, WORDS * 8 bytes a program: as int64 words where the
    member starts 16-byte aligned in the arena (packing pads every member to
    that) and its destination is too (`wide`), byte by byte otherwise."""
    member = tl.program_id(0)
    header = arena.to(tl.pointer_type(tl.int64))
    start = tl.load(header + 2 * member)
    length = tl.load(header + 2 * member + 1)
    address = tl.load(destinations + member)
    span = WORDS * 8
    base = tl.program_id(1).to(tl.int64) * span
    if base < length:
        if tl.load(wide + member) != 0:
            words = length // 8
            index = base // 8 + tl.arange(0, WORDS)
            source = (arena + tl.multiple_of(start, 16)).to(tl.pointer_type(tl.int64))
            target = tl.multiple_of(address, 16).to(tl.pointer_type(tl.int64))
            tl.store(
                target + index, tl.load(source + index, index < words), index < words
            )
            # The last program also moves the 0-7 tail bytes.
            if base + span >= length:
                tail = words * 8 + tl.arange(0, 8)
                tl.store(
                    address.to(tl.pointer_type(tl.uint8)) + tail,
                    tl.load(arena + start + tail, tail < length),
                    tail < length,
                )
        else:
            offset = base + tl.arange(0, WORDS * 8)
            tl.store(
                address.to(tl.pointer_type(tl.uint8)) + offset,
                tl.load(arena + start + offset, offset < length),
                offset < length,
            )


@triton.jit
def scatter_packed_rows(arena, destinations, wide, pitches, WORDS: tl.constexpr):
    """`scatter_packed_words` for a publish that carries a row plan
    (`PackedCopy._submit_planned`). The header holds main's (start, byte
    count) pairs for every member, then a (row bytes, row map offset) pair per
    member. Row bytes 0: a contiguous prefix, scattered by
    `scatter_packed_words` itself. Otherwise the payload holds rows of that
    many bytes (a multiple of 16), each stored at the start of the destination
    row the int32 row map names, ``pitches`` bytes apart (a row member's
    destination is 16-byte aligned)."""
    member = tl.program_id(0)
    header = arena.to(tl.pointer_type(tl.int64))
    rows_at = header + 2 * tl.num_programs(0) + 2 * member
    row_bytes = tl.load(rows_at)
    if row_bytes == 0:
        scatter_packed_words(arena, destinations, wide, WORDS)
    else:
        start = tl.load(header + 2 * member)
        length = tl.load(header + 2 * member + 1)
        base = tl.program_id(1).to(tl.int64) * (WORDS * 8)
        if base < length:
            index = base // 8 + tl.arange(0, WORDS)
            mask = index < length // 8
            source = (arena + tl.multiple_of(start, 16)).to(tl.pointer_type(tl.int64))
            values = tl.load(source + index, mask)
            row_words = row_bytes // 8
            row_map = (arena + tl.load(rows_at + 1)).to(tl.pointer_type(tl.int32))
            row = index // row_words
            column = index - row * row_words
            target_row = tl.load(row_map + row, mask, other=0).to(tl.int64)
            pitch_words = tl.load(pitches + member) // 8
            address = tl.load(destinations + member)
            target = tl.multiple_of(address, 16).to(tl.pointer_type(tl.int64))
            tl.store(target + target_row * pitch_words + column, values, mask)


def _pad16(nbytes):
    return -(-nbytes // 16) * 16


# A step that carries a row plan rounds its DMA up to a multiple of this many
# bytes, capped at the arena (the extra bytes are never scattered), and caches
# that many prefix view pairs of its own; plan sizes vary from step to step.
_PLANNED_DMA_STEP = 32 << 10
_PLANNED_PREFIX_CACHE = 64


class PackedCopy:
    """Persistent byte storage; the publication owner fences reuse of the arena."""

    @classmethod
    def create(cls, members, device):
        if device.type != "cuda" or len(members) < 2:
            return None, "packing requires multiple GPU destinations"
        if any(
            not b.source.is_contiguous() or not b.destination.is_contiguous()
            for b in members
        ):
            return None, "strided regions use checked direct"
        result = cls()
        result.members = members
        result.header_bytes = 16 * len(members)  # int64 (offset, byte count)
        # A planned step's header: main's int64 (offset, byte count) pairs, then
        # an int64 (row bytes, row map offset) pair per member
        # (`_submit_planned`).
        result.row_header_bytes = 32 * len(members)
        # Members that may take a row plan (`RowPlan`): two-dimensional rows of
        # whole 16-byte words at a word-aligned destination.
        result.row_members = tuple(
            b.unit == "rows"
            and b.source.ndim == 2
            and b.bytes_per_count % 16 == 0
            and b.destination.data_ptr() % 16 == 0
            for b in members
        )
        # Every member starts 16-byte aligned (`submit` pads), for the word
        # copy; a planned step also needs a row map of one int32 a row.
        capacity = result.row_header_bytes + sum(
            _pad16(b.capacity * b.bytes_per_count)
            + (_pad16(4 * b.row_capacity) if rows else 0)
            for b, rows in zip(members, result.row_members)
        )
        result.host = torch.empty(
            capacity, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        result.device = torch.empty(capacity, dtype=torch.uint8, device=device)
        result.header = result.host[: result.header_bytes].view(torch.int64).numpy()
        result.payload = memoryview(result.host.numpy())
        # Byte views avoid typed copies canonicalizing bool or casting BF16.
        result.sources = tuple(
            memoryview(b.source.reshape(-1).view(torch.uint8).numpy()) for b in members
        )
        result.destinations = torch.tensor(
            [b.destination.data_ptr() for b in members],
            dtype=torch.int64,
            device=device,
        )
        # Destinations the word copy may store to as int64 (16-byte aligned).
        result.wide = torch.tensor(
            [int(b.destination.data_ptr() % 16 == 0) for b in members],
            dtype=torch.int32,
            device=device,
        )
        result.prefixes = {}
        result.launch = scatter_packed_words
        result.kernel = None
        # The planned path (`plan_rows`, `_submit_planned`).
        result.capacity = capacity
        result.row_header = (
            result.host[: result.row_header_bytes].view(torch.int64).numpy()
        )
        result.arena = result.host.numpy()
        result.rows = tuple(
            (
                b.source.view(torch.uint8)
                .numpy()
                .reshape(b.row_capacity, b.bytes_per_count)
                if rows
                else None
            )
            for b, rows in zip(members, result.row_members)
        )
        result.pitches = torch.tensor(
            [
                b.bytes_per_count if rows else 0
                for b, rows in zip(members, result.row_members)
            ],
            dtype=torch.int64,
            device=device,
        )
        result.plans = [None] * len(members)
        result.planned_prefixes = {}
        result.row_launch = scatter_packed_rows
        result.row_kernel = None
        from atom.utils.block_tables import WHOLE_PREFIX_BYTES

        if any(
            rows and b.capacity * b.bytes_per_count > WHOLE_PREFIX_BYTES
            for b, rows in zip(members, result.row_members)
        ):
            # A member that can take plans: compile the planned kernel now, not
            # on the first planned publish mid-serving. Over an all-zero header
            # every member is an empty prefix, so it stores nothing.
            result.device[: result.row_header_bytes].zero_()
            result.row_kernel = result.row_launch[(len(members), 1, 1)](
                result.device,
                result.destinations,
                result.wide,
                result.pitches,
                WORDS=1024,
                num_warps=8,
            )
        return result, None

    def submit(self, counts):
        """Enqueue copies and return whether the packed arena is borrowed."""
        active = [i for i, count in enumerate(counts) if count]
        if len(active) == 1:
            # A single live member already takes one DMA; no packing/scatter.
            i = active[0]
            self.members[i]._copy(counts[i])
            return False
        offset = self.header_bytes
        largest = 0
        for i, (binding, count) in enumerate(zip(self.members, counts)):
            length = 0 if count is None else count * binding.bytes_per_count
            self.header[2 * i] = offset
            self.header[2 * i + 1] = length
            if length:
                self.payload[offset : offset + length] = self.sources[i][:length]
                offset += -(-length // 16) * 16
                largest = max(largest, length)
        prefix = self.prefixes.get(offset)
        if prefix is None:
            prefix = self.host[:offset], self.device[:offset]
            if len(self.prefixes) == 4:
                del self.prefixes[next(iter(self.prefixes))]
            self.prefixes[offset] = prefix
        # Header and payload share this single asynchronous H2D.
        prefix[1].copy_(prefix[0], non_blocking=True)
        # 1024 words (8 KiB) a program.
        grid = (len(self.members), (largest + 8191) // 8192, 1)
        if self.kernel is None:
            self.kernel = self.launch[grid](
                self.device, self.destinations, self.wide, WORDS=1024, num_warps=8
            )
        else:
            # Both argument storages and their layouts are fixed for this
            # binding. Reuse its compiled kernel; counts and grid stay dynamic.
            self.kernel[grid](self.device, self.destinations, self.wide)
        return True

    def plan_rows(self, i, plan):
        """Record member ``i``'s `RowPlan` (or None) for the next publish. A
        plan routes that one publish through `_submit_planned`; the plain
        `submit` stays untouched."""
        self.plans[i] = plan
        if any(p is not None for p in self.plans):
            self.submit = self._submit_planned
        elif "submit" in self.__dict__:
            del self.submit

    def _submit_planned(self, counts):
        """`submit` for one publish where a member carries a `RowPlan`: only
        the planned rows' leading bytes and an int32 row map for it, the
        plain prefix for the others, through `scatter_packed_rows`. Consumes
        the plans."""
        plans = self.plans
        self.plans = [None] * len(self.members)
        del self.submit
        active = [i for i, count in enumerate(counts) if count]
        if len(active) == 1:
            # A single live member already takes one DMA; no packing/scatter.
            i = active[0]
            self.members[i]._copy(counts[i])
            return False
        header = self.row_header
        rows_at = 2 * len(self.members)  # (row bytes, row map) after the pairs
        offset = self.row_header_bytes
        largest = 0
        for i, (binding, count) in enumerate(zip(self.members, counts)):
            plan = plans[i] if count else None
            header[2 * i] = offset
            if plan is not None:
                # Only the planned rows' leading bytes; the rest of every
                # destination row already holds the source's bytes.
                rows, row_bytes = plan.rows, plan.row_bytes
                length = len(rows) * row_bytes
                block = self.arena[offset : offset + length].reshape(
                    len(rows), row_bytes
                )
                gather_rows(block, self.rows[i][:, :row_bytes], rows)
                row_map = offset + _pad16(length)
                self.arena[row_map : row_map + 4 * len(rows)].view(np.int32)[:] = rows
                header[2 * i + 1] = length
                header[rows_at + 2 * i] = row_bytes
                header[rows_at + 2 * i + 1] = row_map
                offset = row_map + _pad16(4 * len(rows))
                largest = max(largest, length)
                continue
            length = 0 if count is None else count * binding.bytes_per_count
            header[2 * i + 1] = length
            header[rows_at + 2 * i] = 0
            if length:
                self.payload[offset : offset + length] = self.sources[i][:length]
                offset += _pad16(length)
                largest = max(largest, length)
        # Plan sizes vary step to step: round the DMA up to a few cached sizes.
        offset = min(-(-offset // _PLANNED_DMA_STEP) * _PLANNED_DMA_STEP, self.capacity)
        prefix = self.planned_prefixes.get(offset)
        if prefix is None:
            prefix = self.host[:offset], self.device[:offset]
            if len(self.planned_prefixes) == _PLANNED_PREFIX_CACHE:
                del self.planned_prefixes[next(iter(self.planned_prefixes))]
            self.planned_prefixes[offset] = prefix
        prefix[1].copy_(prefix[0], non_blocking=True)
        grid = (len(self.members), (largest + 8191) // 8192, 1)
        if self.row_kernel is None:
            self.row_kernel = self.row_launch[grid](
                self.device,
                self.destinations,
                self.wide,
                self.pitches,
                WORDS=1024,
                num_warps=8,
            )
        else:
            self.row_kernel[grid](
                self.device, self.destinations, self.wide, self.pitches
            )
        return True
