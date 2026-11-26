# SPDX-License-Identifier: MIT
"""Pack metadata into one pinned arena, upload once, scatter to stable addresses."""

import torch
import triton
import triton.language as tl


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
        # Every member starts 16-byte aligned (`submit` pads), for the word copy.
        capacity = result.header_bytes + sum(
            -(-(b.capacity * b.bytes_per_count) // 16) * 16 for b in members
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
