# SPDX-License-Identifier: MIT
"""Pack metadata into one pinned arena, upload once, scatter to stable addresses."""

import torch
import triton
import triton.language as tl


@triton.jit
def scatter_packed_bytes(arena, destinations, BLOCK: tl.constexpr):
    member = tl.program_id(0)
    header = arena.to(tl.pointer_type(tl.int64))
    start = tl.load(header + 2 * member)
    length = tl.load(header + 2 * member + 1)
    destination = tl.load(destinations + member).to(tl.pointer_type(tl.uint8))
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(arena + start + offset, offset < length, other=0)
    tl.store(destination + offset, values, offset < length)


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
        capacity = result.header_bytes + sum(
            b.capacity * b.bytes_per_count for b in members
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
        result.prefixes = {}
        result.launch = scatter_packed_bytes
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
                offset += length
                largest = max(largest, length)
        prefix = self.prefixes.get(offset)
        if prefix is None:
            prefix = self.host[:offset], self.device[:offset]
            if len(self.prefixes) == 4:
                del self.prefixes[next(iter(self.prefixes))]
            self.prefixes[offset] = prefix
        # Header and payload share this single asynchronous H2D.
        prefix[1].copy_(prefix[0], non_blocking=True)
        grid = (len(self.members), (largest + 1023) // 1024, 1)
        if self.kernel is None:
            self.kernel = self.launch[grid](self.device, self.destinations, BLOCK=1024)
        else:
            # Both argument storages and their layouts are fixed for this
            # binding. Reuse its compiled kernel; counts and grid stay dynamic.
            self.kernel[grid](self.device, self.destinations)
        return True
