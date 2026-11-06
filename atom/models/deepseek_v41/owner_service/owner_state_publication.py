"""Publish disjoint native STATE/PAGE readers in bounded owner packets."""

from dataclasses import dataclass

COPY_BLOCK = 4096
PACKET_BYTES = 32 * 1024 * 1024
DESCRIPTOR_ROWS = 16384


@dataclass(frozen=True)
class CopyRegion:
    owner: int
    offset: int
    size: int


@dataclass(frozen=True)
class PublicationChunk:
    owner: int
    regions: tuple[CopyRegion, ...]
    size: int
    tiles: int


def publication_chunks(
    regions, extent, *, packet_bytes=PACKET_BYTES, descriptor_rows=DESCRIPTOR_ROWS
):
    """Only independent regions commute with the original request order.

    An alias retains the serial native publication before any cache write.
    Byte and descriptor capacities are fixed bounds, not tuning parameters.
    """
    regions = tuple(regions)
    if packet_bytes < COPY_BLOCK or descriptor_rows < 1:
        raise ValueError("Invalid state publication capacity")
    for region in regions:
        if (
            not 0 <= region.owner < 4
            or region.offset < 0
            or region.size <= 0
            or region.offset + region.size > extent
        ):
            raise ValueError("State publication region escapes its native cache")
    previous_end = 0
    for region in sorted(regions, key=lambda item: item.offset):
        if region.offset < previous_end:
            return None
        previous_end = region.offset + region.size
    result = []
    for owner in range(4):
        parts, size, tiles = [], 0, 0
        for region in regions:
            if region.owner != owner:
                continue
            offset, remaining = region.offset, region.size
            while remaining:
                available = min(
                    packet_bytes - size, (descriptor_rows - tiles) * COPY_BLOCK
                )
                if not available:
                    result.append(PublicationChunk(owner, tuple(parts), size, tiles))
                    parts, size, tiles = [], 0, 0
                    continue
                take = min(remaining, available)
                part = CopyRegion(owner, offset, take)
                parts.append(part)
                size += take
                tiles += (take + COPY_BLOCK - 1) // COPY_BLOCK
                remaining -= take
                offset += take
        if parts:
            result.append(PublicationChunk(owner, tuple(parts), size, tiles))
    if sum(chunk.size for chunk in result) != sum(region.size for region in regions):
        raise ValueError("State publication did not cover every native byte")
    return tuple(result)


def cache_regions(cache, dirty, selected):
    """Offsets refer to the original allocation, including all index planes."""
    base = cache.backing.data_ptr()
    regions = []
    paged = cache.unit_regions()
    for request_id in selected:
        owner, slot, pages = dirty[request_id]
        state = cache.state_bytes[slot]
        if not state.is_contiguous() or state.element_size() != 1:
            raise ValueError("Native state publication requires dense byte regions")
        regions.append(CopyRegion(owner, state.data_ptr() - base, state.numel()))
        for page in pages:
            regions.extend(
                CopyRegion(owner, address - base + page * size, size)
                for address, size in paged
            )
    return tuple(regions)


def copy_descriptors(chunk):
    """Expand segments into one bounded byte-copy tile per device program."""
    import numpy as np

    sizes = np.asarray([region.size for region in chunk.regions], dtype=np.int64)
    counts = (sizes + COPY_BLOCK - 1) // COPY_BLOCK
    segment = np.repeat(np.arange(len(sizes)), counts)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts[:-1])))
    tile = np.arange(chunk.tiles) - np.repeat(starts, counts)
    within = tile * COPY_BLOCK
    offsets = np.asarray([region.offset for region in chunk.regions], dtype=np.int64)
    packet = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(sizes[:-1])))
    return np.stack(
        (
            offsets[segment] + within,
            packet[segment] + within,
            np.minimum(COPY_BLOCK, sizes[segment] - within),
        ),
        axis=1,
    )


class StatePacketPublisher:
    def __init__(
        self, group, *, packet_bytes=PACKET_BYTES, descriptor_rows=DESCRIPTOR_ROWS
    ):
        import torch

        from atom.utils import CpuGpuBuffer

        from .owner_state_publication_kernels import copy_bytes

        self.group = group
        self.packet_bytes, self.descriptor_rows = packet_bytes, descriptor_rows
        self.packet = torch.empty(packet_bytes, dtype=torch.uint8, device="cuda")
        self.descriptors = CpuGpuBuffer(
            descriptor_rows,
            3,
            dtype=torch.int64,
            device=self.packet.device,
            pin_memory=True,
        )
        self.uploaded = torch.cuda.Event()
        self.upload_recorded = False
        self.copy = copy_bytes

    def publish(self, cache, dirty, selected, counters):
        import torch.distributed as dist

        if (
            cache.backing.dtype != self.packet.dtype
            or not cache.backing.is_contiguous()
        ):
            raise ValueError("Bind the original contiguous byte allocation")
        regions = cache_regions(cache, dirty, selected)
        chunks = publication_chunks(
            regions,
            cache.backing.numel(),
            packet_bytes=self.packet_bytes,
            descriptor_rows=self.descriptor_rows,
        )
        if chunks is None:
            counters["packet_publication_alias_fallbacks"] += 1
            return False
        for chunk in chunks:
            if self.upload_recorded:
                # Only the pinned descriptor reader must finish. Default
                # stream order protects packet/cache reuse through unpack.
                self.uploaded.synchronize()
            self.descriptors.np[: chunk.tiles] = copy_descriptors(chunk)
            descriptors = self.descriptors.copy_to_gpu(chunk.tiles)
            self.uploaded.record()
            self.upload_recorded = True
            self.copy[(chunk.tiles,)](
                cache.backing,
                self.packet,
                descriptors,
                PACK=True,
                BLOCK=COPY_BLOCK,
                num_warps=4,
            )
            dist.broadcast(
                self.packet[: chunk.size],
                src=self.group.ranks[chunk.owner],
                group=self.group.device_group,
            )
            self.copy[(chunk.tiles,)](
                cache.backing,
                self.packet,
                descriptors,
                PACK=False,
                BLOCK=COPY_BLOCK,
                num_warps=4,
            )
            counters["packet_broadcasts"] += 1
            counters["packet_publication_bytes"] += chunk.size
        counters["packet_publication_regions"] += len(regions)
        counters["reconciliation_bytes"] += sum(region.size for region in regions)
        counters["packet_reconciliation_steps"] += 1
        return True
