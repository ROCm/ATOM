"""DSV4 group/schema mapping composed over generic byte-chunk arenas."""

from __future__ import annotations

from dataclasses import dataclass, field

from atom.kv_cache.pools.chunk_arena import (
    ArenaEmpty,
    ChunkArena,
    ChunkBackedFreeList,
)

OWNER_SWA = "swa"
OWNER_COMPRESS = "compress"

GROUP_C4 = "c4"
GROUP_C128 = "c128"
GROUP_DENSE = "dense"


def group_of_ratio(ratio: int) -> str:
    if ratio == 4:
        return GROUP_C4
    if ratio == 128:
        return GROUP_C128
    return GROUP_DENSE


@dataclass(frozen=True)
class ArenaGroupSpec:
    """Physical geometry for one DSV4 ratio group."""

    name: str
    num_chunks: int
    bytes_per_chunk: int
    chunk_rows: int
    owners: dict[str, int] = field(default_factory=dict)
    csa_state_off: int | None = None

    @property
    def compress_page_bytes(self) -> int | None:
        return self.owners.get(OWNER_COMPRESS)

    @property
    def has_compress(self) -> bool:
        return OWNER_COMPRESS in self.owners

    @property
    def max_compressed_blocks(self) -> int:
        page_bytes = self.compress_page_bytes
        return (
            self.num_chunks * (self.bytes_per_chunk // page_bytes) if page_bytes else 0
        )

    @classmethod
    def coerce(cls, spec: "ArenaGroupSpec | dict") -> "ArenaGroupSpec":
        if isinstance(spec, cls):
            return spec
        return cls(
            name=spec["name"],
            num_chunks=int(spec["num_chunks"]),
            bytes_per_chunk=int(spec["bytes_per_chunk"]),
            chunk_rows=int(spec.get("chunk_rows", spec["bytes_per_chunk"])),
            owners={key: int(value) for key, value in spec["owners"].items()},
            csa_state_off=spec.get("csa_state_off"),
        )


class ArenaGroup:
    """One DSV4 group composed with a generic :class:`ChunkArena`."""

    def __init__(
        self,
        name: str,
        arena: ChunkArena,
        owner_page_bytes: dict[str, int],
    ):
        self.name = name
        self.arena = arena
        self.free = {
            owner: ChunkBackedFreeList(arena, page_bytes=page_bytes)
            for owner, page_bytes in owner_page_bytes.items()
            if page_bytes > 0
        }
        self.phys: dict[str, dict[int, int]] = {owner: {} for owner in self.free}


class Dsv4UnifiedArena:
    """Map DSV4 logical compressed/SWA IDs to per-group physical pages."""

    def __init__(
        self,
        *,
        block_size: int,
        group_specs: list[dict] | list[ArenaGroupSpec],
    ):
        self.block_size = int(block_size)
        self.enabled = bool(group_specs) and self.block_size > 0
        self.groups: dict[str, ArenaGroup] = {}
        for raw_spec in group_specs:
            spec = ArenaGroupSpec.coerce(raw_spec)
            chunk_arena = ChunkArena(
                num_chunks=spec.num_chunks,
                bytes_per_chunk=spec.bytes_per_chunk,
            )
            self.groups[spec.name] = ArenaGroup(spec.name, chunk_arena, spec.owners)

    def _alloc_owners(self, owners: list[str], logical_id: int) -> None:
        granted: list[tuple[ArenaGroup, str]] = []
        try:
            for group in self.groups.values():
                for owner in owners:
                    free_list = group.free.get(owner)
                    if free_list is None or logical_id in group.phys[owner]:
                        continue
                    group.phys[owner][logical_id] = free_list.pop()
                    granted.append((group, owner))
        except ArenaEmpty:
            for group, owner in granted:
                group.free[owner].free(group.phys[owner].pop(logical_id))
            raise

    def _free_owners(self, owners: list[str], logical_id: int) -> None:
        for group in self.groups.values():
            for owner in owners:
                free_list = group.free.get(owner)
                if free_list is None:
                    continue
                page_id = group.phys[owner].pop(logical_id, None)
                if page_id is not None:
                    free_list.free(page_id)

    def _is_backed(self, owner: str, logical_id: int) -> bool:
        for group in self.groups.values():
            if owner in group.free:
                return logical_id in group.phys[owner]
        return True

    def _max_for(self, owner: str) -> int:
        capacities = [
            group.arena.num_chunks * group.free[owner].pages_per_chunk
            for group in self.groups.values()
            if owner in group.free
        ]
        return min(capacities) if capacities else 0

    def _available_for(self, owner: str) -> int:
        capacities = [
            group.free[owner].available()
            for group in self.groups.values()
            if owner in group.free
        ]
        return min(capacities) if capacities else (1 << 30)

    def max_compressed_blocks(self) -> int:
        return self._max_for(OWNER_COMPRESS) if self.enabled else 0

    def max_swa_blocks(self) -> int:
        return self._max_for(OWNER_SWA) if self.enabled else 0

    def compress_pages_per_chunk(self) -> int:
        if not self.enabled:
            return 0
        pages_per_chunk = [
            group.free[OWNER_COMPRESS].pages_per_chunk
            for group in self.groups.values()
            if OWNER_COMPRESS in group.free
        ]
        return min(pages_per_chunk) if pages_per_chunk else 0

    def compressed_available(self) -> int:
        return self._available_for(OWNER_COMPRESS) if self.enabled else (1 << 30)

    def swa_available(self) -> int:
        return self._available_for(OWNER_SWA) if self.enabled else (1 << 30)

    def can_alloc_compressed(self, count: int) -> bool:
        return not self.enabled or self.compressed_available() >= count

    def can_alloc_swa(self, count: int) -> bool:
        return not self.enabled or self.swa_available() >= count

    def alloc_compressed(self, block_id: int) -> None:
        if self.enabled:
            self._alloc_owners([OWNER_COMPRESS], block_id)

    def free_compressed(self, block_id: int) -> None:
        if self.enabled:
            self._free_owners([OWNER_COMPRESS], block_id)

    def is_compressed_backed(self, block_id: int) -> bool:
        return self._is_backed(OWNER_COMPRESS, block_id) if self.enabled else True

    def alloc_swa(self, swa_id: int) -> None:
        if self.enabled:
            self._alloc_owners([OWNER_SWA], swa_id)

    def free_swa(self, swa_id: int) -> None:
        if self.enabled:
            self._free_owners([OWNER_SWA], swa_id)

    def is_swa_backed(self, swa_id: int) -> bool:
        return self._is_backed(OWNER_SWA, swa_id) if self.enabled else True

    def group_names(self) -> list[str]:
        return list(self.groups)

    def compress_group_of_ratio(self, ratio: int) -> str:
        return group_of_ratio(ratio)

    def _physical_table(
        self, group: str, owner: str, logical_table: list[int]
    ) -> list[int]:
        arena_group = self.groups.get(group)
        if arena_group is None or owner not in arena_group.free:
            return [max(0, block_id) for block_id in logical_table]
        physical = arena_group.phys[owner]
        return [
            physical.get(block_id, 0) if block_id >= 0 else 0
            for block_id in logical_table
        ]

    def physical_compress_table(
        self, group: str, logical_table: list[int]
    ) -> list[int]:
        return self._physical_table(group, OWNER_COMPRESS, logical_table)

    def physical_swa_table(self, group: str, logical_swa_table: list[int]) -> list[int]:
        return self._physical_table(group, OWNER_SWA, logical_swa_table)

    def compress_page(self, group: str, block_id: int) -> int:
        return self.groups[group].phys[OWNER_COMPRESS][block_id]

    def swa_page(self, group: str, swa_id: int) -> int:
        return self.groups[group].phys[OWNER_SWA][swa_id]


# Historical name used by model-engine callers and external tests.
UnifiedKvArena = Dsv4UnifiedArena

__all__ = [
    "ArenaEmpty",
    "ArenaGroup",
    "ArenaGroupSpec",
    "Dsv4UnifiedArena",
    "GROUP_C4",
    "GROUP_C128",
    "GROUP_DENSE",
    "OWNER_COMPRESS",
    "OWNER_SWA",
    "UnifiedKvArena",
    "group_of_ratio",
]
