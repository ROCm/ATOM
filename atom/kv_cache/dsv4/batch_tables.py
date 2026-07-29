"""Build DeepSeek-V4 physical batch tables from logical cache IDs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from atom.kv_cache.batch import KvBatchTables


class ArenaBatchTranslator(Protocol):
    """Narrow arena surface required for scheduler-side table translation."""

    enabled: bool

    def group_names(self) -> list[str]: ...

    def physical_compress_table(
        self, group: str, logical_table: list[int]
    ) -> list[int]: ...

    def physical_swa_table(self, group: str, logical_table: list[int]) -> list[int]: ...

    def swa_page(self, group: str, logical_id: int) -> int: ...

    def is_swa_backed(self, logical_id: int) -> bool: ...


def _csa_physical_swa_page(arena: ArenaBatchTranslator, logical_id: int) -> int:
    """Translate a CSA source/destination without unsafe page-zero fallback."""
    logical_id = int(logical_id)
    if logical_id < 0 or not arena.is_swa_backed(logical_id):
        return -1
    return int(arena.swa_page("c4", logical_id))


def build_dsv4_batch_tables(
    *,
    arena: ArenaBatchTranslator | None,
    block_tables: Sequence[Sequence[int]],
    swa_block_tables: Sequence[Sequence[int]],
    v4_csa_boundary_source_ids: Sequence[int] | np.ndarray,
) -> KvBatchTables:
    """Translate logical DSV4 tables into arena physical pages.

    Bulk arena tables preserve the arena's existing page-zero fallback because
    their kernels use padded table entries.  CSA capture and restore do not:
    logical ``-1`` and unbacked IDs must remain ``-1`` so kernels skip them.
    """
    logical_sources = np.asarray(v4_csa_boundary_source_ids, dtype=np.int32)
    if arena is None or not getattr(arena, "enabled", False):
        return KvBatchTables.empty(boundary_source_ids=logical_sources)

    physical_blocks: dict[str, list[list[int]]] = {}
    physical_swa: dict[str, list[list[int]]] = {}
    for group in arena.group_names():
        physical_blocks[group] = [
            arena.physical_compress_table(group, list(table)) for table in block_tables
        ]
        physical_swa[group] = [
            arena.physical_swa_table(group, list(table)) for table in swa_block_tables
        ]

    csa_pages: list[list[int]] = []
    source_pages = logical_sources
    if "c4" in physical_swa:
        csa_pages = [
            [_csa_physical_swa_page(arena, block_id) for block_id in table]
            for table in swa_block_tables
        ]
        source_pages = np.asarray(
            [_csa_physical_swa_page(arena, source_id) for source_id in logical_sources],
            dtype=np.int32,
        )

    # Main and indexer share the c4 chunk.  Keep object identity so callers
    # cannot accidentally diverge the two views during migration.
    return KvBatchTables(
        arena_block_tables=physical_blocks,
        arena_swa_block_tables=physical_swa,
        csa_main_page_tables=csa_pages,
        csa_idx_page_tables=csa_pages,
        v4_csa_boundary_source_main=source_pages,
        v4_csa_boundary_source_idx=source_pages,
    )
