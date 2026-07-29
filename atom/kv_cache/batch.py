"""Physical KV-cache tables attached to a scheduled batch."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np


def _empty_sources() -> np.ndarray:
    return np.empty(0, dtype=np.int32)


@dataclass(frozen=True, slots=True)
class KvBatchTables:
    """Backend-specific physical tables for one scheduled batch.

    Logical block tables remain on ``ScheduledBatch``.  Dense managers return
    an empty instance, while layout-specific managers populate only the fields
    their backend consumes.
    """

    arena_block_tables: dict[str, list[list[int]]] = field(default_factory=dict)
    arena_swa_block_tables: dict[str, list[list[int]]] = field(default_factory=dict)
    csa_main_page_tables: list[list[int]] = field(default_factory=list)
    csa_idx_page_tables: list[list[int]] = field(default_factory=list)
    v4_csa_boundary_source_main: np.ndarray = field(default_factory=_empty_sources)
    v4_csa_boundary_source_idx: np.ndarray = field(default_factory=_empty_sources)

    @classmethod
    def empty(
        cls,
        *,
        num_sequences: int = 0,
        boundary_source_ids: Sequence[int] | np.ndarray | None = None,
    ) -> "KvBatchTables":
        """Build empty physical tables with logical CSA source passthrough."""
        if boundary_source_ids is None:
            sources = np.full(num_sequences, -1, dtype=np.int32)
        else:
            sources = np.asarray(boundary_source_ids, dtype=np.int32)
        # Main and indexer intentionally share the same source array.
        return cls(
            v4_csa_boundary_source_main=sources,
            v4_csa_boundary_source_idx=sources,
        )

    @property
    def is_empty(self) -> bool:
        return not (
            self.arena_block_tables
            or self.arena_swa_block_tables
            or self.csa_main_page_tables
            or self.csa_idx_page_tables
        )
