"""Dense single-pool KV-cache manager."""

from __future__ import annotations

from collections.abc import Iterable
from copy import copy
from typing import Any

from atom.kv_cache.base_manager import BaseKvCacheManager
from atom.kv_cache.batch import KvBatchTables


class DenseKvCacheManager(BaseKvCacheManager):
    """Primary paged-cache manager with no DSV4 sidecar lifecycle."""

    def __init__(self, config: Any):
        # Keep the caller's cross-process layout immutable.  The compatibility
        # core self-guards these fields, but clearing them here ensures a stale
        # DSV4 geometry cannot instantiate an arena on a dense manager.
        dense_config = copy(config)
        dense_config.num_swa_blocks = 0
        dense_config.swa_window_size = 0
        dense_config.v4_arena_group_specs = None
        dense_config.enable_v4_csa_prefix_state_cache = False
        super().__init__(dense_config)

    def materialize_window(self, seq: Any, seq_len: int) -> None:
        del seq, seq_len

    def ensure_window_for_tokens(
        self, seq: Any, num_cached_tokens: int, num_new_tokens: int
    ) -> None:
        del seq, num_cached_tokens, num_new_tokens

    def finish_prefill_chunk(self, seq: Any) -> None:
        del seq

    def build_batch_tables(self, seqs: Iterable[Any]) -> KvBatchTables:
        return KvBatchTables.empty(num_sequences=sum(1 for _ in seqs))
