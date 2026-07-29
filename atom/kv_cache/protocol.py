"""Scheduler-facing KV-cache manager contract."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from atom.kv_cache.batch import KvBatchTables


@runtime_checkable
class KvCacheManager(Protocol):
    """Only the control-plane surface used by Scheduler."""

    block_size: int

    def can_allocate(self, seq: Any) -> int: ...

    def allocate(self, seq: Any, num_cached_blocks: int = 0) -> None: ...

    def deallocate(self, seq: Any) -> None: ...

    def can_append(self, seq: Any, num_new_tokens: int = 1) -> bool: ...

    def may_append(self, seq: Any, num_new_tokens: int = 1) -> None: ...

    def hash_blocks(self, seq: Any, num_new_tokens: int) -> None: ...

    def materialize_window(self, seq: Any, seq_len: int) -> None: ...

    def ensure_window_for_tokens(
        self, seq: Any, num_cached_tokens: int, num_new_tokens: int
    ) -> None: ...

    def finish_prefill_chunk(self, seq: Any) -> None: ...

    def build_batch_tables(self, seqs: Iterable[Any]) -> KvBatchTables: ...

    def kv_usage(self) -> float: ...

    @property
    def num_total_blocks(self) -> int: ...

    @property
    def num_free_per_req_cache_groups(self) -> int: ...

    def get_block(self, block_id: int) -> Any: ...

    @property
    def kv_events_enabled(self) -> bool: ...

    def take_events(self) -> list[Any]: ...

    def clear_cache(self) -> None: ...

    def record_remote_store(
        self,
        block_hashes: list[int],
        token_ids: list[int],
        parent_block_hash: int | None = None,
    ) -> None: ...
