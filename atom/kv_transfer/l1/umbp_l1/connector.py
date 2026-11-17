# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`umbp_l1`: a KV connector that hands the paged KV pool to MORI-UMBP.

Unlike a transfer or offload connector it moves nothing around the pool: it
provides the pool. The worker side answers `kv_pool_memory` (where the paged
pool's bytes come from: this process's allocation, with UMBP adopting the part
it manages); the scheduler side answers `make_block_pool` (the pool that decides
which block a request gets and what is evicted). Both are the same at every
share ratio: a ratio of 1 only leaves the private part empty. Every per-step
transfer hook is a no-op.

Enabled through `kv_transfer_config` (`settings.py`), and combinable with a P/D
or offload connector under `multi`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import ConnectorMetadata
from atom.kv_transfer.l1.umbp_l1.settings import UmbpL1Settings

logger = logging.getLogger("atom")


def _settings(config: Any) -> UmbpL1Settings:
    return UmbpL1Settings.from_transfer_config(
        getattr(config, "kv_transfer_config", None)
    )


def pool_sharing(kv_transfer_config) -> dict[str, Any]:
    """What DP routing may count on (`KVConnectorFactory.pool_sharing`): a
    rank copies a prefix it lacks from another rank's pool when fetching is on,
    and sessions move off a crowded rank when a margin is set. From the
    settings alone; nothing is built and mori is not imported."""
    settings = UmbpL1Settings.from_transfer_config(kv_transfer_config)
    return {
        "loads": settings.fetch,
        "session_migrate_margin": (
            settings.session_migrate_margin if settings.fetch else None
        ),
    }


class UmbpL1Connector(KVConnectorBase):
    """Worker side: the paged pool's memory."""

    is_producer = False

    def __init__(self, config: Any) -> None:
        self.config = config
        self.settings = _settings(config)

    def kv_pool_memory(self, config: Any, rank: int):
        from atom.kv_transfer.l1.umbp_l1.pool import make_umbp_pool_memory

        return make_umbp_pool_memory(config, self.settings, rank)

    def register_kv_caches(self, kv_caches, transfer_tensors=None, num_blocks=None):
        pass  # UMBP adopted its part of the pool in `bind_layout`

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        pass

    def get_finished(self) -> tuple[set, set]:
        return set(), set()


class UmbpL1ConnectorScheduler(KVConnectorSchedulerBase):
    """Scheduler side: the block pool."""

    is_producer = False

    def __init__(self, config: Any) -> None:
        self.config = config
        self.settings = _settings(config)
        self.pool = None

    def make_block_pool(
        self,
        config: Any,
        num_blocks: int,
        *,
        on_evict: Callable[[int], None] | None = None,
        on_remap: Callable[[int], None] | None = None,
        cache_policy: str = "lru",
        protected_ratio: float = 0.5,
    ):
        from atom.kv_transfer.l1.umbp_l1.pool import make_umbp_block_pool

        # One pool for every share ratio: at 1 its private part is empty.
        self.pool = make_umbp_block_pool(
            config,
            self.settings,
            num_blocks,
            on_evict=on_evict,
            on_remap=on_remap,
            cache_policy=cache_policy,
            protected_ratio=protected_ratio,
        )
        return self.pool

    # --------------------- transfer hooks: nothing to move -------------------- #
    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        return 0, False

    def build_connector_meta(self) -> ConnectorMetadata:
        return ConnectorMetadata()

    def update_state_after_alloc(self, seq: Any) -> None:
        pass

    def request_finished(self, seq: Any) -> None:
        pass

    def should_defer_free(self, seq: Any) -> bool:
        return False

    def send_finished(self, req_id: Any) -> None:
        pass

    def source_blocks_released(self, seq: Any) -> None:
        pass

    def drop_cached_kv(self) -> bool:
        # Old-weight KV must not be found again, here or by another rank.
        if self.pool is not None:
            self.pool.clear_index()
        return True
