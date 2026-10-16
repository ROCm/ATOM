# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The ``mooncake_store`` registration: one worker and scheduler per layout.

The model picks them, as it picks the ``lmcache_offload`` codec: dense KV
goes through :mod:`.worker` and :mod:`.scheduler`, Kimi-K3 through
:mod:`.kimi_k3`, which adds its KDA state images. Config only, so the
scheduler process and every worker pick the same pair.
"""

from __future__ import annotations

from typing import Any

from atom.kv_transfer.offload import config as offcfg


def _is_kimi_k3(config: Any) -> bool:
    return offcfg.select_offload_layout(config) == "kimi_k3"


def _worker_class(config: Any) -> type:
    if _is_kimi_k3(config):
        from atom.kv_transfer.offload.mooncake_store.kimi_k3 import (
            KimiK3MooncakeStoreConnector,
        )

        return KimiK3MooncakeStoreConnector
    from atom.kv_transfer.offload.mooncake_store.worker import (
        MooncakeStoreOffloadConnector,
    )

    return MooncakeStoreOffloadConnector


def _scheduler_class(config: Any) -> type:
    if _is_kimi_k3(config):
        from atom.kv_transfer.offload.mooncake_store.kimi_k3 import (
            KimiK3MooncakeStoreScheduler,
        )

        return KimiK3MooncakeStoreScheduler
    from atom.kv_transfer.offload.mooncake_store.scheduler import (
        MooncakeStoreOffloadScheduler,
    )

    return MooncakeStoreOffloadScheduler


class MooncakeStoreConnector:
    """Builds the Store worker for the model's offload layout."""

    def __new__(cls, config: Any) -> Any:
        return _worker_class(config)(config)

    @classmethod
    def kv_budget_reserve_bytes(cls, config: Any) -> int:
        return _worker_class(config).kv_budget_reserve_bytes(config)


class MooncakeStoreConnectorScheduler:
    """Builds the Store scheduler for the model's offload layout."""

    def __new__(cls, config: Any) -> Any:
        return _scheduler_class(config)(config)


__all__ = ["MooncakeStoreConnector", "MooncakeStoreConnectorScheduler"]
