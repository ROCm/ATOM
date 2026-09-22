# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Capability-selected public LMCache multiprocess connector.

The configured connector name stays ``lmcache_mp``. The scheduler selects its
implementation when ``BlockManager`` exposes a PAGE-backed checkpoint
coordinator; each worker makes the same choice when the attention backend
publishes checkpoint geometry plus its copy callback. No model name or
architecture string participates in the decision.
"""

from __future__ import annotations

from typing import Any

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.kv_transfer.offload._offload_common import validated_kv_role
from atom.kv_transfer.offload.mp import backend


def _publishes_native_state(transfer_tensors: Any) -> bool:
    """Validate and detect the reusable PAGE-backed native-state contract."""

    spec = getattr(transfer_tensors, "paged_state_checkpoint_spec", None)
    copy = getattr(transfer_tensors, "execute_paged_state_copies", None)
    if spec is None and copy is None:
        return False
    if spec is None:
        raise ValueError(
            "execute_paged_state_copies requires paged_state_checkpoint_spec"
        )
    if not callable(copy):
        raise TypeError(
            "paged_state_checkpoint_spec requires a callable execute_paged_state_copies"
        )
    return True


class LMCacheMPConnector(KVConnectorBase):
    """Worker shell selecting PAGE-only or native-state transport by capability."""

    is_producer = False

    def __init__(self, config: Any) -> None:
        backend._validate_mp_config(config)
        validated_kv_role(getattr(config, "kv_transfer_config", {}) or {})
        self._config = config
        self._impl: KVConnectorBase | None = None

    def _require_impl(self) -> KVConnectorBase:
        if self._impl is None:
            raise RuntimeError("lmcache_mp KV caches are not registered")
        return self._impl

    def register_kv_caches(
        self,
        kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        if self._impl is not None:
            raise RuntimeError("lmcache_mp KV caches are already registered")
        if _publishes_native_state(transfer_tensors):
            from atom.kv_transfer.offload.mp.native_state_worker import (
                NativeStateLMCacheMPConnector,
            )

            impl: KVConnectorBase = NativeStateLMCacheMPConnector(self._config)
        else:
            impl = backend.LMCacheMPConnector(self._config)
        impl.register_kv_caches(kv_caches, transfer_tensors, num_blocks)
        self._impl = impl

    def start_load_kv(self, metadata: Any) -> None:
        self._require_impl().start_load_kv(metadata)

    def get_finished(self) -> KVConnectorOutput:
        if self._impl is None:
            return KVConnectorOutput()
        return self._impl.get_finished()

    def get_finished_recv_blocks(self) -> list[int]:
        if self._impl is None:
            return []
        return self._impl.get_finished_recv_blocks()

    def __getattr__(self, name: str) -> Any:
        impl = self.__dict__.get("_impl")
        if impl is None:
            raise AttributeError(name)
        return getattr(impl, name)


class LMCacheMPConnectorScheduler(KVConnectorSchedulerBase):
    """Scheduler shell selecting transport after checkpoint capability exists."""

    is_producer = False
    is_offload = True

    def __init__(self, config: Any) -> None:
        backend._validate_mp_config(config)
        validated_kv_role(getattr(config, "kv_transfer_config", {}) or {})
        self._config = config
        self._impl: KVConnectorSchedulerBase | None = None
        self._block_manager: Any = None

    def _require_impl(self) -> KVConnectorSchedulerBase:
        if self._impl is None:
            raise RuntimeError("lmcache_mp scheduler is not bound to BlockManager")
        return self._impl

    def bind_block_manager(self, block_manager: Any) -> None:
        if self._block_manager is block_manager:
            return
        if self._block_manager is not None:
            raise RuntimeError(
                "lmcache_mp scheduler is already bound to a block manager"
            )
        coordinator = getattr(block_manager, "paged_state_checkpoints", None)
        if coordinator is None:
            impl: KVConnectorSchedulerBase = backend.LMCacheMPConnectorScheduler(
                self._config
            )
        else:
            from atom.kv_transfer.offload.mp.native_state_scheduler import (
                NativeStateLMCacheMPConnectorScheduler,
            )

            impl = NativeStateLMCacheMPConnectorScheduler(self._config)
        bind = getattr(impl, "bind_block_manager", None)
        if callable(bind):
            bind(block_manager)
        self._impl = impl
        self._block_manager = block_manager

    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        return self._require_impl().get_num_new_matched_tokens(seq)

    def build_connector_meta(self) -> Any:
        return self._require_impl().build_connector_meta()

    def update_state_after_alloc(self, seq: Any) -> None:
        self._require_impl().update_state_after_alloc(seq)

    def request_finished(self, seq: Any) -> None:
        self._require_impl().request_finished(seq)

    def should_defer_free(self, seq: Any) -> bool:
        return self._require_impl().should_defer_free(seq)

    def send_finished(self, req_id: Any) -> None:
        self._require_impl().send_finished(req_id)

    def source_blocks_released(self, seq: Any) -> None:
        self._require_impl().source_blocks_released(seq)

    def __getattr__(self, name: str) -> Any:
        impl = self.__dict__.get("_impl")
        if impl is None:
            raise AttributeError(name)
        return getattr(impl, name)


__all__ = ["LMCacheMPConnector", "LMCacheMPConnectorScheduler"]
