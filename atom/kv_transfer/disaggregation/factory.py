# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
KV Connector Factory — registry-based instantiation.

Enables pluggable KV transfer backends without hard-coding class imports
in the engine.  The default backend (``"moriio"``) is registered at module
load time; additional backends can be added via :meth:`KVConnectorFactory.register`.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)

logger = logging.getLogger("atom")


class KVConnectorFactory:
    """Registry + factory for KV connector backends.

    Usage::

        # Registration (happens once, typically at import time)
        KVConnectorFactory.register(
            "moriio",
            worker_module="atom.kv_transfer.disaggregation.moriio.moriio_connector",
            worker_class="MoRIIOConnector",
            scheduler_module="atom.kv_transfer.disaggregation.moriio.moriio_connector",
            scheduler_class="MoRIIOConnectorScheduler",
        )

        # Instantiation (called from forward_context.py)
        connector = KVConnectorFactory.create_connector(config, role="worker")
    """

    _registry: dict[str, dict[str, str]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        worker_module: str,
        worker_class: str,
        scheduler_module: str,
        scheduler_class: str,
        mem_pool_module: str | None = None,
        mem_pool_func: str | None = None,
    ) -> None:
        """Register a KV connector backend.

        Args:
            name: Short identifier (e.g. ``"moriio"``).
            worker_module: Fully qualified module path for the worker connector.
            worker_class: Class name within *worker_module*.
            scheduler_module: Fully qualified module path for the scheduler connector.
            scheduler_class: Class name within *scheduler_module*.
            mem_pool_module: Optional module path providing a custom KV
                allocation pool for this backend (see
                :meth:`get_kv_mem_pool_context`). Omit if the backend can
                register/transfer ordinary torch memory (e.g. RDMA).
            mem_pool_func: Name of a ``func(config) -> context-manager`` in
                *mem_pool_module* returning a ``torch.cuda.use_mem_pool``
                context (or nullcontext when not applicable).
        """
        cls._registry[name] = {
            "worker_module": worker_module,
            "worker_class": worker_class,
            "scheduler_module": scheduler_module,
            "scheduler_class": scheduler_class,
            "mem_pool_module": mem_pool_module,
            "mem_pool_func": mem_pool_func,
        }

    @classmethod
    def create_connector(
        cls, config: Any, role: str = "worker"
    ) -> KVConnectorBase | KVConnectorSchedulerBase:
        """Instantiate a connector for the given *role*.

        The backend name is read from
        ``config.kv_transfer_config.get("kv_connector", "moriio")``.

        Args:
            config: Engine configuration object.
            role: ``"worker"`` or ``"scheduler"``.

        Returns:
            A concrete :class:`KVConnectorBase` or
            :class:`KVConnectorSchedulerBase` instance.
        """
        kv_cfg = getattr(config, "kv_transfer_config", {}) or {}
        backend_name = kv_cfg.get("kv_connector", "moriio")

        if backend_name not in cls._registry:
            raise ValueError(
                f"Unknown KV connector backend {backend_name!r}. "
                f"Available: {list(cls._registry.keys())}"
            )

        entry = cls._registry[backend_name]

        if role == "worker":
            mod = importlib.import_module(entry["worker_module"])
            klass = getattr(mod, entry["worker_class"])
        elif role == "scheduler":
            mod = importlib.import_module(entry["scheduler_module"])
            klass = getattr(mod, entry["scheduler_class"])
        else:
            raise ValueError(f"Unknown role {role!r}, expected 'worker' or 'scheduler'")

        logger.debug(
            "Creating KV connector: backend=%s, role=%s, class=%s",
            backend_name,
            role,
            klass.__name__,
        )
        return klass(config)

    @classmethod
    def get_kv_mem_pool_context(cls, config: Any):
        """Return a KV-allocation context for the active backend, else no-op.

        Some P/D transfer engines require KV cache tensors to live in a special
        allocation pool — e.g. mori-io fabric needs fabric-exportable VMM
        memory; an NVLink/cuMem engine would need its own pool. A backend opts
        in by registering ``mem_pool_module``/``mem_pool_func``; that provider
        inspects *config* and returns either a ``torch.cuda.use_mem_pool(...)``
        context or a ``nullcontext``. Engine code (``allocate_kv_cache``) calls
        this once around KV allocation and stays backend-agnostic, so adding a
        new engine's allocator needs no change here or in the model runner.

        This mirrors SGLang's ``maybe_init_custom_mem_pool`` hook — generic,
        not tied to any single transfer engine.
        """
        from contextlib import nullcontext

        # No KV transfer configured -> nothing to import or wrap.
        kv_cfg = getattr(config, "kv_transfer_config", None) or {}
        if not kv_cfg:
            return nullcontext()

        backend_name = kv_cfg.get("kv_connector", "moriio")
        entry = cls._registry.get(backend_name)
        if entry is None:
            return nullcontext()

        mem_pool_module = entry.get("mem_pool_module")
        mem_pool_func = entry.get("mem_pool_func")
        if not mem_pool_module or not mem_pool_func:
            return nullcontext()

        provider = getattr(importlib.import_module(mem_pool_module), mem_pool_func)
        return provider(config)


# ---------------------------------------------------------------------------
# Built-in backend registration
# ---------------------------------------------------------------------------

KVConnectorFactory.register(
    "moriio",
    worker_module="atom.kv_transfer.disaggregation.moriio.moriio_connector",
    worker_class="MoRIIOConnector",
    scheduler_module="atom.kv_transfer.disaggregation.moriio.moriio_connector",
    scheduler_class="MoRIIOConnectorScheduler",
    # FABRIC backend (UALink scale-up) needs KV in fabric-exportable VMM memory.
    mem_pool_module="atom.kv_transfer.disaggregation.moriio.moriio_common",
    mem_pool_func="maybe_fabric_kv_mem_pool_ctx",
)

KVConnectorFactory.register(
    "mooncake",
    worker_module="atom.kv_transfer.disaggregation.mooncake.mooncake_connector",
    worker_class="MooncakeConnector",
    scheduler_module="atom.kv_transfer.disaggregation.mooncake.mooncake_connector",
    scheduler_class="MooncakeConnectorScheduler",
)

# Composite backend: fans out to several sub-connectors listed under
# kv_transfer_config["connectors"] (e.g. moriio P/D + lmcache_offload on one
# prefill node). Lightweight import — no heavy deps until a sub is built.
KVConnectorFactory.register(
    "multi",
    worker_module="atom.kv_transfer.disaggregation.multi.multi_connector",
    worker_class="MultiConnector",
    scheduler_module="atom.kv_transfer.disaggregation.multi.multi_connector",
    scheduler_class="MultiConnectorScheduler",
)


# ATOM standalone CPU/NVMe KV offload backend (registers "lmcache_offload").
# Import is lightweight (offload/__init__ only records module paths as strings;
# the connector module is imported lazily by create_connector when selected).
try:
    import atom.kv_transfer.offload  # noqa: F401,E402
except Exception as _e:  # pragma: no cover - offload optional (needs lmcache)
    logger.debug("lmcache_offload backend not registered: %s", _e)
