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
    ) -> None:
        """Register a KV connector backend.

        Args:
            name: Short identifier (e.g. ``"moriio"``).
            worker_module: Fully qualified module path for the worker connector.
            worker_class: Class name within *worker_module*.
            scheduler_module: Fully qualified module path for the scheduler connector.
            scheduler_class: Class name within *scheduler_module*.
        """
        cls._registry[name] = {
            "worker_module": worker_module,
            "worker_class": worker_class,
            "scheduler_module": scheduler_module,
            "scheduler_class": scheduler_class,
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


# ---------------------------------------------------------------------------
# Built-in backend registration
# ---------------------------------------------------------------------------

KVConnectorFactory.register(
    "moriio",
    worker_module="atom.kv_transfer.disaggregation.moriio.moriio_connector",
    worker_class="MoRIIOConnector",
    scheduler_module="atom.kv_transfer.disaggregation.moriio.moriio_connector",
    scheduler_class="MoRIIOConnectorScheduler",
)

KVConnectorFactory.register(
    "mooncake",
    worker_module="atom.kv_transfer.disaggregation.mooncake.mooncake_connector",
    worker_class="MooncakeConnector",
    scheduler_module="atom.kv_transfer.disaggregation.mooncake.mooncake_connector",
    scheduler_class="MooncakeConnectorScheduler",
)
