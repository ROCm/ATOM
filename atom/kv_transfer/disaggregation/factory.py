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
from collections.abc import Iterable
from typing import Any, ClassVar

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

    _registry: ClassVar[dict[str, dict[str, str]]] = {}
    _aliases: ClassVar[dict[str, str]] = {}
    _requires_pd_staging: ClassVar[dict[str, bool]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        worker_module: str,
        worker_class: str,
        scheduler_module: str,
        scheduler_class: str,
        aliases: Iterable[str] = (),
        requires_pd_staging: bool = True,
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
        canonical = name.casefold()
        cls._aliases[canonical] = name
        for alias in aliases:
            normalized = alias.strip().casefold()
            if not normalized:
                raise ValueError("KV connector aliases must be non-empty")
            existing = cls._aliases.get(normalized)
            if existing is not None and existing != name:
                raise ValueError(
                    f"KV connector alias {alias!r} is already registered for "
                    f"{existing!r}"
                )
            cls._aliases[normalized] = name
        cls._requires_pd_staging[name] = bool(requires_pd_staging)

    @classmethod
    def canonical_name(cls, value: object, *, path: str = "kv_transfer_config") -> str:
        """Resolve a configured connector name through the shared registry.

        Configuration parsing, attention-pool allocation, and connector
        construction must use the same aliases.  Keeping this in the factory
        prevents model backends from inspecting the private registry.
        """

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} requires a non-empty 'kv_connector' string")
        normalized = value.strip().casefold()
        canonical = cls._aliases.get(normalized)
        if canonical is None:
            available = sorted(cls._registry)
            raise ValueError(
                f"{path} has unknown KV connector {value!r}; available: {available}"
            )
        return canonical

    @classmethod
    def topology_uses_pd_staging(
        cls,
        kv_transfer_config: dict[str, Any] | None,
        *,
        path: str = "kv_transfer_config",
    ) -> bool:
        """Return whether the configured connector needs compressor P/D staging."""

        if kv_transfer_config is None or kv_transfer_config == {}:
            return False
        if not isinstance(kv_transfer_config, dict):
            raise TypeError(
                "kv_transfer_config must be a dict or None, "
                f"got {type(kv_transfer_config).__name__}"
            )
        connector = cls.canonical_name(
            kv_transfer_config.get("kv_connector"), path=path
        )
        return cls._requires_pd_staging.get(connector, True)

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
        backend_name = cls.canonical_name(
            kv_cfg.get("kv_connector", "moriio"), path="kv_transfer_config"
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

KVConnectorFactory.register(
    "native",
    worker_module="atom.kv_transfer.disaggregation.native.native_connector",
    worker_class="NativeConnector",
    scheduler_module="atom.kv_transfer.disaggregation.native.native_connector",
    scheduler_class="NativeConnectorScheduler",
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
