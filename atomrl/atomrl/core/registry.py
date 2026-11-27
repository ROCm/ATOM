# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Registry for workers, algorithms, environments, and reward functions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class Registry:
    """Generic registry for named components.

    Usage::

        worker_registry = Registry("worker")
        worker_registry.register("actor", AtomActorWorker)
        cls = worker_registry.get("actor")
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, Any] = {}

    def register(self, key: str, value: Any) -> None:
        if key in self._registry:
            logger.warning(
                "Overwriting %s registry entry '%s': %s -> %s",
                self.name,
                key,
                self._registry[key],
                value,
            )
        self._registry[key] = value

    def get(self, key: str) -> Any:
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"Unknown {self.name} '{key}'. Available: [{available}]")
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def keys(self) -> list[str]:
        return list(self._registry.keys())

    def decorator(self, key: str) -> Callable:
        """Use as a class/function decorator to register under `key`."""

        def wrapper(cls_or_fn: Any) -> Any:
            self.register(key, cls_or_fn)
            return cls_or_fn

        return wrapper


WORKER_REGISTRY = Registry("worker")
ALGORITHM_REGISTRY = Registry("algorithm")
REWARD_REGISTRY = Registry("reward")
ENV_REGISTRY = Registry("environment")
