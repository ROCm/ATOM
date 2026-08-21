# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Auto-discovered model plugins for LMCache multiprocess offload."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any, ClassVar
import importlib
import pkgutil


class MPModelConnectorPlugin:
    """Declarative entry point for one model-specific MP connector."""

    name: ClassVar[str]
    model_types: ClassVar[frozenset[str]]
    worker_module: ClassVar[str]
    worker_class: ClassVar[str]
    scheduler_module: ClassVar[str]
    scheduler_class: ClassVar[str]

    @classmethod
    def build_worker(cls, config: Any) -> Any:
        module = importlib.import_module(cls.worker_module)
        connector_cls = getattr(module, cls.worker_class)
        return connector_cls(config)

    @classmethod
    def build_scheduler(cls, config: Any) -> Any:
        module = importlib.import_module(cls.scheduler_module)
        connector_cls = getattr(module, cls.scheduler_class)
        return connector_cls(config)


def _validate_plugin(plugin: type[MPModelConnectorPlugin]) -> None:
    if not getattr(plugin, "name", ""):
        raise ValueError(f"{plugin.__name__} must declare a non-empty name")
    model_types = getattr(plugin, "model_types", frozenset())
    if not model_types or not all(
        isinstance(model_type, str) and model_type for model_type in model_types
    ):
        raise ValueError(f"{plugin.__name__} must declare non-empty model_types")
    for attribute in (
        "worker_module",
        "worker_class",
        "scheduler_module",
        "scheduler_class",
    ):
        if not getattr(plugin, attribute, ""):
            raise ValueError(f"{plugin.__name__} must declare {attribute}")


@cache
def _plugins_by_model_type() -> dict[str, type[MPModelConnectorPlugin]]:
    """Import model subpackages and index their plugin subclasses."""

    plugins: dict[str, type[MPModelConnectorPlugin]] = {}
    package_dir = Path(__file__).parent
    for module in pkgutil.iter_modules([str(package_dir)]):
        if not module.ispkg or module.name.startswith("_"):
            continue
        imported = importlib.import_module(f"{__package__}.{module.name}")
        for value in vars(imported).values():
            if (
                not isinstance(value, type)
                or value is MPModelConnectorPlugin
                or not issubclass(value, MPModelConnectorPlugin)
            ):
                continue
            _validate_plugin(value)
            for model_type in value.model_types:
                previous = plugins.setdefault(model_type, value)
                if previous is not value:
                    raise ValueError(
                        f"LMCache MP model_type={model_type!r} is registered by "
                        f"both {previous.__name__} and {value.__name__}"
                    )
    return plugins


def model_type_from_config(config: Any) -> str | None:
    """Read the Hugging Face model type visible to both engine roles."""

    hf_config = getattr(config, "hf_config", None)
    text_config = getattr(hf_config, "text_config", hf_config)
    return getattr(text_config, "model_type", None)


def resolve_plugin(config: Any) -> type[MPModelConnectorPlugin]:
    """Return the plugin registered for ``config`` or fail explicitly."""

    model_type = model_type_from_config(config)
    plugin = _plugins_by_model_type().get(model_type)
    if plugin is None:
        supported = sorted(_plugins_by_model_type())
        raise NotImplementedError(
            f"lmcache_mp has no connector for model_type={model_type!r}; "
            f"supported model types: {supported}"
        )
    return plugin


__all__ = [
    "MPModelConnectorPlugin",
    "model_type_from_config",
    "resolve_plugin",
]
