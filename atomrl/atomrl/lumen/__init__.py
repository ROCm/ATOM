# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("atomrl") or None
except PackageNotFoundError:
    __version__ = None

if __version__ is None:
    try:
        __version__ = version("atom")
    except PackageNotFoundError:
        __version__ = "0.1.0"

__all__ = ["LumenConfig", "float8_e4m3", "float8_e5m2", "quantize"]


def __getattr__(name: str):
    if name == "quantize":
        return import_module("atomrl.lumen.quantize")
    if name == "LumenConfig":
        from atomrl.lumen.config import LumenConfig

        return LumenConfig
    if name in {"float8_e4m3", "float8_e5m2"}:
        float8 = import_module("atomrl.lumen.core.float8")
        return getattr(float8, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, "__version__"])
