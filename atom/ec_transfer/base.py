# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The interface image embeddings move from an encoder (E) to prefill (P) by.

E's ``EcWriter.put`` stores a request's embedding rows and returns a handle;
the handle travels to P inside the request as ``ec_transfer_params``, and P's
``EcReader.get`` returns the rows, or None when P must run its own vision
tower. ATOMesh and the API carry the handle without reading it.

A handle is a JSON object. Every backend fills the same top-level fields and
keeps its own under its name::

    {"transport": "mooncake", "key": <cache_seed>, "rows": 4,
     "dtype": "bfloat16", "mooncake": {...}}
"""

from __future__ import annotations

from typing import Any, Protocol

import torch


class EcStoreFull(RuntimeError):
    """E has no room for the rows; the request goes on without a handle."""


class EcWriter(Protocol):
    def put(self, key: int, rows: torch.Tensor) -> dict[str, Any]:
        """Store ``rows`` under ``key`` and return the handle P reads them by.
        Raises ``EcStoreFull`` when there is no room."""
        ...


class EcReader(Protocol):
    def get(
        self, handle: dict[str, Any], key: int, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """The rows ``handle`` names, or None if they cannot be read, are not
        ``key``'s or not ``dtype``: P then encodes the images itself."""
        ...


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def handle_matches(handle: Any, transport: str, key: int, dtype: torch.dtype) -> bool:
    """Whether ``handle``'s common fields name ``key``'s rows in ``dtype`` on
    ``transport``. A handle from elsewhere may be malformed: never raise."""
    return (
        isinstance(handle, dict)
        and handle.get("transport") == transport
        and handle.get("key") == key
        and handle.get("dtype") == dtype_name(dtype)
        and type(handle.get("rows")) is int
        and handle["rows"] > 0
        and isinstance(handle.get(transport), dict)
    )


def _backends() -> dict[str, tuple[type, type]]:
    # Imported on use: a backend's library is needed only where it runs.
    from atom.ec_transfer.mooncake import MooncakeEcReader, MooncakeEcWriter

    return {"mooncake": (MooncakeEcWriter, MooncakeEcReader)}


def create_writer(
    transport: str, size_bytes: int, device: torch.device | str, **options: Any
) -> EcWriter:
    """E's writer on ``transport``, holding up to ``size_bytes``."""
    backend = _backends().get(transport)
    if backend is None:
        raise ValueError(f"unknown image embedding transport {transport!r}")
    return backend[0](size_bytes, device, **options)


def create_reader(
    transport: str, size_bytes: int, device: torch.device | str, **options: Any
) -> EcReader:
    """P's reader on ``transport``. Raises when the transport is unknown or
    cannot start here; the caller then encodes images itself."""
    backend = _backends().get(transport)
    if backend is None:
        raise ValueError(f"unknown image embedding transport {transport!r}")
    return backend[1](size_bytes, device, **options)
