# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Settings of the UMBP offload connector (``umbp_store``).

Copied from the Mooncake Store offload's ``config`` module and cut down
to what UMBP needs; a shared layer is planned. They come from the
connector's ``kv_connector_extra_config`` -- the top-level transfer config
when it has none -- under keys prefixed ``umbp_store.``. The scheduler and
every worker parse the same dict, so both sides agree on the chunk size and
on the server they address. No environment variable is read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.umbp_store.layouts import require_umbp_layout

KEY_PREFIX = "umbp_store."
_MIB = 1 << 20


@dataclass(frozen=True)
class UMBPStoreOffloadConfig:
    """Parsed ``umbp_store.*`` settings."""

    # The standalone server's gRPC address, e.g. ``unix:///tmp/umbp/server.sock``.
    # Started outside ATOM; every worker and the scheduler connect to it.
    address: str
    # How long a client waits for the server to answer when it connects.
    startup_timeout_ms: int
    chunk_tokens: int
    load_pool_mib: int
    save_pool_mib: int
    save_abandon_timeout_s: float

    @property
    def load_pool_bytes(self) -> int:
        return self.load_pool_mib * _MIB

    @property
    def save_pool_bytes(self) -> int:
        return self.save_pool_mib * _MIB


# The known setting names, after KEY_PREFIX.
_FIELDS = tuple(field.name for field in fields(UMBPStoreOffloadConfig))


def store_chunk_tokens(kv_transfer_config: Mapping[str, Any] | None) -> int:
    """The transfer granularity shared by routing and the UMBP connector."""
    return offcfg.connector_integer(
        offcfg.connector_extras(kv_transfer_config), f"{KEY_PREFIX}chunk_tokens", 256
    )


def parse_umbp_store_config(
    kv_transfer_config: Mapping[str, Any] | None,
) -> UMBPStoreOffloadConfig:
    """Parse and validate the ``umbp_store.*`` keys of a transfer config.

    Keys outside that prefix belong to other parts of ATOM and are ignored.

    Raises:
        ValueError: An unknown ``umbp_store.*`` key, a value of the wrong type
            or out of range, or no server address.
    """
    extra = offcfg.connector_extras(kv_transfer_config)
    values: dict[str, Any] = {}
    for key, value in extra.items():
        if not (isinstance(key, str) and key.startswith(KEY_PREFIX)):
            continue
        name = key[len(KEY_PREFIX) :]
        if name not in _FIELDS:
            raise ValueError(
                f"unknown UMBP offload setting {key!r}; known: "
                f"{', '.join(KEY_PREFIX + field for field in _FIELDS)}"
            )
        values[name] = value

    address = values.get("address")
    if address is None:
        raise ValueError(
            f"the UMBP offload needs {KEY_PREFIX}address: the standalone "
            "server's address, e.g. unix:///tmp/umbp/server.sock"
        )
    return UMBPStoreOffloadConfig(
        address=_text(f"{KEY_PREFIX}address", address),
        startup_timeout_ms=offcfg._strict_integer(
            f"{KEY_PREFIX}startup_timeout_ms",
            values.get("startup_timeout_ms", 10000),
            minimum=1,
        ),
        chunk_tokens=offcfg.connector_integer(extra, f"{KEY_PREFIX}chunk_tokens", 256),
        load_pool_mib=offcfg._strict_integer(
            f"{KEY_PREFIX}load_pool_mib", values.get("load_pool_mib", 1024), minimum=1
        ),
        save_pool_mib=offcfg._strict_integer(
            f"{KEY_PREFIX}save_pool_mib", values.get("save_pool_mib", 256), minimum=1
        ),
        save_abandon_timeout_s=_positive_seconds(
            f"{KEY_PREFIX}save_abandon_timeout_s",
            values.get("save_abandon_timeout_s", 300.0),
        ),
    )


def check_engine_compatibility(cfg: UMBPStoreOffloadConfig, config: Any) -> None:
    """Accept dense and M3 PAGE bytes; refuse stateful layouts and DCP.

    M3 uses complete PAGE regions, including its NSA index cache. Stateful
    layouts require a separate state tier, while a DCP rank's pages cover a
    virtual block that the chunk keys do not name.

    Raises:
        ValueError: The model uses neither dense nor M3 PAGE layout, DCP > 1,
            or the chunk is not a whole number of KV blocks.
    """
    require_umbp_layout(offcfg.select_offload_layout(config))
    dcp = int(getattr(config, "decode_context_parallel_size", 1) or 1)
    if dcp > 1:
        raise ValueError(
            "the UMBP offload does not support decode context parallelism "
            f"(decode_context_parallel_size={dcp})"
        )
    block_size = offcfg._strict_integer(
        "KV cache block size", config.kv_cache_block_size, minimum=1
    )
    if cfg.chunk_tokens % block_size:
        raise ValueError(
            f"{KEY_PREFIX}chunk_tokens={cfg.chunk_tokens} must be a multiple "
            f"of the KV cache block size {block_size}"
        )


# Every bad setting is a ValueError, whatever is wrong with it, as
# `offcfg._strict_integer` raises them.


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without spaces")
    return value


def _positive_seconds(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number of seconds")  # noqa: TRY004
    seconds = float(value)
    # 0 would switch off every leak reclaimer of the engine, not just this one.
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return seconds


__all__ = [
    "KEY_PREFIX",
    "UMBPStoreOffloadConfig",
    "check_engine_compatibility",
    "parse_umbp_store_config",
]
