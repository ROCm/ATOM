# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Settings of the native Mooncake Store offload connector.

They come from the connector's ``kv_connector_extra_config`` -- the top-level
transfer config when it has none, as ``max_pending_saves`` is read -- under
keys prefixed ``mooncake_store.``. The scheduler and every worker parse the
same dict, so both sides agree on the chunk size and on the Store they address.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any, NamedTuple

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.mooncake_store.nic import parse_device_list

KEY_PREFIX = "mooncake_store."
_MIB = 1 << 20
_PROTOCOLS = ("rdma", "tcp")


class StorePool(NamedTuple):
    """The master of one Store pool."""

    master: str
    metadata: str


@dataclass(frozen=True)
class MooncakeStoreOffloadConfig:
    """Parsed ``mooncake_store.*`` settings; see the offload README's table."""

    # One shared pool: its master's RPC address and metadata server. Both None
    # when `pools` is set.
    master: str | None
    metadata: str | None
    # Per-NIC pools keyed by RDMA device; empty for one shared pool.
    pools: Mapping[str, StorePool]
    # Without pools, a worker NIC in this list is refused.
    owner_rdma_devices: tuple[str, ...]
    # Override of the PCI-local NIC: one for every GPU, or one per GPU ordinal.
    rdma_devices: tuple[str, ...]
    # Worker transfer protocol. The scheduler's lookup client is always tcp.
    protocol: str
    local_hostname: str
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

    def store_masters(self) -> list[StorePool]:
        """Every distinct ``(master, metadata)`` a lookup must ask."""
        if not self.pools:
            return [self.pool_of(None)]
        return sorted(set(self.pools.values()))

    def pool_of(self, nic: str | None) -> StorePool:
        """The pool of the worker whose Store client uses RDMA device ``nic``.

        Raises:
            ValueError: Per-NIC pools have none for ``nic``.
        """
        if not self.pools:
            return StorePool(str(self.master), str(self.metadata))
        pool = self.pools.get(nic)
        if pool is None:
            raise ValueError(
                f"mooncake_store.pools has no pool for RDMA device {nic!r} "
                f"(pools: {', '.join(sorted(self.pools))})"
            )
        return pool


# The known setting names, after KEY_PREFIX.
_FIELDS = tuple(field.name for field in fields(MooncakeStoreOffloadConfig))


def parse_mooncake_store_config(
    kv_transfer_config: Mapping[str, Any] | None,
) -> MooncakeStoreOffloadConfig:
    """Parse and validate the ``mooncake_store.*`` keys of a transfer config.

    Keys outside that prefix belong to other parts of ATOM and are ignored.

    Raises:
        ValueError: An unknown ``mooncake_store.*`` key, a value of the wrong
            type or out of range, neither a shared master nor pools, or pools
            without rdma.
    """
    kvc = kv_transfer_config or {}
    extra = kvc.get("kv_connector_extra_config", kvc) or {}
    if not isinstance(extra, Mapping):
        raise ValueError(  # noqa: TRY004  # a settings error, like every other
            "kv_connector_extra_config must be a JSON object"
        )
    values: dict[str, Any] = {}
    for key, value in extra.items():
        if not (isinstance(key, str) and key.startswith(KEY_PREFIX)):
            continue
        name = key[len(KEY_PREFIX) :]
        if name not in _FIELDS:
            raise ValueError(
                f"unknown Mooncake Store offload setting {key!r}; known: "
                f"{', '.join(KEY_PREFIX + field for field in _FIELDS)}"
            )
        values[name] = value

    protocol = _choice(
        "mooncake_store.protocol", values.get("protocol", "rdma"), _PROTOCOLS
    )
    pools = _pools(values.get("pools"))
    master = values.get("master")
    metadata = values.get("metadata")
    if pools:
        if master is not None or metadata is not None:
            raise ValueError(
                "set either mooncake_store.master and mooncake_store.metadata "
                "(one shared pool) or mooncake_store.pools (one pool per NIC), "
                "not both"
            )
        if protocol != "rdma":
            raise ValueError(
                "mooncake_store.pools keys the Store pools by RDMA device; a tcp "
                "Store client has none"
            )
    else:
        if master is None or metadata is None:
            raise ValueError(
                "the Mooncake Store offload needs mooncake_store.master and "
                "mooncake_store.metadata, or mooncake_store.pools"
            )
        master = _address("mooncake_store.master", master)
        metadata = _text("mooncake_store.metadata", metadata)

    local_hostname = values.get("local_hostname")
    if local_hostname is None:
        from atom.utils.network import get_ip

        local_hostname = get_ip()

    return MooncakeStoreOffloadConfig(
        master=master,
        metadata=metadata,
        pools=pools,
        owner_rdma_devices=_devices(
            "mooncake_store.owner_rdma_devices", values.get("owner_rdma_devices", "")
        ),
        rdma_devices=_devices(
            "mooncake_store.rdma_devices", values.get("rdma_devices", "")
        ),
        protocol=protocol,
        local_hostname=_text("mooncake_store.local_hostname", local_hostname),
        chunk_tokens=offcfg._strict_integer(
            "mooncake_store.chunk_tokens", values.get("chunk_tokens", 256), minimum=1
        ),
        load_pool_mib=offcfg._strict_integer(
            "mooncake_store.load_pool_mib", values.get("load_pool_mib", 1024), minimum=1
        ),
        save_pool_mib=offcfg._strict_integer(
            "mooncake_store.save_pool_mib", values.get("save_pool_mib", 256), minimum=1
        ),
        save_abandon_timeout_s=_positive_seconds(
            "mooncake_store.save_abandon_timeout_s",
            values.get("save_abandon_timeout_s", 300.0),
        ),
    )


def check_engine_compatibility(cfg: MooncakeStoreOffloadConfig, config: Any) -> None:
    """Refuse what phase 1 does not move: non-dense layouts, DCP, odd chunks.

    The Store holds one object per (rank, chunk) of the dense codec's opaque
    block bytes. A hybrid, M3 or Kimi-K3 layout needs PAGE regions or a state
    tier this connector does not carry, and a DCP rank's pages cover a virtual
    block that the chunk keys do not name.

    Raises:
        ValueError: The model is not on the dense layout, DCP > 1, or the
            chunk is not a whole number of KV blocks.
    """
    layout = offcfg.select_offload_layout(config)
    if layout != "dense":
        raise ValueError(
            f"the Mooncake Store offload moves only the dense KV layout; this "
            f"model resolves to {layout!r}"
        )
    dcp = int(getattr(config, "decode_context_parallel_size", 1) or 1)
    if dcp > 1:
        raise ValueError(
            "the Mooncake Store offload does not support decode context "
            f"parallelism (decode_context_parallel_size={dcp})"
        )
    block_size = offcfg._strict_integer(
        "KV cache block size", config.kv_cache_block_size, minimum=1
    )
    if cfg.chunk_tokens % block_size:
        raise ValueError(
            f"mooncake_store.chunk_tokens={cfg.chunk_tokens} must be a multiple "
            f"of the KV cache block size {block_size}"
        )


# Every bad setting is a ValueError, whatever is wrong with it, as
# `offcfg._strict_integer` raises them: a caller handling configuration errors
# should not need to know which of them were type errors.


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without spaces")
    return value


def _address(name: str, value: Any) -> str:
    """A ``host:port`` RPC address."""
    text = _text(name, value)
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"{name} must be host:port, got {value!r}")
    return text


def _devices(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a comma-separated string")  # noqa: TRY004
    return tuple(parse_device_list(value))


def _pools(value: Any) -> dict[str, StorePool]:
    """``mooncake_store.pools``, RDMA device -> its pool; empty when unset."""
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            "mooncake_store.pools must be a non-empty JSON object keyed by RDMA "
            "device"
        )
    pools = {}
    for device, pool in value.items():
        name = f"mooncake_store.pools[{device!r}]"
        if not isinstance(pool, Mapping):
            raise ValueError(  # noqa: TRY004
                f'{name} must be {{"master": "host:port", "metadata": "<url>"}}'
            )
        pools[_text(name, device)] = StorePool(
            _address(f"{name}.master", pool.get("master")),
            _text(f"{name}.metadata", pool.get("metadata")),
        )
    return pools


def _choice(name: str, value: Any, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")
    return value


def _positive_seconds(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number of seconds")  # noqa: TRY004
    seconds = float(value)
    # 0 would switch off every leak reclaimer of the engine, not just this one.
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return seconds
