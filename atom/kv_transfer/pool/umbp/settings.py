# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The `umbp_pool` connector's settings, read off its `kv_transfer_config`.

Example::

    --kv-transfer-config '{"kv_connector": "umbp_pool",
                           "kv_connector_extra_config": {
                               "address": "unix:///tmp/umbp-l1/server.sock",
                               "share_ratio": 0.5}}'

`address`        gRPC address of the UMBP standalone server (required).
`namespace`      prefix of this engine's pool names; two engines on one
                 server need different ones. Default "atom".
`group`          share group; pools exchange blocks only within one. Derived
                 from the model and KV format when empty.
`fetch`          copy prefixes in from other ranks' pools. Default true.
`share_ratio`    fraction of the paged blocks UMBP manages and shares, in
                 (0, 1]. This process always allocates the pool; UMBP adopts
                 that fraction of its blocks and the rest stays private. 1
                 (default): every block is UMBP's.
`placement`      below a share ratio of 1, where fresh content goes. "private"
                 (default): the private part first; content it is about to
                 lose moves into UMBP's part, where other ranks can load it.
                 "shared": UMBP's part first, so other ranks can load what was
                 just computed; content UMBP is about to reclaim from this
                 pool moves into the private part, for this rank alone.
                 "write_through": the private part first, and every block
                 computed there is also copied into UMBP's part, where other
                 ranks can load it; nothing moves when either part evicts.
`device_offset`  added to this process's device indices to get the UMBP
                 server's. 0 (default) when both see the same devices in the
                 same order. An engine that sees only its own devices while
                 the server sees those of several engines names where its
                 first one stands in the server's list (e.g. 4 for an engine
                 on devices 4-7 of a server on 0-7).
`session_migrate_margin`
                 with session affinity, move a session off a rank running this
                 fraction of the mean more requests than the mean (its history
                 follows through the shared pool). Unset (default): no moves.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from atom.kv_transfer.offload.config import connector_extras


@dataclass(frozen=True)
class UmbpPoolSettings:
    address: str
    namespace: str = "atom"
    group: str = ""
    fetch: bool = True
    share_ratio: float = 1.0
    placement: str = "private"
    session_migrate_margin: float | None = None
    device_offset: int = 0

    @classmethod
    def from_transfer_config(
        cls, kv_transfer_config: Mapping[str, Any] | None
    ) -> UmbpPoolSettings:
        extra = connector_extras(kv_transfer_config)
        address = str(extra.get("address", "") or "")
        if not address:
            raise ValueError(
                "the umbp_pool connector needs 'address', the gRPC address of the "
                "UMBP standalone server (e.g. unix:///tmp/umbp-l1/server.sock)"
            )
        ratio = float(extra.get("share_ratio", 1.0))
        if not 0 < ratio <= 1:
            raise ValueError(f"umbp_pool share_ratio={ratio}: a share in (0, 1]")
        placement = str(extra.get("placement", "private") or "private").lower()
        if placement not in ("private", "shared", "write_through"):
            raise ValueError(
                f"umbp_pool placement={placement!r}: "
                "'private', 'shared' or 'write_through'"
            )
        margin = extra.get("session_migrate_margin")
        if margin is not None:
            margin = float(margin)
            if margin < 0:
                raise ValueError(f"umbp_pool session_migrate_margin={margin} < 0")
        device_offset = int(extra.get("device_offset", 0) or 0)
        if device_offset < 0:
            raise ValueError(f"umbp_pool device_offset={device_offset} < 0")
        fetch = extra.get("fetch", True)
        if isinstance(fetch, str):
            fetch = fetch.strip().lower() not in ("0", "false", "no", "off")
        return cls(
            address=address,
            namespace=str(extra.get("namespace", "atom") or "atom"),
            group=str(extra.get("group", "") or ""),
            fetch=bool(fetch),
            share_ratio=ratio,
            placement=placement,
            session_migrate_margin=margin,
            device_offset=device_offset,
        )
