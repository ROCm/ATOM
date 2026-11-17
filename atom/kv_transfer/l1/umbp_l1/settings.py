# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The `umbp_l1` connector's settings, read off its `kv_transfer_config`.

Example::

    --kv-transfer-config '{"kv_connector": "umbp_l1",
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
class UmbpL1Settings:
    address: str
    namespace: str = "atom"
    group: str = ""
    fetch: bool = True
    share_ratio: float = 1.0
    session_migrate_margin: float | None = None

    @classmethod
    def from_transfer_config(
        cls, kv_transfer_config: Mapping[str, Any] | None
    ) -> UmbpL1Settings:
        extra = connector_extras(kv_transfer_config)
        address = str(extra.get("address", "") or "")
        if not address:
            raise ValueError(
                "the umbp_l1 connector needs 'address', the gRPC address of the "
                "UMBP standalone server (e.g. unix:///tmp/umbp-l1/server.sock)"
            )
        ratio = float(extra.get("share_ratio", 1.0))
        if not 0 < ratio <= 1:
            raise ValueError(f"umbp_l1 share_ratio={ratio}: a share in (0, 1]")
        margin = extra.get("session_migrate_margin")
        if margin is not None:
            margin = float(margin)
            if margin < 0:
                raise ValueError(f"umbp_l1 session_migrate_margin={margin} < 0")
        fetch = extra.get("fetch", True)
        if isinstance(fetch, str):
            fetch = fetch.strip().lower() not in ("0", "false", "no", "off")
        return cls(
            address=address,
            namespace=str(extra.get("namespace", "atom") or "atom"),
            group=str(extra.get("group", "") or ""),
            fetch=bool(fetch),
            share_ratio=ratio,
            session_migrate_margin=margin,
        )
