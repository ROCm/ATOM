# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The layouts the Mooncake Store offload moves, and the codec for each.

Dense KV, M3 complete PAGEs, V4.1 replay PAGEs and Kimi-K3's dense KV: every
row of the store-agnostic table in :mod:`atom.kv_transfer.offload.kv_store.layouts`.
A layout with a state tier (`store_state_tier`) adds checkpoint images beside.
"""

from __future__ import annotations

from typing import Any

from atom.kv_transfer.offload.kv_store.layouts import (
    DENSE_CODEC_ID,
    KIMI_K3_CODEC_ID,
    REPLAY_PAGE_CODEC_ID,
    STORE_LAYOUTS,
    StoreLayout,
    StoreLayouts,
)

_LAYOUTS = StoreLayouts(
    ("dense", "m3", "dsv41", "kimi_k3"),
    refusal=(
        "the Mooncake Store offload moves the dense KV layout, M3 PAGE "
        "layout, V4.1's replay PAGEs, and Kimi-K3's KV with its state"
    ),
)

MOONCAKE_LAYOUTS: dict[str, StoreLayout] = {
    name: STORE_LAYOUTS[name] for name in _LAYOUTS.names
}


def require_mooncake_layout(name: str) -> StoreLayout:
    """The row for `name`, or the refusal the startup check reports."""
    return _LAYOUTS.require(name)


def mooncake_codec_id(config: Any) -> str:
    """The codec version folded into this server's Store namespace."""
    return _LAYOUTS.codec_id(config)


def build_mooncake_codec(
    config: Any,
    kv_caches: Any,
    transfer_tensors: Any,
    num_blocks: int | None,
    *,
    permit_per_request_state: bool,
) -> Any:
    """Build the codec for the layout `config` resolves to."""
    return _LAYOUTS.build_codec(
        config,
        kv_caches,
        transfer_tensors,
        num_blocks,
        permit_per_request_state=permit_per_request_state,
    )


__all__ = [
    "DENSE_CODEC_ID",
    "KIMI_K3_CODEC_ID",
    "MOONCAKE_LAYOUTS",
    "REPLAY_PAGE_CODEC_ID",
    "build_mooncake_codec",
    "mooncake_codec_id",
    "require_mooncake_layout",
]
