# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The layouts the Mooncake Store offload moves, and the codec for each.

Dense KV, M3 complete PAGEs and V4.1 replay PAGEs: every row of the
store-agnostic table in :mod:`atom.kv_transfer.offload.kv_store.layouts`.
``ATOM_MOONCAKE_STORE_DRAFT_WINDOW`` moves M3 to its draft-window split.
"""

from __future__ import annotations

from typing import Any

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.kv_store.layouts import (
    DENSE_CODEC_ID,
    REPLAY_PAGE_CODEC_ID,
    STORE_LAYOUTS,
    StoreLayout,
    StoreLayouts,
    build_draft_codec,
)
from atom.utils import envs

_LAYOUTS = StoreLayouts(
    ("dense", "m3", "dsv41"),
    refusal=(
        "the Mooncake Store offload moves the dense KV layout, M3 PAGE "
        "layout, and V4.1's replay PAGEs"
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
    return _LAYOUTS.codec_id(config, draft_window=envs.ATOM_MOONCAKE_STORE_DRAFT_WINDOW)


def build_mooncake_codec(
    config: Any,
    kv_caches: Any,
    transfer_tensors: Any,
    num_blocks: int | None,
    *,
    permit_per_request_state: bool,
) -> Any:
    """Build the codec for the layout `config` resolves to (of the target
    objects alone under ATOM_MOONCAKE_STORE_DRAFT_WINDOW)."""
    return _LAYOUTS.build_codec(
        config,
        kv_caches,
        transfer_tensors,
        num_blocks,
        permit_per_request_state=permit_per_request_state,
        draft_window=envs.ATOM_MOONCAKE_STORE_DRAFT_WINDOW,
    )


def build_mooncake_draft_codec(
    config: Any, transfer_tensors: Any, num_blocks: int | None
) -> Any | None:
    """The codec of this rank's draft objects under the split M3 layout; None
    when it is off or this rank holds no draft regions."""
    if not envs.ATOM_MOONCAKE_STORE_DRAFT_WINDOW:
        return None
    _LAYOUTS.require(offcfg.select_offload_layout(config), draft_window=True)
    return build_draft_codec(transfer_tensors, num_blocks)


__all__ = [
    "DENSE_CODEC_ID",
    "MOONCAKE_LAYOUTS",
    "REPLAY_PAGE_CODEC_ID",
    "build_mooncake_codec",
    "build_mooncake_draft_codec",
    "mooncake_codec_id",
    "require_mooncake_layout",
]
