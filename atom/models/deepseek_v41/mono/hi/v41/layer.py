# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1 layer: the names the engine hook imports.

The recipe is ``recipe`` (the blocks of each kind, the placement passes), the schedule ``schedules`` (every knob, its
(kind, rows) entries), a rank's weights ``layer_impl.pack``; their names are re-exported here unchanged.
"""

from .layer_impl import pack
from .recipe import (
    CHAIN_PEER,
    EPS,
    EXPERTS,
    HIDDEN,
    Source,
    chain_peer_first,
    chain_pend,
    channels,
    engram_kv_spill_ok,
    engram_kv_stages,
    moe_route_early,
    residual_layout,
    sgpr_spill_ok,
    window_layer_blocks,
)

__all__ = [
    "CHAIN_PEER", "EPS", "EXPERTS", "HIDDEN", "Source", "chain_peer_first", "chain_pend", "channels",
    "engram_kv_spill_ok", "engram_kv_stages", "moe_route_early", "pack", "residual_layout", "sgpr_spill_ok",
    "window_layer_blocks",
]  # fmt: skip
