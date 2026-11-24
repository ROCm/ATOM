# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""4-rank pull collectives over IPC-mapped pools, FlyDSL (gfx950): the
sequence-parallel prefill's hand-off transport. The torch contract is
``reference.py``. ``FencedAllGatherComm`` unpacks the attention output's
all-to-all into o_proj_full's input; ``AllGatherComm`` gathers row-major fields
with device-owned sequence counters; ``SlotsReduceScatterComm`` adds the
graph-safe MoE slot combine + reduce-scatter (+ shard norm/quant) and the dense
reduce-scatter; ``NormAllGatherComm`` fuses the dense norm/quant into a gather."""

from .host import WORLD, Collectives, Pools
from .slots4_host import SlotsReduceScatterComm
from .all_gather_fence_h6144 import FencedAllGatherComm
from .gather_host import AllGatherComm
from .norm_gather_h6144 import NormAllGatherComm

__all__ = ["WORLD", "Collectives", "Pools", "SlotsReduceScatterComm", "AllGatherComm", "FencedAllGatherComm",
           "NormAllGatherComm"]
