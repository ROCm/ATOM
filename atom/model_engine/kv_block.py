# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


class Block:
    """One physical KV cache block of the compressed pool (BlockManager).

    Lives in its own module because `block_pool.py` and `block_manager.py` both
    need it and would otherwise import each other."""

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


# Name of the sub-pool sizing class backing the per-request state slots
# (GDN/Mamba recurrent state, the DeepSeek-V4 compressor ring). Owned next to
# the block/slot machinery that consumes the count, not in the sizing layer.
STATE_SLOT_CLASS = "state"
