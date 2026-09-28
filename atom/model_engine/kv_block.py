# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import array


class Block:
    """One physical KV cache block of the compressed pool (BlockManager).

    Lives in its own module because `block_pool.py` and `block_manager.py` both
    need it and would otherwise import each other."""

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        # A slice of `Sequence.token_ids`, kept only so a hash hit can be
        # checked against the tokens it claims to stand for. See `update` for
        # why the type is pinned.
        self.token_ids: array.array = array.array("i")

    def update(self, hash: int, token_ids: array.array):
        # Pinned rather than accepted as either: a list here fails twice and
        # neither failure says so. It never compares equal to the `array("i")`
        # the other publish paths store, so every hit on this block reads as a
        # collision; and a list of ids is one traversal slot per token for the
        # collector, which across a full pool is a stop-the-world pause of
        # ~200ms per gen-2 pass against ~5ms for arrays.
        assert isinstance(
            token_ids, array.array
        ), f"Block.token_ids must be an array('i'), got {type(token_ids).__name__}"
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = array.array("i")


# Name of the sub-pool sizing class backing the per-request state slots
# (GDN/Mamba recurrent state, the DeepSeek-V4 compressor ring). Owned next to
# the block/slot machinery that consumes the count, not in the sizing layer.
STATE_SLOT_CLASS = "state"
