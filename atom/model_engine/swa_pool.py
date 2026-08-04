# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The sizing name for a sliding-window KV cache.

This module used to hold `SlidingWindowPool`: a content-addressed block pool
with its own free list and hash index, driven in lockstep with the compressed
pool so out-of-window blocks could be freed while compressed ones persisted.
That existed because a sliding window has to be shareable across requests for a
prefix-cache hit to be usable, and the only sharing mechanism available was
content addressing.

The window is now a per-request ring inside each layer's `unified_kv`, indexed
by the request's state slot. Sharing comes from the state checkpoint copying the
ring into the resuming request's slot (`copy_state_entries`) rather than from
two requests pointing at one block. That removes the free list, the hash index,
the window-freeing walk, the per-request block table, and the admission term —
none of which a ring has an analogue for.

What survives is the name. `sub_pool_spec.py` deliberately defines no
architecture vocabulary: an entry class is named by whatever consumes its count,
and the backend declaring the class imports the name from there. A sliding
window is that consumer, so the name lives here even though there is no longer a
pool object behind it.
"""

SWA_POOL_CLASS = "swa"
