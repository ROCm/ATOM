# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where this step's tokens go: the per-token index arrays a forward needs.

The counterpart of `..pool_layout`, on the other axis. That package answers
where a byte lives and is a function of the config; this one answers where a
token goes and is a function of the batch, so it is rebuilt every step. The
membership rule is the same one, stated in full in `..pool_layout`: by topic,
arithmetic rather than staging, and reachable without aiter or the rest of atom
-- which is what makes the arithmetic checkable against a naive reference while
the staging around it is not.

Four modules, split by what differs and what does not. `prefill` and `decode`
shape the token axis -- ragged for a chunk, rectangular for a speculative step
-- and that is the whole difference between them. `slots` and `batch_ids` are
what both sides then want off that axis: which KV slot each token is written
to, and which sequence it belongs to. Neither of those cares which side shaped
the axis, so neither is duplicated per side; both are the kind of answer that
goes wrong in silence, landing a write or a lookup on another request's row.

Not everything per-token is here. A shape that only one caller has stays with
that caller: V4's decode positions are ragged (a DSpark step verifies a
different number of drafts per request), the one-token decode slot mapping is
derived from `last_block_num_tokens` rather than from a position, and DCP's is
a per-rank filter rather than an address. Pulling any of them in would put two
derivations behind one name.

Nothing is re-exported.
"""
