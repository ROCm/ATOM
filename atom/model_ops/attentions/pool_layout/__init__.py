# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where a byte lives in the cache pools: sizing, addressing, and moving.

One topic across five modules -- how many entries a byte budget buys
(`sub_pool_spec`), which row of the unified pool a DeepSeek-V4 layer's window
or compressed group occupies (`v4_pool_geometry`), where a checkpoint image's
bytes land in the MLA paged pool (`page_unit_geometry`), one request's whole
state as a contiguous run (`state_arena`), and how that run is scattered across
PAGE units and gathered back (`paged_state_copy`). All of it answers a question
about the pool, none of it about a particular step.

`..token_layout` is the other axis, and the two share a membership rule with
three parts.

**By topic.** The axis decides, not the dependencies. Half this tree imports
neither aiter nor atom without belonging anywhere near here.

**Arithmetic, not staging.** A member computes an answer; it does not write a
`forward_vars` mirror or upload one. `page_unit_geometry` is a mixin over
`self.model_runner` and still qualifies because it only reads -- taking `self`
is not the line, having a side effect on the step's buffers is. Move a stager
in and the package becomes a second builder.

**Reachable without aiter or the rest of atom**, so a member is importable on a
runner with no AITER build and no GPU. CI is such a runner, and one import
failure there aborts collection for every test rather than one, so
`tests/test_layout_packages.py` enforces this part over both packages. It is
the part a test can check, which is exactly why it must not be mistaken for the
whole rule.

Not everything on this topic is here, and one absence is worth naming:
`v4_kernels/pool_index.py` is `v4_pool_geometry`'s device-side half, the same
row formulas as `@triton.jit` device functions for the eight kernels that
address the pool. It stays with the kernels -- the split is by execution tier,
which is a real axis too -- but a layout change lands in both places, and
`tests/test_pool_index.py` is what pins the two sides together.

Nothing is re-exported here: a convenience import would pull all five in
whenever one is wanted.
"""
