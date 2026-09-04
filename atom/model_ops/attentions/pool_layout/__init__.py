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

Like `..token_layout`, this package imports neither `aiter` nor any other
`atom` module, so it is importable on a runner with no AITER build and no GPU.
`tests/test_layout_packages.py` enforces that over both packages and says why.
Nothing is re-exported here: a convenience import would pull all five in
whenever one is wanted.
"""
