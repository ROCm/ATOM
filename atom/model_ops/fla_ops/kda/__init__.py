# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project
# (version 0.5.2). The original source code was licensed under the MIT
# license and included the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
"""Vendored KDA chunk forward, fused for ATOM's Kimi-K3 prefill path.

Upstream `fla.ops.kda.chunk_kda` allocates its own output and indexes the
recurrent state densely, which forces the caller to gather the initial state,
scatter the final state, and copy the output. This package threads a slot-index
indirection and an output buffer through the two innermost kernels so all three
happen inside them.

WARNING: these kernels are base-2 (`exp2`). KDA pre-scales its gate by RCP_LN2.
Do not interchange them with the base-e siblings in the parent package
(`fla_ops/chunk_delta_h.py`, `fla_ops/chunk_delta_h_vk.py`, `fla_ops/chunk_o_vk.py`),
which serve GDN and do not pre-scale. The mismatch produces `decay ** 1.4427`
and raises nothing.

The closest collision is `fla_ops/chunk_delta_h.py`: same filename as
`kda/chunk_delta_h.py`, one directory up, opposite base. Its two public names
are the unsuffixed originals; the copies here carry a `_log2` suffix
(`chunk_gated_delta_rule_fwd_kernel_h_blockdim64_log2`,
`chunk_gated_delta_rule_fwd_h_log2`) so a mis-copied import line fails to
resolve instead of silently computing in the wrong base. Keep the suffix.
"""

from .chunk import chunk_kda

__all__ = ["chunk_kda"]
