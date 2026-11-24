# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MiniMax-M3.1 DSpark draft context-KV tail (FlyDSL, gfx950): norm + RoPE +
fp8 cache insert of the fused context-KV GEMM output, in one launch."""

from .host import kv_norm_rope_insert_multilayer, supports_multilayer

__all__ = ["kv_norm_rope_insert_multilayer", "supports_multilayer"]
