# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared gate/up GEMM (``StoreSharedFp32``'s bits) when each workgroup owns one 256x256 parent tile, so there is no
previous-tile epilogue; the grid must have at most 256 parents: ceil(M / 256) * 6 <= 256."""
import functools

from .shared_dense import StoreSharedFp32
from .mxfp8.shared_dense_4wave import compile_mxfp8_gemm_4w_gateup
from amd_kernels.warm_up import launcher


class StoreSharedFp32SingleTile(StoreSharedFp32):
    def thunks_rm(self, c_gate, c_up, base_row, base_col):
        return [], []


@functools.cache
@launcher
def compile_shared_gemm1_a8w8_single_tile():
    return compile_mxfp8_gemm_4w_gateup(
        K=6144, alpha=1.702, beta=1.0, limit=7.0,
        epilogue_type=StoreSharedFp32SingleTile,
        kernel_prefix='shared_gemm1_a8w8_swiglu_quant_single_tile_h6144_i768_',
        tiles=(4, 4),
    )
