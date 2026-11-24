# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared gate/up GEMM (``StoreSharedFp32``'s bits) storing each previous-tile 16-row group right after its
accumulator reads, so no activated FP32 tile stays live across the next tile's first MFMA step."""
import functools

from .shared_dense import StoreSharedFp32
from .mxfp8.shared_dense_4wave import compile_mxfp8_gemm_4w_gateup
from amd_kernels.warm_up import launcher


class StoreSharedFp32Early(StoreSharedFp32):
    def thunks_rm(self, c_gate, c_up, base_row, base_col):
        # both previous-tile quadrants are read before step 0's groups 2 / 3 zero their AGPRs: store in that phase
        return self.thunks(c_gate, c_up, base_row, base_col), []


@functools.cache
@launcher
def compile_shared_gemm1_a8w8_epilogue_early():
    return compile_mxfp8_gemm_4w_gateup(
        K=6144, alpha=1.702, beta=1.0, limit=7.0,
        epilogue_type=StoreSharedFp32Early,
        kernel_prefix='shared_gemm1_a8w8_swiglu_quant_early_h6144_i768_',
        tiles=(4, 4),
    )
