# SPDX-License-Identifier: Apache-2.0
"""p1/draft-fusion overlay: amd-kernels' Bf16ArNormComm with ATOM's draft MXFP8 output contract (row-major [M, 192]
E8M0 scales, row-major codes, ATOM's fp8_groups rule; optional v_rsq). Needs amd_kernels (fd233ef) on PYTHONPATH."""
from .host import DraftArNormComm

__all__ = ["DraftArNormComm"]
