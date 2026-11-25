# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Grouped GEMM package.

``epilogue`` and ``fp8_output`` are optional. They are re-exported when those
modules are present, and omitted otherwise so ``grouped_gemm`` stays importable.
"""

from atomrl.lumen.ops.gemm.grouped_gemm import grouped_gemm, grouped_gemm_wgrad

__all__ = ["grouped_gemm", "grouped_gemm_wgrad"]

try:
    from atomrl.lumen.ops.gemm.epilogue import (
        GemmEpilogue,
        gemm_with_epilogue,
        is_fp8_output_enabled,
    )
except ImportError:
    pass
else:
    __all__ += ["GemmEpilogue", "gemm_with_epilogue", "is_fp8_output_enabled"]

try:
    from atomrl.lumen.ops.gemm.fp8_output import gemm_fp8_output, gemm_scaled_mm
except ImportError:
    pass
else:
    __all__ += ["gemm_fp8_output", "gemm_scaled_mm"]
