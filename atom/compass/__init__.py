# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Device-free performance estimates for ATOM.

The only configuration implemented here is DeepSeek-V4-Pro, tensor parallel
8, DSpark. It is the analytic tier: geometry and a machine spec, no kernel
launch and no device allocation. It does not replace ``ModelRunner``.
"""

from atom.compass.dsv4_tp8_dspark import Estimate, estimate

__all__ = ["Estimate", "estimate"]
