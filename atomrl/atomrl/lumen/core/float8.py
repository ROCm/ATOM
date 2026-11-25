# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################


import torch

from atomrl.lumen.core.utils import get_device_compute_capability

__all__ = ["float8_e4m3", "float8_e5m2"]


def is_fp8_dtype(dtype):
    TORCH_FP8_DTYPE = [
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    ]
    return dtype in TORCH_FP8_DTYPE


def check_fp8_support() -> tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (9, 4):
        return True, ""
    return (
        False,
        "Device compute capability gfx942 or higher required for FP8 execution.",
    )


def check_fp8_ocp_support() -> tuple[bool, str]:
    """Return if fp8 ocp support is available"""
    if get_device_compute_capability() >= (9, 5):
        return True, ""
    return (
        False,
        "Device compute capability gfx950 or higher required for FP8 OCP format.",
    )


###################################################

try:
    if check_fp8_ocp_support()[0]:
        float8_e4m3 = torch.float8_e4m3fn
        float8_e5m2 = torch.float8_e5m2
    else:
        float8_e4m3 = torch.float8_e4m3fnuz
        float8_e5m2 = torch.float8_e5m2fnuz
except AttributeError:
    raise RuntimeError("Your PyTorch build does not support FP8 types.")

###################################################
