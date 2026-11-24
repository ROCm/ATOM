# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for atomrl.lumen.core.float8 — FP8 dtype detection and support checks.

Covers:
  - is_fp8_dtype recognises all four torch FP8 types
  - is_fp8_dtype rejects non-FP8 types (bf16, fp32, int8, …)
  - check_fp8_support returns a (bool, str) tuple
  - check_fp8_ocp_support returns a (bool, str) tuple
  - float8_e4m3 / float8_e5m2 module-level constants are valid FP8 dtypes
"""

import pytest
import torch
from atomrl.lumen.core.float8 import (
    check_fp8_ocp_support,
    check_fp8_support,
    float8_e4m3,
    float8_e5m2,
    is_fp8_dtype,
)

# ===================================================================
# is_fp8_dtype: positive cases
# ===================================================================


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    ],
)
def test_is_fp8_dtype_positive(dtype):
    """All four torch FP8 types should be recognised."""
    assert is_fp8_dtype(dtype)


# ===================================================================
# is_fp8_dtype: negative cases
# ===================================================================


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    ],
)
def test_is_fp8_dtype_negative(dtype):
    """Non-FP8 types should be rejected."""
    assert not is_fp8_dtype(dtype)


# ===================================================================
# check_fp8_support / check_fp8_ocp_support
# ===================================================================


def test_check_fp8_support_returns_tuple():
    """check_fp8_support returns (bool, str)."""
    result = check_fp8_support()
    assert isinstance(result, tuple) and len(result) == 2
    supported, msg = result
    assert isinstance(supported, bool)
    assert isinstance(msg, str)


def test_check_fp8_ocp_support_returns_tuple():
    """check_fp8_ocp_support returns (bool, str)."""
    result = check_fp8_ocp_support()
    assert isinstance(result, tuple) and len(result) == 2
    supported, msg = result
    assert isinstance(supported, bool)
    assert isinstance(msg, str)


# ===================================================================
# Module-level dtype constants
# ===================================================================


def test_float8_e4m3_is_valid():
    """float8_e4m3 constant is a recognised FP8 dtype."""
    assert is_fp8_dtype(float8_e4m3)


def test_float8_e5m2_is_valid():
    """float8_e5m2 constant is a recognised FP8 dtype."""
    assert is_fp8_dtype(float8_e5m2)


def test_float8_e4m3_and_e5m2_differ():
    """E4M3 and E5M2 constants are distinct dtypes."""
    assert float8_e4m3 != float8_e5m2
