# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for atomrl.lumen.core.utils — compute capability detection.

Covers:
  - get_device_compute_capability returns a (major, minor) tuple
  - Both values are non-negative integers
  - Result is consistent across repeated calls (lru_cache)
"""

import torch
from atomrl.lumen.core.utils import get_device_compute_capability


def test_returns_tuple():
    """get_device_compute_capability returns a 2-tuple."""
    cc = get_device_compute_capability()
    assert isinstance(cc, tuple)
    assert len(cc) == 2


def test_values_are_nonneg_ints():
    """Major and minor versions are non-negative integers."""
    major, minor = get_device_compute_capability()
    assert isinstance(major, int) and major >= 0
    assert isinstance(minor, int) and minor >= 0


def test_consistent_result():
    """Repeated calls return the same value (lru_cache)."""
    a = get_device_compute_capability()
    b = get_device_compute_capability()
    assert a == b


def test_matches_torch_properties():
    """Result agrees with torch.cuda.get_device_properties."""
    major, minor = get_device_compute_capability()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    assert major == props.major
    assert minor == props.minor
