# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The platform helpers the host code needs: ``on_gfx950`` (as ``vllm.platforms.rocm``) and
``initialize_xcd_topology`` (the XCD count from HIP, a fail-closed check and a per-device cache)."""

import ctypes
import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

# hipDeviceAttributeNumberOfXccs; the XCD-aware placement uses the value as the
# count of independently cached XCDs.
_HIP_DEVICE_ATTRIBUTE_NUMBER_OF_XCCS = 10018
_SUPPORTED_XCD_COUNT = 8
_HIP_LIB_NAME = "libamdhip64.so"


@functools.cache
def on_gfx950() -> bool:
    """vllm.platforms.rocm.on_gfx950: device 0 is a gfx950."""
    if not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


def hip_runtime_path() -> str:
    """torch's HIP runtime, or the bare soname when torch bundles none (one runtime in the process)."""
    torch_hip = os.path.join(os.path.dirname(torch.__file__), "lib", _HIP_LIB_NAME)
    return torch_hip if os.path.exists(torch_hip) else _HIP_LIB_NAME


def _query_xcd_count(device_index: int) -> int:
    """Return the device's XCD count, or zero when HIP cannot report it."""
    try:
        get_attribute = ctypes.CDLL(hip_runtime_path()).hipDeviceGetAttribute
        get_attribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        get_attribute.restype = ctypes.c_int
        value = ctypes.c_int()
        status = get_attribute(
            ctypes.byref(value),
            _HIP_DEVICE_ATTRIBUTE_NUMBER_OF_XCCS,
            device_index,
        )
    except (AttributeError, OSError, TypeError, ctypes.ArgumentError) as exc:
        logger.warning(
            "Could not query the XCD count through %s (%s); assuming no XCDs",
            hip_runtime_path(),
            exc,
        )
        return 0
    if status != 0 or value.value <= 0:
        return 0
    return value.value


def _supported_xcd_count(*, is_gfx950: bool, queried_count: int) -> int:
    """Fail closed outside the topology on which the mapping was measured."""
    if is_gfx950 and queried_count == _SUPPORTED_XCD_COUNT:
        return queried_count
    return 0


@functools.cache
def _initialize_xcd_topology(device_index: int) -> int:
    """Discover and validate the topology once for each logical device."""
    return _supported_xcd_count(
        is_gfx950=on_gfx950(), queried_count=_query_xcd_count(device_index)
    )


def initialize_xcd_topology() -> int:
    """The current device's XCD count (8 on MI355X), 0 = no XCD grouping."""
    return _initialize_xcd_topology(torch.accelerator.current_device_index())
