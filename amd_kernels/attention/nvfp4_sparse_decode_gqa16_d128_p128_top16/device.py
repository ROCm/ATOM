# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The host's device queries: the XCD count (``initialize_xcd_topology``) and the e4m3 table (``nvfp4_e4m3_lut``)."""

import ctypes
import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

_HIP_LIB_NAME = "libamdhip64.so"
# hipDeviceAttributeNumberOfXccs: HIP's XCC count, used as the count of independently cached XCDs
_HIP_DEVICE_ATTRIBUTE_NUMBER_OF_XCCS = 10018
_SUPPORTED_XCD_COUNT = 8


def _on_gfx950() -> bool:
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


def hip_runtime_path() -> str:
    """Path of torch's HIP runtime, or the bare soname when torch bundles none.

    Images ship several ``libamdhip64.so`` copies (/opt/rocm, ``torch/lib``, triton); a bare ``CDLL`` can bind a second,
    uninitialized runtime once another library has loaded the /opt/rocm copy, so the query binds torch's."""
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
            str(exc),
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
    queried_count = _query_xcd_count(device_index)
    is_gfx950 = _on_gfx950()
    xcd_count = _supported_xcd_count(is_gfx950=is_gfx950, queried_count=queried_count)
    logger.info(
        "XCD-aware sparse attention on device %d: %d XCDs (gfx950=%s, reported %d)",
        device_index,
        xcd_count,
        is_gfx950,
        queried_count,
    )
    return xcd_count


def initialize_xcd_topology() -> int:
    """Initialize XCD-aware sparse attention for the current logical device."""
    return _initialize_xcd_topology(torch.accelerator.current_device_index())


_NVFP4_E4M3_LUT: dict[torch.device, torch.Tensor] = {}


def nvfp4_e4m3_lut(device: torch.device) -> torch.Tensor:
    """[256] int64: byte j of entry s is e4m3_rn_satfinite(m_j * e4m3(s)) for the E2M1 magnitudes m = 0, 0.5, 1,
    1.5, 2, 3, 4, 6; the kernels index it by block scale byte and code magnitude."""
    lut = _NVFP4_E4M3_LUT.get(device)
    if lut is None:
        # device ops only (no host copy), so a first call under graph capture is fine
        scales = torch.arange(256, dtype=torch.uint8, device=device)
        scales = scales.view(torch.float8_e4m3fn).float()
        # unwritten tokens can carry NaN scale bytes, and P = 0 must not multiply a NaN V
        scales = torch.nan_to_num(scales, nan=0.0)
        code = torch.arange(8, device=device)
        exp, man = code // 2, (code % 2).float()
        # E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
        mags = torch.where(
            exp == 0, man * 0.5, torch.exp2((exp - 1).float()) * (1.0 + man * 0.5)
        )
        # e2m1 * e4m3 is exact in fp32; saturate, then round once to E4M3.
        values = (scales[:, None] * mags[None, :]).clamp(-448.0, 448.0)
        lut = values.to(torch.float8_e4m3fn).view(torch.int64).reshape(256)
        _NVFP4_E4M3_LUT[device] = lut
    return lut
