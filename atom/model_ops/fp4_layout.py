# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 GEMM backend and scale-layout contract."""

from dataclasses import dataclass
from enum import IntEnum

import torch
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime

from atom.utils import envs

try:
    from aiter.ops.mx_scale_layout import (
        mx_scale_buffer_shape as mxfp4_scale_buffer_shape,
    )
    from aiter.ops.mx_scale_layout import to_mx_scale_layout as to_mxfp4_scale_layout
    from aiter.utility.mx_types import MXScaleLayoutInt as MXScaleLayout

    _EXPLICIT_MX_LAYOUT_SUPPORTED = True
except ImportError:
    # Keep unrelated models importable while the dependent AITER PR is absent.
    # Any MXFP4 layer fails below before it can use this compatibility mirror.
    from aiter.utility import fp4_utils

    class MXScaleLayout:
        ROW_MAJOR = 0
        AITER_E8M0 = 1
        OPUS_F4 = 2

    def mxfp4_scale_buffer_shape(
        rows: int, k_groups: int, layout: int
    ) -> tuple[int, int]:
        if layout == MXScaleLayout.ROW_MAJOR:
            return rows, k_groups
        if layout == MXScaleLayout.AITER_E8M0:
            return ((rows + 255) // 256 * 256, (k_groups + 7) // 8 * 8)
        if layout == MXScaleLayout.OPUS_F4:
            return ((rows + 31) // 32 * 32, (k_groups + 3) // 4 * 4)
        raise ValueError(f"unknown MXFP4 scale layout: {layout}")

    def to_mxfp4_scale_layout(scale: torch.Tensor, layout: int) -> torch.Tensor:
        if layout == MXScaleLayout.ROW_MAJOR:
            return scale.contiguous()
        if layout == MXScaleLayout.AITER_E8M0:
            return fp4_utils.e8m0_shuffle(scale)
        if layout == MXScaleLayout.OPUS_F4:
            from aiter.ops.shuffle import shuffle_scale_f4

            return shuffle_scale_f4(scale, intype=7)
        raise ValueError(f"unknown MXFP4 scale layout: {layout}")

    _EXPLICIT_MX_LAYOUT_SUPPORTED = False

__all__ = [
    "Fp4BackendKind",
    "Fp4BackendSpec",
    "MXScaleLayout",
    "decode_fp4_backend_layout_code",
    "mxfp4_scale_buffer_shape",
    "resolve_current_fp4_backend_spec",
    "resolve_fp4_backend_spec",
    "to_mxfp4_scale_layout",
]


class Fp4BackendKind(IntEnum):
    DEFAULT_AITER = 0
    TRITON_PRESHUFFLE = 1
    TRITON_NONSHUFFLE = 2


@dataclass(frozen=True)
class Fp4BackendSpec:
    kind: Fp4BackendKind
    weight_scale_layout: int
    activation_scale_layout: int

    @property
    def backend_layout_code(self) -> int:
        """One custom-op scalar carrying backend and both scale layouts."""
        return (
            int(self.kind)
            | (int(self.activation_scale_layout) << 8)
            | (int(self.weight_scale_layout) << 16)
        )


def resolve_fp4_backend_spec(
    arch: str,
    params_dtype: torch.dtype | None,
    triton_gemm: bool,
    nonshuffle_triton_gemm: bool,
) -> Fp4BackendSpec | None:
    if params_dtype != dtypes.fp4x2:
        return None
    if not _EXPLICIT_MX_LAYOUT_SUPPORTED:
        raise RuntimeError(
            "MXFP4 explicit scale layouts require an AITER build containing "
            "ROCm/aiter#6294"
        )
    if nonshuffle_triton_gemm:
        kind = Fp4BackendKind.TRITON_NONSHUFFLE
        layout = MXScaleLayout.ROW_MAJOR
    elif triton_gemm:
        kind = Fp4BackendKind.TRITON_PRESHUFFLE
        layout = MXScaleLayout.AITER_E8M0
    else:
        kind = Fp4BackendKind.DEFAULT_AITER
        layout = (
            MXScaleLayout.OPUS_F4 if arch == "gfx1250" else MXScaleLayout.AITER_E8M0
        )
    return Fp4BackendSpec(kind, layout, layout)


def resolve_current_fp4_backend_spec(
    params_dtype: torch.dtype | None,
) -> Fp4BackendSpec | None:
    """Resolve once during module initialization/loading, never in forward."""
    return resolve_fp4_backend_spec(
        get_gfx_runtime(),
        params_dtype,
        envs.ATOM_USE_TRITON_GEMM,
        envs.ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM,
    )


def decode_fp4_backend_layout_code(
    backend_layout_code: int,
) -> tuple[Fp4BackendKind, int, int]:
    if backend_layout_code < 0:
        raise ValueError("MXFP4 custom op requires a resolved backend layout")
    return (
        Fp4BackendKind(backend_layout_code & 0xFF),
        (backend_layout_code >> 8) & 0xFF,
        (backend_layout_code >> 16) & 0xFF,
    )
