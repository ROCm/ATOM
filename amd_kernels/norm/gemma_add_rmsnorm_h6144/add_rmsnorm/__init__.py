# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Gemma add + RMSNorm at hidden 6144 (bf16 or MXFP8 out), FlyDSL (gfx950): the Triton kernels' bits with a shorter
dependency chain."""

import functools

H = 6144


@functools.cache
def available() -> bool:
    """gfx950 with FlyDSL importable."""
    import torch

    if (
        not torch.cuda.is_available()
        or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    ):
        return False
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


def __getattr__(name):
    if name in __all__ and name != "available":
        from . import host

        return getattr(host, name)
    raise AttributeError(name)


__all__ = [
    "add_rmsnorm_ok",
    "available",
    "gemma_add_rmsnorm",
    "gemma_add_rmsnorm_mxfp8",
    "gemma_add_rmsnorm_mxfp8_row_major",
    "code_tile_rows_for",
    "gemma_add_rmsnorm_mxfp8_for",
    "warmup_for_gemm",
    "warmup_add_rmsnorm_mxfp8",
]
