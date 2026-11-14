# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The one conv3d entry point of the Wan VAE, and the backend each call takes.

Three rungs, fastest first:

* ``flydsl`` -- AITER's FlyDSL implicit-GEMM conv
  (``aiter.ops.flydsl.conv_kernels.flydsl_conv_implicit``): bf16 only, and only
  on gfx950 / gfx1250. AITER ships a Wan2.1 VAE tile table for it
  (``wan21_vae_bf16_tuned_conv3d.csv``), tuned NDHWC in and out at 368x544
  and 480x832.
* ``triton`` -- AITER's Triton conv3d (``aiter.ops.triton.conv.conv3d``):
  fp16 / bf16, groups=1, zero padding -- which every Wan VAE conv is.
* ``torch`` -- ``torch.nn.functional.conv3d``, which runs anything.

``ATOM_DIFFUSION_CONV3D_BACKEND`` names the highest rung a call may take; it
then takes the first rung at or below that one which supports it. The decision
is made per call on the tensors, not on whether ``aiter`` imports: aiter
imports fine in a CPU process and then fails inside the kernel. AITER is
imported lazily, on the first GPU call that wants it, so this module imports
on a CPU runner. A kernel that is selected and then raises is a real error and
propagates; only an AITER build that lacks the kernel falls through.

Tensors are logical NCDHW throughout. Whether they are *physically* NDHWC
(``torch.channels_last_3d``) is a property of the tensor, and the output keeps
the input's: an NDHWC input reaches FlyDSL as a free ``permute`` view, and
Triton's ``layout="ndhwc"`` takes it as it is.
"""

import contextlib
import functools
import logging
from collections import Counter
from collections.abc import Callable, Iterator

import torch
import torch.nn.functional as F

from atom.utils import envs

logger = logging.getLogger(__name__)

# Fastest first; the index is the rung.
CONV3D_BACKENDS = ("flydsl", "triton", "torch")
FLYDSL_CONV3D_GFX = ("gfx950", "gfx1250")

_backend_override: str | None = None
_call_counts: Counter = Counter()


def configured_conv3d_backend() -> str:
    """The highest rung calls may take: the override, else the env var."""
    name = _backend_override or envs.ATOM_DIFFUSION_CONV3D_BACKEND
    if name not in CONV3D_BACKENDS:
        raise ValueError(
            f"ATOM_DIFFUSION_CONV3D_BACKEND must be one of {CONV3D_BACKENDS}, "
            f"got {name!r}"
        )
    return name


@contextlib.contextmanager
def conv3d_backend_override(name: str) -> Iterator[None]:
    """Cap the backend for the duration of a block, overriding the env var."""
    global _backend_override
    if name not in CONV3D_BACKENDS:
        raise ValueError(f"backend must be one of {CONV3D_BACKENDS}, got {name!r}")
    previous = _backend_override
    _backend_override = name
    try:
        yield
    finally:
        _backend_override = previous


def conv3d_call_counts() -> dict[tuple[str, str], int]:
    """Calls so far, keyed by ``(backend, layout)``; layout is NCDHW / NDHWC."""
    return dict(_call_counts)


def reset_conv3d_call_counts() -> None:
    _call_counts.clear()


def is_channels_last(x: torch.Tensor) -> bool:
    """Whether a logical-NCDHW tensor has channels innermost in memory.

    A stride test rather than ``is_contiguous(channels_last_3d)``: a temporal
    slice of a batched NDHWC tensor is no longer contiguous in either format,
    but it is still channels-innermost, and that is what picks the layout.
    """
    return x.dim() == 5 and x.shape[1] > 1 and x.stride(1) == 1


def memory_format_of(x: torch.Tensor) -> torch.memory_format:
    return torch.channels_last_3d if is_channels_last(x) else torch.contiguous_format


@functools.cache
def _gfx_arch(device_index: int) -> str:
    props = torch.cuda.get_device_properties(device_index)
    return getattr(props, "gcnArchName", "").split(":", 1)[0]


@functools.cache
def _aiter_kernel(backend: str) -> Callable | None:
    try:
        if backend == "flydsl":
            from aiter.ops.flydsl.conv_kernels import flydsl_conv_implicit as kernel
        else:
            from aiter.ops.triton.conv.conv3d import conv3d as kernel
    except ImportError as exc:
        logger.warning(
            "AITER %s conv3d is unavailable (%s); conv3d falls back a rung",
            backend,
            exc,
        )
        return None
    return kernel


def _flydsl_supports(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and _gfx_arch(x.device.index) in FLYDSL_CONV3D_GFX
    )


def _triton_supports(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return x.dtype in (torch.bfloat16, torch.float16) and weight.dtype == x.dtype


_SUPPORTS = {"flydsl": _flydsl_supports, "triton": _triton_supports}


def select_conv3d_backend(x: torch.Tensor, weight: torch.Tensor) -> str:
    """The backend :func:`conv3d` will run these tensors on."""
    ceiling = CONV3D_BACKENDS.index(configured_conv3d_backend())
    # HIP tensors report device type "cuda".
    if x.device.type != "cuda":
        return "torch"
    for name in CONV3D_BACKENDS[ceiling:-1]:
        if _SUPPORTS[name](x, weight) and _aiter_kernel(name) is not None:
            return name
    return "torch"


def _triple(value: int | tuple[int, ...]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"expected an int or 3 ints, got {value!r}")
    return tuple(int(v) for v in value)


def conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
) -> torch.Tensor:
    """``F.conv3d`` (groups=1, zero padding) on the backend selected for it.

    ``x`` is logical ``[N, C, D, H, W]`` in either memory format and ``weight``
    is ``[K, C, kT, kH, kW]``; the output is logical ``[N, K, Do, Ho, Wo]`` in
    the input's memory format.
    """
    stride = _triple(stride)
    padding = _triple(padding)
    channels_last = is_channels_last(x)
    backend = select_conv3d_backend(x, weight)
    _call_counts[(backend, "NDHWC" if channels_last else "NCDHW")] += 1

    if backend == "flydsl":
        kernel = _aiter_kernel("flydsl")
        if channels_last:
            y = kernel(
                x.permute(0, 2, 3, 4, 1).contiguous(),
                weight,
                bias,
                stride=stride,
                padding=padding,
                input_layout="NDHWC",
                output_layout="NDHWC",
            )
            return y.permute(0, 4, 1, 2, 3)
        return kernel(x.contiguous(), weight, bias, stride=stride, padding=padding)

    if backend == "triton":
        kernel = _aiter_kernel("triton")
        if channels_last:
            return kernel(x, weight, bias, stride, padding, layout="ndhwc")
        return kernel(x.contiguous(), weight, bias, stride, padding, layout="ncdhw")

    y = F.conv3d(x, weight, bias, stride, padding)
    return y.contiguous(memory_format=memory_format_of(x))
