# SPDX-License-Identifier: MIT
"""Trace the FlyDSL kernels of the fp8 block-scale GEMM before serving.

FlyDSL keys its cache on the traced IR, so the first launch of each kernel
config in a process re-runs the kernel's Python body before the cached binary
can be found; for the 8-wave GEMM at large K that takes seconds. The tuned
table picks a different config per M bucket, so prefills keep landing on
configs nobody has traced yet, and while one DP rank traces, its peers wait
for it in the next collective. Launching every config once at startup moves
that cost out of serving.
"""

import logging
import time
from collections.abc import Callable

import torch
from torch import nn

from atom.utils import envs

logger = logging.getLogger("atom")

_BLOCK = 128


def mxscale_gemm_shapes(model: nn.Module) -> dict:
    """(N, K) -> (weight dtype, scale dtype, scale shape) of every 2-D fp8
    weight with a one-byte 128x128 block scale: the operands the gfx950
    mxscale bpreshuffle GEMM serves."""
    shapes = {}
    for module in model.modules():
        weight = getattr(module, "weight", None)
        scale = getattr(module, "weight_scale", None)
        if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
            continue
        if weight.dim() != 2 or scale.dim() != 2:
            continue
        if weight.element_size() != 1 or scale.element_size() != 1:
            continue
        n, k = weight.shape
        if tuple(scale.shape) != (-(-n // _BLOCK), -(-k // _BLOCK)):
            continue
        shapes.setdefault((n, k), (weight.dtype, scale.dtype, tuple(scale.shape)))
    return shapes


def first_m_per_kernel(lookup: Callable[[int], dict | None], max_m: int) -> list:
    """The smallest M in [1, max_m] at which each tuned kernel is picked.
    Untuned M (no row) are left to the runtime's heuristic."""
    first = {}
    for m in range(1, max_m + 1):
        config = lookup(m)
        if config is not None:
            first.setdefault(str(config["kernelName"]), m)
    return sorted(first.values())


def pretrace_mxscale_gemms(model: nn.Module, max_m: int, label: str = "") -> None:
    try:
        from aiter import gemm_a8w8_blockscale_bpreshuffle
        from aiter.jit.utils.chip_info import get_gfx
        from aiter.ops.gemm_op_a8w8 import get_mxscale_bpreshuffle_config
    except ImportError:
        return
    # Without the preshuffled weights these layers take another GEMM.
    if get_gfx() != "gfx950" or not envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE:
        return
    shapes = mxscale_gemm_shapes(model)
    if not shapes:
        return

    start = time.time()
    device = torch.cuda.current_device()
    launched = 0
    for (n, k), (w_dtype, s_dtype, s_shape) in sorted(shapes.items()):
        ms = first_m_per_kernel(
            lambda m, n=n, k=k: get_mxscale_bpreshuffle_config(m, n, k, "128x128"),
            max_m,
        )
        if not ms:
            continue
        weight = torch.empty((n, k), dtype=w_dtype, device=device)
        w_scale = torch.ones(s_shape, dtype=torch.uint8, device=device).view(s_dtype)
        for m in ms:
            x = torch.empty((m, k), dtype=w_dtype, device=device)
            x_scale = torch.ones((m, s_shape[1]), dtype=torch.uint8, device=device)
            gemm_a8w8_blockscale_bpreshuffle(
                x, weight, x_scale.view(s_dtype), w_scale, torch.bfloat16
            )
            launched += 1
        del weight, w_scale
    torch.cuda.synchronize()
    logger.info(
        f"{label}: traced {launched} fp8 block-scale GEMM kernels for "
        f"{len(shapes)} weight shapes in {time.time() - start:.1f} s"
    )
