# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What production's dispatch runs for the LM head GEMM of a rank's vocab shard
at a width, so the step tail sums it the same way: AITER's tuned row for
``[rows, HIDDEN] x [VR, HIDDEN]^T`` bf16 names the kernel and its split-K.
Asked through AITER's own lookup, so a retuned table moves the mirror with it;
a kernel whose arithmetic is not mirrored refuses the width."""

import re

from atom.models.kimi_k3.step_tail import layout as L

_FLYDSL = re.compile(
    r"flydsl_hgemm_abf16_wbf16_bf16_t(\d+)x(\d+)x(\d+)x(\d+)_ks(\d+)_.*_gfx1250"
)
_OPUS = re.compile(
    r"opus_gemm_gfx1250_splitk_clusterlaunch_tdm_ws_(\d+)x(\d+)x(\d+)x(\d+)_.*"
)


def parse_head_kernel(libtype, kernel_name, split_k) -> tuple[str, int, int]:
    """``(kind, split_k, block_k)`` of a tuned row, as ``layout.split_steps``
    takes them; ValueError for a kernel the tail does not mirror.

    'flydsl': ``gemm_a16w16_kernel_gfx1250`` -- WMMA 16x16x32 over K in order
    from 0, ``ks`` equal slices of K summed in slice order. 'opus': the
    split-K workspace kernel -- the same sums over slices of whole ``B_K``
    tiles written as fp32 partials, then ``splitk_reduce_kernel_gfx1250``
    adding them in order from +0. Both round the fp32 sum to bf16."""
    name = str(kernel_name or "")
    if libtype == "flydsl":
        m = _FLYDSL.fullmatch(name)
        if m is not None:
            return "flydsl", int(m[5]), int(m[3])
    elif libtype == "opus":
        m = _OPUS.fullmatch(name)
        if m is not None:
            return "opus", max(1, int(split_k or 1)), int(m[4])
    raise ValueError(f"LM head on {libtype} kernel {name!r}")


def head_splits(rows: int) -> tuple[int, ...]:
    """The K steps at which production's LM head GEMM of ``rows`` rows starts
    a new partial sum (``layout.split_steps``); ValueError where the tail has
    no mirror of what production runs."""
    from aiter.tuned_gemm import get_GEMM_A16W16_config

    cfg = get_GEMM_A16W16_config(
        M=rows,
        N=L.VR,
        K=L.HIDDEN,
        bias=False,
        dtype="torch.bfloat16",
        otype="torch.bfloat16",
        scaleAB=False,
        bpreshuffle=False,
        **_arch(),
    )
    kind, split_k, block_k = parse_head_kernel(
        cfg.get("libtype"), cfg.get("kernelName"), cfg.get("splitK")
    )
    return L.split_steps(kind, split_k, block_k)


def _arch() -> dict:
    """The arch and CU count ``tgemm.mm`` looks the row up with."""
    import torch
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx
    from aiter.ops.opus._arch import SUPPORTED_OPUS_ARCHES, _device_arch_and_cu

    gfx, cu_num = _device_arch_and_cu(torch.device("cuda", torch.cuda.current_device()))
    if gfx not in SUPPORTED_OPUS_ARCHES:
        gfx, cu_num = get_gfx(), get_cu_num()
    return {"gfx": gfx, "cu_num": cu_num}
