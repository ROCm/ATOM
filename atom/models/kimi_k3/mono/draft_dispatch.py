# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What production's dispatch picks for the DSpark draft's two MLA GEMMs at a
width, so the gfx1250 draft kernels sum them the same way: the FlyDSL
mxfp8_128 split-K of ``fused_qkv_a_proj`` (E8M0 activations), and the gluon
blockscale GEMM's split-K of ``q_b_proj`` (fp32 activation scales). Asked
through AITER's own lookups, so a retuned table moves the mirror with it."""

from atom.mono.runtime.consensus import MonoUnsupported

QKV_A_ROWS = 2176
HIDDEN = 7168
Q_LORA = 1536
K_GROUP = 128


def qkv_a_split_k(rows: int) -> int:
    """``gemm_a8w8_blockscale_bpreshuffle``'s split_k for fused_qkv_a_proj
    (E8M0 x E8M0 on gfx1250): the tuned FlyDSL row's, else the untuned
    fallback kernel's. Refused where the dispatch would leave FlyDSL."""
    from aiter.ops.gemm_op_a8w8 import (
        _flydsl_mxfp8_fallback_kernel,
        get_mxscale_bpreshuffle_config,
    )

    from atom.models.kimi_k3.mono.kernels.stages.mla_draft_front_layout import (
        qkv_a_split_k as parse,
    )

    cfg = get_mxscale_bpreshuffle_config(rows, QKV_A_ROWS, HIDDEN, "128x128")
    if cfg is not None:
        if cfg.get("libtype") != "flydsl":
            raise MonoUnsupported(
                f"{rows} rows: fused_qkv_a_proj on {cfg.get('libtype')}"
            )
        split = parse(cfg.get("kernelName"))
        if split is None or "_fsk" in str(cfg.get("kernelName")):
            raise MonoUnsupported(
                f"{rows} rows: fused_qkv_a_proj {cfg.get('kernelName')}"
            )
        return split
    kernel = _flydsl_mxfp8_fallback_kernel(rows, QKV_A_ROWS, HIDDEN)
    if kernel is None:
        raise MonoUnsupported(f"{rows} rows: no FlyDSL kernel for fused_qkv_a_proj")
    return int(kernel.split_k)


def q_b_split_groups(rows: int, out_rows: int) -> int:
    """K groups (128 wide) a split of production's q_b_proj GEMM sums: the
    gluon ``gemm_a8w8_blockscale_preshuffle``'s SPLITK_BLOCK_SIZE / 128 at
    the width (fp32 x_scale and the E8M0 weight scales widened, gfx1250).
    Refused where the dispatch would run another kernel."""
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.gemm_op_a8w8 import _hip_blockscale_supported
    from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8_blockscale import (
        _get_config,
    )
    from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params

    if get_gfx() != "gfx1250":
        raise MonoUnsupported("q_b_proj's fp32-scale GEMM is gfx1250's gluon one")
    backend = _q_b_backend(rows, out_rows)
    if backend != "gluon":
        raise MonoUnsupported(f"{rows} rows: q_b_proj on the {backend} backend")
    if not _hip_blockscale_supported():
        # the dispatch's explicit fallback config: one split, 128-wide K blocks
        return Q_LORA // K_GROUP
    cfg, _ = _get_config(rows, out_rows, Q_LORA, True, backend=backend)
    kernel_type = cfg.get("kernel_type") or "bandwidth_bound"
    if kernel_type != "bandwidth_bound":
        raise MonoUnsupported(f"{rows} rows: q_b_proj's {kernel_type} kernel")
    compute_splitk_params(cfg, Q_LORA)
    if cfg["BLOCK_SIZE_K"] != K_GROUP or cfg["SPLITK_BLOCK_SIZE"] % K_GROUP:
        raise MonoUnsupported(f"{rows} rows: q_b_proj blocks {cfg}")
    return cfg["SPLITK_BLOCK_SIZE"] // K_GROUP


def _q_b_backend(rows: int, out_rows: int) -> str:
    """The backend ``gemm_a8w8_blockscale_bpreshuffle`` hands the fp32 path
    on gfx1250: a tuned row's ``kernelName`` hint, else gluon where the arch
    has the preshuffle gluon kernels."""
    import torch
    from aiter.jit.core import AITER_CONFIGS
    from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config
    from aiter.ops.opus._arch import _device_arch_and_cu
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import _GLUON_PRESHUFFLE_ARCHS
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    gfx, cu = _device_arch_and_cu(torch.device("cuda", torch.cuda.current_device()))
    config = get_CKGEMM_config(
        rows,
        out_rows,
        Q_LORA,
        AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE,
        gfx=gfx,
        cu_num=cu,
    )
    name = str(config.get("kernelName", "")) if config is not None else ""
    if name in ("triton", "gluon"):
        return name
    return "gluon" if get_arch() in _GLUON_PRESHUFFLE_ARCHS else "triton"
