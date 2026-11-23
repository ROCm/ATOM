# SPDX-License-Identifier: MIT
"""gfx942 delayed mHC seam for prefill row counts (off: ATOM_DSV41_MHC_ROWS=0).

The Triton seam is a split-K GEMM whose programs read 64-128 B of each row per
load; on MI308X that runs far below bandwidth. SGLang's gfx942 boundary splits
the seam instead (hc_post_rows_gfx942.hip there): a row kernel does the post,
the collapse, both square sums and here also the sublayer RMSNorm; one
hipBLASLt GEMM against fn split into bf16 hi + lo gives the mixing dots in
fp32; a one-thread-per-row kernel computes the gates. Same outputs as
`mhc_fused_post_pre_delayed_rmsnorm`, not bitwise (operation order, exact
divisions for the gates).
"""

import ctypes
import logging
import os
import subprocess
from functools import cache

import torch

from atom.model_ops import hip_gfx942
from atom.utils import envs

logger = logging.getLogger(__name__)

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mhc_rows_gfx942.hip")
_HC = 4
_MIX = 24
# Below this many rows (decode, short chunks) the Triton seam is as fast.
_MIN_ROWS = 1024
# hipBLASLt on MI308X, K = 20480: N padded to 64 wins below this many rows,
# the transposed product (fn rows as M) above it.
_TRANSPOSED_MIN_ROWS = 12288


class _RowArgs(ctypes.Structure):
    _fields_ = (
        [
            (n, ctypes.c_void_p)
            for n in (
                "x",
                "res",
                "post_in",
                "comb_in",
                "pre_prev",
                "norm_w",
                "res_out",
                "y",
                "sq",
            )
        ]
        + [(n, ctypes.c_int64) for n in ("x_stride", "res_stride")]
        + [(n, ctypes.c_int32) for n in ("m", "h", "w_f32")]
        + [("norm_eps", ctypes.c_float)]
    )


class _GateArgs(ctypes.Structure):
    _fields_ = (
        [
            (n, ctypes.c_void_p)
            for n in ("dots", "sq", "scale", "base", "pre", "post", "comb")
        ]
        + [(n, ctypes.c_int64) for n in ("dot_t", "dot_i")]
        + [(n, ctypes.c_int32) for n in ("m", "iters")]
        + [
            (n, ctypes.c_float)
            for n in ("inv_k", "rms_eps", "pre_eps", "sk_eps", "post_mult")
        ]
    )


@cache
def _kernels():
    """(hip, {name: function}), or None when the module cannot be built or loaded."""
    names = ("mhc_rows_gfx942_post", "mhc_rows_gfx942_nopost", "mhc_gates_rows_gfx942")
    try:
        # the post-mix keeps separate multiplies and adds, as the Triton seam stores it
        return hip_gfx942.load(_SRC, names, ("-ffp-contract=off",))
    except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
        logger.warning(
            "gfx942 mHC rows seam unavailable, keeping the Triton seam: %s", e
        )
        return None


@cache
def _arch_ok():
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx942"


def _hilo(fn):
    """[64, 4H] bf16: fn's hi rows, lo rows and zero rows, cached on the weight."""
    cached = getattr(fn, "_atom_gfx942_hilo", None)
    if cached is not None and cached[0] == fn._version:
        return cached[1]
    hi = fn.to(torch.bfloat16)
    pad = hi.new_zeros(64 - 2 * fn.shape[0], fn.shape[1])
    hilo = torch.cat([hi, (fn - hi.float()).to(torch.bfloat16), pad], 0).contiguous()
    fn._atom_gfx942_hilo = (fn._version, hilo)
    return hilo


def applies(
    residual, hc_fn, norm_weight, sublayer_output, w_preshuffle_bf16, fuse_quant
):
    rows, hc_mult, hidden = residual.view(
        -1, residual.shape[-2], residual.shape[-1]
    ).shape
    if (
        not envs.ATOM_DSV41_MHC_ROWS
        or rows < _MIN_ROWS
        or w_preshuffle_bf16
        or fuse_quant
        or hc_mult != _HC
        or hidden % 512
        or residual.dtype != torch.bfloat16
        or not residual.is_contiguous()
        or residual.data_ptr() % 16
        or hc_fn.dtype != torch.float32
        or tuple(hc_fn.shape) != (_MIX, _HC * hidden)
        or not hc_fn.is_contiguous()
        or norm_weight.dtype not in (torch.bfloat16, torch.float32)
        or not norm_weight.is_contiguous()
        or norm_weight.data_ptr() % 16
        or (
            torch.cuda.is_current_stream_capturing()
            and getattr(hc_fn, "_atom_gfx942_hilo", None) is None
        )
    ):
        return False
    if sublayer_output is not None:
        flat = sublayer_output.reshape(rows, hidden)
        if flat.stride(1) != 1 or flat.stride(0) % 8 or flat.data_ptr() % 16:
            return False
    return _arch_ok() and _kernels() is not None


def pre_delayed_rows(
    residual,
    pre_mix,
    hc_fn,
    hc_scale,
    hc_base,
    norm_weight,
    *,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    post_mult,
    norm_eps,
    sublayer_output=None,
    post_mix=None,
    combination=None,
):
    """`(new_residual, normed, pre, post, comb)` of the seam; new_residual is
    `residual` itself when no post is owed. Shapes as the Triton seam's."""
    hip, fns = _kernels()
    hc_mult, hidden = residual.shape[-2:]
    flat = residual.view(-1, hc_mult, hidden)
    rows = flat.shape[0]
    dev = residual.device
    with_post = sublayer_output is not None
    pre_in = pre_mix.reshape(rows, hc_mult).float().contiguous()
    normed = torch.empty(rows, hidden, dtype=torch.bfloat16, device=dev)
    sq = torch.empty(rows, 2, dtype=torch.float32, device=dev)
    if with_post:
        x = sublayer_output.reshape(rows, hidden)
        post_in = post_mix.reshape(rows, hc_mult).float().contiguous()
        comb_in = combination.reshape(rows, hc_mult * hc_mult).float().contiguous()
        new_residual = torch.empty_like(flat)
    else:
        new_residual = flat
    args = _RowArgs(
        x.data_ptr() if with_post else 0,
        flat.data_ptr(),
        post_in.data_ptr() if with_post else 0,
        comb_in.data_ptr() if with_post else 0,
        pre_in.data_ptr(),
        norm_weight.data_ptr(),
        new_residual.data_ptr() if with_post else 0,
        normed.data_ptr(),
        sq.data_ptr(),
        x.stride(0) if with_post else 0,
        flat.stride(0),
        rows,
        hidden,
        norm_weight.dtype == torch.float32,
        norm_eps,
    )
    kernel = "mhc_rows_gfx942_post" if with_post else "mhc_rows_gfx942_nopost"
    hip.launch(fns[kernel], (rows + 3) // 4, args)

    copies = new_residual.view(rows, hc_mult * hidden)
    hilo = _hilo(hc_fn)
    if rows >= _TRANSPOSED_MIN_ROWS:
        dots = torch.mm(hilo[: 2 * _MIX], copies.t(), out_dtype=torch.float32)
        dot_t, dot_i = 1, dots.stride(0)
    else:
        dots = torch.mm(copies, hilo.t(), out_dtype=torch.float32)
        dot_t, dot_i = dots.stride(0), 1

    pre = torch.empty(rows, hc_mult, dtype=torch.float32, device=dev)
    post = torch.empty(rows, hc_mult, dtype=torch.float32, device=dev)
    comb = torch.empty(rows, hc_mult, hc_mult, dtype=torch.float32, device=dev)
    scale = hc_scale.float().contiguous()
    base = hc_base.float().contiguous()
    gargs = _GateArgs(
        dots.data_ptr(),
        sq.data_ptr(),
        scale.data_ptr(),
        base.data_ptr(),
        pre.data_ptr(),
        post.data_ptr(),
        comb.data_ptr(),
        dot_t,
        dot_i,
        rows,
        int(sinkhorn_iters),
        1.0 / (hc_mult * hidden),
        rms_eps,
        hc_eps,
        hc_eps,
        post_mult,
    )
    hip.launch(fns["mhc_gates_rows_gfx942"], (rows + 255) // 256, gargs)
    return new_residual.view_as(residual), normed, pre, post, comb
