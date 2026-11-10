# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step kernel's expert weights. The routed experts it reads in place, as the production MoE keeps them; the FP8
shared expert in its own plane (``pack_shared``): per expert ``EXPU`` units of ``UNIT8`` bytes -- 18 x 160 gate/up
units ((group, K step, tile) order) then 320 x 5 down units ((tile, K step) order).
"""

import logging

import numpy as np
import torch

from atom.model_ops.linear import weight_is_stored_preshuffled
from atom.models.deepseek_v41.mono.gfx1250.kernels.dims import MOE_IPAD
from atom.mono.runtime.consensus import MonoUnsupported

logger = logging.getLogger("atom")

H, I = 5120, 576  # hidden, the rank's intermediate slice (TP4)
PAD = MOE_IPAD  # the MoE's intermediate width, padded (the kernel's strides assume it)
UGU, NUG, DNU, NDN = 160, 18, 5, 320
EXPU = NUG * UGU + NDN * DNU

# the routed experts as aiter's gfx1250 FlyDSL fused_moe keeps them (Mxfp4MoEMethod, FP8 activations: gate / up rows
# interleaved, then the 16 x 16 tile shuffle; scales n32k4), uint8 [E] + shape; the kernel's moe.py addresses them
ROUTED = {
    "mw13": ("w13_weight", (2 * PAD, H // 2)),
    "ms13": ("w13_weight_scale", (2 * PAD // 32, H // 32 * 32)),
    "mw2": ("w2_weight", (H, PAD // 2)),
    "ms2": ("w2_weight_scale", (H // 32, PAD // 32 * 32)),
}


def routed_experts(experts) -> dict[str, torch.Tensor]:
    """A MoE layer's routed experts (its ``FusedMoE``) as the kernel reads them, refused unless in that layout."""
    method = experts.quant_method
    if (
        getattr(method, "use_triton", True)
        or not getattr(method, "is_guinterleave", False)
        or not getattr(experts.w13_weight, "is_shuffled", False)
    ):
        raise MonoUnsupported(
            "single-launch step: routed experts not in the FlyDSL fused_moe GUGU layout"
        )
    out = {}
    for key, (name, shape) in ROUTED.items():
        t = getattr(experts, name).data.view(torch.uint8)
        if tuple(t.shape[1:]) != shape or not t.is_contiguous():
            raise MonoUnsupported(
                f"single-launch step: {name} {tuple(t.shape)}, expected [E]{list(shape)}"
            )
        out[key] = t
    return out


UNIT8 = 2176  # an FP8 unit: 32 lanes x 64 weight bytes + 32 lanes x 4 E8M0 scale codes


def unshuffle_weight(w):
    """The checkpoint row-major ``[N, K]`` FP8 bytes of a weight ``shuffle_weights`` rearranged in place for the
    preshuffled GEMM (aiter ``shuffle_weight``, layout (16, 16), one byte an element: ``[N/16][16][K/32][2][16]`` ->
    ``[N/16][K/32][2][16][16]``)."""
    n, k = w.shape
    x = w.view(torch.uint8).reshape(n // 16, k // 32, 2, 16, 16)
    return x.permute(0, 3, 1, 2, 4).reshape(n, k).contiguous()


def checkpoint_weight(linear):
    """A loaded FP8 linear's weight as checkpoint row-major bytes ``[N, K]`` (uint8), undoing the in-place shuffle
    ``LinearBase.process_weights_after_loading`` applies when the preshuffled group32 GEMM reads it (decided the way
    the loader decided it)."""
    w = linear.weight
    rows = getattr(linear, "native_a8_group_rows", None)
    shuffled = rows is not None and weight_is_stored_preshuffled(
        linear.quant_type,
        linear.params_dtype,
        needs_preshuffled_weight=getattr(linear, "needs_preshuffled_weight", False),
        native_group_rows=rows,
    )
    return unshuffle_weight(w.data) if shuffled else w.data.view(torch.uint8)


def pack_shared(shared):
    """The FP8 shared expert plane of one MoE block's ``shared_experts`` module (gate_up_proj, w2)."""
    return pack_shared_local(
        checkpoint_weight(shared.gate_up_proj),
        shared.gate_up_proj.weight_scale.data,
        checkpoint_weight(shared.w2),
        shared.w2.weight_scale.data,
    )


def _maps8():
    lane = np.arange(32)
    r = lane % 16
    bb = np.arange(16)
    wofs = (
        16 * (lane // 16)[:, None, None]
        + 32 * np.arange(4)[None, :, None]
        + bb[None, None, :]
    )  # [32][4][16]
    tp = np.arange(NUG * 4)
    ug_rows = np.where(
        r[None, :] < 8,
        tp[:, None] * 8 + r[None, :],
        I + tp[:, None] * 8 + (r[None, :] - 8),
    )
    ug_off = np.arange(40)[:, None, None, None] * 128 + wofs[None]
    g, tl = tp // 4, tp % 4
    ug_u = (g[:, None] * 40 + np.arange(40)[None, :]) * 4 + tl[:, None]
    dn_rows = np.arange(NDN)[:, None] * 16 + r[None, :]
    dn_off = np.arange(DNU)[:, None, None, None] * 128 + wofs[None]
    dn_u = NUG * UGU + np.arange(NDN)[:, None] * 5 + np.arange(DNU)[None, :]
    sc_ug = np.arange(40)[:, None] * 4 + np.arange(4)[None, :]
    sc_dn = np.arange(DNU)[:, None] * 4 + np.arange(4)[None, :]
    return ug_rows, ug_off, ug_u, dn_rows, dn_off, dn_u, sc_ug, sc_dn


def pack_shared_local(gate_up, gate_up_scale, w2, w2_scale):
    """The FP8 shared expert's plane ``[EXPU][UNIT8]`` (same unit order as the routed planes; lane l: row l % 16, K
    chunks 16 (l // 16) + 32 c, c < 4) from the rank's loaded parameters, checkpoint layout (``unshuffle_weight``
    first if the loader shuffled them): gate_up ``[2 I][H]`` (gate rows then up rows) with its 32 x 32 block scales
    ``[2 I / 32][H / 32]``, w2 ``[H][I]`` with ``[H / 32][I / 32]``."""
    dev = gate_up.device
    ug_rows, ug_off, ug_u, dn_rows, dn_off, dn_u, sc_ug, sc_dn = [
        torch.as_tensor(a, device=dev) for a in _maps8()
    ]
    ug = gate_up.view(torch.uint8)
    ugs = gate_up_scale.view(torch.uint8).repeat_interleave(32, 0)  # a code a row
    dn = w2.view(torch.uint8)
    dns = w2_scale.view(torch.uint8).repeat_interleave(32, 0)
    out = torch.empty((EXPU, UNIT8), dtype=torch.uint8, device=dev)
    out[ug_u.reshape(-1), :2048] = ug[
        ug_rows[:, None, :, None, None], ug_off[None]
    ].reshape(72 * 40, 2048)
    out[ug_u.reshape(-1), 2048:] = ugs[
        ug_rows[:, None, :, None], sc_ug[None, :, None, :]
    ].reshape(72 * 40, 128)
    kd = dn.shape[1]
    w = dn[dn_rows[:, None, :, None, None], dn_off.clamp(max=kd - 1)[None]]
    w = torch.where(
        (dn_off < kd)[None], w, torch.zeros((), dtype=torch.uint8, device=dev)
    )
    out[dn_u.reshape(-1), :2048] = w.reshape(NDN * DNU, 2048)
    sc = dns[dn_rows[:, None, :, None], sc_dn.clamp(max=17)[None, :, None, :]]
    sc = torch.where(
        (sc_dn < 18)[None, :, None, :],
        sc,
        torch.full((), 127, dtype=torch.uint8, device=dev),
    )
    out[dn_u.reshape(-1), 2048:] = sc.reshape(NDN * DNU, 128)
    return out
