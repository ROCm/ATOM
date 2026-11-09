# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The routed experts of every MoE layer (target and draft) in the single-launch kernel's plane layout, packed on the
GPU from the loaded weights before ``Mxfp4MoEMethod`` shuffles them in place (the prefill keeps the shuffled ones).

A plane: per expert ``EXPU`` units of ``UNIT`` bytes -- 18 x 160 gate/up units ((group, K step, tile) order) then
320 x 5 down units ((tile, K step) order); a unit is 32 lanes x 32 weight bytes + 32 lanes x 4 E8M0 scale codes, lane l
holding row l % 16 of the unit's 16-row tile and K chunks (2 c + l // 16) x 16 bytes. Byte-identical to the
investigation packer (pack_plane.py / pack_plane_torch.py, checked against the HIP reference).
"""

import logging

import numpy as np
import torch

logger = logging.getLogger("atom")

H, I = 5120, 576  # hidden, the rank's intermediate slice (TP4)
UNIT, UGU, NUG, DNU, NDN = 1152, 160, 18, 5, 320
EXPU = NUG * UGU + NDN * DNU

# layer prefix ("layers.3.ffn", "mtp.0.ffn") -> uint8 [E * EXPU * UNIT] on the layer's device
PLANES: dict[str, torch.Tensor] = {}


def _maps():
    l = np.arange(32)
    r = l % 16
    bb = np.arange(16)
    wofs = (2 * np.arange(2)[None, :, None] + (l // 16)[:, None, None]) * 16 + bb[
        None, None, :
    ]  # [32][2][16]
    tp = np.arange(NUG * 4)
    ug_rows = np.where(
        r[None, :] < 8,
        tp[:, None] * 8 + r[None, :],
        I + tp[:, None] * 8 + (r[None, :] - 8),
    )
    ug_off = np.arange(40)[:, None, None, None] * 64 + wofs[None]
    dn_rows = np.arange(NDN)[:, None] * 16 + r[None, :]
    dn_off = np.arange(DNU)[:, None, None, None] * 64 + wofs[None]
    g = tp // 4
    tl = tp % 4
    ug_u = (g[:, None] * 40 + np.arange(40)[None, :]) * 4 + tl[:, None]
    dn_u = NUG * UGU + np.arange(NDN)[:, None] * 5 + np.arange(DNU)[None, :]
    sc_ug = np.arange(40)[:, None] * 4 + np.arange(4)[None, :]
    sc_dn = np.arange(DNU)[:, None] * 4 + np.arange(4)[None, :]
    return ug_rows, ug_off, dn_rows, dn_off, ug_u, dn_u, sc_ug, sc_dn


def _pack(ug, ugs, dn, dns):
    """ug [E][1152][2560], ugs [E][1152][160], dn [E][5120][288], dns [E][5120][18] (uint8) -> [E][EXPU][UNIT]."""
    dev = ug.device
    ug_rows, ug_off, dn_rows, dn_off, ug_u, dn_u, sc_ug, sc_dn = [
        torch.as_tensor(a, device=dev) for a in _maps()
    ]
    E = ug.shape[0]
    out = torch.empty((E, EXPU, UNIT), dtype=torch.uint8, device=dev)
    w = ug[:, ug_rows[:, None, :, None, None], ug_off[None]]
    # [E][72][40][32 lanes][2][16] -> [..][2][32 lanes][16]: a lane's 2 x 16 B 512 B apart, so each A load of the
    # kernel (lda) spans 4 cache lines, not 8 (CT11 11-08: fixture Zipf -4.2..-5.2 us/layer)
    w = w.transpose(3, 4)
    out[:, ug_u.reshape(-1), :1024] = w.reshape(E, 72 * 40, 1024)
    sc = ugs[:, ug_rows[:, None, :, None], sc_ug[None, :, None, :]]
    out[:, ug_u.reshape(-1), 1024:] = sc.reshape(E, 72 * 40, 128)
    kd = dn.shape[2]
    w = dn[:, dn_rows[:, None, :, None, None], dn_off.clamp(max=kd - 1)[None]]
    w = torch.where(
        (dn_off < kd)[None, None], w, torch.zeros((), dtype=torch.uint8, device=dev)
    )
    w = w.transpose(3, 4)  # (the same lane-split layout as the UG units)
    out[:, dn_u.reshape(-1), :1024] = w.reshape(E, NDN * DNU, 1024)
    sc = dns[:, dn_rows[:, None, :, None], sc_dn.clamp(max=17)[None, :, None, :]]
    sc = torch.where(
        (sc_dn < 18)[None, None, :, None, :],
        sc,
        torch.full((), 127, dtype=torch.uint8, device=dev),
    )
    out[:, dn_u.reshape(-1), 1024:] = sc.reshape(E, NDN * DNU, 128)
    return out


def pack_routed_local(w13, s13, w2, s2, chunk=16):
    """The plane from one rank's MXFP4 expert parameters as loaded (uint8 views): w13 [E][2 pad][2560] (w1 rows
    0..575, w3 rows pad..pad+575), s13 [E][2 pad][160], w2 [E][5120][pad / 2] (bytes 0..287), s2 [E][5120][pad / 32]
    (codes 0..17)."""
    pad = w13.shape[1] // 2
    if pad < I or w13.shape[2] != H // 2 or w2.shape[1] != H or w2.shape[2] * 2 != pad:
        raise ValueError(
            f"unexpected expert shapes {tuple(w13.shape)} {tuple(w2.shape)}"
        )
    parts = []
    for e0 in range(0, w13.shape[0], chunk):
        e1 = min(w13.shape[0], e0 + chunk)
        ug = torch.cat([w13[e0:e1, :I], w13[e0:e1, pad : pad + I]], 1)
        ugs = torch.cat([s13[e0:e1, :I], s13[e0:e1, pad : pad + I]], 1)
        parts.append(
            _pack(ug, ugs, w2[e0:e1, :, : I // 2], s2[e0:e1, :, : I // 32]).reshape(-1)
        )
    return torch.cat(parts)


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
    from atom.model_ops.linear import weight_is_stored_preshuffled

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
    ``[2 I / 32][H / 32]``, w2 ``[H][I]`` with ``[H / 32][I / 32]``. Byte-identical to the investigation packer
    (pack_plane.py pack_shared)."""
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


def _u8(t):
    return t.data.view(torch.uint8)


def capture(prefix: str):
    """A FusedMoE ``_mono_pre_shuffle`` hook: pack the layer's experts into PLANES[prefix] (before the shuffle)."""

    def hook(layer):
        with torch.no_grad():
            PLANES[prefix] = pack_routed_local(
                _u8(layer.w13_weight),
                _u8(layer.w13_weight_scale),
                _u8(layer.w2_weight),
                _u8(layer.w2_weight_scale),
            )
        logger.debug("mono single: packed %s (%d B)", prefix, PLANES[prefix].numel())

    return hook
