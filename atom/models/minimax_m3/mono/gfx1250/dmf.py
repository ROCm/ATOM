# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host wrapper of DMF (``kernels/dmf_gfx1250.py``): a MiniMax-M3.1 dense decoder layer's MLP half -- o_proj + k5
AR1 + residual + RMSNorm, gate_up + SwiGLU-OAI + MXFP8, down -- as one launch, with production's bits. The down's
bf16 output (the layer's per-rank MLP output) lands in ``DmfOutputs.down``; the next layer's input all-reduce takes
it as it takes production's ``down_proj`` output."""

from __future__ import annotations

import torch

from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled, _run_compiled, current_stream_arg, ptr_arg

_SYNCED = set()
_SCRATCH = {}
ROLES = 6  # gate_up + down (bit 0, the router, is off: a dense layer has none)
# gate_up layout: 1 = one workgroup a 32-column h group with its four K splits (252 workgroups, all resident);
# 0 = MF's (group, split) workgroups (548). Same bits.
GW = 1
# the down: 1 = on the 128 o_proj workgroups after their push (48 columns each, K stages handed on through LDS), no
# down workgroups (228 in all); 0 = 24 down workgroups; 2 = 1 with the down's h fragments in VGPRs before its relay.
# Same bits.
DG = 2
# 1: the gate_up's x blocks and the down's h chunks are read in one round, one poll loop. Same bits.
PR = 0


def dmf_scratch(device: torch.device, build=None):
    """The rank's hand-off buffers (3 copies each, armed 0xff), launch counts and timeout counter, shared by the
    dense layers of the rank (allocated once, outside any graph capture). One set a build (``build``: (gw, bq, dg),
    default (GW, 0, DG)): the launch counts and hand-off copies of two layouts never mix."""
    from .kernels import dmf as K

    key = (torch.device(device), (GW, 0, DG) if build is None else tuple(build))
    if key not in _SCRATCH:
        _SCRATCH[key] = dict(
            xb=torch.full((K.NCOPY * K.XBUF,), 0xFF, dtype=torch.uint8, device=device),
            lg=torch.full((K.NCOPY * K.LGBUF,), 0xFF, dtype=torch.uint8, device=device),
            hb=torch.full((K.NCOPY * K.HCOPY,), 0xFF, dtype=torch.uint8, device=device),
            gp=torch.zeros((K.GPBUF,), dtype=torch.uint8, device=device),
            rp=torch.zeros((8,), dtype=torch.uint8, device=device),
            cnt=torch.zeros((K.GRID,), dtype=torch.int32, device=device),
            err=torch.zeros((4,), dtype=torch.int32, device=device),
            x8=torch.full((K.NCOPY * K.X8COPY,), 0xFF, dtype=torch.uint8, device=device),
        )
    return _SCRATCH[key]


class DmfOutputs:
    """A dense layer's outputs (read after the launch): the down output, and the h it was computed from."""

    def __init__(self, device):
        from .kernels import dmf as K

        self.down = torch.zeros((8, 6144), dtype=torch.bfloat16, device=device)
        # ar2: the next layer's input (E4M3 rows, row-major E8M0 scales) and the residual stream after this layer
        self.x2 = torch.zeros((8, 6144), dtype=torch.uint8, device=device)
        self.x2s = torch.zeros((8, 6144 // 32), dtype=torch.uint8, device=device)
        self.r2 = torch.zeros((8, 6144), dtype=torch.bfloat16, device=device)
        self.h = torch.zeros((8, K.S_INTER), dtype=torch.uint8, device=device)  # E4M3, row-major
        self.hs = torch.zeros((8, K.N_GROUPS), dtype=torch.uint8, device=device)  # E8M0, row-major
        # the router's outputs, unused (a dense layer has no router; the table keeps MF's layout)
        self._unused = torch.zeros((8 * 6144,), dtype=torch.uint8, device=device)


def dmf_table(gu_w, gu_s, dn_w, dn_s, outs: DmfOutputs, scratch, next_norm_w=None):
    """The layer's table: gate_up E4M3 [2 x inter, 6144] (16, 16) preshuffled + its E8M0 scales (n32k4 preshuffled),
    down E4M3 [6144, inter] + scales (the same layouts); inter = the kernel's ``S_INTER`` a rank."""
    from .kernels import dmf as K

    assert tuple(gu_w.shape) == (2 * K.S_INTER, 6144) and tuple(dn_w.shape) == (6144, K.S_INTER)
    words = [0] * K.T_WORDS
    u = outs._unused.data_ptr()
    words[K.T_RW] = words[K.T_RB] = u
    words[K.T_GW] = gu_w.data_ptr()
    words[K.T_GS] = gu_s.data_ptr()
    words[K.T_DW] = dn_w.data_ptr()
    words[K.T_DS] = dn_s.data_ptr()
    words[K.T_IDS] = words[K.T_WTS] = words[K.T_XQ] = words[K.T_XS] = u
    words[K.T_SH] = outs.down.data_ptr()
    words[K.T_XB] = scratch["xb"].data_ptr()
    words[K.T_LG] = scratch["lg"].data_ptr()
    words[K.T_HB] = scratch["hb"].data_ptr()
    words[K.T_CNT] = scratch["cnt"].data_ptr()
    words[K.T_ERR] = scratch["err"].data_ptr()
    words[K.T_GP] = scratch["gp"].data_ptr()
    words[K.T_RP] = scratch["rp"].data_ptr()
    words[K.T_HP] = outs.h.data_ptr()
    words[K.T_HSP] = outs.hs.data_ptr()
    words[K.T_X8] = scratch["x8"].data_ptr()
    # ar2: the next layer's input norm (Gemma, bf16 [6144]) and where its MXFP8 input and the residual go
    words[K.T_NW2] = u if next_norm_w is None else next_norm_w.data_ptr()
    words[K.T_X2Q] = outs.x2.data_ptr()
    words[K.T_X2S] = outs.x2s.data_ptr()
    words[K.T_R2] = outs.r2.data_ptr()
    return torch.tensor(words, dtype=torch.int64, device=gu_w.device)


def dmf_supported(rows: int, inter: int, k: int) -> str | None:
    """None when DMF takes a dense layer's MLP half: 8 rows, ``inter`` h columns a rank (the kernel's), o_proj K."""
    from .kernels import dmf as K

    if rows != 8 or inter != K.S_INTER or k != 2048:
        return f"rows {rows}, inter {inter}, o_proj K {k}"
    return None


def flydsl_dmf(
    x, x_scale, w, w_scale, res, norm_w, eps, inbox, ctr, slot, rank, tab, *,
    alpha: float, beta: float, limit: float, maxit: int = 20000000, out=None, res_out=None, max_rows: int = 256,
    timeline=None, stream=None, gw: int | None = None, bq: int = 0, dg: int | None = None, pr: int | None = None,
    ar2: int = 0,
):  # fmt: skip
    """Stage O's operands at 8 rows (x [8, 2048] E4M3 + E8M0 row scales; o_proj in the decode GEMM's layout), the
    norm, production's k5 context and the layer's table. Returns (out, res_out): the normed rows and the residual
    stream after AR1; the MLP's down output lands in the table's ``DmfOutputs.down``."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from .kernels.dmf import launch_mf

    m, k = x.shape
    assert m == 8 and w.shape[0] == 6144 and len(inbox) == 4
    assert x.stride(1) == 1 and x.stride(0) % 16 == 0 and x_scale.is_contiguous()
    if out is None:
        out = torch.empty((m, 6144), dtype=torch.bfloat16, device=x.device)
    if res_out is None:
        res_out = torch.empty((m, 6144), dtype=torch.bfloat16, device=x.device)
    if stream is None:
        stream = current_stream_arg()
    gw = GW if gw is None else int(gw)
    dg = DG if dg is None else int(dg)
    pr = PR if pr is None else int(pr)
    spec = (k, float(alpha), float(beta), float(limit), 1, ROLES, max_rows, 0 if timeline is None else 1, 1, 0, 0,
            int(bq), gw, dg, pr, int(ar2))
    args = (
        ptr_arg(x), ptr_arg(x_scale), ptr_arg(w), ptr_arg(w_scale), ptr_arg(res), ptr_arg(norm_w),
        ptr_arg(out), ptr_arg(res_out), flyc.from_c_void_p(fx.Int32, int(ctr)), ptr_arg(tab, fx.Int64),
        int(inbox[0]), int(inbox[1]), int(inbox[2]), int(inbox[3]),
        int(x.stride(0)), int(slot), int(rank), int(maxit), float(eps),
        stream,
        *spec,
        ptr_arg(timeline if timeline is not None else tab, fx.Int32),
    )  # fmt: skip
    if spec not in _SYNCED:
        # first launch of this build: compile without launching, then barrier the TP group (no barrier in a capture),
        # so no rank's JIT holds its peers' polls (as ``flydsl_mf``)
        _preload_compiled(launch_mf, *args)
        if torch.distributed.is_initialized() and not torch.cuda.is_current_stream_capturing():
            torch.distributed.barrier()
        _SYNCED.add(spec)
    _run_compiled(launch_mf, *args, specialization_key=("dmf",) + spec)
    return out, res_out
