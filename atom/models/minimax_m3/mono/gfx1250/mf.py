# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host wrapper of the MoE-front launch (``kernels/mf_gfx1250.py``): o_proj + k5 AR1 + norm || router || shared
expert, one launch on the main stream."""

from __future__ import annotations

import torch

from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled, _run_compiled, current_stream_arg, ptr_arg

_SYNCED = set()
_SCRATCH = {}


def mf_scratch(device: torch.device, build=None):
    """The rank's hand-off buffers (3 copies each, armed: every byte 0xff), launch counts and timeout counter,
    shared by every layer of the rank (allocated once, outside any graph capture). One set a build (``build``: a
    key of the launch options that change which workgroups count launches, e.g. ``(rmode, shell)``): the launch
    counts of two builds must never mix."""
    from .kernels import mf as K

    key = (torch.device(device), build)
    if key not in _SCRATCH:
        _SCRATCH[key] = dict(
            xb=torch.full((K.NCOPY * K.XBUF,), 0xFF, dtype=torch.uint8, device=device),
            lg=torch.full((K.NCOPY * K.LGBUF,), 0xFF, dtype=torch.uint8, device=device),
            hb=torch.full((K.NCOPY * K.HCOPY,), 0xFF, dtype=torch.uint8, device=device),
            gp=torch.zeros((K.GPBUF,), dtype=torch.uint8, device=device),
            rp=torch.zeros((K.RPBUF,), dtype=torch.uint8, device=device),
            cnt=torch.zeros((K.GRID,), dtype=torch.int32, device=device),
            err=torch.zeros((4,), dtype=torch.int32, device=device),
        )
    return _SCRATCH[key]


class MfOutputs:
    """A layer's outputs (read after the launch): routes, MXFP4 rows, the shared expert's output."""

    def __init__(self, device):
        self.topk_ids = torch.full((8, 4), -1, dtype=torch.int32, device=device)
        self.topk_weights = torch.zeros((8, 4), dtype=torch.float32, device=device)
        self.xq = torch.zeros((8, 6144 // 2), dtype=torch.uint8, device=device)
        self.xs = torch.zeros((8, 6144 // 32), dtype=torch.uint8, device=device)
        self.shared = torch.zeros((8, 6144), dtype=torch.bfloat16, device=device)


def mf_table(gate_w, bias, gu_w, gu_s, dn_w, dn_s, outs: MfOutputs, scratch, num_valid_rows=None):
    """The layer's table: router weight bf16 [128, 6144], bias f32 [128], shared gate_up E4M3 [1536, 6144] (16, 16)
    preshuffled + its E8M0 scales (n32k4 preshuffled), down E4M3 [6144, 768] + scales (the same layouts)."""
    from .kernels import mf as K

    assert gate_w.dtype == torch.bfloat16 and tuple(gate_w.shape) == (128, 6144) and gate_w.is_contiguous()
    assert bias.dtype == torch.float32 and bias.numel() == 128
    assert tuple(gu_w.shape) == (1536, 6144) and tuple(dn_w.shape) == (6144, 768)
    words = [0] * K.T_WORDS
    words[K.T_RW] = gate_w.data_ptr()
    words[K.T_RB] = bias.data_ptr()
    words[K.T_NVR] = 0 if num_valid_rows is None else num_valid_rows.data_ptr()
    words[K.T_GW] = gu_w.data_ptr()
    words[K.T_GS] = gu_s.data_ptr()
    words[K.T_DW] = dn_w.data_ptr()
    words[K.T_DS] = dn_s.data_ptr()
    words[K.T_IDS] = outs.topk_ids.data_ptr()
    words[K.T_WTS] = outs.topk_weights.data_ptr()
    words[K.T_XQ] = outs.xq.data_ptr()
    words[K.T_XS] = outs.xs.data_ptr()
    words[K.T_SH] = outs.shared.data_ptr()
    words[K.T_XB] = scratch["xb"].data_ptr()
    words[K.T_LG] = scratch["lg"].data_ptr()
    words[K.T_HB] = scratch["hb"].data_ptr()
    words[K.T_CNT] = scratch["cnt"].data_ptr()
    words[K.T_ERR] = scratch["err"].data_ptr()
    words[K.T_GP] = scratch["gp"].data_ptr()
    words[K.T_RP] = scratch["rp"].data_ptr()
    return torch.tensor(words, dtype=torch.int64, device=gate_w.device)


def flydsl_mf(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    res: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    inbox: list[int],
    ctr: int,
    slot: int,
    rank: int,
    tab: torch.Tensor,
    *,
    maxit: int = 20000000,
    out: torch.Tensor | None = None,
    res_out: torch.Tensor | None = None,
    roles: int = 7,
    max_rows: int = 256,
    timeline: torch.Tensor | None = None,
    poll_sleep: int = 1,
    rmode: int = 0,
    shell: int = 0,
    stream=None,
):
    """Stage O's operands at 8 rows (x [8, 2048] E4M3 + E8M0 row scales; o_proj in the decode GEMM's layout) and the
    layer's table. Returns (out, res_out); the routes, rows and shared output land in the table's outputs."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from .kernels.mf import launch_mf

    m, k = x.shape
    assert m == 8 and w.shape[0] == 6144 and len(inbox) == 4
    assert x.stride(1) == 1 and x.stride(0) % 16 == 0 and x_scale.is_contiguous()
    if out is None:
        out = torch.empty((m, 6144), dtype=torch.bfloat16, device=x.device)
    if res_out is None:
        res_out = torch.empty((m, 6144), dtype=torch.bfloat16, device=x.device)
    if stream is None:
        stream = current_stream_arg()
    spec = (k, 1.702, 1.0, 7.0, 1, roles, max_rows, 0 if timeline is None else 1, int(poll_sleep), int(rmode), int(shell))
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
        # first launch of this build: compile without launching, then barrier the TP group (no barrier in a capture)
        _preload_compiled(launch_mf, *args)
        if torch.distributed.is_initialized() and not torch.cuda.is_current_stream_capturing():
            torch.distributed.barrier()
        _SYNCED.add(spec)
    _run_compiled(launch_mf, *args, specialization_key=("mf",) + spec)
    return out, res_out
