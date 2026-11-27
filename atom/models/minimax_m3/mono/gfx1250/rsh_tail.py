# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of launch A's router + shared gate_up tail phase (``kernels/rsh_tail_gfx1250.py``).

``RshOutputs`` holds what the phase writes (launch B's inputs) and its scratch; ``rsh_table`` packs the
device table the phase reads (one int64 word per pointer / int, ``T_*``). Build the table once per layer
outside graph capture: only data pointers go in it, and they must stay put across replays.
"""

from __future__ import annotations

import torch

from .kernels import rsh_tail as k


class RshOutputs:
    """Launch B's inputs + the phase's cluster scratch (and optional per-CTA stamps)."""

    def __init__(self, device, tl=False):
        u8 = dict(dtype=torch.uint8, device=device)
        self.topk_ids = torch.empty((k.ROWS, k.TOPK), dtype=torch.int32, device=device)
        self.topk_weights = torch.empty((k.ROWS, k.TOPK), dtype=torch.float32, device=device)
        self.xq = torch.empty((k.ROWS, k.HIDDEN // 2), **u8)  # MXFP4 token rows (router quant_rows)
        self.xs = torch.empty((k.ROWS, k.HIDDEN // 32), **u8)  # their E8M0
        self.h = torch.empty((k.ROWS, k.S_INTER), **u8)  # shared intermediate, E4M3
        self.hs = torch.empty((k.ROWS, k.N_GROUPS), **u8)  # its E8M0, row-major
        self.scratch = torch.zeros(k.SCR_WORDS, dtype=torch.float32, device=device)
        # roles: 104 (clusters of 8) or 112 (clusters of 16)
        self.stamps = torch.zeros((k.N_ROLES + 8, k.TL_SLOTS), dtype=torch.int64, device=device) if tl else None

    def router_logits_fp32(self):
        """The router partials summed in split order (fp32 [8, 128]): bf16-round for the logits."""
        p = self.scratch[: k.SCR_R].view(k.R_SPLIT, k.ROWS, k.EXPERTS)
        acc = p[0].clone()
        for i in range(1, k.R_SPLIT):
            acc = acc + p[i]
        return acc

    def gate_up_bf16(self, cl=8):
        """The shared gate_up (bf16 [8, 1536]) from the scratch partials, summed in split order (``cl``: the
        phase's cluster size)."""
        p = self.scratch[k.SCR_R :].view(-1, cl // k.S_SPLIT, k.S_SPLIT, 4, k.ROWS, 16)  # c, gl, s, tile, tok, row
        acc = p[:, :, 0].clone()
        for i in range(1, k.S_SPLIT):
            acc = acc + p[:, :, i]
        acc = acc.to(torch.bfloat16)  # [c, gl, tile, tok, 16]
        g = acc.permute(3, 0, 1, 2, 4).reshape(k.ROWS, k.N_GROUPS, 4, 16)  # tok, group, tile, row
        gate = g[..., :8].reshape(k.ROWS, k.S_INTER)
        up = g[..., 8:].reshape(k.ROWS, k.S_INTER)
        return torch.cat([gate, up], dim=1)


def rsh_table(
    x: torch.Tensor,
    router_w: torch.Tensor,
    router_bias: torch.Tensor,
    shared_w: torch.Tensor,
    shared_ws: torch.Tensor,
    out: RshOutputs,
    num_valid_rows: torch.Tensor | None = None,
    x_prev: torch.Tensor | None = None,
) -> torch.Tensor:
    """The phase's device table. ``x``: bf16 [>= 8, 6144] rows (R1's normed output, unit inner stride);
    ``router_w``: bf16 [128, 6144] contiguous; ``router_bias``: fp32 [128] (``e_score_correction_bias``);
    ``shared_w``: the shared gate_up's E4M3 [1536, 6144] in production's preshuffled (16, 16) layout;
    ``shared_ws``: its E8M0 [1536, 192] in production's n32k4 layout; ``num_valid_rows``: int32 [1] or None;
    ``x_prev`` (data-as-flag builds): the x buffer of the previous launch that used the phase (same shape and
    stride as ``x``), re-armed to 0xff by this launch's router CTAs; every x buffer starts filled with 0xff."""
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.shape[1] == k.HIDDEN and x.shape[0] >= k.ROWS
    assert router_w.dtype == torch.bfloat16 and router_w.is_contiguous() and tuple(router_w.shape) == (k.EXPERTS, k.HIDDEN)
    assert router_bias.dtype == torch.float32 and router_bias.numel() == k.EXPERTS
    assert shared_w.element_size() == 1 and shared_w.is_contiguous() and tuple(shared_w.shape) == (2 * k.S_INTER, k.HIDDEN)
    assert shared_ws.element_size() == 1 and shared_ws.is_contiguous() and shared_ws.numel() == 2 * k.S_INTER * k.HIDDEN // 32
    words = [0] * k.T_WORDS
    words[k.T_X] = x.data_ptr()
    words[k.T_LDX] = x.stride(0)
    words[k.T_RW] = router_w.data_ptr()
    words[k.T_RB] = router_bias.data_ptr()
    words[k.T_NVR] = 0 if num_valid_rows is None else num_valid_rows.data_ptr()
    words[k.T_SW] = shared_w.data_ptr()
    words[k.T_SS] = shared_ws.data_ptr()
    words[k.T_IDS] = out.topk_ids.data_ptr()
    words[k.T_WTS] = out.topk_weights.data_ptr()
    words[k.T_XQ] = out.xq.data_ptr()
    words[k.T_XS] = out.xs.data_ptr()
    words[k.T_H] = out.h.data_ptr()
    words[k.T_HS] = out.hs.data_ptr()
    words[k.T_SCR] = out.scratch.data_ptr()
    words[k.T_TL] = 0 if out.stamps is None else out.stamps.data_ptr()
    if x_prev is not None:
        assert x_prev.shape == x.shape and x_prev.stride() == x.stride() and x_prev.data_ptr() != x.data_ptr()
    words[k.T_XPREV] = 0 if x_prev is None else x_prev.data_ptr()
    return torch.tensor(words, dtype=torch.int64, device=x.device)


def rsh_launch(tab: torch.Tensor, x: torch.Tensor | None = None, stream=None, tl: bool = False, cl: int = 16,
               shared_only: bool = False):
    """Run the phase as its own launch (``kernels.rsh_tail_gfx1250.compile_rsh_launch``) on the table of
    ``rsh_table(x=k5 output, ...)``: router ids / weights + MXFP4 rows, and the shared expert's MXFP8 h + E8M0.
    ``x`` (contiguous bf16 [8, 6144]): pass it per call when its buffer is not the table's (graph capture).
    ``cl``: 16 (the router in moe_router_topk's 16-workgroup geometry, the default) or 8 (launch A's layout).
    ``shared_only``: only the shared gate_up + SwiGLU + MXFP8 h (96 CTAs), e.g. on the shared expert's side stream."""
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, current_stream_arg, ptr_arg

    stream = current_stream_arg() if stream is None else stream
    if x is None:
        _run_compiled(k.compile_rsh_launch(tl=tl, cl=cl, shared_only=shared_only), ptr_arg(tab, fx.Int64), stream)
        return
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and tuple(x.shape) == (k.ROWS, k.HIDDEN)
    _run_compiled(k.compile_rsh_launch(tl=tl, x_arg=True, cl=cl, shared_only=shared_only), ptr_arg(tab, fx.Int64),
                  ptr_arg(x, fx.BFloat16), stream)


__all__ = ["RshOutputs", "rsh_launch", "rsh_table"]
