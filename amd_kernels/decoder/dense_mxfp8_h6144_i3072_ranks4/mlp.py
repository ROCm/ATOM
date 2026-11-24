# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The dense MLP block on its gate_up input codes: gate_up GEMM -> SwiGLU-OAI + MXFP8 quant -> down GEMM (TP
partial), or at 1-8 rows the fused gate_up + SwiGLU kernel -> down on its bf16 output. The row-count dispatch and
the code layouts (``gemm.mxfp8_common.prequant_code_tile_rows`` of each GEMM) live here; the GEMM calls are the ones
``gemm.mxfp8_common``'s linear integration makes for codes / bf16 inputs, so the bits equal the per-kernel chain."""

import torch

from amd_kernels.warm_up import entry

H, I = 6144, 3072
FUSED_GATEUP_MAX_ROWS = 8  # gemm.mxfp8_gateup_swiglu_n6144_k6144_i3072's rows
MAX_ROWS = 2048
MXFP8_SWIGLU_FORMS = ("bf16_chain", "fp32")


def _gemm_codes(q, s, w, ws, m):
    """The MXFP8 GEMM on pre-quantized codes, as the linear integration picks it (decode row on tiled codes)."""
    from amd_kernels.gemm.mxfp8_common import gemm as G

    return G.mxfp8_gemm_prequant(q, s, w, ws, torch.bfloat16, m).reshape(m, w.shape[0])


def mlp_supported(gate_up_w: torch.Tensor, down_w: torch.Tensor) -> bool:
    """Whether ``mxfp8_swiglu_oai_mlp`` takes these weights: preshuffled MXFP8 gate_up [6144, 6144] (gate | up) and
    down [6144, 3072]."""
    return (
        tuple(gate_up_w.shape) == (2 * I, H)
        and tuple(down_w.shape) == (H, I)
        and gate_up_w.dtype == down_w.dtype == torch.float8_e4m3fn
    )


def _mlp(q, s, m, gate_up_w, gate_up_ws, down_w, down_ws, alpha, beta, limit, form="bf16_chain"):
    """down's output [m, 6144] bf16 (the TP partial: no reduction) of the MLP on gate_up's input ``q`` / ``s`` (the
    MXFP8 codes and shuffled scales of m rows in the layout ``prequant_code_tile_rows(m, 6144, 6144)`` names, as
    ``allreduce_gemma_bf16_mxfp8_h6144`` / ``vocab_gather_gemma_mxfp8_h6144`` write them). Weights: preshuffled
    MXFP8 (``gemm.mxfp8_common.preshuffle_mxfp8_weight``), gate_up [6144, 6144] (gate | up), down [6144, 3072]. ``form``:
    the SwiGLU numerics (``bf16_chain`` = each op rounded to bf16, ``fp32``). The first call of each form compiles
    every row count's variants (1-2048 rows)."""
    from amd_kernels.activation.swiglu_oai_mxfp8_i3072 import swiglu_oai_mxfp8
    from amd_kernels.gemm.mxfp8_common import gemm as G

    assert 1 <= m <= MAX_ROWS and form in MXFP8_SWIGLU_FORMS
    assert tuple(gate_up_w.shape) == (2 * I, H) and tuple(down_w.shape) == (H, I)
    if m <= FUSED_GATEUP_MAX_ROWS and q.shape[0] == 16:
        from amd_kernels.gemm.mxfp8_gateup_swiglu_n6144_k6144_i3072 import mxfp8_gateup_swiglu_bf16

        act = mxfp8_gateup_swiglu_bf16(q, s, gate_up_w, gate_up_ws, m, alpha, beta, limit, form=form)
        return G.mxfp8_linear(act.reshape(-1, I), down_w, down_ws, torch.bfloat16).reshape(m, H)
    h = _gemm_codes(q, s, gate_up_w, gate_up_ws, m)
    tile = G.prequant_code_tile_rows(m, H, I, torch.bfloat16)
    qd, sd = swiglu_oai_mxfp8(h, alpha, beta, limit, tile, form=form)
    return _gemm_codes(qd, sd, down_w, down_ws, m)


def _mode(q, s, m, gate_up_w, gate_up_ws, down_w, down_ws, alpha, beta, limit, form="bf16_chain"):
    return form


def _inputs(mode, n, q, s, m, gate_up_w, gate_up_ws, down_w, down_ws, alpha, beta, limit, form="bf16_chain"):
    """``n`` rows of zero codes in the layout the gate_up GEMM reads at n rows (the caller's weights; nothing the
    call writes is the caller's)."""
    from amd_kernels.gemm.mxfp8_common import gemm as G

    tile = G.prequant_code_tile_rows(n, 2 * I, H, torch.bfloat16)
    rows = -(-n // max(tile, 1)) * max(tile, 1)
    qn = torch.zeros((rows, H), dtype=torch.float8_e4m3fn, device=q.device)
    sn = torch.full((-(-rows // 32) * 32, H // 32), 127, dtype=torch.uint8, device=q.device)
    return (qn, sn, n, gate_up_w, gate_up_ws, down_w, down_ws, alpha, beta, limit), {"form": mode}


mxfp8_swiglu_oai_mlp = entry("decoder/dense_mxfp8_h6144_i3072_ranks4:mlp", mode=_mode,
                             sizes=lambda mode: range(1, MAX_ROWS + 1), inputs=_inputs)(_mlp)
