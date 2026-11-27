# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoE front (MF v2, MONO): Stage O's o_proj -> k5 all-reduce + residual + Gemma RMSNorm and, in the same
launch, the layer's router (biased sigmoid top-4 of 128 with its MXFP4 token rows for GEMM1) and the whole
shared expert (gate_up -> SwiGLU -> down), aiter ``flydsl_mf``.

At one request x 8 rows of a layer whose MoE runs the dual-stream op with the glue epilogue, the MoE then runs
on one stream (``MiniMaxM3MoE._front_moe_forward``): production's routed GEMM1 on the launch's routes and rows,
the launch's shared output as the gather's addend with no side-stream join, so P6's ``moe_tail_ar`` takes
GEMM2 + gather + routed scale + shared add + AR2 + norm in the next all-reduce. Production's bits: the launch's
router reads the real-row count production's router reads (``pass_scheduled_tokens_gpu``), so rows past it (a
replay of a step with fewer real rows) route to -1 with weight 0, as production's. Every other call is Stage O's
(up to 32 rows)."""

import logging

import torch

from . import stage_o

logger = logging.getLogger("atom")

DEFERS_PA = False  # production's PA runs in the attention op; the launch starts at o_proj
MAX_ROWS = stage_o.MAX_ROWS
ROWS = 8
# One switch read at call time (WIRE's served-step A/B flips it per graph config); off = Stage O + production's
# dual-stream MoE.
ENABLED = True

# (id(moe), num_valid_rows pointer or 0) -> (table, outputs, kept tensors) or None (the MoE is not the shape it
# takes). The router masks rows past the step's real count (``num_valid_rows``) as production's does, so a table
# holds the count tensor production's router reads for that call (the step's ``batch_info`` entry, read in place).
_TABLES: dict = {}
_OUTPUTS: dict = {}  # id(moe) -> MfOutputs (one set a layer, whatever the count tensor)


def available() -> bool:
    if not stage_o.available():
        return False
    try:
        from atom.models.minimax_m3.mono.gfx1250.mf import flydsl_mf  # noqa: F401
    except ImportError:
        return False
    return True


def hosts(traits) -> bool:
    return False  # no PA launch is deferred


def rows_ok(m: int) -> bool:
    return stage_o.rows_ok(m)


def _takes(moe) -> bool:
    """The MoE the launch's table describes: router bf16 [128, 6144] + fp32 bias, sigmoid top-4 renormalized,
    the shared expert as a native group-32 E4M3 gate_up [1536, 6144] and down [6144, 768] (preshuffled), MXFP4
    routed experts with the glue epilogue on the dual-stream op (where the one-stream path is taken)."""
    from atom.model_ops.moe import Mxfp4MoEMethod

    se = getattr(moe, "shared_experts", None)
    experts = getattr(moe, "experts", None)
    if se is None or experts is None or getattr(moe, "fuse_shared_experts", True):
        return False
    gu, dn = se.gate_up_proj, se.down_proj
    return bool(
        getattr(moe, "_use_dual_stream", False)
        and getattr(moe, "alt_stream", None) is not None
        and getattr(moe, "_glue_epilogue", None) is not None
        and getattr(moe, "_shared_fold_weights", None) is None
        and isinstance(experts.quant_method, Mxfp4MoEMethod)
        and experts.scoring_func == "sigmoid"
        and experts.top_k == 4
        and experts.renormalize
        and not experts.use_grouped_topk
        and experts.custom_routing_function is None
        and experts.num_fused_shared_experts == 0
        and getattr(experts, "swiglu_limit", None) == 7.0
        and moe.e_score_correction_bias is not None
        and moe.gate.weight.dtype == torch.bfloat16
        and tuple(moe.gate.weight.shape) == (128, 6144)
        and getattr(gu, "native_a8_group_rows", None) == 1
        and getattr(gu.weight, "is_shuffled", False)
        and tuple(gu.weight.shape) == (1536, 6144)
        and getattr(dn.weight, "is_shuffled", False)
        and dn.weight.dtype == torch.float8_e4m3fn
        and tuple(dn.weight.shape) == (6144, 768)
    )


def _num_valid_rows(m: int, dev: torch.device):
    """(ok, tensor or None): the real-row count production's router takes for an ``m``-row call
    (``pass_scheduled_tokens_gpu``); not ok when it is not an int32 count on this device."""
    from atom.utils.forward_context import pass_scheduled_tokens_gpu

    nv = pass_scheduled_tokens_gpu(m)
    if nv is None:
        return True, None
    ok = nv.dtype == torch.int32 and nv.numel() >= 1 and nv.device == dev
    return ok, nv


def _tables(moe, nv):
    """The MoE's launch table for the count tensor ``nv`` (None: every row real), built once a (MoE, count
    tensor) outside any capture (a warm-up call precedes the capture); None when the MoE is not the shape the
    launch takes, or inside a capture for a count tensor no eager call met."""
    key = (id(moe), 0 if nv is None else nv.data_ptr())
    if key in _TABLES:
        return _TABLES[key]
    if torch.cuda.is_current_stream_capturing():
        logger.info("MiniMax-M3 MoE front: %s: no table for this row-count tensor at capture; Stage O",
                    getattr(moe, "prefix", "?"))
        return None
    got = None
    if _takes(moe):
        from atom.models.minimax_m3.mono.gfx1250.mf import MfOutputs, mf_scratch, mf_table

        se = moe.shared_experts
        gu, dn = se.gate_up_proj, se.down_proj
        dev = moe.gate.weight.device
        outs = _OUTPUTS.get(id(moe))
        if outs is None:
            outs = _OUTPUTS[id(moe)] = MfOutputs(dev)
        keep = (
            moe.gate.weight.data.contiguous(),
            moe.e_score_correction_bias.data.float().contiguous(),
            gu.weight.data.view(torch.uint8),
            gu.weight_scale.data.view(torch.uint8),
            dn.weight.data.view(torch.uint8),
            dn.weight_scale.data.view(torch.uint8),
        )
        tab = mf_table(*keep, outs, mf_scratch(dev), num_valid_rows=nv)
        got = (tab, outs, keep + (nv,))
    else:
        logger.info("MiniMax-M3 MoE front: %s keeps production's MoE (not the shape the launch takes)",
                    getattr(moe, "prefix", "?"))
    _TABLES[key] = got
    return got


def run(call):
    from atom.model_ops.module_dispatch_ops import _dual_stream_is_active
    from atom.models.minimax_m3 import launch_a

    m = call.x.shape[0]
    moe = call.seam.moe
    if (
        ENABLED
        and m == ROWS
        and moe is not None
        and call.x.stride(1) == 1
        and call.x.stride(0) % 16 == 0
        and call.x_scale.is_contiguous()
        and _dual_stream_is_active(moe, m)
    ):
        nv_ok, nv = _num_valid_rows(m, call.x.device)
        got = _tables(moe, nv) if nv_ok else None
        if got is not None:
            from atom.models.minimax_m3.mono.gfx1250.mf import flydsl_mf

            tab, outs, _ = got
            k5 = call.k5
            out, res_out = flydsl_mf(
                call.x,
                call.x_scale,
                call.seam.o_proj.weight,
                call.seam.o_proj.weight_scale,
                call.residual.reshape(m, -1),
                call.norm_w,
                call.eps,
                k5.inbox,
                k5.ctr,
                k5.slot,
                k5.rank,
                tab,
                max_rows=k5.max_rows,
                maxit=launch_a.k5_maxit(),
            )
            launch_a.set_moe_front(moe, out, outs)
            return out, res_out
    return stage_o.run(call)
