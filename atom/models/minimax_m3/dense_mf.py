# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dense MLP front (DMF): a MiniMax-M3.1 dense decoder layer's post-attention half -- o_proj, the k5 all-reduce +
residual + Gemma RMSNorm + MXFP8, gate_up, SwiGLU-OAI + MXFP8 and down -- as one launch (aiter ``flydsl_dmf``) at
C4 decode steps of one request x 8 rows on gfx1250 TP4, in place of production's five launches, with their bits.

The launch streams the gate_up and down weights from entry while o_proj and the all-reduce hop run (the MoE
layers' MF front with the router off and the dense MLP's sizes). The down's bf16 output is the layer's per-rank
MLP output; the next layer's input all-reduce takes it as it takes production's ``down_proj`` output. Every other
call (prefill, other row counts, a route that is not the k5 one-shot, a missing table) runs production's calls in
the layer's order."""

import importlib
import logging
import weakref

import torch

from atom.utils.custom_register import direct_register_custom_op

logger = logging.getLogger("atom")

ROWS = 8
HIDDEN = 6144
O_PROJ_K = 2048
# Which forward the op stands in for (the layer's production path):
TAIL_AR_NORM = 0  # MiniMaxM3DecoderLayer.forward: fused_allreduce_gemma_rms_norm[_mxfp8_quant] + MLP
TAIL_SP_AR_NORM = 1  # _forward_sp_ar (ATOM_M3_PREFILL_SP_AR): sp_fused_allreduce_gemma_rms_norm[_mxfp8_quant] + MLP
ENABLED = True  # measurement builds only flip it (off: every call runs production's launches)
# The next layer's input all-reduce + norm + MXFP8 in the same launch (aiter DMF ``ar2``), where that layer's input
# all-reduce is production's MXFP8 one: its call then takes the launch's outputs (`_install_ar_hook`). Read per call.
# Off by default: byte-exact against production's k5 (1 GPU, 11-27 04:58), but its in-launch receive costs more than
# the k5 launch it replaces there (-1.64 us a layer with it vs -1.88 without); on for a served A/B arm.
AR2 = False
# calls by path: dmf = the one launch; dmf_production = production's launches through the op; dmf_ar2 = launches
# that also ran the next layer's input AR; ar2_taken / ar2_missed = that AR's call took the outputs / ran itself
STATS = {"dmf": 0, "dmf_production": 0, "dmf_ar2": 0, "ar2_taken": 0, "ar2_missed": 0}
_DONE: dict = {}  # down output address -> (residual address, norm weight address, (E4M3, residual out, E8M0))
_HOOK: dict = {}

_SEAMS: dict = {}
_TABLES: dict = {}  # seam name -> (table, outputs, kept tensors) or None (the layer is not the shape DMF takes)
_ROUTE: dict = {}  # residual shape -> production's post-attention AR + norm + MXFP8 is k5 in Gluon order
_LOGGED: set = set()


class Seam:
    """One dense decoder layer's seam: what the op reaches by name."""

    def __init__(self, name, layer):
        self.name = name
        self.layer = weakref.ref(layer)
        self.layer_num = getattr(layer.self_attn, "layer_num", -1)
        self.o_proj = layer.self_attn.o_proj
        self.mlp = layer.mlp
        self.norm = layer.post_attention_layernorm


def _log_once(key, msg, *args):
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(msg, *args)


def available() -> bool:
    try:
        from atom.models.minimax_m3.mono.gfx1250.dmf import flydsl_dmf  # noqa: F401
    except ImportError:
        return False
    return True


def bind(layer, prefix: str) -> str | None:
    """The seam name of ``layer`` (a dense MiniMaxM3DecoderLayer), or None when it keeps its traced path: a dense
    layer whose sparse attention writes the fast o_proj's MXFP8 input and whose MLP takes the AR's MXFP8 output and
    the SwiGLU-OAI MXFP8 fusion, TP4, gfx1250, no sequence parallelism / plugin mode, aiter's DMF present. Per-step
    conditions are the op's."""
    from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
    from aiter.jit.utils.chip_info import get_gfx

    from atom.distributed.ulysses_sp import get_sp_world_size
    from atom.plugin.prepare import is_plugin_mode
    from atom.utils import envs

    attn = layer.self_attn
    mlp = getattr(layer, "mlp", None)
    if not (
        not getattr(layer, "is_moe_layer", True)
        and mlp is not None
        and callable(getattr(attn, "attend_mxfp8", None))
        and getattr(attn, "sparse_pa_mxfp8_out", False)
        and getattr(attn, "fast_oproj", False)
        and envs.ATOM_MINIMAX_M3_GLUON_AR_MXFP8
        and getattr(mlp, "fuse_swiglu_mxfp8", False)
        and get_tensor_model_parallel_world_size() == 4
        and get_sp_world_size() == 1
        and not is_plugin_mode()
        and get_gfx() == "gfx1250"
        and available()
    ):
        return None
    name = f"{prefix}.dmf"
    if name in _SEAMS:
        raise ValueError(f"Duplicate DMF seam: {name}")
    _SEAMS[name] = Seam(name, layer)
    _log_once("bind", "MiniMax-M3 DMF seam: dense layers' MLP half as one launch at C4 decode steps of 1 request x %d "
              "rows", ROWS)
    return name


def stats() -> dict:
    return dict(STATS)


# ---------------------------------------------------------------- per-call conditions
def _decode_step() -> bool:
    from atom.utils.forward_context import get_forward_context

    ctx = getattr(get_forward_context(), "context", None)
    return ctx is not None and not getattr(ctx, "is_prefill", True)


def _k5_gluon_mxfp8_route(residual: torch.Tensor, norm_w: torch.Tensor, mode: int = TAIL_AR_NORM) -> bool:
    """Whether production's AR + Gemma RMSNorm + MXFP8 of this call (aiter ``fused_allreduce_rmsnorm_mxfp8_quant``)
    is the k5 one-shot in Gluon order with its MXFP8 epilogue: Stage O's protocol and sum order. Shape-only (no
    deferred MoE tail rides on o_proj's output), so cached."""
    key = (tuple(residual.shape), mode)
    got = _ROUTE.get(key)
    if got is not None:
        return got
    ok = False
    try:
        from aiter.dist.parallel_state import get_tp_group

        # not `import aiter.dist....communicator_cuda as cc`: the `aiter.dist` attribute is torch.distributed
        cc = importlib.import_module("aiter.dist.device_communicators.communicator_cuda")
        comm = get_tp_group().device_communicator
        ca = getattr(comm, "ca_comm", None)
        x = residual  # o_proj's output has the residual's shape, dtype and layout
        n = norm_w.numel()
        if (
            ca is not None and x.dim() == 2 and x.is_contiguous() and x.shape[-1] == n and n % 32 == 0
            # the sequence-parallel op takes the plain AR below its owner-row split's size
            and (mode == TAIL_AR_NORM or not comm.ar2s_sp_eligible(x, norm_w))
        ):
            lamport = ca.should_lamport_ar_rms(x, x, norm_w, n)
            gluon = cc._use_gluon_ar_norm(comm, x, x, norm_w, lamport=lamport)
            ok = bool(gluon and ca.should_k5_ar(x, x, norm_w))
    except Exception as e:  # an aiter without this dispatch: production's path
        _log_once("route_err", "MiniMax-M3 DMF: AR route check failed (%s); production path", e)
        ok = False
    _ROUTE[key] = ok
    _log_once(f"route{key}", "MiniMax-M3 DMF: post-attention AR + MXFP8 of %s is k5 Gluon: %s", key, ok)
    return ok


def _mlp_takes(mlp) -> bool:
    """The MLP production runs as DMF's: MXFP8 input, native group-32 E4M3 gate_up [2 x inter, 6144] and down
    [6144, inter] (preshuffled, n32k4 scales), SwiGLU-OAI + MXFP8 between them, inter = DMF's."""
    from atom.models.minimax_m3.mono.gfx1250.dmf import dmf_supported

    from atom.models.minimax_m3 import model as m3

    gu, dn = mlp.gate_up_proj, mlp.down_proj
    reduces_partials = gu.splitk_consumer_max_m > 0 and m3._linear_consumes_per_token_fp8(gu)
    inter = dn.weight.shape[1] if dn.weight.dim() == 2 else -1
    return bool(
        m3._linear_consumes_mxfp8(gu)
        and not m3._linear_consumes_per_token_fp8(gu)
        and getattr(gu, "native_a8_group_rows", None) == 1
        and m3._linear_consumes_native_mxfp8(dn)
        and mlp.fuse_swiglu_mxfp8
        and not reduces_partials
        and getattr(gu, "bias", None) is None
        and getattr(dn, "bias", None) is None
        and gu.weight.dtype == torch.float8_e4m3fn
        and dn.weight.dtype == torch.float8_e4m3fn
        and getattr(gu.weight, "is_shuffled", False)
        and getattr(dn.weight, "is_shuffled", False)
        and tuple(gu.weight.shape) == (2 * inter, HIDDEN)
        and tuple(gu.weight_scale.shape) == (2 * inter, HIDDEN // 32)
        and tuple(dn.weight_scale.shape) == (HIDDEN, inter // 32)
        and dmf_supported(ROWS, inter, O_PROJ_K) is None
    )


def _next_input_norm(seam):
    """The next layer's input Gemma RMSNorm when that layer's input all-reduce is production's MXFP8 one (what
    the launch's ``ar2`` computes), with this layer's eps; else None."""
    from atom.models.minimax_m3 import model as m3
    from atom.utils import envs

    layer = seam.layer()
    ref = getattr(layer, "_next_layer", None) if layer is not None else None
    nxt = ref() if ref is not None else None
    attn = getattr(nxt, "self_attn", None)
    qkv = getattr(attn, "qkv_proj", None)
    norm = getattr(nxt, "input_layernorm", None)
    w = getattr(norm, "weight", None)
    if not (
        isinstance(nxt, m3.MiniMaxM3DecoderLayer)
        and qkv is not None
        and isinstance(norm, m3.GemmaRMSNorm)
        and envs.ATOM_MINIMAX_M3_GLUON_AR_MXFP8
        and m3._linear_consumes_mxfp8(qkv)
        and not m3._linear_consumes_per_token_fp8(qkv)
        and w is not None
        and w.dtype == torch.bfloat16
        and w.numel() == HIDDEN
        and float(norm.variance_epsilon) == float(seam.norm.variance_epsilon)
    ):
        return None
    return norm


def _build(ar2):
    """The aiter build a table of this ``ar2`` runs (its scratch key): (gw, bq, dg, rd, ar2)."""
    from atom.models.minimax_m3.mono.gfx1250 import dmf

    return (getattr(dmf, "GW", 1), 0, getattr(dmf, "DG", 1), getattr(dmf, "PR", 0), int(ar2))


def _table(seam):
    """The layer's DMF table, built once outside any capture (a warm-up call precedes it), or None."""
    if seam.name in _TABLES:
        return _TABLES[seam.name]
    if torch.cuda.is_current_stream_capturing():
        return None
    got = None
    if _mlp_takes(seam.mlp):
        from atom.models.minimax_m3.mono.gfx1250.dmf import DmfOutputs, dmf_scratch, dmf_table

        gu, dn = seam.mlp.gate_up_proj, seam.mlp.down_proj
        dev = gu.weight.device
        outs = DmfOutputs(dev)
        keep = (
            gu.weight.data.view(torch.uint8),
            gu.weight_scale.data.view(torch.uint8),
            dn.weight.data.view(torch.uint8),
            dn.weight_scale.data.view(torch.uint8),
        )
        nxt = _next_input_norm(seam) if _aiter_has_ar2() else None
        tab = dmf_table(*keep, outs, dmf_scratch(dev, build=_build(nxt is not None)),
                        **({"next_norm_w": nxt.weight.data} if nxt is not None else {}))
        got = (tab, outs, keep, nxt)
    else:
        _log_once(f"takes{seam.name}", "MiniMax-M3 DMF: %s keeps production's MLP (not the shape DMF takes)",
                  seam.name)
    _TABLES[seam.name] = got
    return got


def _run(seam, x, x_scale, residual, norm_w, eps, mode):
    """(mlp output, residual out) from the one launch, or None when this call does not qualify."""
    if not (ENABLED and _decode_step()):
        return None
    from atom.models.minimax_m3 import launch_a
    from atom.models.minimax_m3 import model as m3

    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    s2 = x_scale.reshape(m, -1)
    if not (
        m == ROWS
        and k == O_PROJ_K
        and x.dtype == torch.float8_e4m3fn
        and x2.stride(1) == 1
        and x2.stride(0) % 16 == 0
        and s2.shape[1] == k // 32
        and s2.is_contiguous()
        and residual.dtype == torch.bfloat16
        and tuple(residual.shape) == (m, HIDDEN)
        and residual.is_contiguous()
        and norm_w.numel() == HIDDEN
        and tuple(seam.o_proj.weight.shape) == (HIDDEN, k)
        and m3._fast_oproj_resolved(seam.o_proj)
        and _k5_gluon_mxfp8_route(residual, norm_w, mode)
    ):
        return None
    from aiter.ops.flydsl.mxfp8_decode_gemm_gfx1250 import mxfp8_decode_gemm_supported

    if not mxfp8_decode_gemm_supported(m, HIDDEN, k):
        return None
    k5 = launch_a._k5()
    if k5 is None or k5.max_rows < m:
        return None
    got = _table(seam)
    if got is None:
        return None
    from atom.models.minimax_m3.mono.gfx1250.dmf import flydsl_dmf

    tab, outs, _, nxt = got
    mlp = seam.mlp
    ar2 = nxt is not None and AR2 and _install_ar_hook()
    _, res_out = flydsl_dmf(
        x2,
        s2,
        seam.o_proj.weight,
        seam.o_proj.weight_scale,
        residual,
        norm_w,
        float(eps),
        k5.inbox,
        k5.ctr,
        k5.slot,
        k5.rank,
        tab,
        alpha=float(mlp.swiglu_alpha),
        beta=float(mlp.swiglu_beta),
        limit=float(mlp.swiglu_limit),
        maxit=launch_a.k5_maxit(),
        max_rows=k5.max_rows,
        **({"ar2": 1} if ar2 else {}),
    )
    _DONE.clear()  # an entry the next AR call did not take is stale now
    if ar2:
        from aiter.utility.dtypes import fp8_e8m0

        STATS["dmf_ar2"] += 1
        _DONE[outs.down.data_ptr()] = (
            res_out.data_ptr(),
            nxt.weight.data_ptr(),
            (outs.x2.view(torch.float8_e4m3fn), outs.r2, outs.x2s.view(fp8_e8m0)),
        )
    return outs.down, res_out


def _aiter_has_ar2() -> bool:
    import inspect

    try:
        from atom.models.minimax_m3.mono.gfx1250.dmf import flydsl_dmf
    except ImportError:
        return False
    return "ar2" in inspect.signature(flydsl_dmf).parameters


def _install_ar_hook() -> bool:
    """Wrap the TP group's MXFP8 all-reduce + RMSNorm entry (``fused_allreduce_rmsnorm_mxfp8_quant`` of its device
    communicator, the call the next layer's input AR makes): a call on a DMF ``ar2`` launch's down output, with that
    launch's residual and norm, returns the launch's outputs (the same bytes: the launch ran this AR in k5's order and
    advanced k5's counters for it). Any other call is production's."""
    if "ok" in _HOOK:
        return _HOOK["ok"]
    ok = False
    try:
        from aiter.dist.parallel_state import get_tp_group

        comm = get_tp_group().device_communicator
        orig = comm.fused_allreduce_rmsnorm_mxfp8_quant

        def fused_allreduce_rmsnorm_mxfp8_quant(input_, res_inp_, weight_, eps, prefill_support=False,
                                                gemma_norm=False):
            ent = _DONE.pop(input_.data_ptr(), None) if _DONE else None
            if ent is not None:
                if gemma_norm and ent[0] == res_inp_.data_ptr() and ent[1] == weight_.data_ptr():
                    STATS["ar2_taken"] += 1
                    return ent[2]
                STATS["ar2_missed"] += 1
                _log_once("ar2_missed", "MiniMax-M3 DMF: an AR call on a DMF launch's output did not take its "
                          "outputs (residual / norm differ); production's AR")
            return orig(input_, res_inp_, weight_, eps, prefill_support, gemma_norm=gemma_norm)

        comm.fused_allreduce_rmsnorm_mxfp8_quant = fused_allreduce_rmsnorm_mxfp8_quant
        ok = True
        _log_once("ar2", "MiniMax-M3 DMF: the next layer's input AR runs in the dense launch (ar2)")
    except Exception as e:  # an aiter without this entry: no ar2
        _log_once("ar2_err", "MiniMax-M3 DMF: no ar2 (%s)", e)
    _HOOK["ok"] = ok
    return ok


# ---------------------------------------------------------------- the op
def minimax_m3_dense_tail(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    seam_name: str,
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """o_proj on the attention's MXFP8 output -> all-reduce + residual + Gemma RMSNorm -> the dense MLP of the seam
    ``seam_name``: (MLP output, TP-partial; residual out). ``mode``: the forward the op stands in for."""
    seam = _SEAMS[seam_name]
    got = _run(seam, x, x_scale, residual, norm_weight, eps, mode)
    if got is not None:
        STATS["dmf"] += 1
        _log_once("dmf", "MiniMax-M3 DMF: first one-launch call (layer %d, %d rows)", seam.layer_num, ROWS)
        return got
    STATS["dmf_production"] += 1
    # production: the calls of the layer's traced path, in its order
    from atom.model_ops.layernorm import (
        fused_allreduce_gemma_rms_norm,
        fused_allreduce_gemma_rms_norm_mxfp8_quant,
        fused_allreduce_gemma_rms_norm_quant,
        sp_fused_allreduce_gemma_rms_norm,
        sp_fused_allreduce_gemma_rms_norm_mxfp8_quant,
    )
    from atom.models.minimax_m3 import model as m3
    from atom.utils import envs

    mlp = seam.mlp
    h = m3._o_proj_mxfp8(seam.o_proj, x, x_scale)
    if mode == TAIL_SP_AR_NORM:
        if envs.ATOM_MINIMAX_M3_GLUON_AR_MXFP8 and m3._linear_consumes_mxfp8(mlp.gate_up_proj):
            h, h_scale, residual = sp_fused_allreduce_gemma_rms_norm_mxfp8_quant(h, residual, seam.norm)
            return mlp(h, x_scale=h_scale), residual
        h, residual = sp_fused_allreduce_gemma_rms_norm(h, residual, seam.norm)
        return mlp(h), residual
    if m3._linear_consumes_per_token_fp8(mlp.gate_up_proj):
        h, h_scale, residual = fused_allreduce_gemma_rms_norm_quant(h, residual, seam.norm)
        return mlp(h, x_scale=h_scale), residual
    if envs.ATOM_MINIMAX_M3_GLUON_AR_MXFP8 and m3._linear_consumes_mxfp8(mlp.gate_up_proj):
        h, h_scale, residual = fused_allreduce_gemma_rms_norm_mxfp8_quant(h, residual, seam.norm)
        return mlp(h, x_scale=h_scale), residual
    h, residual = fused_allreduce_gemma_rms_norm(h, residual, seam.norm)
    return mlp(h), residual


def _minimax_m3_dense_tail_fake(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    seam_name: str,
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(residual), torch.empty_like(residual)


direct_register_custom_op(
    op_name="minimax_m3_dense_tail",
    op_func=minimax_m3_dense_tail,
    mutates_args=(),
    fake_impl=_minimax_m3_dense_tail_fake,
    # o_proj reads row-major E4M3 rows and contiguous E8M0 scales (as its own op)
    tags=(torch.Tag.needs_fixed_stride_order,),
)

__all__ = ["ENABLED", "STATS", "bind", "minimax_m3_dense_tail", "stats"]
