# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MiniMax-M3.1 decode on gfx1250 TP4: Stage O.

A MoE decoder layer whose sparse attention writes o_proj's MXFP8 input (the
fast o_proj) runs its post-attention part as one custom op, opaque to
torch.compile:

    attention (QKV, index top-k, sparse PA -> MXFP8 rows)
      -> ``minimax_m3_attn_tail``: o_proj -> all-reduce + residual + RMSNorm

Per call, from shapes only (so every TP rank decides alike), the op takes either

* aiter's ``flydsl_oproj_ar``: the o_proj GEMV, the k5 one-shot all-reduce and
  the residual + Gemma RMSNorm in one launch, on production's k5 context
  (inboxes, counters), with production's bits. Taken when production's
  all-reduce of the call is the k5 kernel in Gluon order with bf16 output,
  i.e. a decode step of at most ``MAX_ROWS`` rows; or
* production's two calls (the decode GEMM, then the fused all-reduce + norm)
  for everything else: prefill, larger batches, other routes.

Stage O runs k5's own inbox / counter protocol, so it interleaves with the
other layers' k5 calls.
"""

from __future__ import annotations

import ctypes
import dataclasses
import importlib
import logging

import torch

from atom.utils.custom_register import direct_register_custom_op

logger = logging.getLogger("atom")

MAX_ROWS = 32
HIDDEN = 6144

_SEAMS: dict[str, Seam] = {}
_ROUTE: dict[tuple, bool] = {}  # residual shape -> production takes the k5 Gluon route
_AVAILABLE: list[bool] = []


@dataclasses.dataclass
class K5:
    """Production's k5 context (``CustomAllreduce._k5_ctx``): the four inboxes,
    this rank's counters, packs a sender a buffer, rank, row cap."""

    inbox: list
    ctr: int
    slot: int
    rank: int
    max_rows: int


class Seam:
    """One decoder layer's Stage O call site: what the op reaches by name."""

    def __init__(self, name, layer):
        self.name = name
        self.o_proj = layer.self_attn.o_proj
        self.norm = layer.post_attention_layernorm


def available() -> bool:
    """aiter's Stage O launch imports (checked once)."""
    if not _AVAILABLE:
        try:
            from aiter.ops.flydsl.oproj_ar import flydsl_oproj_ar  # noqa: F401
        except ImportError:
            _AVAILABLE.append(False)
        else:
            _AVAILABLE.append(True)
    return _AVAILABLE[0]


def bind(layer, prefix: str) -> str | None:
    """The Stage O seam name of ``layer`` (a MiniMaxM3DecoderLayer), or None
    when the layer keeps its traced path: a MoE layer whose sparse attention
    writes the fast o_proj's MXFP8 input, TP4, gfx1250, no sequence parallelism
    or plugin mode, and aiter's launch present. Per-call conditions are the
    op's."""
    attn = layer.self_attn
    if not (
        getattr(layer, "is_moe_layer", False)
        and callable(getattr(attn, "attend_mxfp8", None))
        and getattr(attn, "sparse_pa_mxfp8_out", False)
        and getattr(attn, "fast_oproj", False)
    ):
        return None
    from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
    from aiter.jit.utils.chip_info import get_gfx
    from atom.distributed.ulysses_sp import get_sp_world_size

    from atom.plugin.prepare import is_plugin_mode

    if not (
        get_tensor_model_parallel_world_size() == 4
        and get_sp_world_size() == 1
        and not is_plugin_mode()
        and get_gfx() == "gfx1250"
        and available()
    ):
        return None
    name = f"{prefix}.stage_o"
    if name in _SEAMS:
        raise ValueError(f"Duplicate Stage O seam: {name}")
    _SEAMS[name] = Seam(name, layer)
    return name


def _k5() -> K5 | None:
    """Production's k5 context, the one its custom all-reduce launches k5 with."""
    from aiter.dist.parallel_state import get_tp_group

    ca = getattr(get_tp_group().device_communicator, "ca_comm", None)
    ctx = int(getattr(ca, "_k5_ctx", 0) or 0) if ca is not None else 0
    if not ctx:
        return None
    # k5ar::Ctx: int rank, int64 max_rows, int64 slot, char* inbox[4], uint32* ctr
    w = [
        int.from_bytes(ctypes.string_at(ctx + o, 8), "little") for o in range(0, 64, 8)
    ]
    return K5(inbox=w[3:7], ctr=w[7], slot=w[2], rank=w[0] & 0xFFFFFFFF, max_rows=w[1])


def _k5_maxit() -> int:
    """Production k5's poll bound: a slow but alive peer is never cut off."""
    car = importlib.import_module("aiter.dist.device_communicators.custom_all_reduce")
    return int(getattr(car, "_K5_AR_MAXIT", 20000000))


def _k5_gluon_route(residual: torch.Tensor, norm_w: torch.Tensor) -> bool:
    """Whether production's all-reduce + norm of this call is the k5 one-shot
    kernel in Gluon order with bf16 output: aiter's ``fused_allreduce_rmsnorm``
    dispatch asked with the call's shapes. Shape-only, so cached; an aiter
    without this dispatch answers False (production's path)."""
    key = tuple(residual.shape)
    got = _ROUTE.get(key)
    if got is None:
        try:
            got = _k5_gluon_route_checked(residual, norm_w)
        except Exception as e:  # noqa: BLE001 -- any failure keeps production
            logger.info("MiniMax-M3 Stage O: AR route check failed (%s)", e)
            got = False
        _ROUTE[key] = got
        logger.info(
            "MiniMax-M3 Stage O: post-attention AR of %s is k5 Gluon: %s", key, got
        )
    return got


def _k5_gluon_route_checked(residual: torch.Tensor, norm_w: torch.Tensor) -> bool:
    from aiter.dist.parallel_state import get_tp_group

    # not `import aiter.dist....communicator_cuda as cc`: the `aiter.dist`
    # attribute is torch.distributed
    cc = importlib.import_module("aiter.dist.device_communicators.communicator_cuda")
    comm = get_tp_group().device_communicator
    ca = getattr(comm, "ca_comm", None)
    x = residual  # o_proj's output has the residual's shape, dtype and layout
    n = norm_w.numel()
    total = x.numel() * x.element_size()
    if ca is None or x.dim() != 2 or not x.is_contiguous() or x.shape[-1] != n:
        return False
    if getattr(comm, "gluon_ar", None) is not None and (
        total <= cc._GLUON_AR_MAX_BYTES and n >= 256 and n & (n - 1) == 0
    ):
        return False
    if cc._AR2S_RMS_MIN_BYTES <= total <= cc._AR2S_RMS_MAX_TOTAL_BYTES and (
        ca.should_ar2s_rms(x, x, norm_w, cc._AR2S_RMS_MIN_BYTES)
    ):
        return False
    split = getattr(cc, "_use_lamport_split_ar_norm", None)
    if split is not None and split(comm, x, x, norm_w, n):
        return False
    lamport = ca.should_lamport_ar_rms(x, x, norm_w, n)
    gluon = cc._use_gluon_ar_norm(comm, x, x, norm_w, lamport)
    return bool(gluon and ca.should_k5_ar(x, x, norm_w))


def _fused(seam, x, x_scale, residual, norm_w):
    """The call's ``flydsl_oproj_ar`` arguments, or None for production's path."""
    from atom.models.minimax_m3.model import _fast_oproj_resolved

    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if not (
        0 < m <= MAX_ROWS
        and x.dtype == torch.float8_e4m3fn
        and x_scale is not None
        and residual.dtype == torch.bfloat16
        and tuple(residual.shape) == (m, HIDDEN)
        and residual.is_contiguous()
        and norm_w.numel() == HIDDEN
        and _fast_oproj_resolved(seam.o_proj)
        and _k5_gluon_route(residual, norm_w)
    ):
        return None
    k5 = _k5()
    if k5 is None or k5.max_rows < m:
        return None
    return x2, x_scale.reshape(m, -1), k5


def minimax_m3_attn_tail(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    seam_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """o_proj on the attention's MXFP8 output -> all-reduce + residual + Gemma
    RMSNorm of the seam ``seam_name``: (output, residual_out)."""
    seam = _SEAMS[seam_name]
    fused = _fused(seam, x, x_scale, residual, norm_weight)
    if fused is not None:
        from aiter.ops.flydsl.oproj_ar import flydsl_oproj_ar

        x2, s2, k5 = fused
        return flydsl_oproj_ar(
            x2,
            s2,
            seam.o_proj.weight,
            seam.o_proj.weight_scale,
            residual,
            norm_weight,
            float(eps),
            k5.inbox,
            k5.ctr,
            k5.slot,
            k5.rank,
            max_rows=k5.max_rows,
            maxit=_k5_maxit(),
        )
    # production: the layer's traced calls, in its order
    from atom.model_ops.layernorm import fused_allreduce_gemma_rms_norm
    from atom.models.minimax_m3.model import _o_proj_mxfp8

    h = _o_proj_mxfp8(seam.o_proj, x, x_scale)
    return fused_allreduce_gemma_rms_norm(h, residual, seam.norm)


def _minimax_m3_attn_tail_fake(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    seam_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(residual), torch.empty_like(residual)


direct_register_custom_op(
    op_name="minimax_m3_attn_tail",
    op_func=minimax_m3_attn_tail,
    mutates_args=(),
    fake_impl=_minimax_m3_attn_tail_fake,
    # o_proj reads row-major E4M3 rows and contiguous E8M0 scales (as its own op)
    tags=(torch.Tag.needs_fixed_stride_order,),
)
