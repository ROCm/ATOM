# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP8 activation producers for the MiniMax-M3.1 DSpark draft's linears.

Every draft linear is native MXFP8 (E4M3 values, an E8M0 scale per 32
columns), and each one quantized its bf16 input in a pass of its own
(`quantize_fp8`) right after the kernel that wrote that input: the Gemma
RMSNorm in front of qkv_proj and gate_up_proj, the SwiGLU-OAI in front of
down_proj. These kernels write the quantized pair directly. The value
quantized is the bf16 the unfused kernel would have written, and the groups
go through the same `fp8_groups`, so the pair matches norm-then-quantize up to
the norm's own fp32 reduction order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from atom.model_ops.blockscale_kernels.quantization import fp8_groups

_E8M0 = torch.float8_e8m0fnu


@triton.jit
def _gemma_rmsnorm_mxfp8_kernel(
    x_ptr,
    res_ptr,  # residual in, or x_ptr when HAS_RES is False
    w_ptr,
    q_ptr,  # [M, K] fp8 out
    s_ptr,  # [M, K / 32] uint8 E8M0 codes out
    res_out_ptr,  # [M, K] bf16 out: x + residual (HAS_RES)
    stride_x,
    stride_r,
    eps,
    K: tl.constexpr,
    G: tl.constexpr,  # K / 32
    GP: tl.constexpr,  # G rounded up to a power of two
    HAS_RES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    groups = tl.arange(0, GP)
    offs = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    mask = offs < K
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        # the norm reads the fp32 sum; the residual carried on is its bf16
        # (aiter's fused_qk_rmsnorm_group_quant, which GemmaRMSNorm runs)
        r = tl.load(res_ptr + row * stride_r + offs, mask=mask, other=0.0)
        x = x + r.to(tl.float32)
        tl.store(res_out_ptr + row * K + offs, x.to(tl.bfloat16), mask=mask)
    var = tl.sum(tl.sum(x * x, 1), 0) / K
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x * tl.rsqrt(var + eps) * (1.0 + w)).to(tl.bfloat16).to(tl.float32)
    q, code, _ = fp8_groups(y)
    tl.store(q_ptr + row * K + offs, q, mask=mask)
    tl.store(s_ptr + row * G + groups, code.to(tl.uint8), mask=groups < G)


class _DraftAr:
    comm = None
    tried = False
    zeros = None  # [MAX_ROWS, 6144] bf16 zero residual (the block embedding's add)


DRAFT_AR_MAX_ROWS = 2048


def draft_ar_comm():
    """ATOM_M3_DRAFT_AK_AR_NORM: the draft's TP4 all-reduce + add + Gemma norm (+ MXFP8) communicator, built and
    warmed up once per rank (a collective on the TP CPU group), or None (flag off, TP 1, or not four real ranks)."""
    from atom.utils import envs

    if not envs.ATOM_M3_DRAFT_AK_AR_NORM:
        return None
    if _DraftAr.tried:
        return _DraftAr.comm
    if torch.cuda.is_current_stream_capturing():
        return None  # built eagerly only (collective + allocations)
    _DraftAr.tried = True
    import logging

    import torch.distributed as dist
    from aiter.dist.parallel_state import get_tp_group

    tp = get_tp_group()
    world = dist.get_world_size(tp.cpu_group)
    if world != 4:
        logging.getLogger("atom").warning(
            "ATOM_M3_DRAFT_AK_AR_NORM: needs 4 real TP ranks (have %d); keeping AITER's all-reduce", world)
        return None
    try:
        from df_draft_ar import DraftArNormComm
    except ImportError as e:
        raise ImportError("ATOM_M3_DRAFT_AK_AR_NORM=1 needs amd-kernels and the df_draft_ar overlay on PYTHONPATH") from e
    dev = torch.device("cuda", torch.cuda.current_device())
    comm = DraftArNormComm(tp.rank_in_group, tp.cpu_group, dev)
    comm.warm_up_atom(torch.zeros(6144, dtype=torch.bfloat16, device=dev), 1e-6, rows=(8, 64, 448))
    _DraftAr.zeros = torch.zeros(DRAFT_AR_MAX_ROWS, 6144, dtype=torch.bfloat16, device=dev)
    torch.cuda.synchronize(dev)
    _DraftAr.comm = comm
    logging.getLogger("atom").info("ATOM_M3_DRAFT_AK_AR_NORM: draft AR + Gemma norm + MXFP8 in one launch")
    return comm


def draft_ar_capture():
    """Around the CUDA-graph captures (every rank alike): registers the captured draft calls' inputs."""
    import contextlib

    comm = draft_ar_comm()
    return comm.capture() if comm is not None else contextlib.nullcontext()


def draft_ar_takes(x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor) -> bool:
    comm = _DraftAr.comm
    if comm is None or x.dim() != 2 or x.shape[1] != 6144 or not 0 < x.shape[0] <= DRAFT_AR_MAX_ROWS:
        return False
    if weight.dtype != torch.bfloat16 or not weight.is_contiguous() or x.dtype != torch.bfloat16:
        return False
    if residual is not None and (residual.stride(1) != 1 or residual.stride(0) % 8):
        return False
    return not torch.cuda.is_current_stream_capturing() or comm.capturing


def draft_ar_norm(x, residual, weight, eps, quant: bool):
    """all_reduce(x) [+ residual] -> Gemma norm, as ``gemma_rmsnorm_mxfp8`` (quant) or ``GemmaRMSNorm`` (bf16) would
    return it after AITER's all-reduce: ``(q, scales, residual out)`` or ``(out, residual out)``. ``residual`` None =
    no add (the residual out is the reduced x). Falls back to AITER's all-reduce + the unfused kernels.
    From 32 rows the communicator runs its owned-residual form: the residual out then holds only this rank's quarter
    of the rows, which is all the next call of the same chain reads (the draft block chains 11 calls of one row
    count, all taken or all not: the conditions are the same for every call of a pass)."""
    if draft_ar_takes(x, residual, weight):
        comm = _DraftAr.comm
        x = x.contiguous()
        r = _DraftAr.zeros[: x.shape[0]] if residual is None else residual
        if quant:
            q, s, res = comm.ar_add_rmsnorm_mxfp8_atom(x, r, weight, eps)
            return q, s.view(_E8M0), res
        return comm.ar_add_rmsnorm_atom(x, r, weight, eps)
    from atom.model_ops.communication_op import tensor_model_parallel_all_reduce

    x = tensor_model_parallel_all_reduce(x)
    if quant:
        q, s, res = gemma_rmsnorm_mxfp8(x, weight, eps, residual)
        return q, s, (x if residual is None else res)
    raise RuntimeError("draft_ar_norm bf16 fallback is the caller's (GemmaRMSNorm)")


def gemma_rmsnorm_mxfp8(
    x: torch.Tensor,  # [M, K] bf16
    weight: torch.Tensor,  # [K]
    eps: float,
    residual: torch.Tensor | None = None,
):
    """``GemmaRMSNorm(x [+ residual])`` quantized to MXFP8: ``(values [M, K]
    E4M3, scales [M, K / 32] E8M0, residual out or None)``."""
    from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE

    M, K = x.shape
    q = torch.empty(M, K, dtype=FP8_DTYPE, device=x.device)
    s = torch.empty(M, K // 32, dtype=torch.uint8, device=x.device)
    res_out = torch.empty_like(x) if residual is not None else None
    if M:
        _gemma_rmsnorm_mxfp8_kernel[(M,)](
            x, x if residual is None else residual, weight, q, s,
            x if res_out is None else res_out,
            x.stride(0), 0 if residual is None else residual.stride(0), float(eps),
            K=K, G=K // 32, GP=triton.next_power_of_2(K // 32),
            HAS_RES=residual is not None, num_warps=8,
        )  # fmt: skip
    return q, s.view(_E8M0), res_out


@triton.jit
def _swiglu_oai_mxfp8_kernel(
    gu_ptr,  # [M, 2I] bf16, [gate | up]
    q_ptr,  # [M, I] fp8
    s_ptr,  # [M, I / 32] uint8
    stride_g,
    alpha,
    beta,
    limit,
    I: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    groups = tile * 32 + tl.arange(0, 32)
    cols = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    mask = cols < I
    gate = tl.load(gu_ptr + row * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gu_ptr + row * stride_g + I + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    y = (gate * tl.sigmoid(alpha * gate) * (up + beta)).to(tl.bfloat16).to(tl.float32)
    q, code, _ = fp8_groups(y)
    tl.store(q_ptr + row * I + cols, q, mask=mask)
    tl.store(s_ptr + row * (I // 32) + groups, code.to(tl.uint8), mask=groups < I // 32)


def swiglu_oai_mxfp8(gate_up: torch.Tensor, alpha: float, beta: float, limit):
    """`swiglu_oai_split` quantized to MXFP8: ``(values [M, I], scales
    [M, I / 32])``."""
    from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE

    M, two_i = gate_up.shape
    inter = two_i // 2
    q = torch.empty(M, inter, dtype=FP8_DTYPE, device=gate_up.device)
    s = torch.empty(M, inter // 32, dtype=torch.uint8, device=gate_up.device)
    if M:
        _swiglu_oai_mxfp8_kernel[(M, triton.cdiv(inter, 1024))](
            gate_up, q, s, gate_up.stride(0), float(alpha), float(beta),
            0.0 if limit is None else float(limit),
            I=inter, HAS_LIMIT=limit is not None, num_warps=4,
        )  # fmt: skip
    return q, s.view(_E8M0)
