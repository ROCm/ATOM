# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Optional gfx1250 AITER backend for ATOM's KDA ReplaySSM entry point."""

from functools import lru_cache

import torch

from atom.utils import envs


@lru_cache(maxsize=1)
def _get_aiter_replayssm():
    # Older AITER installations remain usable with ATOM's native backend.
    try:
        from aiter.ops.triton.attention.kda_replayssm import (
            replayssm_sigmoid_gating_delta_rule,
        )
    except (ImportError, AttributeError):
        return None
    return (
        replayssm_sigmoid_gating_delta_rule
        if callable(replayssm_sigmoid_gating_delta_rule)
        else None
    )


def _supported(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    ckpt,
    buf_k,
    buf_u,
    buf_g,
    write_pos,
    slot_idx,
    cu_seqlens,
    max_query_len,
    o,
):
    # Inspect metadata only: this path must also work during graph capture.
    if not q.is_cuda or q.ndim != 4 or q.shape[0] != 1:
        return False
    if getattr(torch.cuda.get_device_properties(q.device), "gcnArchName", "").split(
        ":"
    )[0] not in (
        "gfx1250",
        "gfx1250-strict",
    ):
        return False
    if q.device.index != torch.cuda.current_device():
        return False
    _, tokens, H, K = q.shape
    if v.ndim != 4 or v.shape[:2] != (1, tokens):
        return False
    HV, V = v.shape[2:]
    if K != 128 or V != 128 or H <= 0 or HV <= 0 or HV % H:
        return False
    if not isinstance(max_query_len, int) or max_query_len < 1:
        return False
    if (
        k.shape != q.shape
        or a.shape != (1, tokens, HV, K)
        or b.shape != (1, tokens, HV)
    ):
        return False
    floating = (torch.bfloat16, torch.float16, torch.float32)
    if any(x.dtype not in floating for x in (q, k, v, a, b, buf_k, buf_u)):
        return False
    if any(x.dtype != torch.float32 for x in (A_log, dt_bias, ckpt, buf_g)):
        return False
    if ckpt.ndim != 4 or ckpt.shape[1:] != (HV, V, K):
        return False
    slots = ckpt.shape[0]
    if slots == 0 or ckpt.stride()[1:] != (V * K, K, 1) or buf_k.ndim != 4:
        return False
    cap = buf_k.shape[2]
    if cap < 2 * max_query_len or cap - max_query_len > 64:
        return False
    if any(
        x.shape != (slots, HV, cap, K) or x.stride(-1) != 1
        for x in (buf_k, buf_u, buf_g)
    ):
        return False
    if A_log.numel() != HV or dt_bias.numel() != HV * K:
        return False
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        return False
    if (
        cu_seqlens.ndim != 1
        or slot_idx.ndim != 1
        or cu_seqlens.numel() != slot_idx.numel() + 1
        or write_pos.shape != (slots,)
    ):
        return False
    if any(
        x.dtype != torch.int32 or not x.is_contiguous()
        for x in (write_pos, slot_idx, cu_seqlens)
    ):
        return False
    tensors = (
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        ckpt,
        buf_k,
        buf_u,
        buf_g,
        write_pos,
        slot_idx,
        cu_seqlens,
    )
    if any(x.device != q.device for x in tensors):
        return False
    # ATOM accepts a three-dimensional output and returns its batch view.
    return o is None or (
        o.shape == (tokens, HV, V)
        and o.device == q.device
        and o.dtype in floating
        and o.is_contiguous()
        and all(
            o.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            for x in tensors
        )
    )


def try_aiter_replayssm(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    ckpt,
    buf_k,
    buf_u,
    buf_g,
    write_pos,
    slot_idx,
    cu_seqlens,
    max_query_len,
    o=None,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    lower_bound=None,
):
    """Return None before launch if unsupported; never retry mutated state."""
    if not envs.ATOM_USE_AITER_REPLAYSSM_KDA or not _supported(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        ckpt,
        buf_k,
        buf_u,
        buf_g,
        write_pos,
        slot_idx,
        cu_seqlens,
        max_query_len,
        o,
    ):
        return None
    backend = _get_aiter_replayssm()
    if backend is None:
        return None
    # Fork checkpoint seeding belongs to the caller and has already run.
    # Exceptions must propagate: a failed launch may have modified records.
    out = backend(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        ckpt=ckpt,
        buf_k=buf_k,
        buf_u=buf_u,
        buf_g=buf_g,
        write_pos=write_pos,
        slot_idx=slot_idx,
        cu_seqlens=cu_seqlens,
        max_query_len=max_query_len,
        o=o,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        lower_bound=lower_bound,
    )
    if out is None:
        raise RuntimeError("AITER ReplaySSM returned no output after launch")
    return out
