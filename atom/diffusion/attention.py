# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Packed varlen attention backends. A real trade, not a fallback ladder.

    asm     aiter ASM v3 varlen. 124.0 TFLOP/s on gfx942, matching the tuned
            fixed-length kernel (123.9). The default.
    triton  aiter Triton varlen. 99.0 TFLOP/s, but the only backend that
            reproduces the sglang reference bit-for-bit. Pin in parity tests.
    sdpa    Segment-wise SDPA. CPU fallback and numerics anchor.

They agree to ~1e-5 cosine per call (ordinary bf16 spread), which over 50 steps
compounds into a different but equally valid sample. No claim that any one is
more accurate; a run comparing pixels against sglang must select ``triton``.
"""

import itertools
import os
from enum import Enum

import torch
from torch import nn

ATTENTION_BACKEND_ENV = "ATOM_DIFFUSION_ATTN_BACKEND"


class AttentionBackend(str, Enum):
    """Packed varlen attention implementation."""

    ASM = "asm"
    TRITON = "triton"
    SDPA = "sdpa"


DEFAULT_ATTENTION_BACKEND = AttentionBackend.ASM


def resolve_attention_backend(
    backend: "AttentionBackend | str | None" = None,
) -> AttentionBackend:
    """Resolve an explicit choice, else ``$ATOM_DIFFUSION_ATTN_BACKEND``, else ASM."""
    if backend is None:
        backend = os.environ.get(ATTENTION_BACKEND_ENV) or None
    if backend is None:
        return DEFAULT_ATTENTION_BACKEND
    if isinstance(backend, AttentionBackend):
        return backend
    try:
        return AttentionBackend(str(backend).strip().lower())
    except ValueError as exc:
        names = ", ".join(b.value for b in AttentionBackend)
        raise ValueError(
            f"unknown attention backend {backend!r}; expected one of {names}"
        ) from exc


def _sdpa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    outs = []
    for start, stop in itertools.pairwise(cu_seqlens.tolist()):
        if stop <= start:
            continue
        qs, ks, vs = (t[start:stop].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        o = nn.functional.scaled_dot_product_attention(qs, ks, vs, scale=softmax_scale)
        outs.append(o.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)


def packed_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
    backend: "AttentionBackend | str | None" = None,
    pad_from: int | None = None,
) -> torch.Tensor:
    """Non-causal varlen attention over a packed multi-segment sequence.

    ``q``/``k``/``v`` are ``[total_tokens, heads, head_dim]``; segments are
    delimited by ``cu_seqlens``.

    ``pad_from`` marks where trailing alignment padding begins; those rows are
    dropped from the call and zeroed in the result. Not a micro-optimisation:
    the ASM grid is ``(heads, num_segments, ceil(max_seqlen / 256))``, sized
    from ``max_seqlen`` rather than per segment, so a 24-row padding segment
    gets a full plane of 2,072 workgroups of which one has work. Dropping it
    halves the grid: 93.0 -> 80.9 ms per layer at H3's shapes.

    Bit-exact: the padding already sits in its own segment, so no real token
    attends to it either way.
    """
    backend = resolve_attention_backend(backend)

    # Dispatch on device, not on whether aiter imports: aiter imports fine on a
    # CPU-only run and then fails inside the kernel with "q must be on CUDA".
    # (HIP tensors report device.type == "cuda", which is the ROCm path here.)
    if q.device.type == "cpu" or backend is AttentionBackend.SDPA:
        return _sdpa_varlen(q, k, v, cu_seqlens=cu_seqlens, softmax_scale=softmax_scale)

    if backend is AttentionBackend.TRITON:
        from aiter.ops.triton.attention.mha import (
            flash_attn_varlen_func as _varlen,
        )
    else:
        try:
            from aiter import flash_attn_varlen_func as _varlen
        except ImportError:
            return _sdpa_varlen(
                q, k, v, cu_seqlens=cu_seqlens, softmax_scale=softmax_scale
            )

    total = int(q.shape[0])
    if (
        pad_from is not None
        and 0 < pad_from < total
        and backend is AttentionBackend.ASM
    ):
        # Prefix views of contiguous tensors stay contiguous, and aiter writes
        # through `out=`, so nothing is copied.
        out = torch.empty_like(q)
        out[pad_from:].zero_()
        _varlen(
            q=q[:pad_from].contiguous(),
            k=k[:pad_from].contiguous(),
            v=v[:pad_from].contiguous(),
            cu_seqlens_q=cu_seqlens[:-1].contiguous(),
            cu_seqlens_k=cu_seqlens[:-1].contiguous(),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=False,
            out=out[:pad_from],
        )
        return out

    out = _varlen(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=softmax_scale,
        causal=False,
    )
    return out[0] if isinstance(out, tuple) else out
