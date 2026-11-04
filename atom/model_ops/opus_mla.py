# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opus flat-row MLA decode for MTP (``ATOM_OPUS_MLA``, off by default).

The target model's causal MTP decode under round-robin decode context
parallelism, and without it (DCP1, see below), runs aiter's
``OpusMlaFlatDecode`` instead of ``mla_decode_fwd``:
a host planner that packs each request's rows (``token * H + head``) into
128-row chunks and balances their KV over the XCDs, a stage-1 kernel and a
bf16 merge. Taken with 96 gathered heads (round-robin DCP), ``float8_e4m3fn`` Q / KV
(gfx950), ``page_size`` 1, query length above 1, a KV cache below 4 GiB and
running batches of up to ``ATOM_OPUS_MLA_MAX_BS`` requests, without decode
TBO. The draft model, non-causal blocks, query length 1, sparse attention,
and an aiter without ``OpusMlaFlatDecode`` (logged once) stay on
``mla_decode_fwd``. The kernel feeds the softmax
probabilities to the PV product as fp8, so its output sits further from an
fp32 reference than ``mla_decode_fwd``'s; its LSE matches as closely.

Three hooks:

1. ``prepare_decode`` builds the step's plan on the host (:func:`update_plan`),
   outside the CUDA graph, and marks the metadata (:func:`make_marker`).
2. ``build_for_cudagraph_capture`` marks every captured batch size that will
   use the kernel (:func:`capture_marker`), with the same test
   (``AiterMLAMetadataBuilder._uses_opus_mla``), so a graph replays the kernel
   it was captured with.
3. ``_forward_decode`` picks the decode function (:func:`wanted`).

The dispatch is decided on the running batch (the captured graph's size) and
the plan is built over the scheduled requests: padded rows get no work. The
plan, partial and output-merge buffers keep fixed addresses, so a captured
graph replays with every later plan. One decoder per process; every decode
call runs on the forward's compute stream, in layer order, after the step's
plan copy.

DCP1 (``--decode-context-parallel-size 1``): the same switch also takes the MTP
target decode without decode context parallelism. Each rank decodes its own
query heads, padded to the MLA minimum (12 or 16, ``DCP1_HEADS``), over its
whole KV, with no ``g_kv_indptr``. The plan is built for those heads with
``cp_world_size`` 1, which switches aiter's planner and merge to their no-CP
form. Every other condition above is the same, and the plan marker carries its
head count so a call only runs the plan built for its own heads.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from atom.utils import envs

logger = logging.getLogger("atom")

try:
    from aiter.ops.opus.mla_flat import OpusMlaFlatDecode
except ImportError:  # aiter without the opus flat-row decode
    OpusMlaFlatDecode = None

HEADS = 96
# DCP1 (no decode context parallelism): a rank's own query heads, padded to the MLA
# minimum: 16, or 12 for a build that lowers the minimum (the kernel runs both).
DCP1_HEADS = (12, 16)
# Widest request a plan takes: ``OpusMlaFlatDecode``'s default 8 chunks of 128
# rows (``q_len * heads``).
MAX_ROWS = 128 * 8
QK_DIM = 576
V_DIM = 512
# The kernel addresses the KV cache through one 32-bit buffer descriptor.
KV_BYTES_LIMIT = 1 << 32

_decoder = None
_logged: set = set()


def _log_once(key, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.info(msg, *args)


def enabled() -> bool:
    if not envs.ATOM_OPUS_MLA:
        return False
    if OpusMlaFlatDecode is None:
        _log_once(
            "no_api",
            "opus MLA decode unavailable: aiter lacks OpusMlaFlatDecode; "
            "decode stays on mla_decode_fwd",
        )
        return False
    return True


def dcp1_shape_ok(num_heads: int, q_len: int) -> bool:
    """Whether a DCP1 decode of ``num_heads`` heads at query length ``q_len``
    is one the kernel takes (``DCP1_HEADS``, at most ``MAX_ROWS`` rows)."""
    return num_heads in DCP1_HEADS and 1 < q_len and q_len * num_heads <= MAX_ROWS


def uses_opus_for_batch(batch_size: int) -> bool:
    """Whether a running / captured-graph batch of this size uses the kernel."""
    return 1 <= batch_size <= envs.ATOM_OPUS_MLA_MAX_BS


def _get_decoder():
    global _decoder
    if _decoder is None:
        _decoder = OpusMlaFlatDecode(max_batch=envs.ATOM_OPUS_MLA_MAX_BS)
        logger.info(
            "opus MLA decode: ready for batches of up to %d",
            envs.ATOM_OPUS_MLA_MAX_BS,
        )
    return _decoder


def update_plan(
    kv_indptr: np.ndarray,
    bs: int,
    q_len: int,
    cp_world_size: int,
    num_heads: int = HEADS,
) -> None:
    """Builds this step's plan from the host rank-local ``kv_indptr`` of the
    ``bs`` scheduled requests and queues its copy on the current stream.
    ``num_heads``: the gathered ``HEADS`` under DCP, or a DCP1 rank's own."""
    _get_decoder().plan(kv_indptr[: bs + 1], q_len, num_heads, cp_world_size)


class PlanMarker:
    """``attn_metadata.opus_plan``: this step's / graph's decode runs the
    published plan, built for ``heads`` query heads."""

    __slots__ = ("bs", "heads", "q_len")

    def __init__(self, bs: int, q_len: int, heads: int = HEADS):
        self.bs, self.q_len, self.heads = bs, q_len, heads


def make_marker(bs: int, q_len: int, heads: int = HEADS) -> PlanMarker:
    _get_decoder()
    return PlanMarker(bs, q_len, heads)


def capture_marker(bs: int, q_len: int, heads: int = HEADS) -> PlanMarker:
    """Marker for a graph capture's metadata, built outside the capture: also
    publishes an empty plan, so the eager warm-up forward before the capture
    runs no stale step plan over the synthetic capture KV."""
    _get_decoder().clear()
    return PlanMarker(bs, q_len, heads)


def wanted(
    attn_metadata,
    *,
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    q_scale,
    kv_scale,
    causal: bool,
    cp_world_size: int,
    max_q_len: int,
    page_size: int,
) -> bool:
    """Whether this ``mla_decode_fwd`` call runs :func:`decode` instead."""
    marker = getattr(attn_metadata, "opus_plan", None)
    return (
        marker is not None
        and envs.ATOM_OPUS_MLA
        and causal
        and (
            # round-robin DCP: the gathered heads
            (cp_world_size > 1 and marker.heads == HEADS)
            # DCP1: a rank's own heads
            or (cp_world_size == 1 and marker.heads in DCP1_HEADS)
        )
        and max_q_len > 1
        and marker.q_len == max_q_len
        and page_size == 1
        and q.dim() == 3
        and q.shape[1] == marker.heads
        and q.shape[2] == QK_DIM
        and q.is_contiguous()
        and q.dtype == torch.float8_e4m3fn
        and kv_buffer.dtype == torch.float8_e4m3fn
        and kv_buffer.is_contiguous()
        and kv_buffer.numel() * kv_buffer.element_size() < KV_BYTES_LIMIT
        and o.dtype == torch.bfloat16
        and o.dim() == 3
        and o.shape[2] == V_DIM
        and o.is_contiguous()
        and q_scale is not None
        and kv_scale is not None
    )


def decode(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    **kw,
):
    """``mla_decode_fwd`` drop-in for the calls :func:`wanted` accepts: writes
    the bf16 output into ``o`` and returns ``(o, lse)``."""
    lse = _get_decoder()(
        q,
        kv_buffer,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kw["sm_scale"],
        kw["q_scale"],
        kw["kv_scale"],
        kw["g_kv_indptr"],
        kw["cp_world_size"],
        kw["cp_rank"],
    )
    return o, lse
