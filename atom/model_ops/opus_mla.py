# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opus flat-row MLA decode for DCP + MTP (``ATOM_OPUS_MLA``, off by default).

The target model's causal MTP decode under round-robin decode context
parallelism runs aiter's ``OpusMlaFlatDecode`` instead of ``mla_decode_fwd``:
a host planner that packs each request's rows (``token * H + head``) into
128-row chunks and balances their KV over the XCDs, a stage-1 kernel and a
bf16 merge. Taken only with 96 gathered heads, ``float8_e4m3fn`` Q / KV
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


def update_plan(kv_indptr: np.ndarray, bs: int, q_len: int, cp_world_size: int) -> None:
    """Builds this step's plan from the host rank-local ``kv_indptr`` of the
    ``bs`` scheduled requests and queues its copy on the current stream."""
    _get_decoder().plan(kv_indptr[: bs + 1], q_len, HEADS, cp_world_size)


class PlanMarker:
    """``attn_metadata.opus_plan``: this step's / graph's decode runs the
    published plan."""

    __slots__ = ("bs", "q_len")

    def __init__(self, bs: int, q_len: int):
        self.bs, self.q_len = bs, q_len


def make_marker(bs: int, q_len: int) -> PlanMarker:
    _get_decoder()
    return PlanMarker(bs, q_len)


def capture_marker(bs: int, q_len: int) -> PlanMarker:
    """Marker for a graph capture's metadata, built outside the capture: also
    publishes an empty plan, so the eager warm-up forward before the capture
    runs no stale step plan over the synthetic capture KV."""
    _get_decoder().clear()
    return PlanMarker(bs, q_len)


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
        and cp_world_size > 1
        and max_q_len > 1
        and marker.q_len == max_q_len
        and page_size == 1
        and q.dim() == 3
        and q.shape[1] == HEADS
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
