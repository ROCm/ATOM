# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One attention layer's fused QKV: the GEMM and the Ulysses exchange run as
an AITER gemm_fusion plan, and the impl continues after the stages the plan
covered. Every per-call choice depends only on shapes, dtypes and the
broadcast batch, so all SP ranks take the same path.
"""

import logging

import torch

from atom.model_ops.sp_attention import (
    attend_exchanged,
    gather_heads,
    ulysses_attention,
)
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

_KV_DTYPES = {"fp8": torch.float8_e4m3fn, "bf16": torch.bfloat16}


def rows2d(t):
    """``[max_pos, 1, 1, rot/2]`` rotary caches as ``[max_pos, rot/2]``."""
    return t if t.dim() == 2 else t.reshape(t.shape[0], -1)


def cache_layout_reason(impl, state, *, page, with_index):
    """Why the bound caches are not the SHUFFLE layout the plan writes through
    raw pointers, or None. K ``[N, kv, hd/x, page, x]``, V ``[N, kv, page/x,
    hd, x]``; FP8 adds float32 ``[N, kv, page]`` scales; the indexer keys are
    FP8 ``[N, 128, 128]``."""
    want = _KV_DTYPES.get(impl.kv_cache_dtype)
    if want is None:
        return "kv_layout"
    x = 16 // want.itemsize
    nkv, hd = impl.num_kv_heads, impl.head_dim
    k = state["k_cache"]
    v = state.get("v_cache_shuffle", state["v_cache"])
    if (
        k.dim() != 5
        or v.dim() != 5
        or k.dtype != want
        or v.dtype != want
        or tuple(k.shape[1:]) != (nkv, hd // x, page, x)
        or tuple(v.shape) != (k.shape[0], nkv, page // x, hd, x)
        or k.stride(4) != 1
        or k.stride(3) != x
        or v.stride(4) != 1
        or v.stride(3) != x
    ):
        return "kv_layout"
    scales = (state["k_scale"], state["v_scale"])
    if impl.kv_cache_dtype == "fp8":
        for s in scales:
            if (
                s is None
                or s.dtype != torch.float32
                or tuple(s.shape) != (k.shape[0], nkv, page)
                or s.stride(2) != 1
            ):
                return "kv_layout"
    elif any(s is not None for s in scales):
        return "kv_layout"
    if with_index:
        ic = impl.index_cache
        if (
            ic is None
            or impl.index_cache_dtype != "fp8"
            or impl.index_head_dim != 128
            or ic.dtype != torch.float8_e4m3fn
            or tuple(ic.shape[1:]) != (128, 128)
            or not ic.is_contiguous()
        ):
            return "index_layout"
    return None


def static_weight_reason(lin):
    """Why the plan cannot take the QKV weight as is (a 16x16-preshuffled FP8
    ``[N, K]`` with at least N scale rows), or None."""
    w = lin.weight
    if getattr(w, "is_shuffled", False) is not True:
        return "weight_layout"
    n, k = lin.output_size, lin.input_size
    if w.dim() != 2 or w.shape[1] != k:
        return "weight_shape"
    # A preshuffled weight padded on N keeps its first N rows contiguous.
    if w.shape[0] < n or (w.shape[0] != n and not lin.is_output_padded):
        return "weight_shape"
    scale = getattr(lin, "weight_scale", None)
    if scale is None or scale.shape[0] < n:
        return "weight_shape"
    return None


def native_reason(*, dummy_run, x, x_scale, input_size, positions=None, slot=None):
    """Why this call must take the native path, or None. ``positions`` and
    ``slot`` are given for a plan that runs the prologue, whose kernel reads
    one position and slot per real token."""
    if dummy_run:
        return "dummy_run"
    if x_scale is None or x.dtype != torch.float8_e4m3fn:
        return "unquantized_input"
    if (
        x_scale.dtype != torch.float32
        or x_scale.numel() != x.shape[0]
        or not x_scale.is_contiguous()
    ):
        return "xscale_layout"
    if x.dim() != 2 or x.shape[1] != input_size or x.stride(1) != 1:
        return "x_layout"
    if positions is None:
        return None
    real = positions.shape[-1]
    if positions.numel() != real or slot.numel() < real:
        return "meta_len"
    for t in (positions, slot):
        if t.dtype != torch.int64 or not t.is_contiguous() or t.numel() == 0:
            return "meta_layout"
    return None


class QkvSite:
    """The fused QKV of one attention layer."""

    def __init__(self, name, attn, linear, spec, mode, runtime, out_width):
        self.name = name
        self.attn = attn
        self.linear = linear
        self.spec = spec
        self.mode = mode
        self.runtime = runtime
        self.out_width = out_width
        # Bucket M -> the plan prepared for it.
        self.plans = {}
        # One site advances the device step counter once per forward.
        self.ticks = False
        self._norm_f32 = {}
        self._logged = set()

    def _f32(self, key, t):
        cached = self._norm_f32.get(key)
        if cached is None or cached[0] is not t:
            cached = (t, t.float().contiguous())
            self._norm_f32[key] = cached
        return cached[1]

    def build_meta(self, state, positions, slot):
        impl = self.attn.impl
        rot = impl.rotary_emb
        meta = {
            "positions": positions,
            "slot_mapping": slot,
            "k_cache": state["k_cache"],
            "v_cache": state.get("v_cache_shuffle", state["v_cache"]),
            "k_scale": state["k_scale"],
            "v_scale": state["v_scale"],
            "cos_cache": rows2d(rot.cos_cache),
            "sin_cache": rows2d(rot.sin_cache),
            "q_norm_w": self._f32("q", impl.q_norm.weight),
            "k_norm_w": self._f32("k", impl.k_norm.weight),
        }
        if self.mode == "indexer":
            meta["index_cache"] = impl.index_cache
            meta["index_q_norm_w"] = self._f32("iq", impl.index_q_norm.weight)
            meta["index_k_norm_w"] = self._f32("ik", impl.index_k_norm.weight)
        return meta

    def _log_once(self, path, reason, m):
        if (path, reason) not in self._logged:
            self._logged.add((path, reason))
            logger.info(
                "gemm_fusion %s: %s (%s) first at local M %d",
                self.name,
                path,
                reason,
                m,
            )

    def run(self, x, x_scale, positions, scale):
        """The layer's per-token FP8 attention output; its scales go to
        ``scale``."""
        rt = self.runtime
        if self.ticks and rt.ready:
            rt.step.add_(1)
        attn, lin = self.attn, self.linear
        m = x.shape[0]
        tokens = rt.world * m
        fwd_ctx = get_forward_context()
        plan = self.plans.get(rt.bucket(self.spec, m)) if rt.ready else None
        state = slot = None
        if plan is None:
            reason = "no_plan"
        else:
            if plan.n_stages == 2:
                state = attn.impl.gemm_fusion_prologue_state(fwd_ctx)
                slot = state["slot_mapping"]
            reason = native_reason(
                dummy_run=fwd_ctx.context.is_dummy_run,
                x=x,
                x_scale=x_scale,
                input_size=lin.input_size,
                positions=positions if state is not None else None,
                slot=slot,
            )
        call_id = None
        if reason is None:
            call_id = rt.call_id(self, tokens, plan)
            if call_id is None:
                reason = "call_id_refused"
        if reason is not None:
            self._log_once("native", reason, m)
            qkv = lin(x, x_scale=x_scale)
            out, rows = ulysses_attention(attn, None, None, None, positions, None, qkv)
            return gather_heads(out, rows, attn.sp_head_exchange, scale)

        self._log_once("fused", f"{plan.vid} n_stages={plan.n_stages}", m)
        stream = torch.cuda.current_stream()
        if plan.n_stages == 2:
            outs = rt.outs[self.mode]
            plan.run(
                x=x,
                x_scale=x_scale,
                w=lin.weight,
                w_scale=lin.weight_scale,
                outs=outs,
                meta=self.build_meta(state, positions, slot),
                res=rt.res,
                stream=stream,
                call_id=call_id,
            )
            real = positions.shape[-1]
            index_q = outs.get("index_q")
            out = attn.impl.forward_after_prologue(
                outs["q"][:real],
                outs["k"][:real] if "k" in outs else None,
                outs["v"][:real] if "v" in outs else None,
                state["k_cache"],
                state["v_cache"],
                state["k_scale"],
                state["v_scale"],
                fwd_ctx,
                index_q=None if index_q is None else index_q[:real],
            )
        else:
            plan.run(
                x=x,
                x_scale=x_scale,
                w=lin.weight,
                w_scale=lin.weight_scale,
                res=rt.res,
                stream=stream,
                call_id=call_id,
            )
            # A view of the receive window, consumed on this stream before
            # this rank's next large launch.
            packed = plan.recv_view(rt.res, m)
            out = attend_exchanged(attn, packed, positions)
        return gather_heads(out, tokens, attn.sp_head_exchange, scale)
