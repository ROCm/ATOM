# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in K2 bindings: standalone halves and the shared merged runner.

The two TP reductions use RCCL between real launch boundaries. This adapter
preserves native compressor/index metadata and delayed state for the next K1.
With a shared Gfx1250DecodeContext, the merged kernel replaces both launches
and both per-layer RCCL reductions using the gfx950 peer/mailbox lifecycle.
GPU validation must precede use in serving; the default path remains disabled.
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from aiter.dist.parallel_state import get_tp_group
from aiter.ops.flydsl.kernels.tensor_shim import check_e8m0

from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_ffn import (
    ffn_gfx1250,
    ffn_scratch_specs,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_layer_post import (
    _tensor_spec,
    attention_projection_gfx1250,
    attention_scratch_specs,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_moe import (
    EXPERTS,
    HIDDEN,
    moe_widths,
)


@triton.jit
def _guard_k2(
    state,
    codes,
    out,
    residual,
    pre,
    post,
    comb,
    STATE_WORDS: tl.constexpr,
    ROWS: tl.constexpr,
    LAYER: tl.constexpr,
    FFN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Keep the first K1/K2 error, and make following rank collectives safe."""
    code = tl.load(state + STATE_WORDS - 1)
    if code != 0:
        # Same-stream K1 guard already wrote codes[LAYER]; retain its error.
        seen = tl.load(codes + LAYER)
        tl.store(codes + LAYER, tl.where(seen == 0, code, seen))
        for start in range(0, ROWS * 5120, BLOCK):
            at = start + tl.arange(0, BLOCK)
            tl.store(out + at, 0, at < ROWS * 5120)
        if FFN:
            for start in range(0, ROWS * 4 * 5120, BLOCK):
                at = start + tl.arange(0, BLOCK)
                tl.store(residual + at, 0, at < ROWS * 4 * 5120)
            at = tl.arange(0, BLOCK)
            tl.store(pre + at, 0, at < ROWS * 4)
            tl.store(post + at, 0, at < ROWS * 4)
            tl.store(comb + at, 0, at < ROWS * 16)


class LocalK2:
    """Bind native parameters in place and retain one buffer set per row width."""

    def __init__(self, block):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[gfx1250 K2 runner] warm layer before capture")
        group = get_tp_group()
        if group.world_size != 2:
            raise ValueError("[gfx1250 K2 runner] runtime bring-up requires TP2")
        if block.ffn_norm.eps != 1e-20:
            raise ValueError("[gfx1250 K2 runner] unsupported FFN norm")
        # The native model owns a shared alt_stream even with side streams
        # disabled. K2 consumes weights and runs its own single-stream body;
        # it never invokes the native FFN forward or forks that stream.
        qm = block.ffn.experts.quant_method
        if (
            not getattr(qm, "is_gfx1250", False)
            or not getattr(qm, "fp8_activations", False)
            or getattr(qm.moe, "use_ep", True)
            or getattr(qm, "use_triton", False)
            or getattr(qm, "use_triton_decode", False)
            or getattr(qm, "use_triton_ep", False)
            or block.ffn.experts.num_fused_shared_experts != 0
            or block.ffn.routed_scaling_factor != 1.5
        ):
            raise ValueError(
                "[gfx1250 K2 runner] requires native TP A8W4 text MoE layout"
            )
        self.interleaved = bool(qm.is_guinterleave)
        self.block, self.group, self.buffers = block, group, {}
        self.tp = group.world_size
        self.context = None
        self.metadata_buffers = {}
        self.merged_bindings = {}
        params = dict(block.named_parameters())
        names = (
            "hc_ffn_fn",
            "hc_ffn_scale",
            "hc_ffn_base",
            "ffn_norm.weight",
            "ffn.gate.weight",
            "ffn.gate.e_score_correction_bias",
            "ffn.experts.w13_weight",
            "ffn.experts.w13_weight_scale",
            "ffn.experts.w2_weight",
            "ffn.experts.w2_weight_scale",
            "ffn.shared_experts.gate_up_proj.weight",
            "ffn.shared_experts.gate_up_proj.weight_scale",
            "ffn.shared_experts.w2.weight",
            "ffn.shared_experts.w2.weight_scale",
            "attn.wo_a.weight",
            "attn.wo_a.weight_scale",
            "attn.wo_b.weight",
            "attn.wo_b.weight_scale",
        )
        self.weights = {name: params[name] for name in names}
        real, padded = moe_widths(self.tp)
        # Validate byte geometry at bind time. Neither this flag nor a shape
        # proves numerical accuracy; native weight preparation owns layout.
        for name, shape, dtype in [
            (
                "ffn.experts.w13_weight",
                (EXPERTS, 2 * padded, HIDDEN // 2),
                torch.float4_e2m1fn_x2,
            ),
            (
                "ffn.experts.w2_weight",
                (EXPERTS, HIDDEN, padded // 2),
                torch.float4_e2m1fn_x2,
            ),
            (
                "ffn.shared_experts.gate_up_proj.weight",
                (2 * real, HIDDEN),
                torch.float8_e4m3fn,
            ),
            ("ffn.shared_experts.w2.weight", (HIDDEN, real), torch.float8_e4m3fn),
        ]:
            _tensor_spec(self.weights[name], shape, dtype)
        for name, shape in [
            ("ffn.experts.w13_weight_scale", (EXPERTS, 2 * padded // 32, HIDDEN)),
            ("ffn.experts.w2_weight_scale", (EXPERTS, HIDDEN // 32, padded)),
        ]:
            check_e8m0(self.weights[name], name, "gfx1250 K2 runner", shape)

    def _buffers(self, rows, device):
        if rows in self.buffers:
            return self.buffers[rows]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[gfx1250 K2 runner] warm row width before capture")

        def allocate(specs):
            return {
                name: (
                    torch.zeros(shape, dtype=dtype, device=device)
                    if name == "state"
                    else torch.empty(shape, dtype=dtype, device=device)
                )
                for name, (shape, dtype) in specs.items()
            }

        a = allocate(attention_scratch_specs(rows, self.block.attn.heads))
        f = allocate(ffn_scratch_specs(rows, self.tp))
        outputs = allocate(
            {
                "attention": ((rows, HIDDEN), torch.bfloat16),
                "ffn": ((rows, HIDDEN), torch.bfloat16),
                "residual": ((rows, 4, HIDDEN), torch.bfloat16),
                "pre": ((rows, 4), torch.float32),
                "post": ((rows, 4), torch.float32),
                "comb": ((rows, 4, 4), torch.float32),
                "normed": ((rows, HIDDEN), torch.bfloat16),
            }
        )
        b = {"attn": a, "ffn": f, "outputs": outputs}
        self.buffers[rows] = b
        return b

    def forward(
        self, query, residual, pre, post, comb, cache, step, rope, codes, image_mask
    ):
        if image_mask is not None:
            raise ValueError("[gfx1250 K2 runner] image routing requires native path")
        rows = query.shape[0]
        if self.context is not None:
            return self._merged_forward(
                query, residual, pre, post, comb, cache, step, rope, codes
            )
        b = self._buffers(rows, query.device)
        a, f, o, w = b["attn"], b["ffn"], b["outputs"], self.weights
        attn, block = self.block.attn, self.block
        prefix, pptr, extend, eptr = cache.attention_indices(attn.spec, step)
        attention_projection_gfx1250(
            query,
            cache.pool,
            cache.ring_view(attn.spec.layer_id),
            prefix,
            pptr,
            extend,
            eptr,
            attn.attn_sink,
            cache.rope_positions(step).reshape(rows),
            rope.cos_cache,
            rope.sin_cache,
            w["attn.wo_a.weight"],
            w["attn.wo_a.weight_scale"],
            w["attn.wo_b.weight"],
            w["attn.wo_b.weight_scale"],
            o["attention"],
            a,
            softmax_scale=attn.softmax_scale,
        )
        _guard_k2[(1,)](
            a["state"],
            codes,
            o["attention"],
            residual,
            pre,
            post,
            comb,
            STATE_WORDS=a["state"].numel(),
            ROWS=rows,
            LAYER=attn.spec.layer_id,
            FFN=False,
            BLOCK=256,
        )
        # A real launch boundary: no RCCL call occurs inside a FlyDSL kernel.
        dist.all_reduce(o["attention"], group=self.group.device_group)
        ffn_gfx1250(
            residual,
            o["attention"],
            pre,
            post,
            comb,
            w["hc_ffn_fn"],
            w["hc_ffn_scale"],
            w["hc_ffn_base"],
            w["ffn_norm.weight"],
            w["ffn.gate.weight"],
            w["ffn.gate.e_score_correction_bias"],
            w["ffn.experts.w13_weight"],
            w["ffn.experts.w13_weight_scale"],
            w["ffn.experts.w2_weight"],
            w["ffn.experts.w2_weight_scale"],
            w["ffn.shared_experts.gate_up_proj.weight"],
            w["ffn.shared_experts.gate_up_proj.weight_scale"],
            w["ffn.shared_experts.w2.weight"],
            w["ffn.shared_experts.w2.weight_scale"],
            o["residual"],
            o["pre"],
            o["post"],
            o["comb"],
            o["normed"],
            o["ffn"],
            f,
            tp=self.tp,
            interleaved=self.interleaved,
        )
        _guard_k2[(1,)](
            f["state"],
            codes,
            o["ffn"],
            o["residual"],
            o["pre"],
            o["post"],
            o["comb"],
            STATE_WORDS=f["state"].numel(),
            ROWS=rows,
            LAYER=attn.spec.layer_id,
            FFN=True,
            BLOCK=256,
        )
        dist.all_reduce(o["ffn"], group=self.group.device_group)
        return block.finish_ffn(
            o["ffn"].unsqueeze(0),
            o["residual"].unsqueeze(0),
            o["pre"].unsqueeze(0),
            o["post"].unsqueeze(0),
            o["comb"].unsqueeze(0),
        )

    def _merged_forward(
        self, query, residual, pre, post, comb, cache, step, rope, codes
    ):
        rows = query.shape[0]
        if rows not in self.merged_bindings:
            raise RuntimeError("[gfx1250 K2 runner] prepare the real binding before the TP chain")
        self.merged_bindings[rows]()
        return self._merged_finish(rows, codes)

    def prepare_binding(self, query, residual, pre, post, comb, cache, rope, codes):
        """Bind and capture one layer before the first mailbox step launch."""
        from atom.models.deepseek_v41.mono.gfx1250.gfx1250_context import ring_source
        from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_merged_post import (
            layer_post_gfx1250,
        )

        rows = query.shape[0]
        ctx, attn, w = self.context, self.block.attn, self.weights
        b = ctx.rows[rows]
        o = b["outputs"]
        ix = b["index"]
        spec = attn.spec
        a = {
            "query": query,
            "pool": cache.pool,
            "ring": ring_source(cache, attn.spec.layer_id),
            "prefix": ix["prefix"] if spec.ratio else ix["prefix"][:0],
            "prefix_indptr": ix["pptr"],
            "extend": ix["extend"],
            "extend_indptr": ix["eptr"],
            "sink": attn.attn_sink,
            "positions": ix["positions"],
            "cos": ctx.rope_tables[id(rope)][0],
            "sin": ctx.rope_tables[id(rope)][1],
            "woa": w["attn.wo_a.weight"],
            "woa_s": w["attn.wo_a.weight_scale"],
            "wob": w["attn.wo_b.weight"],
            "wob_s": w["attn.wo_b.weight_scale"],
            "out": o["attention"],
        }
        f = {
            "residual": residual,
            "attention": o["attention"],
            "pre": pre,
            "post": post,
            "comb": comb,
            "fn": w["hc_ffn_fn"],
            "scale": w["hc_ffn_scale"],
            "bias": w["hc_ffn_base"],
            "nw": w["ffn_norm.weight"],
            "gate_w": w["ffn.gate.weight"],
            "gate_bias": w["ffn.gate.e_score_correction_bias"],
            "w13": w["ffn.experts.w13_weight"],
            "w13_s": w["ffn.experts.w13_weight_scale"],
            "w2": w["ffn.experts.w2_weight"],
            "w2_s": w["ffn.experts.w2_weight_scale"],
            "sgu": w["ffn.shared_experts.gate_up_proj.weight"],
            "sgu_s": w["ffn.shared_experts.gate_up_proj.weight_scale"],
            "sw2": w["ffn.shared_experts.w2.weight"],
            "sw2_s": w["ffn.shared_experts.w2.weight_scale"],
            "residual_out": o["residual"],
            "pre_out": o["pre"],
            "post_out": o["post"],
            "comb_out": o["comb"],
            "normed": o["normed"],
            "out": o["ffn"],
        }
        binding = layer_post_gfx1250(
            a,
            f,
            b["scratch"],
            ctx.peers,
            ctx.metadata[attn.spec.layer_id],
            ctx.mailboxes._state,
            codes,
            cache._gfx1250_diagnostic_sticky,
            tp=self.tp,
            softmax_scale=attn.softmax_scale,
            interleaved=self.interleaved,
            argument_tables=ctx.argument_tables,
            _prepare_only=True,
            index={
                "i_query": ctx.k1_buffers[rows]["iq8"],
                "i_weights": ctx.k1_buffers[rows]["swiw"],
                "i_plane": cache.index_units[spec.kv_owner].values if spec.ratio else cache.backing[:0],
                "i_visible": ix["visible"][spec.ratio] if spec.ratio else ix["pptr"][:rows],
                "i_table": ix["table"],
                "i_keymeta": cache._gfx1250_k1_meta[rows][1],
                "i_config": ctx.index_config[spec.layer_id],
                "i_selection": ix["selection"][spec.topk_owner] if spec.ratio else ix["prefix"].view(rows, 512),
                "i_candidates": ix["candidates"],
            },
            index_bound_max=ctx.index_bound_max,
            timeline=ctx.timeline[attn.spec.layer_id] if ctx.timeline is not None else None,
        )
        self.merged_bindings[rows] = binding
        binding.runtime.prepare_graph((binding.table,))
        self._merged_guard(rows, codes, prepare=True)
        return binding

    def _merged_guard(self, rows, codes, *, prepare=False):
        b = self.context.rows[rows]
        o = b["outputs"]
        args = (b["scratch"]["state"], codes, o["ffn"], o["residual"],
                o["pre"], o["post"], o["comb"])
        options = dict(STATE_WORDS=b["scratch"]["state"].numel(), ROWS=rows,
                       LAYER=self.block.attn.spec.layer_id, FFN=True, BLOCK=256)
        if prepare:
            _guard_k2.warmup(*args, grid=(1,), **options)
        else:
            _guard_k2[(1,)](*args, **options)

    def _merged_finish(self, rows, codes):
        o = self.context.rows[rows]["outputs"]
        self._merged_guard(rows, codes)
        return self.block.finish_ffn(
            o["ffn"].unsqueeze(0),
            o["residual"].unsqueeze(0),
            o["pre"].unsqueeze(0),
            o["post"].unsqueeze(0),
            o["comb"].unsqueeze(0),
        )
