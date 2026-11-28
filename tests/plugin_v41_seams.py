# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The DeepSeek-V4.1 seam specs ATOM's CPU tests declare (``conftest.py``).

ATOM declares no model's seams; plugin_poc's V4.1 model declares these through
entry point group ``atom.seam_specs``. The host and recipe tests exercise the
routing with this copy (the call sites they name are ATOM's shared ops).
"""

from __future__ import annotations

from atom.plugin_host import SeamSpec

_V41 = "atom/models/deepseek_v41_plugin/"


def _seam(
    name, contract, kind, form, inputs, kwargs, returns, key, graph, target, **kw
):
    return SeamSpec(
        name, contract, kind, form, inputs, kwargs, returns, key, graph, target, **kw
    )


# The module path each seam's call site is in (recipe reachability checks).
_SCOPES = {
    "v41.block.ffn": "layers.*",
    "v41.attn.out": "layers.*.attn",
    "v41.attn.qk_norm": "layers.*.attn",
    "v41.attn.project_out": "layers.*.attn",
    "v41.attn.out_full": "layers.*.attn",
    "v41.attn.attend_out": "layers.*.attn",
    "v41.attn.attend": "layers.*.attn",
    "v41.attn.index_topk": "layers.*.attn.indexer",
    "v41.moe.shared_act": "layers.*.ffn.shared_experts",
    "v41.moe.shared_expert": "layers.*.ffn",
    "v41.moe.topk": "layers.*.ffn.experts",
}


# Alternative spans of the attention tail (not nested): the wider one makes the
# narrower unreachable when its owner serves.
_COVERS = {
    "v41.attn.project_out": ("v41.attn.out",),
    "v41.attn.out_full": ("v41.attn.out",),
    "v41.attn.attend_out": ("v41.attn.attend",),
}
_EXCLUDES: dict[str, tuple[str, ...]] = {}


def specs() -> list[SeamSpec]:
    from dataclasses import replace

    out = [
        _seam(
            "v41.block.ffn",
            1,
            "region",
            "bind",
            ("x",),
            (),
            ("y",),
            ("x",),
            "opaque_op",
            _V41 + "model.py",
            doc="the FFN sublayer, Block.ffn(normed): bf16 [T,H] -> bf16 [T,H]"
            "; flags: layer_id, module",
        ),
        _seam(
            "v41.attn.out",
            1,
            "region",
            "bind",
            ("x_fp8", "x_scale", "wo_a_w", "wo_a_ws"),
            ("dtype",),
            ("y", "y_scale"),
            ("x_fp8", "x_scale"),
            "capture_safe",
            _V41 + "attention.py",
            doc="wo_a: wo_b's input pair (e4m3 + e8m0 g32) or (bf16, None)",
        ),
        _seam(
            "v41.attn.qk_norm",
            1,
            "op",
            "bind",
            ("q", "k", "q_weight", "k_weight", "eps_q"),
            ("eps_k", "amax_floor"),
            ("q_fp8", "q_scale", "k_out"),
            ("q", "k", "eps_q"),
            "capture_safe",
            "atom/model_ops/layernorm.py",
            doc="dual RMSNorm, q to MXFP8: the kernel DualRMSNormMXFP8 calls",
        ),
        _seam(
            "v41.moe.shared_act",
            1,
            "region",
            "bind",
            ("combined", "weights"),
            (),
            ("act", "act_scale"),
            ("combined", "weights"),
            "opaque_op",
            _V41 + "v4.py",
            parent="v41.block.ffn",
            doc="shared expert clamp-act(+quant): w2's input pair or (bf16, None)",
        ),
        _seam(
            "v41.moe.shared_expert",
            1,
            "region",
            "bind",
            ("x", "weights", "w13_w", "w13_ws"),
            (),
            ("act", "act_scale"),
            ("x", "weights"),
            "capture_safe",
            _V41 + "v4.py",
            parent="v41.block.ffn",
            doc="shared expert gate_up + clamp-act(+quant): w2's input pair or "
            "(bf16, None); flags: swiglu_limit, native_fused_act, gate_up_n, "
            "gate_up_k, gate_up_group_rows",
        ),
        _seam(
            "v41.attn.project_out",
            1,
            "region",
            "bind",
            ("x_fp8", "x_scale", "wo_a_w", "wo_a_ws", "wo_b_w", "wo_b_ws"),
            ("dtype",),
            ("out",),
            ("x_fp8", "x_scale"),
            "capture_safe",
            _V41 + "attention.py",
            boundary="collective",
            doc="wo_a + wo_b + TP all-reduce: (e4m3 [M,G,K], e8m0 g32) -> the "
            "TP-reduced bf16 [M, hidden] on every rank",
        ),
        _seam(
            "v41.attn.out_full",
            1,
            "region",
            "bind",
            ("o", "positions", "cos", "sin", "wo_a_w", "wo_a_ws"),
            ("dtype",),
            ("y", "y_scale"),
            ("o",),
            "capture_safe",
            _V41 + "attention.py",
            doc="inverse RoPE + g32 quant + wo_a from bf16 attention rows: "
            "wo_b's input pair or (bf16, None)",
        ),
        _seam(
            "v41.attn.attend_out",
            1,
            "region",
            "bind",
            (
                "query",
                "pool",
                "kv_indices_prefix",
                "kv_indptr_prefix",
                "kv",
                "kv_indices_extend",
                "kv_indptr_extend",
                "attn_sink",
                "softmax_scale",
                "out",
                "positions",
                "cos_cache",
                "sin_cache",
                "x_fp8",
                "x_scale",
            ),
            ("num_groups",),
            ("x_fp8", "x_scale"),
            (
                "query",
                "pool",
                "kv",
                "attn_sink",
                "softmax_scale",
                "out",
                "positions",
                "cos_cache",
                "sin_cache",
                "x_fp8",
                "x_scale",
            ),
            "capture_safe",
            _V41 + "attention.py",
            doc="FP8-KV decode attention + inverse RoPE + MXFP8 g32, written "
            "into x_fp8/x_scale (wo_a's input pair); out is the BF16 scratch",
        ),
        _seam(
            "v41.moe.topk",
            1,
            "op",
            "call",
            (
                "topk_weights",
                "topk_ids",
                "router_logits",
                "e_score_correction_bias",
                "renormalize",
                "route_scale",
            ),
            ("score_func", "num_valid_rows"),
            (),
            (
                "router_logits",
                "topk_weights",
                "topk_ids",
                "e_score_correction_bias",
                "renormalize",
                "route_scale",
            ),
            "capture_safe",
            "atom/model_ops/moe.py",
            parent="v41.block.ffn",
            doc="topk_gating in FusedMoE.select_experts, in place",
        ),
        _seam(
            "v41.op.quant_fp8_g32",
            1,
            "op",
            "call",
            ("x",),
            ("dequantize",),
            ("x_fp8", "x_scale"),
            ("x",),
            "capture_safe",
            "atom/model_ops/blockscale.py",
            doc="quantize_fp8: e4m3 + e8m0 [..., K/32]; dequantize=True is QAT",
        ),
        _seam(
            "v41.attn.index_topk",
            1,
            "op",
            "call",
            ("scores", "group", "row_ends", "out", "count", "stride0", "stride1"),
            ("k", "stable", "row_starts", "plane_width"),
            (),
            ("scores", "group", "row_ends", "out", "count", "stride0", "stride1"),
            "capture_safe",
            "atom/model_ops/indexer/score.py",
            doc="packed top_k_per_row_decode, in place into out",
        ),
        _seam(
            "v41.attn.attend",
            1,
            "op",
            "call",
            (
                "q",
                "unified_kv",
                "kv_indices_prefix",
                "kv_indptr_prefix",
                "kv",
                "kv_indices_extend",
                "kv_indptr_extend",
                "attn_sink",
                "softmax_scale",
            ),
            (
                "out",
                "unified_kv_rope",
                "q_packed",
                "q_rope",
                "k_packed",
                "k_rope",
                "prefix",
            ),
            ("out",),
            ("q", "unified_kv", "kv", "attn_sink", "softmax_scale"),
            "capture_safe",
            "atom/model_ops/v4_kernels/paged_2src.py",
            doc="sparse_attn_v4_paged_2src; out= is written and returned",
        ),
        _seam(
            "v41.draft.ctx",
            1,
            "region",
            "bind",
            ("taps", "rows", "running_tokens"),
            (),
            ("aux", "aux_scale"),
            ("taps", "rows", "running_tokens"),
            "capture_safe",
            _V41 + "dspark.py",
            doc="DSpark context taps -> main_proj's input pair or (bf16, None)",
        ),
        _seam(
            "v41.draft.markov",
            1,
            "region",
            "call",
            (
                "base_logits",
                "anchor_ids",
                "markov_w1",
                "markov_w2",
                "out_ids",
                "markov_embeds",
                "vocab_start",
                "tp_group",
            ),
            ("use_custom",),
            (),
            (
                "base_logits",
                "anchor_ids",
                "out_ids",
                "markov_embeds",
                "vocab_start",
            ),
            "capture_safe",
            _V41 + "dspark.py",
            boundary="collective",
            doc="DSpark sharded Markov T-step argmax block, in place into "
            "out_ids and markov_embeds",
        ),
        _seam(
            "v41.spec.verify_sample",
            1,
            "region",
            "call",
            (
                "logits",
                "temperatures",
                "target_logits_indices",
                "bonus_logits_indices",
                "cu_num_draft_tokens",
            ),
            ("all_greedy", "top_ks", "top_ps", "needs_independent_noise"),
            ("bonus_token_ids", "target_token_ids", "target_logits"),
            (
                "logits",
                "temperatures",
                "target_logits_indices",
                "bonus_logits_indices",
                "cu_num_draft_tokens",
            ),
            "capture_safe",
            "atom/model_engine/model_runner.py",
            doc="speculative verification + bonus sampling of a verify step",
        ),
        _seam(
            "v41.mono.install",
            1,
            "region",
            "call",
            ("model", "atom_config", "drafter"),
            ("metadata_builder",),
            ("model",),
            (),
            "eager",
            "atom/model_engine/model_runner.py",
            boot_only=True,
            doc="mono_decode_installers' V4.1 installer -> the model to serve",
        ),
    ]
    return [
        replace(
            s,
            scope=_SCOPES.get(s.name),
            covers=_COVERS.get(s.name, ()),
            excludes=_EXCLUDES.get(s.name, ()),
        )
        for s in out
    ]
