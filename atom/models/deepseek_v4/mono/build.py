# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: prepare-time plan/argument extraction for ATOM, 2026-10-22.
"""Prepare-time argument assembly shared by runner and prepared variants."""

from atom.models.deepseek_v4.mono.config import LAUNCH_PHASES, shape_dims
from atom.mono.runtime.consensus import MonoUnsupported


def launch_plan(rows: int, *, tokens_per_seq: int = 1, attention_mode="causal") -> dict:
    """Exact-row topology; deployment admission and residency are separate."""
    if rows == 1:
        plan = {
            "launch_shape": "one",
            "qkv_batch_limit": None,
            "qkv_samples_per_pass": None,
            "qkv_norm_handoff": False,
        }
    elif rows in (*range(2, 9), 14):
        samples_per_pass = 1
        if rows == tokens_per_seq == 7:
            samples_per_pass = 7
        elif rows == 4 and tokens_per_seq == 1 and attention_mode == "causal":
            samples_per_pass = 4
        elif tokens_per_seq == 4 and rows in (4, 8) and attention_mode == "causal":
            samples_per_pass = rows
        elif rows == 8 and tokens_per_seq == 1:
            samples_per_pass = 2
        plan = {
            # S7/q7 combined attention can spill; split that complete request.
            "launch_shape": (
                "three" if rows == 14 or rows == tokens_per_seq == 7 else "two"
            ),
            "qkv_batch_limit": 1,
            # Ordinary S4/q1 and causal S4/S8 q4 consume a full tile in one
            # weight pass. Geometry selection does not grant DSpark admission.
            "qkv_samples_per_pass": samples_per_pass,
            "qkv_norm_handoff": True,
        }
    else:
        raise MonoUnsupported(
            "the decode builder supports exact 1..8-row and candidate q7 S14 buckets"
        )
    if attention_mode == "draft_block":
        # Draft forward dependencies require all producers before attention.
        plan = dict(
            plan,
            launch_shape="three",
            qkv_samples_per_pass=(
                1 if rows == tokens_per_seq == 7 else plan["qkv_samples_per_pass"]
            ),
        )
    return plan


def kernel_kwargs(
    cfg,
    *,
    tokens_per_seq,
    timeline,
    moe_mode,
    index_topk_refresh,
    qkv_batch_limit=None,
    qkv_samples_per_pass=None,
    qkv_norm_handoff=False,
    attention_mode="causal",
):
    """Identical geometry and numerical contract for every launch phase."""
    return dict(
        scale=cfg.softmax_scale,
        timeline=timeline,
        moe_mode=moe_mode,
        tokens_per_seq=tokens_per_seq,
        compressor_ring_rows=cfg.state_ring_rows(tokens_per_seq),
        swiglu_limit=cfg.swiglu_limit,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
        bf16_kv_roundtrip=cfg.bf16_kv_roundtrip,
        quant_query_fp8=cfg.quant_query_fp8,
        window_rows=cfg.cache_rows,
        index_topk_refresh=index_topk_refresh,
        qkv_batch_limit=qkv_batch_limit,
        qkv_samples_per_pass=qkv_samples_per_pass,
        qkv_norm_handoff=qkv_norm_handoff,
        attention_mode=attention_mode,
        **shape_dims(cfg),
    )


def _build_launches(cfg, rows, npes, *, launch_shape, **kwargs):
    """Build the planned phases; callers own compilation, cache and TP admission."""
    from atom.models.deepseek_v4.mono.kernels.layer import build_dsv4_kernel

    return tuple(
        build_dsv4_kernel(rows, cfg.heads, npes, cfg.window, phase=phase, **kwargs)
        for phase in LAUNCH_PHASES[launch_shape]
    )
