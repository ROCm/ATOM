# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The registered :class:`ModelSpec` entries.

Separate from ``model_registry`` so the registry carries no bridge imports.
Registration order is resolution order, most specific first: DSv4, DeepSeek-V3,
any other config with routed experts, dense catch-all.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

from atomrl.engine.training.bridges import dsv3, dsv4, gpt
from atomrl.engine.training.model_registry import (
    MODEL_REGISTRY,
    ModelCaps,
    ModelSpec,
    hf_num_experts,
    resolve_head_dim,
)


def _effective_num_experts(hf: Mapping[str, Any], ec: Mapping[str, Any]) -> int:
    """Expert count after the engine_config override, which wins over HF."""
    return int(ec.get("num_experts") or hf_num_experts(hf) or 0)


def _export_dsv4(engine, **_):
    """Gathers like Qwen3-MoE but renames to the DSv4 *checkpoint* names, which
    the rollout feeds straight to vLLM's ``load_weights``."""
    named = itertools.chain(
        engine._full_megatron_named_params_moe(),
        engine._router_bias_buffers(),
    )
    return dsv4.megatron_to_dsv4_native(named)


def _export_dsv3(engine, **_):
    """MLA export. The router bias is a buffer, so it is chained in explicitly:
    noaux_tc top-k reads it, and a rollout without it picks other experts.

    Via the engine, not the bridge's raw walker: the engine helper renumbers
    stage-local layers to global ones and broadcasts across PP. At PP=1 the two
    agree; above it the raw walker yields colliding local names and omits every
    layer outside the publishing rank's stage."""
    named = itertools.chain(
        engine._full_megatron_named_params_moe(),
        engine._router_bias_buffers(),
    )
    return dsv3.megatron_to_hf(named)


def _export_moe(engine, stack_experts: bool = False, **_):
    return gpt.megatron_to_hf(
        engine._full_megatron_named_params_moe(stack_experts),
        engine._dims,
    )


def _export_dense(engine, **_):
    return gpt.megatron_to_hf(engine._full_megatron_named_params(), engine._dims)


def _dense_dims(hf: Mapping[str, Any]) -> gpt.GPTDims:
    return gpt.GPTDims(
        num_layers=hf["num_hidden_layers"],
        hidden=hf["hidden_size"],
        num_heads=hf["num_attention_heads"],
        num_kv_groups=hf["num_key_value_heads"],
        head_dim=resolve_head_dim(hf),
        ffn=hf["intermediate_size"],
        vocab=hf["vocab_size"],
    )


# --- DeepSeek-V4 -------------------------------------------------------------
# 4-D hyper-connection residual stream, heterogeneous attention per layer, hash
# routing on the first layers: nothing the generic config path can describe. Its
# block-quantized FP8 is also unreadable by the HF bridge, hence build_dims=None.
DSV4 = MODEL_REGISTRY.register(
    ModelSpec(
        name="deepseek_v4",
        detect=lambda hf, ec: dsv4.is_dsv4(hf),
        caps=ModelCaps(
            supports_hf_bridge=False,
            supports_dynamic_batch=False,
            builds_own_config=True,
            # Its topology check must run before every forward, and the generic
            # non-pipelined shortcut cannot host it.
            requires_pipeline_forward=True,
            packed_stream_is_single_sequence=True,
        ),
        build_dims=None,
        # Lazy on purpose: these functions ship with the DSv4 branch, not main,
        # so binding them at import time would break this module for every model.
        # build_config matches what Megatron's parser produces from
        # deepseek-v4-flash.sh, the config the DSv4 references were measured on.
        build_config=lambda *a, **kw: dsv4.build_dsv4_config(*a, **kw),
        build_layer_spec=lambda *a, **kw: dsv4.build_dsv4_spec(*a, **kw),
        sequence_alignment=lambda tfcfg: dsv4.sequence_alignment(tfcfg),
        # Refuses the topologies DSv4 gets wrong; see _dsv4_check_topology.
        pre_forward_check=lambda engine: engine._dsv4_check_topology(),
        export_weights=_export_dsv4,
    )
)


# --- DeepSeek-V3 family (V3 / V3.1 / R1, and the Kimi K2 line) ---------------
# Before gpt_moe: a DSv3 config declares routed experts, so the generic entry
# would claim it and build a plain TransformerConfig with fused QKV -- wrong for
# MLA. Detects on ``architectures``, not MLA fields, which DSv4 also has.
DSV3 = MODEL_REGISTRY.register(
    ModelSpec(
        name="deepseek_v3",
        detect=lambda hf, ec: dsv3.is_dsv3(hf),
        caps=ModelCaps(
            # Construction needs MLATransformerConfig, not TransformerConfig.
            builds_own_config=True,
            supports_hf_bridge=True,
        ),
        build_dims=dsv3.build_dsv3_dims,
        build_config=dsv3.build_dsv3_config,
        # MLA is a parameter of the stock TE builder, so no custom layer spec.
        build_layer_spec=None,
        routing_defaults={"moe_router_pre_softmax": False},
        export_weights=_export_dsv3,
    )
)


# --- Qwen3-MoE (and any config declaring routed experts) ---------------------
# HF ``norm_topk_prob=True`` (softmax -> top-k -> renormalize) is identical to
# Megatron's ``moe_router_pre_softmax=False``, since the full-softmax denominator
# cancels under renormalization. ``True`` would leave gate weights summing to <1
# and diverge from vLLM -- a large rollout/train log-prob gap.
QWEN3_MOE = MODEL_REGISTRY.register(
    ModelSpec(
        name="gpt_moe",
        detect=lambda hf, ec: _effective_num_experts(hf, ec) > 1,
        build_dims=gpt.build_moe_dims,
        routing_defaults={"moe_router_pre_softmax": False},
        export_weights=_export_moe,
    )
)


# --- Dense catch-all ---------------------------------------------------------
# GQA + SwiGLU. Last, so it only sees what nothing else claimed.
QWEN3_DENSE = MODEL_REGISTRY.register(
    ModelSpec(
        name="gpt_dense",
        detect=lambda hf, ec: True,
        build_dims=_dense_dims,
        export_weights=_export_dense,
    )
)
