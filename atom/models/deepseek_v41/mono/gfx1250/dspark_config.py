# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Deployments and graph shapes enabled for the gfx1250 DSpark backend."""

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.mono.runtime import deployment
from atom.mono.runtime.deployment import first_refusal
from atom.utils import envs

SUPPORTED_TP = (4,)
DRAFT_ROWS = 5
VERIFY_ROWS = DRAFT_ROWS + 1
MAX_REQUESTS = 8
MAX_ROWS = MAX_REQUESTS * VERIFY_ROWS
MAX_MOE_ROWS = 96
# The request counts routed to mono. A verify step of 3-7 requests (18-42
# rows) stays native: its mono builds, and padding it to 48 rows, both measured
# slower than native on CT16 (C05 width results).
MONO_REQUESTS = (1, 2, 8)
TARGET_WIDTHS = tuple(n * VERIFY_ROWS for n in MONO_REQUESTS)
DRAFT_WIDTHS = tuple(n * DRAFT_ROWS for n in MONO_REQUESTS)
ATTENTION_KEYS_MAX = 64 * 16
INDEXER_SHAPE = (ip.TOPK, ip.TOPK_BLOCKS, ip.BLOCK_ROWS, ip.HEADS, ip.DIM)


def _deployment_checks(atom_config):
    """The shared deployment contracts, specialized to this TP4 backend.

    Keep gfx950's common_refusals and architecture gate unchanged. Runtime
    plugin/weight-update/graph checks use the existing common predicates.
    """
    tp = atom_config.tensor_parallel_size
    dpa = deployment.dp_attention_ranks(atom_config)
    if dpa:
        parallel = [(False, f"DP attention {dpa} not in ()")]
    else:
        shard = deployment.shard_refusal(Dims, tp)
        parallel = [
            (shard is None, shard),
            (tp in SUPPORTED_TP, f"TP {tp} not in {SUPPORTED_TP}"),
            (not atom_config.enable_expert_parallel, "expert parallel"),
            (atom_config.parallel_config.data_parallel_size == 1, "DP > 1"),
        ]
    gfx = deployment._gfx()
    return [
        (gfx == "gfx1250", f"arch {gfx}"),
        *parallel,
        (atom_config.pipeline_parallel_size == 1, "PP > 1"),
        (atom_config.decode_context_parallel_size == 1, "decode CP"),
        (atom_config.prefill_context_parallel_size == 1, "prefill CP"),
        (not (atom_config.enable_tbo or atom_config.enable_tbo_decode), "TBO"),
        (not (deployment.is_vllm() or deployment.is_sglang()), "plugin mode"),
        (
            not deployment._takes_weight_updates(atom_config),
            "rollout weight updates (mono caches copies a sync cannot reach)",
        ),
        (
            getattr(getattr(atom_config, "compilation_config", None), "level", None)
            != deployment.CompilationLevel.DYNAMO_AS_IS,
            "DYNAMO_AS_IS compilation (fullgraph breaks on data_ptr)",
        ),
        (not deployment._uses_tr_hybrid(atom_config), "TR_HYBRID cudagraph"),
    ]


def config_refusal(atom_config) -> str | None:
    """Use the shared mono switch and decline deployments outside C05's scope."""
    if not envs.ATOM_MONO_ENABLE:
        return "switched off"
    spec = atom_config.speculative_config
    hf = atom_config.hf_config
    dspark = getattr(atom_config, "dspark", None)
    checks = _deployment_checks(atom_config)
    checks += [
        (atom_config.kv_cache_dtype == "bf16", "gfx1250 requires bf16 KV cache"),
        (atom_config.index_cache_dtype == "fp8", "gfx1250 requires fp8 index cache"),
        (spec is not None and spec.method == "dspark", "gfx1250 requires DSpark"),
        (
            spec is not None and spec.num_speculative_tokens == DRAFT_ROWS,
            "gfx1250 requires five draft tokens",
        ),
        (
            not getattr(dspark, "confidence_schedule", False)
            and not getattr(dspark, "ragged", False),
            "gfx1250 requires fixed-width DSpark",
        ),
        (atom_config.compilation_config.level == 0, "gfx1250 requires level 0"),
        (
            hf.sliding_window + hf.index_topk <= ATTENTION_KEYS_MAX,
            "too many attention keys",
        ),
        (
            (
                hf.index_topk,
                hf.candidate_topk_blocks,
                hf.candidate_block_size,
                hf.index_n_heads,
                hf.index_head_dim,
            )
            == INDEXER_SHAPE,
            "indexer shape",
        ),
        (envs.ATOM_DSV41_SIDE_STREAMS == 0, "ATOM_DSV41_SIDE_STREAMS"),
        (
            not (
                envs.ATOM_MONO_CHECK or envs.ATOM_MONO_DEBUG or envs.ATOM_MONO_TIMELINE
            ),
            "gfx1250 has no split-K2/debug/timeline build",
        ),
    ]
    return first_refusal(checks)
