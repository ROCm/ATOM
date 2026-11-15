# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The one Qwen3.8-Flash-Next deployment Mono serves, and why any other is refused.

Mono serves the full 48-layer Quark MXFP4 checkpoint at TP1 with MTP3 verify on MI355X: the target's 36 GDN and 12
QSA decoder layers as one program a layer for verify steps of 1-4 requests x ``SPEC`` rows, and the BF16 MTP
drafter's layer. ``config_refusal`` names the first condition a deployment misses. It is asked at the PLE table's
construction (``ple.make_host_embedding``) and at install (``dispatch.install_mono_decode``), both before any state
is written; a refused deployment runs Native for the whole model.

The GDN programs run the MTP verify ATOM picks: ReplaySSM (its default under MTP: one checkpoint a request plus
``(k, u, g)`` records) or the classic verify (``ATOM_ENABLE_REPLAYSSM=0``: the state snapshotted per verify row). The
kernels assume an fp32 GDN state, FP8 K / V rows with fp32 scales on 64-token pages, BF16 index caches and BF16 draft
caches.
"""

import torch

from atom.models.qwen38.mono.kernels.geometry import REPLAY_MAX
from atom.mono.plan.shard import Shard
from atom.mono.runtime.deployment import (
    common_refusals,
    first_refusal,
    resident_grid_refusals,
)
from atom.utils import envs

SPEC = 4  # MTP3 verify rows per request (and the drafter's first step)
BATCHES = (1, 2, 3, 4)  # requests a served target or draft step carries
# draft rows on the batched W4 head (1 row: the global-certificate head; others Native)
HEAD_ROWS = (2, 4)
KV_BLOCK = 64
SUPPORTED_TP = (1,)
# KV connectors served beside the programs: LMCache CPU offload's transfers are bounded gather / scatter kernels and
# host copies on its own streams, fenced by events, that never wait on the model
OFFLOAD_CONNECTORS = ("lmcache_offload",)

LAYERS = 48
HIDDEN = 2560
HC = 4
HC_RANK = 320
EXPERTS = 512
TOPK = 10
INTERMEDIATE = 640
QSA_HEADS = 24
QSA_KV_HEADS = 2
QSA_HEAD_DIM = 256
QSA_INDEX = (4, 1, 128, 2048, 4)  # heads, KV heads, head dim, budget, compress ratio
GDN = (16, 48, 128, 128, 4)  # key heads, value heads, key / value head dim, conv width
LAYER_TYPES = tuple(
    "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
    for layer in range(LAYERS)
)


def _shape_checks(hf) -> list[tuple[bool, str]]:
    layer_types = tuple(
        "full_attention" if kind == "qwen_sparse_attention" else kind
        for kind in getattr(hf, "layer_types", ())
    )
    ple = tuple(getattr(hf, "ple_layer_ids", ()) or ())
    gdn = tuple(
        getattr(hf, k, None)
        for k in (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        )
    )
    index = tuple(
        getattr(hf, k, None)
        for k in (
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        )
    )
    qsa = tuple(
        getattr(hf, k, None)
        for k in ("num_attention_heads", "num_key_value_heads", "head_dim")
    )
    return [
        (getattr(hf, "num_hidden_layers", None) == LAYERS, "not 48 layers"),
        (getattr(hf, "hidden_size", None) == HIDDEN, "hidden size"),
        (getattr(hf, "hc_count", None) == HC, "HC count"),
        (getattr(hf, "hc_lowrank", None) == HC_RANK, "HC rank"),
        (getattr(hf, "num_experts", None) == EXPERTS, "expert count"),
        (getattr(hf, "num_experts_per_tok", None) == TOPK, "top-k"),
        (
            getattr(hf, "moe_intermediate_size", None) == INTERMEDIATE,
            "expert intermediate",
        ),
        (
            getattr(hf, "shared_expert_intermediate_size", None) == INTERMEDIATE,
            "shared expert intermediate",
        ),
        (layer_types == LAYER_TYPES, "layer type table"),
        (gdn == GDN, "GDN shape"),
        (qsa == (QSA_HEADS, QSA_KV_HEADS, QSA_HEAD_DIM), "QSA shape"),
        (index == QSA_INDEX, "QSA index shape"),
        (ple == (2,), f"PLE layers {ple}"),
        (getattr(hf, "ple_conv_kernel_size", None) == 4, "PLE convolution"),
        (getattr(hf, "ngram_size", None) == 3, "PLE ngram size"),
        (getattr(hf, "rms_norm_eps", None) == 1e-6, "RMS epsilon"),
        (getattr(hf, "hidden_act", None) == "silu", "hidden activation"),
        (getattr(hf, "mamba_ssm_dtype", None) == "float32", "GDN state dtype"),
        (getattr(hf, "norm_topk_prob", True) is True, "router renormalization"),
        (getattr(hf, "output_gate_type", None) == "sigmoid", "QSA output gate"),
    ]


def _quark_mxfp4(atom_config) -> bool:
    """The Quark MXFP4 checkpoint family (MXFP4 experts; attention FP8 as stored or quantized online as ATOM does)."""
    quant = getattr(atom_config, "quant_config", None)
    hf_quant = getattr(quant, "hf_quant_config", None) or {}
    if getattr(quant, "quant_method", None) != "quark":
        return False
    if hf_quant.get("global_quant_config", {}).get("weight", {}).get("dtype") != "fp4":
        return False
    layer = hf_quant.get("layer_quant_config") or {}
    fp8 = {
        k
        for k, v in layer.items()
        if ((v or {}).get("weight") or {}).get("dtype") == "fp8_e4m3"
    }
    acts = ((hf_quant.get("global_quant_config") or {}).get("input_tensors") or {}).get(
        "dtype"
    )
    return {"*linear_attn.*", "*self_attn.*"} <= fp8 or (not fp8 and acts == "fp4")


def _cache_formats(atom_config) -> tuple[str, str]:
    """(K / V, index) cache formats as ATOM's QSA resolves them (``qwen4_exp.ops.qsa.qsa_cache_formats``)."""
    kv = getattr(atom_config, "kv_cache_dtype", "bf16") or "bf16"
    kv = "bf16" if kv in ("auto", "bfloat16") else kv
    index = getattr(atom_config, "index_cache_dtype", None) or kv
    return kv, "bf16" if index in ("auto", "bfloat16") else index


def _deployment_checks(atom_config) -> list[tuple[bool, str]]:
    """The checkpoint, parallelism, speculative decode and cache format Mono serves."""
    spec = getattr(atom_config, "speculative_config", None)
    checks = common_refusals(atom_config, Shard, SUPPORTED_TP)
    # every program's CTAs are resident, one a CU, and spin on each other
    checks += resident_grid_refusals(atom_config, connectors=OFFLOAD_CONNECTORS)
    checks += [
        (
            getattr(spec, "method", None) == "mtp"
            and getattr(spec, "num_speculative_tokens", None) == SPEC - 1,
            "not MTP3 speculative decode",
        ),
        (
            _cache_formats(atom_config) == ("fp8", "bf16"),
            f"KV / index caches {_cache_formats(atom_config)}",
        ),
        (
            atom_config.kv_cache_block_size == KV_BLOCK,
            f"KV block {atom_config.kv_cache_block_size}, not {KV_BLOCK}",
        ),
        (_quark_mxfp4(atom_config), "not the Quark MXFP4 checkpoint"),
        (atom_config.torch_dtype == torch.bfloat16, "activation dtype is not BF16"),
        *_shape_checks(atom_config.hf_config),
    ]
    return checks


def _switch_checks() -> list[tuple[bool, str]]:
    """The engine switches the kernels assume (each names the setting that serves)."""
    return [
        (
            envs.ATOM_ENABLE_REPLAYSSM is False
            or envs.ATOM_REPLAYSSM_CACHE_LEN <= REPLAY_MAX,
            f"ATOM_REPLAYSSM_CACHE_LEN={envs.ATOM_REPLAYSSM_CACHE_LEN} (Mono replays up to {REPLAY_MAX} records)",
        ),
        (
            envs.ATOM_GDN_SSM_DTYPE == "fp32",
            f"ATOM_GDN_SSM_DTYPE={envs.ATOM_GDN_SSM_DTYPE} (Mono keeps an fp32 GDN state)",
        ),
        (
            envs.ATOM_QSA_DRAFT_KV_BF16,
            "the MTP draft caches must stay BF16 (ATOM_QSA_DRAFT_KV_BF16=1)",
        ),
    ]


def config_refusal(atom_config) -> str | None:
    """Why this deployment cannot run Qwen Mono, or ``None``."""
    if not envs.ATOM_MONO_ENABLE:
        return "ATOM_MONO_ENABLE is off"
    return first_refusal(_deployment_checks(atom_config) + _switch_checks())


def refused_by_switch_only(atom_config) -> bool:
    """A deployment Mono serves but for an engine switch: the install then warns, naming the setting."""
    return (
        bool(envs.ATOM_MONO_ENABLE)
        and first_refusal(_deployment_checks(atom_config)) is None
        and first_refusal(_switch_checks()) is not None
    )
