# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
DeepSeek-V4 model for ATOM (PR1: skeleton + tiny-config eager forward).

Architecture reference: /data/DeepSeek-V4-Pro/inference/model.py
Tech report: /app/logs_claude/deepseek_v4/DeepSeek_V4.pdf

This file is the PR1 skeleton. It mirrors the reference implementation's class
structure so dummy state_dicts produced by the reference can be loaded directly
into ATOM modules for numerical parity validation. Production paths (FP8/FP4
weight loading, tensor parallelism, AITER kernels, KV cache integration, MTP
spec decode, torch.compile, server) land in PR2-PR6.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, Tuple, cast

if TYPE_CHECKING:
    from atom.model_ops.attentions.deepseek_v4_attn import AttentionMetaData_DSV4

import aiter
import torch
import torch.nn.functional as F
from aiter import (
    cp_gather_indexer_k_quant_cache,
    dtypes,
    rope_rotate_activation,
)
from aiter import silu_and_mul as aiter_silu_and_mul
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce,
)
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from atom.distributed.pcp_utils import (
    get_pcp_world_size,
    pcp_allgather_rerange,
    pcp_pad_len,
    pcp_round_robin_split,
)
from aiter.ops.topk import top_k_per_row_decode, top_k_per_row_prefill
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.fusions.fused_clamp_act_mul import (
    fused_clamp_act_mul,
)
from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16
from atom.config import (
    Config,
    LayerQuantConfig,
    QuantizationConfig,
    QuantType,
    get_current_atom_config,
)
from atom.model_loader.loader import WeightsMapper

# Side-effect import: registers `torch.ops.aiter.maybe_dual_stream_forward`
# (shared with deepseek_v2) and `torch.ops.aiter.indexer_score_topk` (V4-only).
# MoE.forward dispatches via the former so torch.compile/Dynamo treats stream
# code as opaque; Indexer.forward_batched dispatches via the latter to hide
# its dynamic-shape internals from Dynamo / fake-tensor mode.
from atom.model_ops import module_dispatch_ops as _module_dispatch_ops  # noqa: F401
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import RMSNorm, rmsnorm2d_fwd_
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.quant_v4 import act_quant_inplace
from atom.model_ops.sparse_attn_v4 import (
    hc_split_sinkhorn,
)
from atom.model_ops.topK import (
    is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config,
)
from atom.model_ops.triton_hash_topk import hash_topk_triton
from atom.model_ops.triton_rmsnorm_nw import rmsnorm_nw
from atom.model_ops.utils import atom_parameter
from atom.model_ops.v4_kernels import (
    CompressPlan,
    csa_translate_pack,
    fused_compress_attn,
    inverse_rope_inplace,
    qk_norm_rope_maybe_quant,
    scale_indexer_weights,
    sparse_attn_v4_paged_decode,
    sparse_attn_v4_paged_prefill,
    swa_write,
    update_compressor_states,
)
from atom.utils import envs, mark_spliting_op
from atom.utils.decorators import support_torch_compile
from atom.utils.forward_context import AttnState, get_forward_context
from torch import nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classical KV cache scatter / gather helpers (PR3-pre2c-B).
#
# Each V4 block (block_size=lcm(m, m')=128 original tokens) holds k_per_block
# compressed entries per layer (k1=32 for CSA, k2=1 for HCA). Compressor.forward
# scatters newly-compressed entries into block-table-indexed slots; sparse_attn
# input gathers all committed entries up to the current position.
#
# In PR3-pre2c-B these helpers run on a single sequence (block_table fetched
# from `forward_context.attn_metadata.block_tables[0]`). PR3-main extends to
# per-seq dispatch.
# ---------------------------------------------------------------------------

# V4 paper §3.6.1: classical-KV block_size = lcm(m, m'). For V4-Pro / V4-Flash
# this is lcm(4, 128) = 128 original tokens. Kept as a constant so Compressor
# code does not need to import the builder.
_V4_BLOCK_SIZE: int = 128

_V4_RMSNORM_BACKEND = os.environ.get("ATOM_V4_RMSNORM_BACKEND", "triton")
_V4_USE_TRITON_RMSNORM = _V4_RMSNORM_BACKEND == "triton"
# Env-gated quant round-trips. Read once at module load — checking each
# forward burns syscalls (V4-Pro: 64 layers × multiple sites per call).
_V4_FORCE_UE8M0_QUANT = os.environ.get("V4_FORCE_UE8M0_QUANT", "0") == "1"
_V4_USE_REF_QUANT = os.environ.get("V4_USE_REF_QUANT", "0") == "1"
# Fused-kernel switches. Default off; flip via env to A/B against the eager path.
_V4_USE_TRITON_FUSION = os.environ.get("ATOM_V4_USE_TRITON_FUSION", "0") == "1"
ENABLE_DS_QKNORM_QUANT_FUSION = envs.ATOM_ENABLE_DS_QKNORM_QUANT_FUSION


def _rmsnorm_nw(x: torch.Tensor, eps: float, dim: int) -> torch.Tensor:
    if _V4_USE_TRITON_RMSNORM:
        return rmsnorm_nw(x, eps)
    ones = torch.ones(dim, dtype=x.dtype, device=x.device)
    return rmsnorm2d_fwd_(x, ones, eps, dim)


def _v4_attention_fake(
    x: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(x)


@mark_spliting_op(is_custom=True, gen_fake=_v4_attention_fake, mutates_args=[])
def v4_attention_with_output(
    x: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    return self.forward_impl(x, positions)


# ---------------------------------------------------------------------------
# Config wrapper
# ---------------------------------------------------------------------------


@dataclass
class DeepseekV4Args:
    """Mirrors `inference/model.py:ModelArgs`. Constructed from `hf_config`.

    Field names match the V4 HuggingFace `config.json` keys where possible;
    aliases are documented inline.
    """

    # Core
    vocab_size: int = 129280
    dim: int = 7168  # hidden_size
    n_layers: int = 61  # num_hidden_layers
    n_mtp_layers: int = 1  # num_nextn_predict_layers
    n_hash_layers: int = 3  # num_hash_layers
    norm_eps: float = 1e-6  # rms_norm_eps
    max_seq_len: int = 1048576  # max_position_embeddings
    max_batch_size: int = 4  # default placeholder; production driven by ATOM scheduler

    # Attention (MQA, single shared KV head)
    n_heads: int = 128  # num_attention_heads
    head_dim: int = 512
    rope_head_dim: int = 64  # qk_rope_head_dim
    q_lora_rank: int = 1536
    o_lora_rank: int = 1024
    o_groups: int = 16
    window_size: int = 128  # sliding_window

    # Per-layer attention type: 0=Dense, 4=CSA, 128 (or other large m')=HCA
    compress_ratios: Tuple[int, ...] = field(default_factory=tuple)

    # Indexer (CSA layers only)
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 1024
    use_index_cache: bool = False
    index_topk_freq: int = 1
    index_topk_pattern: Optional[Any] = None

    # MoE
    moe_inter_dim: int = 3072  # moe_intermediate_size
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6  # num_experts_per_tok
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 2.5  # routed_scaling_factor
    swiglu_limit: float = 10.0

    # Hyper-Connections (mHC)
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # YaRN RoPE
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0  # rope_scaling.factor
    original_seq_len: int = 65536  # rope_scaling.original_max_position_embeddings
    beta_fast: int = 32
    beta_slow: int = 1

    # Quantization (PR1 ignores; PR2+ uses)
    dtype: Literal["bf16", "fp8"] = "bf16"
    expert_dtype: Optional[Literal["fp4", "fp8"]] = None
    scale_fmt: Optional[Literal["ue8m0"]] = None

    # V4QuantizationConfig — Linear layers auto-build the right (FP8 / FP4
    # / BF16) weight + scale params. Set by DeepseekV4ForCausalLM at init.
    quant_config: Optional[Any] = None

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "DeepseekV4Args":
        # Use getattr with sensible defaults so we work whether the HF config is
        # a real V4 PretrainedConfig (all fields present) or a V3 PretrainedConfig
        # populated with extra V4 attrs (some fields may live only in the raw
        # config_dict, not on the config object — `transformers` strips unknown
        # kwargs unless they're in the schema).
        def g(k, default=None):
            return getattr(hf_config, k, default)

        rope_scaling = g("rope_scaling", {}) or {}
        return cls(
            vocab_size=g("vocab_size"),
            dim=g("hidden_size"),
            n_layers=g("num_hidden_layers"),
            n_mtp_layers=g("num_nextn_predict_layers", 1),
            n_hash_layers=g("num_hash_layers", 0),
            norm_eps=g("rms_norm_eps", 1e-6),
            max_seq_len=g("max_position_embeddings", 2048),
            n_heads=g("num_attention_heads"),
            head_dim=g("head_dim", 512),
            rope_head_dim=g("qk_rope_head_dim", 64),
            q_lora_rank=g("q_lora_rank", 1536),
            o_lora_rank=g("o_lora_rank", 256),
            o_groups=g("o_groups", 16),
            window_size=g("sliding_window", 128),
            compress_ratios=tuple(g("compress_ratios", (0,))),
            index_n_heads=g("index_n_heads", 64),
            index_head_dim=g("index_head_dim", 128),
            index_topk=g("index_topk", 1024),
            use_index_cache=bool(g("use_index_cache", False)),
            index_topk_freq=int(g("index_topk_freq", 1)),
            index_topk_pattern=g("index_topk_pattern", None),
            moe_inter_dim=g("moe_intermediate_size", 2048),
            n_routed_experts=g("n_routed_experts", 256),
            n_shared_experts=g("n_shared_experts", 1),
            n_activated_experts=g("num_experts_per_tok", 6),
            score_func=g("scoring_func", "sqrtsoftplus"),
            route_scale=g("routed_scaling_factor", 1.5),
            swiglu_limit=g("swiglu_limit", 10.0),
            hc_mult=g("hc_mult", 4),
            hc_sinkhorn_iters=g("hc_sinkhorn_iters", 20),
            hc_eps=g("hc_eps", 1e-6),
            rope_theta=g("rope_theta", 10000.0),
            compress_rope_theta=g("compress_rope_theta", 160000.0),
            rope_factor=rope_scaling.get("factor", 1.0),
            original_seq_len=rope_scaling.get("original_max_position_embeddings", 0),
            beta_fast=rope_scaling.get("beta_fast", 32),
            beta_slow=rope_scaling.get("beta_slow", 1),
            # Default to "ue8m0" matching reference ModelArgs (inference/model.py:40);
            # HF config.json does not carry this field, only inference/config.json does.
            scale_fmt=g("scale_fmt", "ue8m0"),
        )


def _v4_index_topk_refreshes(args: DeepseekV4Args, layer_id: int) -> bool:
    index_topk_pattern = args.index_topk_pattern
    if index_topk_pattern is not None:
        return not (
            0 <= layer_id < len(index_topk_pattern)
            and index_topk_pattern[layer_id] == "S"
        )

    index_topk_freq = int(args.index_topk_freq)
    if index_topk_freq <= 0:
        raise ValueError("index_topk_freq must be a positive integer")
    csa_ordinal = (
        sum(1 for ratio in args.compress_ratios[: layer_id + 1] if ratio == 4) - 1
    )
    if csa_ordinal < 0:
        return False
    return csa_ordinal % index_topk_freq == 0


def _should_skip_v4_index_topk(args: DeepseekV4Args, layer_id: int) -> bool:
    if not args.use_index_cache:
        return False
    if args.compress_ratios[layer_id] != 4:
        return False
    if _v4_index_topk_refreshes(args, layer_id):
        return False

    # V4 writes CSA indices into a shared per-forward buffer and immediately
    # consumes it. A skip layer is safe only after an earlier CSA refresh layer
    # has populated that buffer in the same forward pass.
    return any(
        args.compress_ratios[prev_layer] == 4
        and _v4_index_topk_refreshes(args, prev_layer)
        for prev_layer in range(layer_id - 1, -1, -1)
    )


# ---------------------------------------------------------------------------
# Module-level constants matching reference inference/model.py module globals
# ---------------------------------------------------------------------------

# PR1 always runs single-rank; TP comes in PR3.
_FP4_BLOCK_SIZE = 32  # matches reference's fp4_block_size


# ---------------------------------------------------------------------------
# V4-specific QuantizationConfig — wired by DeepseekV4ForCausalLM in PR3c
# ---------------------------------------------------------------------------


def _wo_a_is_bf16_on_disk(model_path):
    """Return True iff this ckpt stores ``layers.0.attn.wo_a.weight`` as BF16
    (already pre-dequantized) with NO companion ``wo_a.scale`` on disk.

    V4-Flash-FP8 ships ``wo_a`` as BF16 directly; V4-Flash-Base / V4-Pro ship
    it as FP8 + UE8M0 block-scale and rely on
    ``DeepseekV4Attention.process_weights_after_loading`` to dequant at load
    time. The ATOM Linear allocator decides FP8 vs BF16 from the quant spec
    at module-init time, so we have to probe the ckpt here BEFORE building
    the model — otherwise the FP8 + scale param shapes mismatch the BF16
    tensor on disk and produce garbage attention output.
    """
    if not model_path or not os.path.isdir(model_path):
        return False
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(idx_path):
        return False
    try:
        with open(idx_path) as f:
            idx = json.load(f)
        wmap = idx.get("weight_map", {})
    except Exception:
        return False
    probe = "layers.0.attn.wo_a.weight"
    if probe not in wmap:
        return False
    scale_present_in_idx = "layers.0.attn.wo_a.scale" in wmap
    # Even when listed in the index, the shard may not actually contain the
    # scale (V4-Flash-FP8 had a stale index entry). Open the shard and verify.
    try:
        from safetensors import safe_open

        with safe_open(os.path.join(model_path, wmap[probe]), framework="pt") as h:
            w = h.get_slice(probe)
            w_dtype = (
                w.get_dtype() if hasattr(w, "get_dtype") else getattr(w, "dtype", None)
            )
            if w_dtype in (torch.bfloat16, "BF16"):
                return True  # BF16 weight; no scale needed regardless of index
            if not scale_present_in_idx:
                return False
            if "layers.0.attn.wo_a.scale" not in h.keys():
                # Index lies. wo_a still FP8 but no scale → loader will fail
                # anyway; safer to fall back to no_spec, although this case is
                # unexpected.
                return True
    except Exception:
        return False
    return False


def make_v4_quant_config(hf_config, model_path=None, online_quant_config=None):
    """Build a QuantizationConfig that knows V4's per-layer quant scheme.

    Two V4 SKUs supported:
      - **V4-Pro** (gfx950 / MI355X): routed experts FP4 e2m1 packed +
        per-1x32 UE8M0 scale (DeepGEMM `gemm_a4w4_quant` path).
      - **V4-Flash-Base** (gfx942 / MI308 + others): routed experts FP8 e4m3
        per-block 128x128 + UE8M0 scale (aiter `gemm_a8w8_blockscale` /
        Triton MoE per_1x128 path).

    The routed-expert spec is auto-detected from the ckpt's quantization
    layout via :func:`_detect_v4_routed_quant_spec`; SKU-agnostic projections
    (wq_a/b, wkv, wo_b, indexer.wq_b) all stay FP8 per-block 128x128.

    V4 checkpoint layout (common):
      - Most projections (wq_a/b, wkv, wo_b, indexer.wq_b, etc.): FP8 e4m3 +
        128x128 ue8m0 block scale. Picked up by ATOM's standard parser.
      - Routed expert weights (`ffn.experts.{N}.w{1,2,3}`): FP4 (V4-Pro) OR
        FP8 per-block (V4-Flash-Base) — auto-detected.
      - `wo_a`: FP8 on disk but loaded as BF16 (convert.py:137-141 dequantizes
        because the grouped-LoRA einsum needs BF16; aiter has no FP8 einsum).
      - `Compressor.wkv` / `Compressor.wgate` / `indexer.weights_proj`: BF16
        (or fp32 internally; reference declares dtype= explicitly). Loaded raw.
      - All RMSNorm weights, attn_sink, hc_*: BF16/fp32 raw, no quant.

    The optional ``online_quant_config`` is forwarded to the base
    QuantizationConfig so V4 models can also be re-quantized at load time
    (e.g. ``ptpc_fp8`` / ``mxfp4``). V4's hardcoded per-layer overrides
    (FP4 routed experts, BF16 compressor / indexer.weights_proj) are
    preserved on BOTH the source lookup AND the online lookup — returning
    the same spec on the online path triggers the FusedMoE/Linear
    ``source == online_target`` early-return so those layers stay untouched.
    """

    base = QuantizationConfig(hf_config, online_quant_config=online_quant_config)

    fp4_spec = LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=dtypes.fp4x2)
    # FP8 per-block 128x128 — V4-Flash-Base routed path.
    # ``dtypes.fp8`` from aiter resolves to ``float8_e4m3fnuz`` on gfx942/gfx94x
    # (MI308) and ``float8_e4m3fn`` on gfx950 / NV — picked at import time.
    fp8_block_spec = LayerQuantConfig(
        quant_type=QuantType.per_1x128,
        quant_dtype=dtypes.fp8,
    )
    no_spec = LayerQuantConfig(quant_type=QuantType.No, quant_dtype=torch.bfloat16)

    # Detect which routed-expert quant scheme this ckpt uses (FP4 or FP8-block).
    # ``base`` is consulted first — if the user's quant_method parser already
    # produced a per_1x128 fp8 spec for ``ffn.experts``, we honor it; only
    # when the parser yields no information do we fall back to V4-Pro's FP4.
    routed_spec = _detect_v4_routed_quant_spec(
        hf_config, base, fp4_spec, fp8_block_spec
    )

    # V4-Flash-FP8 ships ``wo_a`` already dequanted to BF16 on disk (no
    # ``.scale`` companion). Probe the ckpt; when wo_a is BF16, allocate it
    # as BF16 directly. Other SKUs (V4-Pro / V4-Flash-Base) keep wo_a as
    # FP8 + UE8M0 scale and rely on the load-time dequant in
    # ``DeepseekV4Attention.process_weights_after_loading``.
    wo_a_is_bf16 = _wo_a_is_bf16_on_disk(model_path)
    if wo_a_is_bf16:
        logger.info(
            "ckpt stores wo_a as BF16 on disk; allocating BF16 "
            "wo_a params (skipping FP8 + scale load-time dequant)."
        )

    orig_lookup = base.get_layer_quant_config

    def overridden(layer_name, use_online_quant=False, *, check_children=False):
        # Routed experts → SKU-detected (FP4 for V4-Pro, FP8-block for V4-Flash).
        # Match both per-expert prefix `layers.N.ffn.experts.M.w{1,2,3}` (used
        # by individual Linear lookups, with trailing `.M.w1`) AND the bare
        # `layers.N.ffn.experts` prefix (used by FusedMoE.__init__ when
        # constructing fused expert params — has NO trailing dot).
        #
        # V4 hardcoded specs apply on BOTH source AND online lookups. When
        # online_quant is enabled, returning the source spec here means
        # FusedMoE/Linear see `source == online_target` and skip the
        # dequant→requant round-trip for these layers (which would either
        # crash on the moe assert or further damage already-quantized weights).
        if ".ffn.experts" in layer_name:
            return routed_spec
        # BF16 / fp32 raw paths
        if (
            ".compressor.wkv" in layer_name
            or ".compressor.wgate" in layer_name
            or ".indexer.weights_proj" in layer_name
        ):
            return no_spec
        # V4-Flash-FP8 layout: wo_a is BF16 on disk — allocate as BF16 directly
        # so the loader receives matching dtype. Other SKUs let wo_a allocate
        # as FP8 + scale and DeepseekV4Attention dequants at load time.
        # When online_quant is enabled, also keep wo_a BF16 so
        # the dequant→requant round-trip is skipped for this layer.
        if ".wo_a" in layer_name and (wo_a_is_bf16 or use_online_quant):
            return no_spec
        return orig_lookup(
            layer_name,
            use_online_quant=use_online_quant,
            check_children=check_children,
        )

    base.get_layer_quant_config = overridden
    return base


def _detect_v4_routed_quant_spec(hf_config, base, fp4_spec, fp8_block_spec):
    """Detect V4 routed-expert quant scheme from HF config + parser output.

    Resolution order:
      1. **HF config ``expert_dtype``** — if the model's config.json declares
         ``expert_dtype`` (e.g. ``"fp8"`` or ``"fp4"``), use it directly.
      2. **Parser-derived spec for ``ffn.experts``** — if the model's
         quant_method parser (quark / generic / fp8 / ...) already produced a
         layer pattern that matches ``ffn.experts.*.w*``, honor it. This is
         the canonical path: the ckpt's own quantization_config dict declares
         ``per_1x128`` (fp8 block) or ``per_1x32`` (fp4 microscaling), and
         the parser turns it into the correct spec.
      3. **Heuristic from ``quant_method``** — when the parser doesn't carry
         per-layer detail (some compressed-tensors ckpts only set a global
         spec), look at ``hf_config.quantization_config.quant_method``:
         strings containing "fp4"/"mxfp4" → FP4; "fp8" → FP8 block.
      4. **V4-Pro default fallback** — historical V4 default (FP4 e2m1).

    Returns the chosen ``LayerQuantConfig`` (always either ``fp4_spec`` or
    ``fp8_block_spec`` — never None).
    """

    # ── 1. HF config expert_dtype hint ──
    expert_dtype = getattr(hf_config, "expert_dtype", None) or ""
    if isinstance(expert_dtype, str):
        ed = expert_dtype.lower()
        if "fp4" in ed:
            return fp4_spec
        if "fp8" in ed:
            return fp8_block_spec

    # ── 2. Parser-derived spec ──
    # Probe a representative routed-expert layer name. The parser's pattern
    # match (fnmatch) returns whatever was declared in the ckpt's
    # quantization_config -> layer_quant_config dict.
    sample = base.get_layer_quant_config("layers.0.ffn.experts.0.w1")
    if sample.is_quantized:
        # FP4: ATOM uses per_1x32 + dtypes.fp4x2 (microscaling FP4)
        if sample.quant_type == QuantType.per_1x32:
            return fp4_spec
        # FP8 per-block: per_1x128 + fp8 dtype
        if sample.quant_type == QuantType.per_1x128:
            return fp8_block_spec
        logger.warning(
            "Routed-expert layer quantized with unsupported quant_type=%s "
            "(expected per_1x32 or per_1x128). Falling through to heuristic.",
            sample.quant_type,
        )

    # ── 3. quant_method heuristic ──
    qc = getattr(hf_config, "quantization_config", None) or {}
    method = (qc.get("quant_method") or "").lower() if isinstance(qc, dict) else ""
    fmt = (qc.get("fmt") or "").lower() if isinstance(qc, dict) else ""
    method_lower = method + " " + fmt
    if "fp4" in method_lower or "mxfp4" in method_lower:
        return fp4_spec
    if "fp8" in method_lower or "deepseek_fp8" in method_lower:
        return fp8_block_spec

    # ── 4. V4-Pro default fallback ──
    logger.info(
        "routed-expert quant not auto-detected; falling back to FP4 (V4-Pro). "
        "Set expert_dtype in config.json to override."
    )
    return fp4_spec


def _dequant_fp8_block_to_bf16(w_fp8, scale, block=128):
    """Dequant block-scaled FP8 e4m3 → BF16 (for wo_a load path).

    Mirrors convert.py:137-141. The wo_a weight is stored FP8 on disk but
    used as BF16 in inference because aiter doesn't support FP8 grouped einsum.
    """
    out_dim, in_dim = w_fp8.shape
    w = w_fp8.unflatten(0, (-1, block)).unflatten(-1, (-1, block)).float()
    s = scale.float()
    deq = w * s[:, None, :, None]
    return deq.flatten(2, 3).flatten(0, 1).bfloat16()


# ---------------------------------------------------------------------------
# Small utilities — port of inference/model.py:183-276
# ---------------------------------------------------------------------------


@lru_cache(2)
def _precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    """Precompute complex exponentials for rotary embeddings with YaRN scaling.

    Port of inference/model.py:199-229. When `original_seq_len > 0`, applies YaRN
    frequency interpolation with a smooth linear ramp between beta_fast and
    beta_slow correction ranges.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_, max_, dim):
        if min_ == max_:
            max_ += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_) / (max_ - min_)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply rotary positional embeddings IN-PLACE (manual complex multiply).

    Port of inference/model.py:232-244. The input tensor `x` is overwritten with
    the rotated values; the same tensor is also returned for chaining.
    `inverse=True` uses the conjugate (un-rotation) — used on the attention
    output to remove absolute-position embedding from the value contribution.

    NOTE: forward RoPE on Q/KV now goes through `_V4RoPE` (aiter kernel). This
    function is kept ONLY for the output inverse step, which aiter does not
    expose.
    """
    y = x
    x_f = x.float()
    x = torch.view_as_complex(x_f.reshape(*x_f.shape[:-1], -1, 2))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


@lru_cache(8)
def _build_cos_sin_cache(
    rotary_dim: int,
    max_seq_len: int,
    base: float,
    factor: float,
    original_seq_len: int,
    beta_fast: int,
    beta_slow: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared cos/sin cache for `_V4RoPE`, keyed by (rope params, dtype, device).

    V4 has only 3 distinct rope param sets (HCA / CSA / Dense) — without
    deduping we'd materialize 62 copies per rank (~16GB at fp32 complex,
    ~8GB at bf16). Per-device caching means each rank holds exactly one
    cos+sin pair per param set. Cache size 8 covers (HCA, CSA, Dense) ×
    (cuda:0..N) headroom.
    """
    freqs = _precompute_freqs_cis(
        rotary_dim,
        max_seq_len,
        original_seq_len,
        base,
        factor,
        beta_fast,
        beta_slow,
    )
    cos = (
        freqs.real.to(device=device, dtype=dtype)
        .contiguous()
        .unsqueeze(-2)
        .unsqueeze(-2)
    )
    sin = (
        freqs.imag.to(device=device, dtype=dtype)
        .contiguous()
        .unsqueeze(-2)
        .unsqueeze(-2)
    )
    return cos, sin


class _V4RoPE(nn.Module):
    """Per-token-positions RoPE wrapper around aiter's `rope_cached_*_fwd_inplace`.

    Builds the cos/sin cache via V4's exact YaRN math (`_precompute_freqs_cis`),
    then dispatches to the aiter HIP kernel. Works on a pre-sliced rope tensor
    (`head_size == rotary_dim`) so callers stay symmetric with the existing
    `_apply_rotary_emb(x[..., -rd:], ...)` pattern.

    `freqs_for_positions(positions)` rebuilds a complex tensor from the cos/sin
    slices for the attention output's inverse RoPE step (which aiter does not
    expose). We deliberately do NOT keep a complex `freqs_cis` buffer: cos/sin
    in bf16 is half the memory of complex64, and 62 layers × 1M positions ×
    32 freqs adds up fast.
    """

    def __init__(
        self,
        rotary_dim: int,
        max_seq_len: int,
        base: float,
        factor: float,
        original_seq_len: int,
        beta_fast: int,
        beta_slow: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.factor = factor
        self.original_seq_len = original_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.dtype = dtype
        # Build cos/sin caches at __init__ via the lru_cached `_build_cos_sin_cache`
        # and store as plain attributes — NOT `register_buffer`. ATOM wraps model
        # construction in `torch.set_default_device(self.device)`, so the lru_cache
        # builds directly on the current GPU device and is shared across all 62
        # layers with the same rope params (V4 has only 3 distinct sets:
        # HCA/CSA/Dense). Plain-attribute storage skips PyTorch's per-buffer
        # `.to()` machinery, which would clone each layer's reference into a
        # separate GPU tensor (62 × 256 MiB ≈ 16 GiB at V4-Pro's
        # max_position_embeddings=1M — verified OOM if we register_buffer).
        # Tradeoff vs aiter/sglang/vllm: those engines accept the per-layer
        # clone because their target models have much smaller max-pos; V4's 1M
        # context window makes dedup essential. Forward path still does zero
        # cache lookups — only attribute reads.
        self.cos_cache, self.sin_cache = _build_cos_sin_cache(
            rotary_dim,
            max_seq_len,
            base,
            factor,
            original_seq_len,
            beta_fast,
            beta_slow,
            dtype,
            torch.empty(0).device,
        )

    def freqs_for_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Rebuild the complex `freqs_cis` slice for the given positions.

        Used by the attention output's inverse RoPE step.
        Returns: complex64 [num_tokens, rotary_dim // 2].
        """
        cos = self.cos_cache.index_select(0, positions).squeeze(-2).squeeze(-2).float()
        sin = self.sin_cache.index_select(0, positions).squeeze(-2).squeeze(-2).float()
        return torch.complex(cos, sin)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> None:
        """In-place RoPE on `query` (and `key` if given). All inputs are the
        rope-slice only (`head_size == rotary_dim`)."""
        # rotate_style=1 → GPT-J / interleaved (matches V4's view_as_complex).
        rotate_style = 1
        num_tokens = positions.numel()
        if key is not None:
            aiter.rope_cached_positions_2c_fwd_inplace(
                query.view(1, num_tokens, -1, self.rotary_dim),
                key.view(1, num_tokens, -1, self.rotary_dim),
                self.cos_cache,
                self.sin_cache,
                positions.view(1, num_tokens),
                rotate_style,
                reuse_freqs_front_part=True,
                nope_first=False,
            )
        else:
            aiter.rope_cached_positions_fwd_inplace(
                query.view(1, num_tokens, -1, self.rotary_dim),
                self.cos_cache,
                self.sin_cache,
                positions.view(1, num_tokens),
                rotate_style,
                reuse_freqs_front_part=True,
                nope_first=False,
            )

    def inverse(self, positions: torch.Tensor, x: torch.Tensor) -> None:
        """In-place inverse RoPE via fused Triton kernel.

        ``x`` must be the rope-slice only (last dim == rotary_dim).
        """
        inverse_rope_inplace(x, self.cos_cache, self.sin_cache, positions)


# ---------------------------------------------------------------------------
# Compressor + Indexer — port of inference/model.py:279-433
# ---------------------------------------------------------------------------


class Compressor(nn.Module):
    """Compresses KV cache via learned gated pooling over `compress_ratio` consecutive tokens.

    Port of inference/model.py:279-377. `overlap=True` (always set when
    ratio==4, used by CSA) uses overlapping windows to smooth block boundaries.

    Forward delegates pool + RMSNorm + RoPE + bf16 kv_cache scatter to a single
    fused Triton kernel (`fused_compress_attn`). Per-source-position dispatch
    inside the kernel (`s >= start_pos` → INPUT, else state cache) handles
    fresh prefill / chunked prefill / single-token decode / MTP-N uniformly.

    !!!! TODO: QUANT NOT YET FUSED — output drifts from training-time numerics !!!!
    The reference model trained with QAT round-trip:
      - CSA path (rotate=False): `act_quant_inplace(kv[..., :-rd], 64, "ue8m0")`
                                 (BF16 → FP8 e4m3 with ue8m0 scale → BF16)
      - Indexer path (rotate=True): `rotate_activation(kv); fp4_act_quant_inplace(kv, 32)`
                                    (Hadamard rotate then BF16 → FP4 e2m1 → BF16)
    Currently the fused kernel writes raw post-RoPE BF16 to kv_cache, skipping
    both. End-to-end testing shows outputs remain coherent (4 prompts from PR
    #650 baseline still produce sensible completions), but they are NOT
    byte-equal to baseline; benchmark accuracy (lm_eval / GSM8K) MAY regress.
    `self.rotate` is preserved on the module as the discriminator for the
    follow-up PR that ports the two quant flavours into the kernel.
    """

    def __init__(
        self,
        args: DeepseekV4Args,
        compress_ratio: int = 4,
        head_dim: int = 512,
        rotate: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.scale_fmt = args.scale_fmt
        self.prefix = prefix
        coff = 1 + self.overlap

        self.ape = atom_parameter(
            torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32)
        )
        # Fused [wkv; wgate]: both BF16 on disk (same dim out per shard).
        # quant_config=None → BF16 weight; forward calls with otype=fp32 to
        # keep the Compressor's softmax-pool path in fp32 accumulate.
        self.wkv_gate = MergedReplicatedLinear(
            self.dim,
            [coff * self.head_dim, coff * self.head_dim],
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wkv_gate",
        )
        self.norm = RMSNorm(self.head_dim, args.norm_eps)

        # Fixed CUDAGraph-stable scratch for `wkv_gate(x)` output on the captured
        # decode path, in TBO, two concurrent ubatch threads never share the
        # same scratch.
        self._combined_cg_buf: dict = {}

        # External tensors — assigned by the owning Attention / Indexer at first forward.
        self.kv_cache: Optional[torch.Tensor] = None
        self.rotary_emb: Optional[_V4RoPE] = None
        # FP8 quant path only: strided fp32 view of the per-block scale region
        # of `self.kv_cache`. Bound by the V4 builder when `kv_cache.dtype` is
        # FP8 (Indexer-inner Compressor); None for BF16 cache (Main path).
        self.cache_scale: Optional[torch.Tensor] = None

        # State cache (per paper §3.6.1 "uncompressed tail + B-side overlap
        # window" portion). Indexed as a single ring buffer of size
        # `ring_size` (≥ coff * compress_ratio) by `pos % ring_size` per token
        # — no segment switching, no roll. The `forward` softmax-pool consumer
        # resolves A-side (current block) vs B-side (previous block) by
        # block-id parity (`comp_id % 2`).
        #
        # PR3-pre2a: a 1-slot register_buffer is kept here so warmup (which
        # runs before allocate_kv_cache → build_kv_cache_tensor) sees a
        # valid tensor; afterwards `DeepseekV4AttentionMetadataBuilder.
        # build_kv_cache_tensor` setattr-replaces these attributes with
        # views of the per-request cache pool whose second dim is the real
        # ring_size = K_pool + max_spec_steps where K_pool = coff * ratio
        # (non-spec collapses to K_pool since max_spec_steps == 0; causal
        # writes guarantee no read-before-overwrite alias). The 1-slot init
        # buffers (≈9 MB total across all layers) are GC'd once replaced
        # before any real kernel call, so the placeholder's smaller second
        # dim never actually flows through the kernel's
        # `state_size >= K_pool` assertion.
        self.register_buffer(
            "kv_state",
            torch.zeros(
                1,
                coff * compress_ratio,
                coff * self.head_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (1, coff * compress_ratio, coff * self.head_dim),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, dim]
        plan: "CompressPlan",
        state_slot_mapping: torch.Tensor,  # [bs] int32
        block_tables: Optional[torch.Tensor] = None,  # [bs, max_blocks_per_seq] int32
    ) -> None:
        """Batched plan-style compress: one fused kernel call for the whole
        fwd's batch (across all seqs).

        Single fused Triton kernel does pool + RMSNorm + RoPE + cache scatter
        in one launch. Each compression boundary across the batch is one row
        in `plan.compress_plan_gpu`. State cache update fires after (write
        order critical — fused kernel reads state-cache-as-of-previous-fwd;
        `update_compressor_states` overwrites for next fwd).

        Quant mode is auto-selected by `self.kv_cache.dtype`:
          - BF16 cache (CSA Main / HCA Main): raw BF16 row write into
            `self.kv_cache` (consumed by paged_decode/paged_prefill via
            `unified_kv` per-fwd indices).
          - FP8 cache (Indexer-inner): per-row amax → ue8m0 scale → fp8 cast
            → preshuffled (MFMA 16x16 tile) write into `self.kv_cache`, plus
            fp32 scale into `self.cache_scale` (a strided view of the same
            allocation built by the V4 builder). Bit-exact with
            `indexer_k_quant_and_cache` / `cp_gather_indexer_k_quant_cache`
            (cache_kernels.cu:1145+).

        Side-effecting only — no return value (cache scatter IS the output).

        TODO: QAT for the BF16 Main path (FP8 round-trip per Compressor
        docstring) is not yet fused. End-to-end accuracy unaffected today
        because the input act_quant simulation is applied upstream.

        Args:
            x:           [num_tokens, dim] flat ragged batch hidden state.
            plan:        CompressPlan from attn_metadata.compress_plans[ratio]
                         (or a synthetic bs=1 plan during warmup).
            state_slot_mapping: [bs] int32 — per-seq state cache slot.
            block_tables: [bs, max_blocks_per_seq] int32 — physical block IDs
                         per seq; None during warmup (skips kv_cache scatter).
                         Required for the Indexer FP8 path (slot resolution).
        """
        assert self.rotary_emb is not None, "compressor.rotary_emb must be set by owner"
        assert (
            x.dim() == 2 and x.shape[-1] == self.dim
        ), f"Compressor expects [num_tokens, {self.dim}], got {tuple(x.shape)}"
        ratio = self.compress_ratio
        overlap = self.overlap
        d = self.head_dim
        rd = self.rope_head_dim

        # Single fused BF16 GEMM via tgemm. (Probing whether dropping the
        # otype=fp32 upcast — relying on fused_compress_attn's internal fp32
        # accumulator instead — is accuracy-neutral.) torch.split returns
        # zero-copy strided views; downstream kernels (fused_compress_attn,
        # update_compressor_states) accept strided kv/score (only inner
        # stride must be 1).
        coff_d = (1 + overlap) * d
        combined = self.wkv_gate(x)
        # ===== PCP (full-KV) =====
        # `x` here is this rank's 1/W round-robin shard (model.forward entry split).
        # The wkv_gate projection above is per-token (parallelizable), but the
        # downstream fused_compress_attn compresses `ratio` CONSECUTIVE tokens
        # into one entry — which round-robin split breaks. So all-gather the
        # projected `combined` back to full sequence order before compression,
        # mirroring SGLang's compute_kv_score (all-gather kv_score after the
        # projection, before the cross-token compress). The plan /
        # state_slot_mapping passed to fused_compress_attn are full-sequence
        # (never split in the builder), so they match the gathered `combined`.
        if _pcp_active():
            combined = pcp_allgather_rerange(combined, get_pcp_world_size())
        # TBO decode: copy `combined` into a fixed-address buffer so CUDAGraph
        # capture/replay see a stable pointer (allocator may re-place it).
        from atom.utils.tbo.ubatching import tbo_active, tbo_current_ubatch_id

        _fc = get_forward_context()
        if getattr(_fc, "in_hipgraph", False) and tbo_active():
            ub = tbo_current_ubatch_id()
            n_tok = combined.shape[0]
            buf = self._combined_cg_buf.get(ub)
            if buf is None or buf.shape[0] < n_tok or buf.shape[1] != combined.shape[1]:
                buf = torch.empty(
                    combined.shape[0],
                    combined.shape[1],
                    dtype=combined.dtype,
                    device=combined.device,
                )
                self._combined_cg_buf[ub] = buf
            buf[:n_tok].copy_(combined)
            combined = buf[:n_tok]
        kv, score = torch.split(combined, [coff_d, coff_d], dim=-1)

        # ====== Unified fused kernel path (CSA + Indexer) ======
        # Order is critical: fused kernel reads state cache as-of-end-of-
        # PREVIOUS-fwd. `update_compressor_states` overwrites them with this
        # fwd's data for the NEXT fwd's overlap — must run AFTER the fused
        # kernel.
        cos_cache, sin_cache = self.rotary_emb.cos_cache, self.rotary_emb.sin_cache
        # Quant path triggers when the bound cache is FP8 (Indexer-inner).
        # `self.cache_scale` is bound alongside `self.kv_cache` by the V4
        # builder when the cache is FP8 (strided fp32 view of the per-block
        # scale region).
        is_quant = self.kv_cache is not None and self.kv_cache.dtype != torch.bfloat16
        # Skip the kernel's cache scatter during warmup (kv_cache/block_tables
        # not yet bound).
        if block_tables is None or self.kv_cache is None:
            scatter_kv_cache = None
            scatter_block_tables = None
        else:
            scatter_kv_cache = self.kv_cache
            scatter_block_tables = block_tables
        fused_compress_attn(
            kv_in=kv,
            score_in=score,
            kv_state=self.kv_state,
            score_state=self.score_state,
            plan=plan,
            state_slot_mapping=state_slot_mapping,
            ape=self.ape,
            rms_weight=self.norm.weight,
            rms_eps=self.norm.eps,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            kv_cache=scatter_kv_cache,
            block_tables=scatter_block_tables,
            k_per_block=_V4_BLOCK_SIZE // ratio,
            overlap=overlap,
            ratio=ratio,
            head_dim=d,
            rope_head_dim=rd,
            quant=is_quant,
            cache_scale=self.cache_scale if is_quant else None,
            use_ue8m0=(self.scale_fmt == "ue8m0"),
            preshuffle=True,
            fp8_max=(torch.finfo(self.kv_cache.dtype).max if is_quant else None),
        )
        update_compressor_states(
            kv,
            score,
            self.ape,
            self.kv_state,
            self.score_state,
            write_plan=plan.write_plan_gpu,
            num_write=plan.num_write,
            state_slot_mapping=state_slot_mapping,
            ratio=ratio,
            overlap=overlap,
        )


class Indexer(nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.

    Port of inference/model.py:380-433. Has its own Compressor (with Hadamard
    rotation + FP4 simulation) to build a separate compressed KV cache used
    only for index scoring; query is also FP4-simulated.
    """

    def __init__(self, args: DeepseekV4Args, compress_ratio: int = 4, prefix: str = ""):
        super().__init__()
        self.prefix = prefix  # Used by V4 attention builder for layer-id parsing.
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.compress_ratio = compress_ratio

        qc = args.quant_config
        # Indexer Q is replicated across TP ranks: the index scoring path
        # needs all 64 heads at every rank to compute the per-token
        # compressed-position topk locally without cross-rank all_reduce.
        # Sharding wq_b would force an extra all_reduce on `index_score`
        # after the per-head sum.
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=qc,
            prefix=f"{prefix}.wq_b",
        )
        # weights_proj: BF16 in reference. Replicated because the layer is
        # tiny (dim × n_heads = 7168 × 64 ≈ 896KB BF16) and column-parallel
        # sharding produces a degenerate N=8 GEMM with no aiter tuned
        # config; full replication keeps N=64.
        self.weights_proj = ReplicatedLinear(
            self.dim,
            self.n_heads,
            bias=False,
            quant_config=qc,
            prefix=f"{prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5
        # Init-time hoists out of `forward_batched`'s hot path.
        # FP8 Q quant is fused into `rope_rotate_activation` (per_1x128 over
        # head_dim); `group_size` is the per-1xN block. head_dim is the index
        # head dim (128), so there is exactly one scale per (token, head).
        self._q_quant_group = self.head_dim
        self._weights_scale = self.softmax_scale * self.n_heads**-0.5
        # `deepgemm_fp8_paged_mqa_logits` decode-path output column count:
        # one indexer slot per `compress_ratio` source tokens.
        self._max_model_len_idx = args.max_seq_len // compress_ratio

        self.compressor = Compressor(
            args,
            compress_ratio,
            self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
        )
        # PR3-pre2c-B: Indexer.kv_cache is bound by the V4 attention builder
        # to a `[num_blocks, k1, head_dim]` per-CSA-layer view of the global
        # `csa_idx_kv` classical KV pool. The 1-slot register_buffer below is
        # a warmup fallback (warmup runs before allocate_kv_cache); it is
        # setattr-replaced post-binding and GC'd. Same pattern as Compressor's
        # kv_state in pre2a / Attention.swa_kv in pre2c-A.
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                1,
                args.max_seq_len // compress_ratio,
                self.head_dim,
            ),
            persistent=False,
        )
        self.rotary_emb: Optional[_V4RoPE] = None

        # Register self in static_forward_context so the
        # `torch.ops.aiter.indexer_score_topk` dispatcher can look us up by
        # `layer_name` (= self.prefix). Same pattern as V4 MoE registration.
        get_current_atom_config().compilation_config.static_forward_context[
            prefix
        ] = self

    def forward_batched(
        self,
        x_full: torch.Tensor,  # [total_tokens, dim]
        qr_full: torch.Tensor,  # [total_tokens, q_lora_rank] — fp8 when qr_full_scale given
        positions: torch.Tensor,  # [total_tokens]
        qr_full_scale: Optional[
            torch.Tensor
        ] = None,  # per_1x128 scale paired with qr_full
    ) -> torch.Tensor:
        """Q proj + RoPE + FP8-quant + weights compute (have module state),
        then dispatch to `torch.ops.aiter.indexer_score_topk`, which calls
        back into `self.indexer_score_topk(q_fp8, weights, self.index_topk)`.

        Caller must invoke `self.compressor` once batched BEFORE this so all
        seqs' Indexer kv_cache is already populated.

        Returns:
          topk_in_seq: `[total_tokens, index_topk] int32` — RAW seq-local row
            indices (each token's column refers to row in its own seq's
            compressed K). Cols past per-token visibility cap hold -1
            sentinels (kernel-native: prefill `top_k_per_row_prefill` and
            decode `top_k_per_row_decode` both write -1 in the tail).
            Consumer (`csa_translate_pack`) skips negative entries via its
            `topk >= 0` write mask.
        """
        assert self.rotary_emb is not None
        rd = self.rope_head_dim
        total_tokens = x_full.size(0)

        # Q proj + RoPE + rotate (batched). rotary_emb internally reshapes
        # to (1, num_tokens, -1, rotary_dim) so the input doesn't need an
        # explicit batch dim. rotate_activation is last-dim-only.
        q = self.wq_b(qr_full, x_scale=qr_full_scale).view(
            total_tokens, self.n_heads, self.head_dim
        )
        # RoPE + Hadamard-rotate + FP8 quant fused in one kernel. Q is online
        # (recomputed each fwd, no cache); the bf16 rotated Q is never read back,
        # so it is quantized in place of being materialized. `out_scale` carries
        # the per-(token, head) fp8 block scale (head_dim == group => one/row).
        # `_weights_scale` precomputed in __init__.
        # self.rotary_emb(positions, q[..., -rd:]); q = rotate_activation(q)
        q_fp8 = torch.empty_like(q, dtype=dtypes.fp8)
        q_scale = torch.empty(
            (total_tokens * self.n_heads, self.head_dim // self._q_quant_group),
            dtype=dtypes.fp32,
            device=q.device,
        )
        rope_rotate_activation(
            q_fp8,
            q,
            self.rotary_emb.cos_cache,
            self.rotary_emb.sin_cache,
            positions,
            rd,
            out_scale=q_scale,
            group_size=self._q_quant_group,
        )
        q_fp8 = q_fp8.view(total_tokens, self.n_heads, self.head_dim)
        q_scale = q_scale.view(total_tokens, self.n_heads, 1)

        # weights = weights_proj * q_scale * (softmax_scale * 1/sqrt(H))
        # weights_proj is BF16 but auto-promotes to fp32 via fp32 q_scale,
        # so no explicit `.float()` cast needed.
        weights = self.weights_proj(x_full)
        weights = scale_indexer_weights(weights, q_scale, self._weights_scale)

        return torch.ops.aiter.indexer_score_topk(
            q_fp8, weights, self.prefix, self.index_topk
        )  # [total_tokens, index_topk] int32

    def indexer_score_topk(
        self,
        q_fp8: torch.Tensor,  # [total_tokens, n_heads, head_dim] fp8
        weights: torch.Tensor,  # [total_tokens, n_heads] fp32
        topk: int,
    ) -> torch.Tensor:
        """Module-side entry invoked by `torch.ops.aiter.indexer_score_topk`.

        Reads `block_tables` and `v4_indexer_meta` from
        `get_forward_context().attn_metadata` (built once per fwd in
        `DeepseekV4AttentionMetadataBuilder._build_v4_indexer_meta`) — the
        per-CSA-layer call has zero CPU index math and zero H2D copies.

        Returns:
          topk_in_seq: `[total_tokens, topk] int32` — RAW seq-local row
            indices into each token's seq's compressed K cache. Cols past
            per-token visibility cap hold -1 sentinels (kernel-native).
            `csa_translate_pack` consumes this layout directly.
        """
        fc = get_forward_context()
        indexer_meta = fc.attn_metadata.indexer_meta
        block_tables = fc.attn_metadata.block_tables  # [bs, max_blocks_per_seq] int32

        # No host-side `if total_committed == 0: return torch.full(-1)`
        # short-circuit — that would freeze a Python branch into the
        # CUDAGraph at capture time. The hot path handles the corner
        # natively: when n_committed == 0 the per-token K bound is 0, the
        # underlying top-k kernels write -1 sentinels across the row, and
        # `csa_translate_pack` skips them via its `topk >= 0` mask.
        if fc.context.is_prefill:
            return self._score_topk_prefill(
                q_fp8, weights, block_tables, indexer_meta, topk
            )  # [total_tokens, topk] int32
        return self._score_topk_decode(
            q_fp8, weights, block_tables, indexer_meta, topk
        )  # [total_tokens, topk] int32

    def _score_topk_prefill(
        self,
        q_fp8: torch.Tensor,  # [total_tokens, n_heads, head_dim] fp8
        weights: torch.Tensor,  # [total_tokens, n_heads] fp32
        block_tables: torch.Tensor,  # [bs, max_blocks_per_seq] int32
        indexer_meta: dict,
        topk: int,
    ) -> torch.Tensor:
        """Variable-K prefill / mixed batch: cp_gather + fp8_mqa_logits.

        Eager-only — total_committed varies per fwd, so output logits shape
        is dynamic and incompatible with CUDAGraph capture.
        """
        device = q_fp8.device
        total_tokens = q_fp8.size(0)
        # K side: cache stores FP8 + 4-byte fp32 scale per row interleaved
        # (uint8 layout written by `indexer_k_quant_and_cache` from the inner
        # Compressor). `cp_gather_indexer_k_quant_cache` does paged-gather
        # + split into separate (FP8, scale) buffers in one kernel — no
        # per-row index list, no online quant.
        total_committed = indexer_meta["total_committed"]
        cu_committed = indexer_meta["cu_committed_gpu"]
        k_fp8 = torch.empty(
            (total_committed, self.head_dim), device=device, dtype=dtypes.fp8
        )
        k_scale = torch.empty((total_committed, 1), device=device, dtype=torch.float32)
        cp_gather_indexer_k_quant_cache(
            self.kv_cache,
            k_fp8,
            k_scale.view(dtypes.fp8),  # 4-byte scale rows treated as fp8 bytes
            block_tables,
            cu_committed,
            preshuffle=True,
        )

        cu_starts = indexer_meta["cu_starts_gpu"]  # [total_tokens] int32
        cu_ends = indexer_meta["cu_ends_gpu"]  # [total_tokens] int32
        logits = fp8_mqa_logits(
            Q=q_fp8,
            KV=k_fp8,
            kv_scales=k_scale,
            weights=weights,
            cu_starts=cu_starts,
            cu_ends=cu_ends,
        )  # [total_tokens, total_committed] fp32; outside [start,end) is -inf

        # aiter `top_k_per_row_prefill` (radix kernel, parametric `k` via the
        # pybind kwarg). Honors per-row [cu_starts[i], cu_ends[i]) so cells
        # outside each row's valid window are never selected; rows shorter
        # than `topk` get -1 sentinels for tail cols.
        #
        # Output is GLOBAL: each cell holds either -1 or
        # `cu_starts[t] + col_in_seq` (= seq_base + seq-local idx). We
        # subtract `seq_base_per_token` to produce the raw seq-local layout
        # `csa_translate_pack` expects. The -1 sentinels are preserved via
        # `torch.where`.
        # [total_tokens, topk] int32 — eager-only path so per-fwd alloc
        # is fine (prefill total_tokens is dynamic; no CG capture here).
        topk_global = torch.empty(
            (total_tokens, topk), dtype=torch.int32, device=device
        )
        top_k_per_row_prefill(
            logits,
            cu_starts,
            cu_ends,
            topk_global,
            None,  # values not needed, only indices
            total_tokens,
            logits.stride(0),
            logits.stride(1),
            k=topk,
        )
        seq_base = indexer_meta["seq_base_per_token_gpu"].unsqueeze(
            1
        )  # [total_tokens, 1] int32
        return torch.where(
            topk_global < 0,
            topk_global,  # preserve -1 sentinel
            topk_global - seq_base,
        )  # [total_tokens, topk] int32, raw seq-local with -1 in tail

    def _score_topk_decode(
        self,
        q_fp8: torch.Tensor,  # [total_tokens, n_heads, head_dim] fp8
        weights: torch.Tensor,  # [total_tokens, n_heads] fp32
        block_tables: torch.Tensor,  # [bs, max_blocks_per_seq] int32
        indexer_meta: dict,
        topk: int,
    ) -> torch.Tensor:
        """Pure-decode path: `deepgemm_fp8_paged_mqa_logits` reads paged FP8
        cache directly, producing fixed-shape `[bs*next_n, max_model_len_idx]`
        logits — CUDAGraph-friendly (no per-fwd `total_committed`-shaped
        allocation). Mirrors V3.2 sparse_attn_indexer decode branch
        (deepseek_v2.py:1047-1084).

        Top-k uses aiter `top_k_per_row_decode` (radix kernel, parametric `k`):
        the kernel honors `n_committed_per_seq` per row, so logits cells past
        each row's valid range are never selected — no `fill_(-inf)` required.
        Rows whose valid range is shorter than `index_topk` get -1 sentinels
        for tail cols. Output is RAW seq-local (each row's cols are 0-indexed
        into that batch's compressed K), exactly the layout
        `csa_translate_pack` consumes.
        """
        total_tokens = q_fp8.size(0)
        n_committed_per_seq_gpu = indexer_meta["n_committed_per_seq_gpu"]  # int32 [bs]
        # NOTE: derive the query batch size from the ACTUAL number of query
        # tokens, NOT from block_tables.size(0). Under TBO the per-ubatch
        # block_tables / n_committed are padded to a DP-unified bucket and will
        # get errors if we try to use the padded rows.
        next_n = max(1, int(get_forward_context().attn_metadata.max_seqlen_q))
        bs = total_tokens // next_n
        # deepgemm requires Q in [bs, next_n, heads, head_dim], KV in
        # [num_blocks, block_size, n_head=1, hidden_dim+scale_dim] (4D).
        q_4d = q_fp8.view(
            bs, next_n, self.n_heads, self.head_dim
        )  # [bs, next_n, n_heads, head_dim] fp8
        kv_cache_4d = self.kv_cache.unsqueeze(
            -2
        )  # [num_blocks, k1_csa, 1, head_dim+scale_dim] uint8
        # Per-fwd write-once GPU scratch — no CPU mirror, no cross-fwd state.
        # Under CUDAGraph capture, torch allocates from the graph's private
        # memory pool and the address is stable across replays at this
        # captured `total_tokens`. No `fill_(-inf)` needed —
        # `top_k_per_row_decode` bounds each row by `n_committed_per_seq[batch]`
        # so unwritten cols are never picked.
        logits = torch.empty(
            total_tokens,
            self._max_model_len_idx,
            dtype=torch.float32,
            device=q_fp8.device,
        )
        deepgemm_fp8_paged_mqa_logits(
            q_4d,
            kv_cache_4d,
            weights,
            logits,
            n_committed_per_seq_gpu,  # int32, sized [bs] (staged in builder)
            block_tables,
            self._max_model_len_idx,
            KVBlockSize=self.kv_cache.size(1),  # k1_csa = 32
            Preshuffle=True,
        )
        # Per-fwd write-once int32 scratch. Kernel writes exactly `index_topk`
        # ints per row (valid seq-local indices then -1 sentinels). CG-safe
        # for the same reason as `logits` above.
        topk_local = torch.empty(
            total_tokens, self.index_topk, dtype=torch.int32, device=q_fp8.device
        )
        top_k_per_row_decode(
            logits,
            next_n,
            n_committed_per_seq_gpu,
            topk_local,
            total_tokens,
            logits.stride(0),
            logits.stride(1),
            k=topk,
        )
        return topk_local  # [total_tokens, index_topk] int32, raw seq-local


# ---------------------------------------------------------------------------
# Stubs — implementations land in tasks #5-#8
# ---------------------------------------------------------------------------


class DeepseekV4Attention(nn.Module):
    """Hybrid attention: MQA + grouped output LoRA + sliding window + attn_sink.

    Port of inference/model.py:436-543. Per-layer behavior driven by
    `compress_ratio` (read from args.compress_ratios[layer_id]):

      - `compress_ratio == 0`: Dense (sliding-window only; no compressor/indexer)
      - `compress_ratio == 4`: CSA (compressor with overlap + indexer for top-k)
      - `compress_ratio >= 8`: HCA (compressor only; topk_idxs pre-computed)

    Layout:
      - Single shared MQA head for KV (head_dim=512). Each query head attends
        to the same compressed/window KV via per-query top-k gather.
      - q_lora_rank low-rank Q projection: wq_a -> q_norm -> wq_b -> RMSNorm-per-head -> RoPE
      - Grouped output LoRA: o_groups groups, each with rank o_lora_rank
      - Sliding window of `args.window_size=128` raw KV entries (BF16, FP8-simulated nope dims)
      - Compressed KV up to `max_seq_len // compress_ratio` entries (when ratio > 0)
      - attn_sink: per-head learnable logit added only to softmax denominator
    """

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV4Args,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        indexer_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        # TP shards heads + groups across ranks. ColumnParallelLinear (wq_b, wo_a)
        # auto-splits output dim, so per-rank counts must be divided by tp_size.
        tp_size = get_tensor_model_parallel_world_size()
        assert (
            args.n_heads % tp_size == 0
        ), f"n_heads={args.n_heads} not divisible by tp={tp_size}"
        assert (
            args.o_groups % tp_size == 0
        ), f"o_groups={args.o_groups} not divisible by tp={tp_size}"
        self.tp_size = tp_size
        self.n_local_heads = args.n_heads // tp_size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps
        self.scale_fmt = args.scale_fmt
        self.skip_topk = False

        qc = args.quant_config
        p = prefix  # e.g. "layers.7.attn"

        # ----- Parameters (names mirror reference for state_dict load) -----
        self.attn_sink = atom_parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        # Fused [wq_a; wkv]: both ReplicatedLinear FP8 sharing input x.
        # On disk still split (`attn.wq_a.{weight,scale}` + `attn.wkv.{weight,scale}`);
        # routed via packed_modules_mapping in DeepseekV4ForCausalLM.
        self.wqkv_a = MergedReplicatedLinear(
            self.dim,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wqkv_a",
        )
        # Fuse q_norm + per_1x128 FP8 quant: kernel emits (qr_fp8, qr_scale)
        # in one launch, both wq_b consumers (outer ColumnParallel + Indexer
        # ReplicatedLinear) skip their own input quant.
        self.q_norm = RMSNorm(
            self.q_lora_rank,
            self.eps,
            fused_quant=True,
            quant_config=qc,
            prefix=f"{p}.q_norm",
        )
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wq_b",
        )
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        # wo_a: grouped LoRA — V4QuantConfig forces this BF16 even though disk is FP8.
        # The grouped einsum (`bsgd,grd->bsgr`) needs BF16 weights; aiter has no FP8 einsum.
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wo_a",
        )
        self.wo_b = RowParallelLinear(
            self.n_groups * args.o_lora_rank,
            self.dim,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wo_b",
        )
        self.softmax_scale = self.head_dim**-0.5

        # ----- Compressor (and Indexer for CSA) -----
        if self.compress_ratio:
            self.compressor = Compressor(
                args,
                self.compress_ratio,
                self.head_dim,
                prefix=f"{p}.compressor",
            )
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio, prefix=f"{p}.indexer")
                self.skip_topk = _should_skip_v4_index_topk(args, layer_id)
            else:
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

        # ----- KV cache splitting (paper §3.6.1) -----
        # State cache (per-request slot, in per_req_cache pool):
        #   `swa_kv`: [num_slots, n_win, head_dim] — most recent n_win window.
        #   Bound by DeepseekV4AttentionMetadataBuilder.build_kv_cache_tensor()
        #   after allocate_kv_cache. The 1-slot register_buffer below is a
        #   warmup fallback (warmup runs before allocate_kv_cache); after
        #   binding it is setattr-replaced with the per_req_cache pool slice
        #   `[max_num_seqs, n_win, head_dim]` and the original buffer is GC'd.
        # `unified_kv` (paged_decode/paged_prefill base) is NOT pre-registered
        # — V4Attention.forward short-circuits the sparse_attn dispatch on
        # `is_dummy_run` so warmup never reads it.
        self.register_buffer(
            "swa_kv",
            torch.zeros(1, args.window_size, self.head_dim),
            persistent=False,
        )
        # Classical KV cache (paper §3.6.1) lives entirely in the global
        # `csa_main_kv` / `hca_main_kv` pool (allocated by the V4 attention
        # builder as `[num_blocks, n_layers, k_per_block, head_dim]`).
        # `Compressor.kv_cache` is bound to a per-layer view of that pool by
        # `DeepseekV4AttentionMetadataBuilder.build_kv_cache_tensor`. The
        # Attention module no longer owns a `kv_cache` attribute (PR3-pre2c-B).

        # ----- RoPE (own per-layer instance, not shared): YaRN for compressed
        # attention layers (long context), plain RoPE for dense (window-only).
        # Wraps aiter's `rope_cached_*_fwd_inplace` kernel so RoPE is driven by
        # per-token `positions` (groundwork for PR3 multi-sequence), while the
        # cos/sin cache uses V4's exact YaRN math via `_precompute_freqs_cis`.
        if self.compress_ratio:
            original_seq_len, rope_theta = (
                args.original_seq_len,
                args.compress_rope_theta,
            )
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        self.rotary_emb = _V4RoPE(
            rotary_dim=self.rope_head_dim,
            max_seq_len=args.max_seq_len,
            base=rope_theta,
            factor=args.rope_factor,
            original_seq_len=original_seq_len,
            beta_fast=args.beta_fast,
            beta_slow=args.beta_slow,
            dtype=torch.bfloat16,
        )
        # Plumb rotary_emb into compressor / indexer here in __init__ rather
        # than lazily in forward — Dynamo can't trace NNModule setattr inside
        # a compiled forward (graph break + backend re-entry).
        if self.compressor is not None:
            self.compressor.rotary_emb = self.rotary_emb
        if self.indexer is not None:
            self.indexer.rotary_emb = self.rotary_emb
            self.indexer.compressor.rotary_emb = self.rotary_emb

        self.alt_stream = alt_stream
        self.indexer_stream = indexer_stream
        self._use_async_compress = (
            self.alt_stream is not None and self.compressor is not None
        )

        self.layer_name = prefix
        atom_config = get_current_atom_config()
        atom_config.compilation_config.static_forward_context[self.layer_name] = self

    def process_weights_after_loading(self) -> None:
        """Dequant wo_a (FP8 + e8m0 block scale) → BF16 in place.

        Called by ATOM's standard loader (atom.model_loader.loader.load_model)
        after all weights are filled. wo_a is allocated as FP8 ColumnParallelLinear
        so both `.weight` (FP8) and `.weight_scale` (e8m0 block scale) load
        correctly via the standard FP8 path. We then dequant to BF16 because
        forward needs `wo_a.weight` as BF16 for the grouped LoRA einsum
        (`bsgd,grd->bsgr`); aiter has no FP8 grouped einsum.

        Idempotent: if wo_a.weight is already BF16 (e.g. dequant was applied
        elsewhere), this is a no-op.
        """
        w = self.wo_a.weight
        if w.dtype == torch.bfloat16:
            return  # already dequanted
        scale = getattr(self.wo_a, "weight_scale", None)
        if w.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz) or scale is None:
            return  # nothing to do
        # Dequant: w (FP8 [out, in]) × scale (e8m0 [out/128, in/128]) → BF16
        bf16 = _dequant_fp8_block_to_bf16(
            w.data, scale.data.to(torch.float32), block=128
        )
        # Replace the weight tensor with BF16, drop the scale param so future
        # loads / introspection don't try to use a stale FP8 scale.
        self.wo_a.weight = atom_parameter(bf16)
        try:
            delattr(self.wo_a, "weight_scale")
        except AttributeError:
            pass
        # CRITICAL: prevent LinearBase.process_weights_after_loading from
        # `shuffle_weights(self.weight)` on the now-BF16 wo_a. That shuffle
        # is for the FP8 CK GEMM layout; applying it to a plain BF16 matrix
        # consumed by `torch.einsum` corrupts the layout (rows get permuted
        # within 16×16 blocks, only rows aligned to the block boundaries
        # stay in place). Iteration order in load_model is parent-first
        # (DeepseekV4Attention before its child wo_a Linear), so our hook
        # runs BEFORE the shuffle — overriding `quant_type` here makes the
        # subsequent LinearBase post-load a no-op for wo_a.
        #
        # TODO(perf): replace dequant-to-BF16 + einsum with FP8 batched BMM
        # (same path as MLA's `_v_up_proj_and_o_proj`). Steps:
        #   1. Dequant FP8 per-128-block → BF16 (this code)
        #   2. Reshape to [n_local_groups, o_lora_rank, d_per_group]
        #   3. Requant via dynamic_per_batched_tensor_quant → FP8 + scalar scale
        #   4. Forward: _aiter_triton_fp8_bmm(o, W_OA, W_OA_scale, group_size=128)
        # This avoids the dequant + einsum overhead and reuses the proven MLA
        # batched-FP8 kernel. See attention_mla.py:211 for reference.
        self.wo_a.quant_type = QuantType.No
        self.wo_a.need_normalize_e4m3fn_to_e4m3fnuz = False

    def maybe_compressors_async(
        self, x, plan, state_slot_mapping, block_tables
    ) -> bool:
        """Fire Compressor(s) on side streams, return immediately.

        Main Compressor → alt_stream (CSA + HCA).
        Indexer Compressor → indexer_stream (CSA only).
        Waits resolve instantly: side streams ~25us, main Q/KV chain ~87us."""
        fc = get_forward_context()
        current_stream = fc.main_stream
        from atom.utils.tbo.ubatching import tbo_active

        use_async_compress = (
            self._use_async_compress and fc.in_hipgraph and not tbo_active()
        )
        has_compressor = self.compressor is not None
        has_indexer = self.indexer is not None and not self.skip_topk
        if use_async_compress:
            if has_compressor:
                self.alt_stream.wait_stream(current_stream)
            if has_indexer:
                self.indexer_stream.wait_stream(current_stream)

            if has_compressor:
                with torch.cuda.stream(self.alt_stream):
                    self.compressor(
                        x,
                        plan=plan,
                        state_slot_mapping=state_slot_mapping,
                        block_tables=block_tables,
                    )
            if has_indexer:
                with torch.cuda.stream(self.indexer_stream):
                    self.indexer.compressor(
                        x,
                        plan=plan,
                        state_slot_mapping=state_slot_mapping,
                        block_tables=block_tables,
                    )
        else:
            if has_compressor:
                self.compressor(
                    x,
                    plan=plan,
                    state_slot_mapping=state_slot_mapping,
                    block_tables=block_tables,
                )
            if has_indexer:
                self.indexer.compressor(
                    x,
                    plan=plan,
                    state_slot_mapping=state_slot_mapping,
                    block_tables=block_tables,
                )
        return use_async_compress

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.aiter.v4_attention_with_output(x, positions, self.layer_name)

    def forward_impl(
        self,
        x: torch.Tensor,  # [num_tokens, dim]  flat ragged-batch hidden state
        positions: torch.Tensor,  # [num_tokens] int  absolute token positions
    ) -> torch.Tensor:  # [num_tokens, dim]  BF16 attention output
        """Compute attention for `x` at absolute token `positions`.

        PR3-main: handles batched multi-sequence input. Linear projections + RoPE
        run once on the flat `[num_tokens, ...]` batch; SWA write, Compressor
        scatter, sparse_attn (gather + score) iterate over sequences using
        per-seq slot + block_table from the V4 attention builder's metadata.
        Per-seq slicing uses `cu_seqlens_q` from `forward_context`.
        """
        assert (
            x.dim() == 2 and x.shape[-1] == self.dim
        ), f"DeepseekV4Attention expects [num_tokens, {self.dim}], got {tuple(x.shape)}"
        # warmup_model runs BEFORE allocate_kv_cache → `unified_kv` is unbound
        # and the new sparse_attn_v4_paged_{decode,prefill} kernels would read
        # OOB. Same pattern as `attention_mha.py:98` — short-circuit dummy_run
        # with a zero output of the correct shape; downstream layers compile
        # on a real fwd. swa_write / Compressor / Indexer are also skipped to
        # avoid touching unbound state caches.
        fc = get_forward_context()
        if fc.context.is_dummy_run:
            return torch.zeros_like(x)
        if os.environ.get("ATOM_V4_BYPASS_ATTN") == "1":
            return torch.zeros_like(x)
        num_tokens = x.size(0)
        cache_size = self.swa_kv.shape[1]
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # ===== Per-fwd metadata (built once in prepare_prefill/decode). =====
        # All per-fwd state read once. Production prepare_decode/prefill
        # always populates these; warmup goes through the same path
        # (`_populate_state_slot_mapping` falls back to slot 0).
        # Cast to V4 typed metadata so V4-specific attribute access (v4_*,
        # compress_plans, ...) is well-typed for pyright.
        attn_md = cast("AttentionMetaData_DSV4", fc.attn_metadata)
        compress_plans = attn_md.compress_plans
        block_tables_gpu = attn_md.block_tables
        state_slot_mapping = attn_md.state_slot_mapping
        plan_for_layer = compress_plans[ratio] if ratio else None

        # ----- Batched ops on full flat tensors -----
        # `_V4_FORCE_UE8M0_QUANT` (module-level): round-trip x/qr to ue8m0-FP8
        # to mirror the reference's `act_quant(scale_fmt="ue8m0")` Linear-input
        # quantization. EXPERIMENT only.
        if _V4_FORCE_UE8M0_QUANT:
            x = x.clone()
            act_quant_inplace(x, 128, "ue8m0")

        # ===== Triple-stream: Q/KV path + both Compressors in parallel =====
        use_async_compress = self.maybe_compressors_async(
            x, plan_for_layer, state_slot_mapping, block_tables_gpu
        )

        # ----- Q/KV projections (main stream) -----
        qkv_a = self.wqkv_a(x)
        q_lora, kv_pre = torch.split(qkv_a, [self.q_lora_rank, self.head_dim], dim=-1)
        assert (
            not _V4_FORCE_UE8M0_QUANT
        ), "_V4_FORCE_UE8M0_QUANT incompatible with fused q_norm quant (qr is already FP8)"
        qr, qr_scale = self.q_norm(q_lora)
        q = self.wq_b(qr, x_scale=qr_scale)
        is_decode = attn_md.state is AttnState.DECODE
        # Single kernel fuses per-head Q RMSNorm (weightless) + KV RMSNorm
        # (weighted) + GPT-J interleaved RoPE on the tail rd dims. Dispatches
        # to flydsl when the shape matches (V4-Pro is always V4-Pro shape →
        # always flydsl). Microbench shows flydsl wins at every measured T
        # from 4 (1.12×) to 32k (1.04×); used for both decode and prefill.
        # Optional FP8 quant outputs left off — downstream sparse_attn /
        # swa_write are still bf16.
        # Decode folds the SWA cache-write into qk_norm_rope_maybe_quant: the
        # post-norm/rope KV row is written into swa_kv[slot, pos%cache, :]
        # (slot = state_slot_mapping[batch_id_per_token[t]]). The flydsl path
        # fuses it into the kernel launch; the Triton fallback emits a separate
        # swa_write internally — either way the bridge owns the SWA write, so
        # no backend dispatch is needed here. Prefill writes its in-chunk SWA
        # tail after sparse_attn, so it passes swa_kv=None and never fuses.
        # For decode, write_per_batch (= min(max_seqlen_q, cache_size)) >=
        # tokens-per-seq, so the fused per-token scatter (gated on batch_id>=0)
        # covers exactly the tokens the old standalone swa_write did.
        q_sa, kv, q_scale, kv_scale = qk_norm_rope_maybe_quant(
            q,
            kv_pre,
            self.kv_norm.weight,
            self.rotary_emb.cos_cache,
            self.rotary_emb.sin_cache,
            positions,
            self.n_local_heads,
            self.head_dim,
            rd,
            self.eps,
            quant_q=False,
            quant_k=False,
            swa_kv=self.swa_kv if is_decode else None,
            state_slot_mapping=state_slot_mapping if is_decode else None,
            batch_id_per_token=attn_md.batch_id_per_token if is_decode else None,
            swa_cu_seqlens_q=attn_md.cu_seqlens_q if is_decode else None,
            swa_cache_size=cache_size if is_decode else None,
            swa_write_per_batch=(
                min(attn_md.max_seqlen_q, cache_size) if is_decode else None
            ),
        )
        if _V4_USE_REF_QUANT:
            act_quant_inplace(kv[..., :-rd], 64, self.scale_fmt)

        # HCA
        if use_async_compress:
            current_stream = fc.main_stream
            if self.compressor is not None:
                current_stream.wait_stream(self.alt_stream)
            if self.indexer is not None:
                current_stream.wait_stream(self.indexer_stream)
        # ===== Compressor + Indexer =====
        if self.indexer is not None and not self.skip_topk:
            indexer_topk_batched = self.indexer.forward_batched(
                x_full=x,
                qr_full=qr,
                qr_full_scale=qr_scale,
                positions=positions,
            )
            # Translate seq-local topk → physical paged offsets and write into
            # the CSA section of either:
            #   - decode buffer `kv_indices_csa` (state is DECODE)
            #   - prefill buffer `kv_indices_prefix_csa` (otherwise)
            # `_fill_csa_paged_compress` dispatches internally on state.
            self._fill_csa_paged_compress(
                attn_md, indexer_topk_batched, positions, num_tokens
            )

        # ===== Sparse attention dispatch =====
        # Decode SWA write fires upstream of this dispatch via the
        # ``swa_write`` call in the decode branch — so ``paged_decode``
        # always sees the current token's K in the ring. Prefill does NOT
        # call swa_write from this layer (prior-chunk K is read from
        # ``unified_kv`` ring via the kv_indices_prefix_swa region).
        if is_decode:
            if ratio == 0:
                kv_indices = attn_md.kv_indices_swa
                kv_indptr = attn_md.kv_indptr_swa
            elif ratio == 4:
                kv_indices = attn_md.kv_indices_csa
                kv_indptr = attn_md.kv_indptr_csa
            else:  # ratio == 128
                kv_indices = attn_md.kv_indices_hca
                kv_indptr = attn_md.kv_indptr_hca
            o = sparse_attn_v4_paged_decode(
                q_sa,
                self.unified_kv,
                kv_indices,
                kv_indptr,
                self.attn_sink,
                self.softmax_scale,
            )  # [S, H, head_dim]
        else:
            # Two-source paged prefill: prefix from `unified_kv` (per-ratio
            # buffer with SWA history + compress section), extend from per-fwd
            # `kv` tensor (in-chunk SWA tail; extend buffer is layer-invariant).
            #
            # ===== PCP (full-KV) =====
            # Under PCP the model.forward entry round-robin-split x/positions to 1/W,
            # so `q_sa` and `kv` here are this rank's 1/W shard. The per-query
            # metadata (kv_indptr/indices_*, indexer_meta) was already reduced
            # to this rank's owned queries in the builder (_apply_pcp_reindex),
            # so `q_sa` + those indices are aligned and used as-is. The only
            # runtime fixups here are on the actual K/V data:
            #   - swa_write must write the FULL sequence SWA ring (every PCP
            #     rank keeps full KV), and
            #   - sparse_attn's extend source must be the FULL `kv` so each 1/W
            #     query can attend the whole in-chunk SWA window.
            # So all-gather `kv` back to full order; positions/cu_seqlens_q/
            # state_slot_mapping for the SWA write stay full (cu_seqlens_q /
            # state_slot_mapping are per-seq, never split; positions_full comes
            # from the forward context which holds the pre-split copy).
            pcp_on = _pcp_active()
            if pcp_on:
                pcp_ws = get_pcp_world_size()
                kv_full = pcp_allgather_rerange(kv, pcp_ws)
                # positions must match kv_full's full-sequence coords for the
                # swa_write ring addressing (`positions[src] % cache_size`).
                # `positions` here is this rank's 1/W shard (split in
                # ForCausalLM.forward); all-gather it back to full order with
                # the same rerange used for kv (NOT fc.context.positions, which
                # the builder reindexed to 1/W).
                positions_full = pcp_allgather_rerange(positions, pcp_ws)
            else:
                kv_full = kv
                positions_full = positions

            if ratio == 0:
                kv_indices_prefix = attn_md.kv_indices_prefix_swa
                kv_indptr_prefix = attn_md.kv_indptr_prefix_swa
            elif ratio == 4:
                kv_indices_prefix = attn_md.kv_indices_prefix_csa
                kv_indptr_prefix = attn_md.kv_indptr_prefix_csa
            elif ratio == 128:
                kv_indices_prefix = attn_md.kv_indices_prefix_hca
                kv_indptr_prefix = attn_md.kv_indptr_prefix_hca
            else:
                raise ValueError(f"Unsupported compress_ratio {ratio}")
            o = sparse_attn_v4_paged_prefill(
                q_sa,
                self.unified_kv,
                kv_indices_prefix,
                kv_indptr_prefix,
                kv_full,
                attn_md.kv_indices_extend,
                attn_md.kv_indptr_extend,
                self.attn_sink,
                self.softmax_scale,
            )  # [S, H, head_dim]
            # swa_write AFTER attn so chunked-prefill prefix SWA reads see
            # prior-chunk's ring contents (current swa_write would overwrite
            # ring slots `pos % cache_size` for positions in this chunk's tail).
            # PCP: write the FULL sequence SWA ring from the gathered kv_full +
            # full positions/cu_seqlens_q (full-KV scheme — every rank holds it).
            swa_write(
                kv_full,
                positions_full,
                attn_md.cu_seqlens_q,
                state_slot_mapping,
                self.swa_kv,
                cache_size,
                min(attn_md.max_seqlen_q, cache_size),
            )

        # Inverse RoPE on output's rope dims to remove absolute-position
        # contribution carried in by the value-side RoPE of the KV entries.
        self.rotary_emb.inverse(positions, o[..., -rd:])
        # ----- Grouped output LoRA (batched on the full flat tensor) -----
        o = o.view(num_tokens, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        y = torch.empty(
            num_tokens,
            self.n_local_groups,
            self.o_lora_rank,
            dtype=o.dtype,
            device=o.device,
        ).transpose(0, 1)
        y = batched_gemm_bf16(o.transpose(0, 1), wo_a, YQ=y)
        x = self.wo_b(y.transpose(0, 1).flatten(1))
        return x

    def _fill_csa_paged_compress(
        self,
        attn_md,
        topk_local_raw: torch.Tensor,
        positions: torch.Tensor,
        total_tokens: int,
    ) -> None:
        """Per-CSA-layer: translate indexer raw `topk_in_seq` → physical paged
        offsets in `unified_kv` and packed-write into the CSA section of the
        active prefix buffer.

        Dispatch:
          - state is DECODE → write into decode buffer `kv_indices_csa`,
                              skip = `window_size` (full SWA prefix per token)
          - prefill / mixed → write into prefill buffer `kv_indices_prefix_csa`,
                              skip = per-token `prefix_swa_count[t]`

        Per doc §6.4:
          block_idx_in_seq = topk_local // csa_block_capacity
          slot_in_block    = topk_local %  csa_block_capacity
          physical_block   = block_tables[batch_id_per_token[t], block_idx_in_seq]
          paged_offset     = swa_pages + physical_block * csa_block_capacity
                             + slot_in_block

        Fully fused into one triton kernel — no [T, index_topk] intermediates,
        no PyTorch fancy index. CG sentinel (batch_id=-1) and OOB clamp are
        handled in-kernel. The kernel derives per-token `valid_k` inline from
        `(positions[t]+1)//ratio` clamped by `n_committed_csa[bid]` and
        `index_topk`, matching Indexer's per-row visibility — so every
        reserved CSA cell gets written and no `-1` sentinel pre-fill is needed.

        Args:
          topk_local_raw: [total_tokens, index_topk] int32 — RAW seq-local
            output of `Indexer.forward_batched`. The leading `valid_k[t]`
            cells are always >= 0; trailing cells are -1 sentinels never
            read by csa_translate_pack (filtered by `k_offs < valid_k`).
          positions: [total_tokens] int — global token positions; forwarded
            to csa_translate_pack so the kernel can compute per-token
            `valid_k` inline.
        """
        # csa_block_capacity = block_size // ratio = 128 // 4 = 32.
        # Derived from constants (not `compressor.kv_cache.size(1)`) because
        # warmup runs before `build_kv_cache_tensor` binds compressor.kv_cache,
        # and this method now fires for both decode and prefill (including
        # warmup batches). Equivalent post-bind: `compressor.kv_cache.size(1)`.
        csa_block_capacity = _V4_BLOCK_SIZE // 4

        if attn_md.state is AttnState.DECODE:
            kv_indptr = attn_md.kv_indptr_csa
            kv_indices = attn_md.kv_indices_csa
            # Decode: skip = `actual_swa_count[t]` = min(pos+1, win) — derived
            # inline by the kernel, so the per-token buffer + its CPU build +
            # H2D in `_attach_v4_paged_decode_meta` are skipped.
            skip_buf = None
            window_size = self.window_size
        else:
            kv_indptr = attn_md.kv_indptr_prefix_csa
            kv_indices = attn_md.kv_indices_prefix_csa
            # Prefill: skip = `prefix_swa_count[t]` (chunked-prefill: depends
            # on `chunk_start[bid]`, not derivable from `positions[t]` alone)
            # — kernel loads from the per-token buffer.
            skip_buf = attn_md.skip_prefix_len_csa
            window_size = 0

        csa_translate_pack(
            topk_local_raw,
            attn_md.block_tables,
            positions,
            kv_indptr,
            attn_md.batch_id_per_token,
            skip_buf,
            kv_indices,
            swa_pages=attn_md.swa_pages,
            csa_block_capacity=csa_block_capacity,
            window_size=window_size,
        )


class Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN (w1, w2, w3). Computation in float32 for stability.

    Port of inference/model.py:587-606. With `swiglu_limit > 0`, clamps both gate
    and up projections (gate clipped above only, up clipped both sides) before
    the SiLU * up product — matches reference behavior exactly.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        swiglu_limit: float = 0.0,
        quant_config: Optional[Any] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        # Fused [w1; w3] (gate_up_proj): both share input x, both ColumnParallel
        # — standard llama/dsv2 fusion. Disk still split; routed via
        # packed_modules_mapping in DeepseekV4ForCausalLM.
        self.gate_up_proj = MergedColumnParallelLinear(
            dim,
            [inter_dim, inter_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.w2 = RowParallelLinear(
            inter_dim,
            dim,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.w2",
        )
        self.swiglu_limit = swiglu_limit
        # Switch: route clamp + silu(gate)*up [+ weights] + per-token FP8 1x128
        # quant through a single aiter triton kernel. The fused kernel emits
        # FP8 + scale; w2 accepts `x_scale` and skips its own quant step.
        self.use_fused_clamp_act_mul = _V4_USE_TRITON_FUSION

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, dim]
        weights: Optional[torch.Tensor] = None,  # [num_tokens, 1]  optional gate
    ) -> torch.Tensor:  # [num_tokens, dim]

        dtype = x.dtype
        # Single fused GEMM. Layout is [gate | up] concat on last dim — matches
        # aiter silu_and_mul's split([d, d], dim=-1) contract. The kernel does
        # silu/clamp/mul in fp32 internally regardless of input dtype, so we
        # feed the bf16 GEMM output directly.
        combined = self.gate_up_proj(x)  # [num_tokens, 2*inter_dim_per_tp]
        if self.use_fused_clamp_act_mul:
            x_fp8, x_scale = fused_clamp_act_mul(
                combined,
                swiglu_limit=self.swiglu_limit,
                activation="silu",
                weights=weights,
                dtype_quant=dtypes.fp8,
                transpose_scale=True,
            )
            return self.w2(x_fp8, x_scale=x_scale)
        out = torch.empty(
            (combined.shape[0], combined.shape[-1] // 2),
            dtype=dtype,
            device=combined.device,
        )
        # limit > 0 enables in-kernel clamp (gate≤limit, up∈[-limit,limit]) via
        # ROCm v_med3_f32 — same semantics as the prior torch.clamp pair.
        aiter_silu_and_mul(out, combined, self.swiglu_limit)
        if weights is not None:
            out = weights.to(dtype) * out
        return self.w2(out)  # [num_tokens, dim]


class MoE(nn.Module):
    """Mixture-of-Experts: top-k routed experts (FusedMoE) + 1 shared expert.

    PR3b: replaces the per-expert nn.Linear list with `FusedMoE` so 384 routed
    experts shard across TP/EP ranks and load FP4 weights via the existing
    `gemm_a4w4_quant` aiter kernel.

    Routing math (`sqrtsoftplus(scores) + bias` topk) is delegated to
    `FusedMoE.select_experts(scoring_func="sqrtsoftplus", e_score_correction_bias=...)`,
    which we extended in atom/model_ops/moe.py to add the V4 path.

    Hash routing for `layer_id < n_hash_layers` (first 3 V4 layers) is NOT yet
    wired through FusedMoE — the `tid2eid` buffer is declared so weight loading
    completes, but inference uses the standard sqrtsoftplus path. Hash layers
    will produce incorrect routing; correct hash routing lands in PR3+.
    """

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV4Args,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.prefix = prefix
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.is_hash_layer = layer_id < args.n_hash_layers
        self.routed_scaling_factor = args.route_scale
        self.swiglu_limit = args.swiglu_limit
        self.tp_size = get_tensor_model_parallel_world_size()
        self.alt_stream = alt_stream
        qc = args.quant_config

        self.gate = ReplicatedLinear(
            self.dim,
            self.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        # V4 hash-routed layers (layer_id < n_hash_layers) use tid2eid lookup,
        # not bias-corrected gate-logit routing — checkpoint has no
        # `gate.bias` for those layers. Only allocate the bias for
        # sqrtsoftplus layers to avoid 3 spurious unloaded-param warnings.
        if not self.is_hash_layer:
            self.gate.e_score_correction_bias = atom_parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32)
            )
        else:
            # tid2eid: per-token-id top-k expert lookup table (V4 first 3
            # layers use this in lieu of gate-logit routing).
            self.gate.tid2eid = atom_parameter(
                torch.empty(
                    args.vocab_size, args.n_activated_experts, dtype=torch.int32
                ),
            )
            # input_ids for hash routing is read from forward_context.context
            # (set by ModelRunner). torch.compile silently drops NNModule
            # attribute mutation across the compile boundary, so stashing on
            # `self.foo` from inside forward is a no-op at runtime.
        assert args.n_shared_experts == 1
        self._fuse_shared_into_routed = (
            is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
                qc,
                shared_expert_prefix=f"{prefix}.shared_experts",
                routed_expert_prefix=f"{prefix}.experts",
            )
        )
        moe_cfg = SimpleNamespace(
            routed_scaling_factor=self.routed_scaling_factor,
            n_shared_experts=(
                args.n_shared_experts if self._fuse_shared_into_routed else 0
            ),
        )
        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=self.n_activated_experts,
            hidden_size=self.dim,
            intermediate_size=args.moe_inter_dim,
            reduce_results=False,
            renormalize=True,
            quant_config=qc,
            use_grouped_topk=False,
            prefix=f"{prefix}.experts",
            scoring_func=args.score_func,  # "sqrtsoftplus"
            e_score_correction_bias=getattr(self.gate, "e_score_correction_bias", None),
            config=moe_cfg,
            shared_expert_prefix=f"{prefix}.shared_experts",
        )
        self.experts.swiglu_limit = args.swiglu_limit

        if not self._fuse_shared_into_routed:
            # self.experts.num_fused_shared_experts = 0
            self.shared_experts = Expert(
                args.dim,
                args.moe_inter_dim,
                swiglu_limit=args.swiglu_limit,
                quant_config=qc,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None
        if self.is_hash_layer:
            # Inject hash routing into FusedMoE.select_experts via the
            # custom_routing_function hook (added in atom/model_ops/moe.py).
            self.experts.custom_routing_function = self._hash_topk

        # Dual-stream: run shared_experts on `alt_stream` in parallel with
        # routed experts on the current stream. Mirrors V2's pattern. Only
        # active when shared_experts exist (not fused into routed) AND the
        # env threshold is positive AND we got an alt_stream from the model.
        # Per-call token count gating happens inside the custom op dispatcher
        # — prefill (large batch) skips dual-stream (overhead > benefit).
        self._use_dual_stream = (
            self.shared_experts is not None
            and self.alt_stream is not None
            and envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD > 0
        )
        if self._use_dual_stream:
            # Register self in static_forward_context so the custom op
            # dispatcher can look us up by `layer_name` (= self.prefix).
            get_current_atom_config().compilation_config.static_forward_context[
                prefix
            ] = self

    def _hash_topk(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """V4 hash routing for first 3 layers.

        topk_ids = tid2eid[input_ids]  (no gate-based selection)
        topk_weights = sqrtsoftplus(router_logits) gathered at topk_ids
        Then renormalize so weights sum to 1 per token.
        """
        fwd_input_ids = get_forward_context().context.input_ids
        assert (
            fwd_input_ids is not None
        ), "forward_context.context.input_ids is None — caller must invoke DeepseekV4ForCausalLM.forward, not DeepseekV4Model.forward directly."
        ids = fwd_input_ids.flatten()
        num_tokens = gating_output.shape[0]
        assert (
            ids.shape[0] == num_tokens
        ), f"input_ids length {ids.shape[0]} does not match gating_output num_tokens {num_tokens}"
        tid2eid = self.gate.tid2eid

        # Fused-shared expert: the custom_routing_function path bypasses
        # select_experts' shared-expert append, so the shared expert (slot
        # n_routed_experts) would never be routed and its ~40% contribution
        # dropped. When shared is fused, write the routed result into the first
        # `topk` columns of the global topK buffer (shared cols pre-filled) and
        # return the full [N, topk + n_shared] view.
        num_fused_shared = getattr(self.experts, "num_fused_shared_experts", 0)
        if num_fused_shared > 0:
            import atom.model_ops.topK as _topK_mod

            assert _topK_mod.aiter_topK_meta_data is not None, (
                "AITER topK meta data is not initialized. "
                "init_aiter_topK_meta_data must run before hash-layer routing."
            )
            total_topk_weights, total_topk_ids = _topK_mod.aiter_topK_meta_data
            assert total_topk_weights.shape[0] >= num_tokens
            hash_topk_triton(
                ids,
                gating_output,
                tid2eid,
                renormalize,
                self.routed_scaling_factor,
                total_topk_ids[:num_tokens, :topk],
                total_topk_weights[:num_tokens, :topk],
            )
            return total_topk_weights[:num_tokens], total_topk_ids[:num_tokens]

        topk_ids = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=gating_output.device
        )
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device=gating_output.device
        )
        hash_topk_triton(
            ids,
            gating_output,
            tid2eid,
            renormalize,
            self.routed_scaling_factor,
            topk_ids,
            topk_weights,
        )
        return topk_weights, topk_ids

    def routed_expert_forward(
        self, x: torch.Tensor  # [num_tokens, dim]
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Gate + FusedMoE routed-expert pass.

        For hash layers the gate's `tid2eid` lookup needs `input_ids`;
        `DeepseekV4ForCausalLM.forward` stashes it on
        `forward_context.context.input_ids` before each forward, and
        `_hash_topk` (FusedMoE's custom_routing_function) reads it there.
        """
        router_logits = self.gate(x)  # [num_tokens, n_routed_experts]
        return self.experts(hidden_states=x, router_logits=router_logits)

    @staticmethod
    def _gather_ids_for_dp(ids: torch.Tensor, ctx) -> torch.Tensor:
        """All-gather input_ids across DP ranks to match gathered hidden_states."""
        from aiter.dist.parallel_state import get_dp_group

        ids_2d = ids.unsqueeze(-1)
        dp_eager_mode = (
            not ctx.context.dp_uniform_decode
        ) and ctx.dp_metadata is not None
        if dp_eager_mode:
            from atom.model_ops.moe import all_gatherv

            sizes = ctx.dp_metadata.get_sizes_across_dp()
            ids_2d = all_gatherv(ids_2d, sizes, get_dp_group())
        else:
            from atom.model_ops.moe import all_gather_with_padding

            ids_2d, _ = all_gather_with_padding(ids_2d, use_cag=False)
        return ids_2d.flatten()

    def combine_outputs(
        self,
        routed: torch.Tensor,  # [num_tokens, dim]
        shared: Optional[torch.Tensor],  # [num_tokens, dim] or None
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Add shared-expert contribution (when not fused into routed) and
        all-reduce across TP ranks.
        """
        if shared is not None:
            routed = routed + shared
        if self.tp_size > 1:
            routed = tensor_model_parallel_all_reduce(routed)
        return routed

    def single_stream_moe_forward(
        self, x: torch.Tensor  # [num_tokens, dim]
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Sequential: shared_experts → routed_experts → combine."""
        shared = self.shared_experts(x) if self.shared_experts is not None else None
        routed = self.routed_expert_forward(x)
        return self.combine_outputs(routed, shared)

    def dual_stream_moe_forward(
        self, x: torch.Tensor  # [num_tokens, dim]
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Run shared_experts on `alt_stream` in parallel with routed_experts
        on the current stream. Mirrors V2's pattern. Both reads of `x` are
        independent; main stream waits on alt_stream's completion before
        combining.
        """
        current_stream = get_forward_context().main_stream
        self.alt_stream.wait_stream(current_stream)
        routed = self.routed_expert_forward(x)
        with torch.cuda.stream(self.alt_stream):
            shared = self.shared_experts.forward(x)
        current_stream.wait_stream(self.alt_stream)
        return self.combine_outputs(routed, shared)

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, dim]  hidden state (post ffn_norm)
    ) -> torch.Tensor:  # [num_tokens, dim]
        # Hash-layer routing reads `input_ids` from forward_context.context
        # inside `_hash_topk` (FusedMoE.custom_routing_function callback);
        # the MoE call itself doesn't need it as a parameter.
        assert (
            x.dim() == 2 and x.shape[-1] == self.dim
        ), f"MoE expects 2D [num_tokens, {self.dim}], got {tuple(x.shape)}"
        if self._use_dual_stream:
            # Shared custom op (also used by V2). Dispatcher reads
            # `_use_dual_stream` + per-call num_tokens vs threshold to pick
            # dual vs single. Custom op = Dynamo barrier so stream context
            # inside `dual_stream_moe_forward` is opaque to torch.compile.
            return torch.ops.aiter.maybe_dual_stream_forward(x, self.prefix)
        return self.single_stream_moe_forward(x)


@dataclass
class HCState:
    residual: torch.Tensor
    post_mix: Optional[torch.Tensor] = None
    comb_mix: Optional[torch.Tensor] = None
    x_prev: Optional[torch.Tensor] = None


class Block(nn.Module):
    """Transformer block with Manifold-Constrained Hyper-Connections (mHC).

    Port of inference/model.py:648-701. ATOM 2D-flat convention: the residual
    stream is widened to `[num_tokens, hc_mult, dim]`. Each sub-layer (attn / ffn):
      1. `hc_pre`: project `[num_tokens, hc_mult, dim]` → `[num_tokens, dim]` via
         Sinkhorn-projected pre-weights (also producing post-weights and combination
         matrix for hc_post).
      2. `attn_norm` + `attn` (or `ffn_norm` + `ffn`): standard sub-layer in
         `[num_tokens, dim]`.
      3. `hc_post`: expand `[num_tokens, dim]` back to `[num_tokens, hc_mult, dim]`
         using the post-weights (gate on the new contribution) + the combination
         matrix applied to the previous residual.
    """

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV4Args,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        indexer_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = DeepseekV4Attention(
            layer_id,
            args,
            prefix=f"{prefix}.attn",
            alt_stream=alt_stream,
            indexer_stream=indexer_stream,
        )
        self.ffn = MoE(layer_id, args, prefix=f"{prefix}.ffn", alt_stream=alt_stream)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        # All HC params stored in fp32 (matches reference's `set_dtype(torch.float32)`).
        self.hc_attn_fn = atom_parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        )
        self.hc_ffn_fn = atom_parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        )
        self.hc_attn_base = atom_parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = atom_parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = atom_parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = atom_parameter(torch.empty(3, dtype=torch.float32))

        # aiter mhc_pre/post kernels assert hidden % 512 == 0 OR hidden % 256 == 0
        # (mhc_kernels.cu:864 calls __builtin_trap on violation). Bind kernel refs
        # at init: present + dim-compatible → use fused; else None → torch fallback.
        # `x.is_cuda` is implicit here — model lives on GPU post-`.to()`; a CPU tensor
        # would have crashed earlier in DeepseekV4Attention.
        _dim_ok = args.dim % 512 == 0 or args.dim % 256 == 0
        self._mhc_pre = getattr(aiter, "mhc_pre", None) if _dim_ok else None
        self._mhc_post = getattr(aiter, "mhc_post", None) if _dim_ok else None
        self._mhc_fused_post_pre = (
            getattr(aiter, "mhc_fused_post_pre", None) if _dim_ok else None
        )
        self.enable_fused_hc = (
            hasattr(aiter, "mhc_fused_post_pre") and not self.layer_id == 0
        )

    # mHC `hc_post_mult_value`: V4 uses `2.0 * sigmoid(post)` for the post gate.
    HC_POST_MULT = 2.0

    def hc_pre(
        self,
        residual: torch.Tensor,  # [num_tokens, hc, dim]  mHC-widened residual
        hc_fn: torch.Tensor,  # [mix_hc, hc*dim]  fp32
        hc_scale: torch.Tensor,  # [3] fp32
        hc_base: torch.Tensor,  # [mix_hc] fp32
        norm_weight: Optional[torch.Tensor] = None,
        norm_eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce mHC residual `[num_tokens, hc, dim]` to sub-layer input `[num_tokens, dim]`.

        Prefers the fused aiter `mhc_pre` kernel (single ROCm op for RMSNorm +
        hc-fn linear + Sinkhorn projection + weighted reduction). Falls back to
        the torch `hc_split_sinkhorn` reference when the aiter kernel is
        unavailable or `dim` doesn't satisfy the `% 256/512 == 0` constraint.

        Returns:
          y:    [num_tokens, dim]      sub-layer input
          post: [num_tokens, hc]       post-gate weights for hc_post
          comb: [num_tokens, hc, hc]   combination matrix for hc_post
        """
        if self._mhc_pre is not None:
            # aiter mhc_pre wants [M, hc, dim] and returns
            # (post [M, hc, 1], comb [M, hc, hc], y [M, dim]).
            post, comb, y = self._mhc_pre(
                residual,
                hc_fn,
                hc_scale,
                hc_base,
                float(self.norm_eps),
                float(self.hc_eps),
                float(self.hc_eps),
                self.HC_POST_MULT,
                int(self.hc_sinkhorn_iters),
                norm_weight,
                norm_eps,
            )
            return y, post.squeeze(-1), comb

        # Torch fallback (no-aiter): mirrors the reference math.
        dtype = residual.dtype
        x_flat = residual.flatten(-2)  # [num_tokens, hc*dim]
        x_normed = _rmsnorm_nw(x_flat, self.norm_eps, x_flat.shape[-1])
        mixes = F.linear(x_normed.float(), hc_fn)  # [num_tokens, mix_hc]
        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = torch.sum(pre.unsqueeze(-1) * residual, dim=-2)  # [num_tokens, dim]
        if norm_weight is not None:
            y = F.rms_norm(y.float(), (y.shape[-1],), norm_weight.float(), norm_eps).to(
                dtype
            )
        return y.to(dtype), post, comb

    def hc_post(
        self,
        x: torch.Tensor,  # [num_tokens, dim]      sub-layer output
        residual: torch.Tensor,  # [num_tokens, hc, dim]  pre-layer residual
        post: torch.Tensor,  # [num_tokens, hc]       from hc_pre
        comb: torch.Tensor,  # [num_tokens, hc, hc]   from hc_pre
    ) -> torch.Tensor:  # [num_tokens, hc, dim]  new residual
        """Expand sub-layer output `[num_tokens, dim]` back to mHC residual
        `[num_tokens, hc, dim]`.

        Prefers the fused aiter `mhc_post` kernel; falls back to the torch
        reference when aiter is unavailable or shape constraints aren't met
        (kernel asserts hidden % 512 == 0 OR hidden % 256 == 0).

        See `/app/logs_claude/deepseek_v4/notes/12_aiter_mhc_post_root_cause.md`
        for past numerical-drift notes on long decode trajectories.
        """
        if self._mhc_post is not None:
            # `out` inherits residual.dtype = x.dtype (residual stream is BF16
            # end-to-end in Block.forward), so no cast needed on the kernel path.
            out = torch.empty_like(residual)
            self._mhc_post(out, x, residual, post.unsqueeze(-1), comb)
            return out

        # Torch fallback. fp32 (post, comb) × BF16 (x, residual) promotes to
        # fp32 by PyTorch promotion rules — cast back to x.dtype before return.
        # post.unsqueeze(-1) * x.unsqueeze(-2): [num_tokens, hc, dim] gating
        # comb.unsqueeze(-1) * residual.unsqueeze(-2): [num_tokens, hc, hc, dim]; sum over hc
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3
        )
        return y.type_as(x)

    def fuse_hc(
        self,
        hc_state: HCState,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: Optional[torch.Tensor] = None,
        norm_eps: float = 1e-6,
    ) -> HCState:
        residual = hc_state.residual
        post = hc_state.post_mix
        comb = hc_state.comb_mix
        x = hc_state.x_prev
        if self.enable_fused_hc and x is not None:
            post, comb, x, res = self._mhc_fused_post_pre(
                x,
                residual,
                post,
                comb,
                hc_fn,
                hc_scale,
                hc_base,
                float(self.norm_eps),
                float(self.hc_eps),
                float(self.hc_eps),
                self.HC_POST_MULT,
                int(self.hc_sinkhorn_iters),
                norm_weight,
                norm_eps,
            )
        else:
            if x is not None:
                res = self.hc_post(x, residual, post, comb)
            else:
                res = residual
            x, post, comb = self.hc_pre(
                res, hc_fn, hc_scale, hc_base, norm_weight, norm_eps
            )
        return HCState(residual=res, post_mix=post, comb_mix=comb, x_prev=x)

    def forward(
        self,
        hc_state: HCState,
        positions: torch.Tensor,  # [num_tokens] int  absolute token positions
    ) -> HCState:  # [num_tokens, hc, dim]  updated residual stream
        # ----- Attention sub-layer with mHC mixing -----
        hc_state = self.fuse_hc(
            hc_state,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.attn_norm.weight,
            self.norm_eps,
        )
        x = hc_state.x_prev
        x = self.attn(x, positions)  # [num_tokens, dim]
        hc_state.x_prev = x
        hc_state = self.fuse_hc(
            hc_state,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.ffn_norm.weight,
            self.norm_eps,
        )
        x = hc_state.x_prev
        x = self.ffn(
            x
        )  # [num_tokens, dim]  (input_ids read from forward_context for hash MoE)
        hc_state.x_prev = x
        return hc_state


class ParallelHead(ParallelLMHead):
    """V4 LM head with mHC reduction; vocab-parallel sharded across TP ranks.

    Port of inference/model.py:704-736. Inherits from `ParallelLMHead` so the
    vocab-axis sharding, `weight_loader`, last-token slicing, bf16 a16w16 GEMM
    (`tgemm.mm`), and TP all-gather come for free. V4 only adds:

    - `forward(...)` taking the mHC residual + hc_head params + final norm
    - `get_logits(...)` so `compute_logits` can call it directly on the
      hidden-state output of `model.forward` (CUDAGraph contract)
    - `hc_head(...)` Sigmoid-gated mHC reduction (vs `Block.hc_pre`'s Sinkhorn)

    Note on weight dtype: the V4 reference (model.py:713-714) keeps the LM head
    in fp32 because the disk weight is bf16; on AMD CDNA3/CDNA4 the bf16 MFMA
    instruction accumulates in fp32 natively, so a bf16 GEMM with the
    bf16-on-disk weight has the same effective precision as the reference's
    fp32 path while halving VRAM and using the faster a16w16 kernel.
    """

    def __init__(
        self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6
    ):
        super().__init__(vocab_size, dim, bias=False)
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps

    def get_logits(
        self, x: torch.Tensor  # [num_tokens, dim]
    ) -> torch.Tensor:  # [bs, vocab]
        """Project to vocab logits via the inherited `ParallelLMHead.forward`,
        which handles last-token slicing (prefill) + tgemm.mm + all-gather.
        """
        assert (
            x.dim() == 2 and x.shape[-1] == self.dim
        ), f"get_logits expects [num_tokens, {self.dim}], got {tuple(x.shape)}"
        return super().forward(x)

    def hc_head(
        self,
        x: torch.Tensor,  # [num_tokens, hc, dim]  mHC residual
        hc_fn: torch.Tensor,  # [hc, hc*dim]  fp32
        hc_scale: torch.Tensor,  # [1] fp32
        hc_base: torch.Tensor,  # [hc] fp32
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Reduce mHC residual `[num_tokens, hc, dim]` → `[num_tokens, dim]`
        via Sigmoid-gated weighted sum (vs Block.hc_pre's Sinkhorn variant).
        """
        _, _, y = aiter.mhc_pre(
            x, hc_fn, hc_scale, hc_base, self.norm_eps, self.hc_eps, sinkhorn_repeat=0
        )
        return y

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, hc, dim]
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: nn.Module,
    ) -> torch.Tensor:  # [bs, vocab]
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)  # [num_tokens, dim]
        # get_logits handles the per-rank vocab shard + all-gather internally.
        return self.get_logits(norm(x))  # [bs, vocab]


def _pcp_active() -> bool:
    """Whether to apply PCP round-robin-split in this forward.

    True only when pcp_size > 1 AND this is a real prefill forward (not decode,
    not dummy/warmup run). Decode runs PCP-redundant (full KV, no split); the
    warmup dummy run has no valid KV cache so it must skip the split path.
    """
    if get_pcp_world_size() <= 1:
        return False
    fc = get_forward_context()
    return fc.context.is_prefill and not fc.context.is_dummy_run


@support_torch_compile
class DeepseekV4Model(nn.Module):
    """Full model: embed -> expand to hc_mult copies -> N blocks -> hc_head -> logits.

    Port of inference/model.py:Transformer (770-810). MTP blocks live in the
    EagleProposer wrapper (`atom.models.deepseek_v4_mtp.DeepseekV4MTP`), not
    on this target. The ckpt's `mtp.*` weights are filtered out at
    `loader.load_model(spec_decode=False)` via the auto-detected
    `need_load_mtp` flag (target has no `mtp.*` params).
    """

    def __init__(
        self,
        *,
        atom_config: Config,
        args: DeepseekV4Args,
    ):
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.hc_mult = args.hc_mult

        # VocabParallelEmbedding shards along vocab dim. At TP=1 weight shape
        # equals nn.Embedding's [vocab_size, dim] so dummy state_dicts load
        # directly. At TP>1 each rank holds vocab_size/tp rows.
        self.embed = VocabParallelEmbedding(args.vocab_size, args.dim)
        # alt_stream: dual-stream MoE (shared_experts // routed_experts) AND
        # Main Compressor overlap. indexer_stream: Indexer Compressor overlap.
        # Both allocated once, shared across all blocks. Attention runs before
        # MoE in each block, so attn and MoE never contend for alt_stream.
        self.alt_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self.indexer_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self.layers = nn.ModuleList(
            [
                Block(
                    layer_id,
                    args,
                    prefix=f"layers.{layer_id}",
                    alt_stream=self.alt_stream,
                    indexer_stream=self.indexer_stream,
                )
                for layer_id in range(args.n_layers)
            ]
        )
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)

        # Top-level hc_head params used to reduce the final hc_mult residual stack
        # before the LM head linear projection.
        hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = atom_parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = atom_parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = atom_parameter(torch.empty(1, dtype=torch.float32))

    def forward(
        self,
        input_ids: torch.Tensor,  # [num_tokens] int  flat ragged-batch token ids
        positions: torch.Tensor,  # [num_tokens] int  abs positions (required)
    ) -> torch.Tensor:  # [num_tokens, hc, dim]  pre-hc_head residual stream
        """Forward over `num_tokens` flat ragged-batch tokens.

        Returns the mHC residual stack `[num_tokens, hc, dim]` BEFORE hc_head
        reduction — `hc_head + RMSNorm + LM head` are all deferred to
        `compute_logits`. Returning the hc-shaped residual lets the (future)
        MTP draft consume it without re-expanding from a dim-reduced state.
        """
        assert input_ids.dim() == 1, f"input_ids must be 1D, got {input_ids.shape}"
        # PCP note: under PCP, `input_ids`/`positions` arrive already round-robin-
        # split to this rank's 1/W shard (done in DeepseekV4ForCausalLM.forward,
        # OUTSIDE the torch.compile boundary — keeping comms / dynamic padding
        # out of the compiled graph). So everything here runs on the 1/W shard;
        # the K/V all-gather inside attention reconstructs full KV per layer,
        # and the final all-gather + un-pad happens back in the caller.
        h = self.embed(input_ids)  # [num_tokens, dim]
        # Expand to hc_mult copies for Hyper-Connections: [num_tokens, hc, dim]
        h = h.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        hc_state = HCState(residual=h, post_mix=None, comb_mix=None, x_prev=None)

        for layer in self.layers:
            hc_state = layer(hc_state, positions)  # [num_tokens, hc, dim]
        h = self.layers[-1].hc_post(
            hc_state.x_prev, hc_state.residual, hc_state.post_mix, hc_state.comb_mix
        )
        return h


class DeepseekV4ForCausalLM(nn.Module):
    """ATOM model contract wrapper.

    Loads via two paths:
    - `model.load_weights(...)` (this file): used by tests + when ModelRunner
      is bypassed. Handles V4 ckpt naming + FP8 wo_a dequant + FusedMoE expert
      dispatch in one place.
    - `atom.model_loader.loader.load_model(...)` (standard ATOM serving): uses
      the `weights_mapping` class attribute below to rename V4 ckpt names into
      shapes the standard FusedMoE expert mapping understands. Wo_a dequant
      and other special cases are handled by the `process_weights_after_loading`
      path on the relevant Linear modules (TODO PR4).
    """

    # Disk-name → param-name renames applied by atom.model_loader.loader.load_model.
    #
    # We use a `WeightsMapper` (prefix/suffix-anchored) for the `model.` prefix
    # injection because the V4 HF checkpoint stores bare names (`norm.weight`,
    # `head.weight`, `embed.weight`, `layers.X.*`, `hc_head_*`, `mtp.X.*`) and
    # our model lives under `self.model = DeepseekV4Model(...)` so all params
    # are accessed via `model.<name>`. The legacy `weights_mapping` substring
    # dict CANNOT express this safely: `"norm.weight" → "model.norm.weight"`
    # also matches inside `attn_norm.weight` / `ffn_norm.weight` / `q_norm.weight`
    # / `compressor.norm.weight` etc. and silently corrupts the lookup
    # (b87f6f, debugged via the `load_model` post-load WARNING).
    #
    # The substring dict is reserved for the renames that ARE legitimately
    # substring-shaped:
    # - `.gate.bias` → `.gate.e_score_correction_bias` (V4's routed-expert
    #   score correction bias has a different name in our model)
    # - `.scale` → `.weight_scale_inv` (V4 ckpt suffix → ATOM's expected name;
    #   load_model then auto-renames `_inv` → `` so the final param is
    #   `.weight_scale`).
    weights_mapper = WeightsMapper(
        orig_to_new_prefix={
            "embed.": "model.embed.",
            "layers.": "model.layers.",
            "norm.weight": "model.norm.weight",
            "head.weight": "model.head.weight",
            "hc_head_": "model.hc_head_",
        }
    )
    weights_mapping = {
        ".gate.bias": ".gate.e_score_correction_bias",
        ".scale": ".weight_scale_inv",
    }
    packed_modules_mapping = {
        "attn.wq_a": ("attn.wqkv_a", 0),
        "attn.wkv": ("attn.wqkv_a", 1),
        "compressor.wkv": ("compressor.wkv_gate", 0),
        "compressor.wgate": ("compressor.wkv_gate", 1),
        "shared_experts.w1": ("shared_experts.gate_up_proj", 0),
        "shared_experts.w3": ("shared_experts.gate_up_proj", 1),
    }

    def __init__(self, config: Config, prefix: str = "") -> None:
        super().__init__()
        self.atom_config = config
        self.hf_config = config.hf_config
        self.args = DeepseekV4Args.from_hf_config(self.hf_config)
        # Build the V4-specific QuantizationConfig (FP8 default + FP4 experts +
        # BF16 wo_a/Compressor) so child Linear layers auto-build the right
        # weight + scale params for real-checkpoint loading. When the HF
        # config lacks `quantization_config` (e.g. dummy / toy validation),
        # this still works — base spec is QuantType.No.
        #
        # Forward the engine-level `online_quant_config` (set via
        # `--online_quant_config` CLI) so V4 weights can be re-quantized at
        # load time. Without this, the engine flag is silently dropped on V4.
        self.args.quant_config = make_v4_quant_config(
            self.hf_config,
            model_path=getattr(config, "model", None),
            online_quant_config=getattr(config, "online_quant_config", None),
        )
        self.atom_config.quant_config = self.args.quant_config
        self.model = DeepseekV4Model(atom_config=config, args=self.args)
        # Tell ModelRunner to size the CG outputs buffer as
        # [max_num_batched_tokens, hc_mult, hidden_size] instead of the
        # default [max_num_batched_tokens, hidden_size]. forward returns
        # the un-reduced mHC residual stack [N, hc, dim].
        self.extra_output_dims: tuple[int, ...] = (self.args.hc_mult,)
        self._need_ids_gather = (
            config.enable_dp_attention
            and not config.enable_expert_parallel
            and self.args.n_hash_layers > 0
        )

    @property
    def disable_fused_shared_loading(self) -> bool:
        """True when shared experts are NOT fused into the routed MoE kernel, so
        the weight loader must keep `ffn.shared_experts.*` on the standalone
        Expert module instead of rewriting them into the fused slot. Read from
        the actual built MoE layers so it always agrees with model structure.
        """
        for m in self.model.modules():
            if m.__class__.__name__ == "MoE":
                return not getattr(m, "_fuse_shared_into_routed", True)
        return False

    def forward(
        self,
        input_ids: torch.Tensor,  # [num_tokens] int
        positions: torch.Tensor,  # [num_tokens] int  required
    ) -> torch.Tensor:  # [num_tokens, dim]  hidden_states
        # Stash input_ids on forward_context for the V4 hash MoE routing
        # callback (`MoE._hash_topk`), which runs inside the Dynamo-opaque
        # `maybe_dual_stream_forward` custom op and can't receive input_ids
        # via the FusedMoE custom-routing-function fixed signature. Setting
        # it here (rather than in ModelRunner) means any caller of
        # `model.forward` — production runner, warmup, benchmarks — gets
        # correct hash routing without a separate setup step.
        ctx = get_forward_context()

        # ===== PCP: round-robin-split the prefill sequence OUTSIDE torch.compile =====
        # PCP splits the prefill query sequence across the PCP group (full-KV
        # scheme). This must happen here in ForCausalLM.forward (NOT in the
        # @support_torch_compile-wrapped DeepseekV4Model.forward) so the
        # cross-rank all-gather + data-dependent padding stay out of the
        # compiled graph (Dynamo mishandles comms / dynamic shapes -> shape
        # desync). Mirrors SGLang, which does cp_round_robin on input_ids in
        # the un-compiled ForCausalLM.forward. We pad tokens to a multiple of
        # pcp_size (dummy tokens, zero-length KV in the builder metadata), then
        # round-robin-split input_ids/positions to this rank's 1/W shard. The model
        # runs entirely on 1/W; the final hidden is all-gathered + un-padded
        # after self.model(...) returns.
        use_pcp = _pcp_active()
        if use_pcp:
            pcp_size = get_pcp_world_size()
            n_global = input_ids.shape[0]
            pad = pcp_pad_len(n_global, pcp_size) - n_global
            if pad > 0:
                input_ids = torch.cat([input_ids, input_ids.new_zeros(pad)], dim=0)
                positions = torch.cat([positions, positions.new_zeros(pad)], dim=0)
            input_ids = pcp_round_robin_split(input_ids, pcp_size)
            positions = pcp_round_robin_split(positions, pcp_size)

        if self._need_ids_gather:
            # DP-attention (no EP) hash routing: input_ids is local but the MoE
            # gate sees DP-gathered gating_output, so gather ids to match. Run
            # the gather INLINE on the compute stream. Running this all-gather on
            # a side stream coordinated it with a DIFFERENT stream/sync than the
            # MoE hidden/router DP gather under TBO → mismatched DP layouts →
            # wrong V4 hash routing (GSM8K 0.95→0.87). NOTE: do NOT wrap this in
            # the TBO ping-pong
            # (tbo_yield_and_switch_*) — injecting an extra yield at forward top
            # desyncs the ping-pong ring and collapses accuracy to ~0.54
            # (measured). The ids tensor is [N,1] int (tiny vs hidden [N,7168]),
            # so inline costs ~nothing in overlap.
            ctx.context.input_ids = MoE._gather_ids_for_dp(input_ids.flatten(), ctx)
        else:
            ctx.context.input_ids = input_ids
        h = self.model(input_ids, positions)

        # ----- PCP: all-gather shards, restore original order, drop pad -----
        if use_pcp:
            h = pcp_allgather_rerange(h, pcp_size)
            if pad > 0:
                h = h[:n_global]
        return h

    def compute_logits(
        self,
        hidden_states: torch.Tensor,  # [num_tokens, hc, dim]  pre-hc_head residual
    ) -> torch.Tensor:  # [bs, vocab]
        # mHC reduce + final RMSNorm + LM head are all here so `model.forward`
        # can return the un-reduced [N, hc, dim] residual stream — the future
        # MTP draft consumes it directly without re-expanding from a dim-reduced
        # state. CG output buffer is sized [N, hc, dim] in ModelRunner via the
        # `extra_output_dims = (hc_mult,)` hook on this class.
        x = self.model.head.hc_head(
            hidden_states,
            self.model.hc_head_fn,
            self.model.hc_head_scale,
            self.model.hc_head_base,
        )
        x = self.model.norm(x)
        return self.model.head.get_logits(x)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Return (param_name, weight_name, expert_id, shard_id) tuples for FusedMoE.

        V4 expert weights on disk are named `ffn.experts.{e}.w{1,2,3}`. Pass
        these as the gate/down/up names to FusedMoE.make_expert_params_mapping.

        When fused shared expert is enabled, FusedMoE allocates one extra expert
        slot (id = n_routed_experts) for the shared expert. Include it in the
        mapping so the loader can dispatch `ffn.shared_experts.w*` (rewritten to
        `ffn.experts.{n_routed_experts}.w*` by the loader) into that slot.
        Otherwise the shared expert weights are dropped and slot N stays
        uninitialized -> garbage MoE output.
        """
        # Whether the shared expert is fused into the routed buffer is decided
        # per-MoE-layer (`_fuse_shared_into_routed`). Read the ACTUAL allocated
        # buffer state from a real MoE layer instead of the global
        # `is_rocm_aiter_fusion_shared_expert_enabled()` — otherwise when fusion
        # is disabled (buffer=256) the mapping would emit 257 entries and
        # mis-load expert weights -> garbage.
        num_fused_shared = 0
        for _m in self.model.modules():
            if _m.__class__.__name__ == "FusedMoE":
                num_fused_shared = getattr(_m, "num_fused_shared_experts", 0)
                break
        num_experts = self.args.n_routed_experts + num_fused_shared
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from an iterable of (name, tensor) pairs.

        Naming conventions (HF V4 checkpoint matches our internal naming 1:1):
            embed.weight
            layers.{i}.attn.{wq_a,q_norm,wq_b,wkv,kv_norm,wo_a,wo_b,attn_sink,...}
            layers.{i}.attn.compressor.{ape,wkv,wgate,norm}
            layers.{i}.attn.indexer.{wq_b,weights_proj}
            layers.{i}.attn.indexer.compressor.{...}
            layers.{i}.ffn.gate.{weight,bias|tid2eid}
            layers.{i}.ffn.experts.{e}.w{1,2,3}
            layers.{i}.ffn.shared_experts.w{1,2,3}
            layers.{i}.{attn_norm,ffn_norm}
            layers.{i}.{hc_attn_*,hc_ffn_*}
            mtp.{i}.{...}                    (same shape as a Block + e_proj/h_proj/...)
            norm.weight, head.weight, hc_head_*

        On-disk quirks:
        - FP8/FP4 scale tensors are named `<param>.scale`; ATOM internally names
          them `<param>.weight_scale`. Remap on lookup.
        - `wo_a` is FP8 + scale on disk but BF16 in our model (V4QuantConfig
          forces no_spec; aiter has no FP8 grouped-einsum). Dequantize the FP8
          weight using the on-disk scale before copying into the BF16 param.

        Returns:
            Set of parameter names successfully loaded.
        """
        loaded: set[str] = set()
        # Index all our params + buffers for fast lookup.
        targets: dict[str, torch.Tensor] = dict(self.model.named_parameters())
        targets.update(dict(self.model.named_buffers()))

        # First pass: bucket on-disk tensors by their candidate target names.
        # Some special-case tensors (wo_a.weight + wo_a.scale → BF16) need to be
        # processed together, so collect all tensors first then resolve.
        scratch: dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            scratch[name] = tensor

        # ----- FusedMoE expert weight dispatch (PR3b) -----
        # Routed expert weights `layers.{i}.ffn.experts.{e}.w{1,2,3}.{weight,scale}`
        # on disk go to FusedMoE's merged `experts.w13_*` / `experts.w2_*` params.
        # The mapping uses substring substitution: `experts.{e}.w1.` (weight_name_part)
        # → `experts.w13_` (param_name_part), keeping the `weight` / `scale` suffix.
        try:
            expert_mapping = self.get_expert_mapping()
        except Exception:
            expert_mapping = []
        # Build longest-first index for unambiguous matching (shared with std loader).
        expert_index: dict[str, tuple[str, int, str]] = {}
        for param_part, weight_part, expert_id, shard_id in expert_mapping:
            expert_index[weight_part] = (param_part, expert_id, shard_id)
        weight_parts_sorted = sorted(expert_index.keys(), key=len, reverse=True)

        consumed: set[str] = set()
        for ckpt_name in list(scratch.keys()):
            if "ffn.experts." not in ckpt_name and "experts." not in ckpt_name:
                continue
            # Skip the routed-gate/non-expert tensors that just live alongside.
            for wpart in weight_parts_sorted:
                if wpart not in ckpt_name:
                    continue
                ppart, expert_id, shard_id = expert_index[wpart]
                tgt_name = ckpt_name.replace(wpart, ppart)
                # FusedMoE expert scales: on-disk `.{shard_id}.scale` → param `_weight_scale`
                # After substring sub `experts.{e}.w1.` → `experts.w13_`, the suffix
                # becomes `_scale`; rename to match FusedMoE's `_weight_scale` param.
                if tgt_name.endswith("_scale"):
                    tgt_name = tgt_name[: -len("_scale")] + "_weight_scale"
                elif tgt_name.endswith(".scale"):
                    tgt_name = tgt_name[: -len(".scale")] + ".weight_scale"
                param = targets.get(tgt_name)
                if param is None:
                    break
                loader = getattr(param, "weight_loader", None)
                if loader is None:
                    break
                tensor = scratch[ckpt_name].to(param.device)
                # Dtype glue:
                # - FP4 packed weights: disk is int8, param is float4_e2m1fn_x2;
                #   FusedMoE._load_w13/w2 already does `.view(torch.uint8)` for fp4x2
                #   params, but only when the loaded tensor dtype matches.
                # - FP8 e8m0 scale: disk is float8_e8m0fnu, param is uint8;
                #   torch's copy_ between mismatched dtypes silently zeros, so
                #   force a uint8 view here.
                if tensor.dtype == torch.float8_e8m0fnu and param.dtype == torch.uint8:
                    tensor = tensor.view(torch.uint8)
                if tensor.dtype == torch.int8 and param.dtype == torch.float4_e2m1fn_x2:
                    tensor = tensor.view(torch.uint8)
                loader(
                    param,
                    tensor,
                    tgt_name,  # weight_name (post-mapping; "scale" substring drives scale dispatch)
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded.add(tgt_name)
                consumed.add(ckpt_name)
                break
        # Drop consumed expert tensors so the second loop doesn't re-process them.
        for k in consumed:
            scratch.pop(k, None)

        for tgt_name, param in targets.items():
            ckpt_name = tgt_name
            # ATOM scale → on-disk scale name
            if ckpt_name.endswith(".weight_scale"):
                alt = ckpt_name.replace(".weight_scale", ".scale")
                if alt in scratch:
                    ckpt_name = alt
            # ATOM `gate.e_score_correction_bias` ↔ on-disk `gate.bias`
            if ckpt_name.endswith(".gate.e_score_correction_bias"):
                alt = ckpt_name.replace(".gate.e_score_correction_bias", ".gate.bias")
                if alt in scratch:
                    ckpt_name = alt
            if ckpt_name not in scratch:
                continue

            # NOTE: previously wo_a had a manual FP8+scale → BF16 dequant special
            # case here. wo_a is now FP8 ColumnParallelLinear in the model so
            # weight + scale load through the standard FP8 path. Dequant happens
            # in DeepseekV4Attention.process_weights_after_loading (called via the
            # post-load hook walk at the end of this method).

            tensor = scratch[ckpt_name].to(param.device)

            # Shape mismatch handling:
            # - When test caps n_routed_experts (e.g. 8 vs disk 384), the on-disk
            #   gate.weight/bias are larger than param. Slice to the first N rows.
            #   Real serving uses full 384 so this is a no-op there.
            # - Other shape mismatches indicate a true wiring bug → skip safely.
            if param.shape != tensor.shape:
                can_slice = param.dim() == tensor.dim() and all(
                    ps <= ts for ps, ts in zip(param.shape, tensor.shape, strict=True)
                )
                if can_slice:
                    slices = tuple(slice(0, s) for s in param.shape)
                    tensor = tensor[slices].contiguous()
                else:
                    continue

            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                loader(param, tensor)
            else:
                if (
                    param.dtype != tensor.dtype
                    and param.dtype == torch.float4_e2m1fn_x2
                ):
                    param.data.view(torch.uint8).copy_(tensor.view(torch.uint8))
                else:
                    param.data.copy_(tensor.to(param.dtype))
            loaded.add(tgt_name)

        # Trigger post-load hooks (e.g. FusedMoE's `process_weights_after_loading`
        # runs `shuffle_weights` so aiter ck_moe sees the right layout). Without
        # this the FP4 ck_moe kernel reads stale layout → HSA crash at forward.
        for module in self.model.modules():
            ppl = getattr(module, "process_weights_after_loading", None)
            if callable(ppl):
                # quant_method.process_weights_after_loading(layer) — quant_method
                # is the FusedMoE attribute, layer is the module itself.
                qm = getattr(module, "quant_method", None)
                if qm is not None and hasattr(qm, "process_weights_after_loading"):
                    qm.process_weights_after_loading(module)
                else:
                    ppl()
        return loaded
