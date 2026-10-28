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
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import lru_cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

if TYPE_CHECKING:
    from atom.model_ops.attentions.deepseek_v4_attn import AttentionMetaData_DSV4

import aiter
import torch
import torch.nn.functional as F
from aiter import (
    QuantType,
    dtypes,
)
from aiter import silu_and_mul as aiter_silu_and_mul
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.batched_gemm_op_a8w8 import (
    batched_gemm_a8w8_mxscale,
    batched_gemm_a8w8_mxscale_bpreshuffle,
)
from aiter.ops.inverse_rope_group_quant import inverse_rope_group_quant
from aiter.ops.triton.fusions.fused_clamp_act_mul import (
    fused_clamp_act_mul,
)
from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16
from torch import nn

from atom.config import (
    Config,
    LayerQuantConfig,
    QuantizationConfig,
    get_current_atom_config,
)
from atom.distributed.pcp_utils import (
    get_pcp_world_size,
    pcp_allgather_rerange,
    pcp_pad_len,
    pcp_round_robin_split,
)
from atom.model_loader.loader import WeightsMapper
from atom.model_loader.weight_utils import local_model_dir

# Side-effect import: registers `torch.ops.aiter.maybe_dual_stream_forward`
# (shared with deepseek_v2) and `torch.ops.aiter.indexer_score_topk` (V4-only).
# MoE.forward dispatches via the former so torch.compile/Dynamo treats stream
# code as opaque; Indexer.forward_batched dispatches via the latter to hide
# its dynamic-shape internals from Dynamo / fake-tensor mode.
from atom.model_ops import module_dispatch_ops as _module_dispatch_ops  # noqa: F401
from atom.model_ops.attentions.pool_layout.v4_pool_fields import (
    FP4_GFX1250_NATURAL,
    fp4_indexer_layout_for_arch,
)
from atom.model_ops.communication_op import (
    tensor_model_parallel_all_reduce,
)
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.indexer import (
    NATURAL,
    PRESHUFFLE,
    IndexCache,
    fold_query_scale,
    quantize_query,
    score_topk,
)
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
from atom.model_ops.topK import topk_meta_rows
from atom.model_ops.triton_hash_topk import hash_topk_triton
from atom.model_ops.triton_rmsnorm_nw import rmsnorm_nw
from atom.model_ops.utils import atom_parameter, shuffle_weights
from atom.model_ops.v4_kernels import (
    CompressPlan,
    QKNormRopeOut,
    csa_translate_pack,
    fp4_indexer_enabled,
    fused_compress_attn,
    hca_persist,
    inverse_rope_inplace,
    qk_norm_rope_maybe_quant,
    sparse_attn_v4_paged_2src,
    sparse_attn_v4_paged_decode,
    swa_write,
    update_compressor_states,
)
from atom.utils import envs, mark_spliting_op
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.decorators import mark_trace, support_torch_compile
from atom.utils.forward_context import (
    AttnState,
    get_forward_context,
    pass_scheduled_tokens_gpu,
)
from atom.utils.region_graph import token_graph
from atom.utils.tbo.ubatching import (
    tbo_active,
    tbo_switch_to_compute_sync,
    tbo_yield_and_switch_from_compute_to_comm,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classical KV cache scatter / gather helpers (PR3-pre2c-B).
#
# Each V4 block (block_size=2*lcm(m, m')=256 original tokens) holds k_per_block
# compressed rows per layer (64 for CSA, 2 for HCA). Compressor.forward
# scatters newly-compressed entries into block-table-indexed slots; sparse_attn
# input gathers all committed entries up to the current position.
#
# In PR3-pre2c-B these helpers run on a single sequence (block_table fetched
# from `forward_context.attn_metadata.block_tables[0]`). PR3-main extends to
# per-seq dispatch.
# ---------------------------------------------------------------------------

# V4 paper §3.6.1: classical-KV block_size = a multiple of lcm(m, m'). For
# V4-Pro / V4-Flash lcm(4, 128) = 128; we use 2*lcm = 256 original tokens so
# a CSA layer keeps 256/4 = 64 rows per block (the FP4 indexer kernels need
# kv_block_size=64). Kept as
# a constant so Compressor code does not need to import the builder. MUST match
# DeepseekV4AttentionMetadataBuilder.block_size and config.kv_cache_block_size.
_V4_BLOCK_SIZE: int = 256

_V4_RMSNORM_BACKEND = os.environ.get("ATOM_V4_RMSNORM_BACKEND", "triton")
_V4_USE_TRITON_RMSNORM = _V4_RMSNORM_BACKEND == "triton"
# Env-gated quant round-trips. Read once at module load — checking each
# forward burns syscalls (V4-Pro: 64 layers × multiple sites per call).
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
    # WIDE split (legacy / FULL cudagraph path): the whole attention (pre + core
    # + post) is a single eager splitting op. Used when NOT doing TR_HYBRID
    # cudagraph — the manual whole-forward FULL capture graphs everything at once
    # so there are no inter-piece tensors to stabilise. Unchanged from baseline.
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    return self.forward_impl(x, positions)


# ---------------------------------------------------------------------------
# Narrow split (TR_HYBRID): the compressor launch is the split op; the rest of
# the attention stays in the dense pieces on either side. A split for the
# compiler only -- its plan is sized by the token count, so a token graph
# records it inline.
# ---------------------------------------------------------------------------


def _v4_attn_compress_launch_fake(x: torch.Tensor, layer_name: str) -> None:
    return None


@mark_spliting_op(
    is_custom=True, gen_fake=_v4_attn_compress_launch_fake, mutates_args=[]
)
def v4_attn_compress_launch(x: torch.Tensor, layer_name: str) -> None:
    """The compressors' launch, opaque to Dynamo: their fused kernels JIT-build
    on first call, which Dynamo cannot trace. Called ahead of the Q/KV
    projections so the side streams overlap them; `_sparse_attention` joins.

    It returns NOTHING, and does not need to: a split op's submodule is the one
    piece the backend leaves uncompiled (`backends.py`, `submod_names_to_compile`
    excludes `is_splitting_graph`), so it never reaches AOT autograd, which is
    the layer that DCEs an effect-free call -- measured: survives under
    `backend="eager"`, dropped under `aot_eager` and `inductor`. Ordering comes
    from `split_graph`'s `keep_original_order=True` and the sequential submodule
    calls it generates, not from a data edge. A regular custom op in a compiled
    piece would need one; this is not that.
    """
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    self._launch_compressors(x)


def _v4_sparse_attention_fake(
    q_sa: torch.Tensor | None,
    kv: torch.Tensor | None,
    q_packed: torch.Tensor | None,
    q_rope: torch.Tensor | None,
    k_packed: torch.Tensor | None,
    k_rope: torch.Tensor | None,
    positions: torch.Tensor,
    idx_q_quant: torch.Tensor | None,
    idx_weights: torch.Tensor | None,
    idx_q_scale: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    # Off the Q, matching the body. Sizing this by `positions` disagrees with it
    # whenever the two token counts differ -- see `_sparse_attention`.
    q_rows = q_sa if q_sa is not None else q_packed
    return q_rows.new_empty(
        (q_rows.shape[0], self.n_local_heads * self.head_dim),
        dtype=torch.bfloat16,
    )


def v4_sparse_attention(
    q_sa: torch.Tensor | None,
    kv: torch.Tensor | None,
    q_packed: torch.Tensor | None,
    q_rope: torch.Tensor | None,
    k_packed: torch.Tensor | None,
    k_rope: torch.Tensor | None,
    positions: torch.Tensor,
    idx_q_quant: torch.Tensor | None,
    idx_weights: torch.Tensor | None,
    idx_q_scale: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    """`_sparse_attention` as a Dynamo-OPAQUE op: CSA pack + the paged attention.

    A REGULAR custom op, not a splitting one. Opacity is the point -- Dynamo
    never traces the body, so the paged kernels inside are reachable from a
    compiled dense piece at all, and the `attn_metadata` they read is live every
    forward rather than frozen to the warmup trace (~16k tokens). The paged
    attention only ever sat in the eager core because torch.compile could not
    trace it; its launch is `N = qo_indptr.numel()-1`, pure token count, so a
    num_tokens-keyed piece holds it correctly.

    Nothing is declared mutated: what it writes (the CSA indices, prefill's SWA
    ring) is read by the next layer, and its return is consumed by
    `_attn_post`, which is the data edge that orders it and blocks DCE.
    """
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    qkn = QKNormRopeOut(
        q_sa=q_sa,
        kv=kv,
        q_packed=q_packed,
        q_rope=q_rope,
        k_packed=k_packed,
        k_rope=k_rope,
    )
    return self._sparse_attention(qkn, positions, idx_q_quant, idx_weights, idx_q_scale)


direct_register_custom_op(
    op_name="v4_sparse_attention",
    op_func=v4_sparse_attention,
    mutates_args=[],
    fake_impl=_v4_sparse_attention_fake,
)


def _qkn_placeholder(layer, q: torch.Tensor, num_tokens: int, *, zeros: bool):
    """A stand-in `QKNormRopeOut`: right shapes, no content.

    One source for the two places that need the shapes without doing the work --
    the op's fake impl (tracing) and the dummy_run short-circuit (warmup, before
    the KV planes are bound). They MUST agree, or the compiled graph fails on
    `assert_size_stride`. `zeros` for warmup, whose output is consumed
    downstream and so has to be finite rather than `empty`'s garbage.

    Which fields are populated is the kv-cache layout; `kv_fp8` is frozen at
    `__init__`, so the shape is fixed per layer.
    """
    from atom.model_ops.v4_kernels.v4_quant import V4_DIM_QK_PACKED, V4_DIM_ROPE

    alloc = q.new_zeros if zeros else q.new_empty
    h, d = layer.n_local_heads, layer.head_dim
    if layer.kv_fp8:
        # The 2buff packed width is NOT `head_dim - rope_head_dim`: that is
        # `V4_DIM_NOPE` (448), and the packed row is the NoPE fp8 plus its inline
        # e8m0 scale plus padding, `V4_DIM_QK_PACKED` (512). Deriving it instead
        # of naming it produced a fake 448 wide against a body 512 wide, which
        # surfaced as `assert_size_stride` inside the compiled graph on an
        # fp8-KV run. These are the same constants `sparse_attn_v4_paged_decode`
        # asserts its Q against, so take them from there.
        return QKNormRopeOut(
            q_packed=alloc((num_tokens, h, V4_DIM_QK_PACKED), dtype=dtypes.fp8),
            q_rope=alloc((num_tokens, h, V4_DIM_ROPE)),
            k_packed=alloc((num_tokens, 1, V4_DIM_QK_PACKED), dtype=dtypes.fp8),
            k_rope=alloc((num_tokens, 1, V4_DIM_ROPE)),
        )
    return QKNormRopeOut(q_sa=alloc((num_tokens, h, d)), kv=alloc((num_tokens, d)))


def _v4_qk_norm_rope_fake(
    q: torch.Tensor,
    kv_pre: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> list[torch.Tensor]:
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    return _qkn_placeholder(self, q, q.shape[0], zeros=False).custom_op_return()


def v4_qk_norm_rope(
    q: torch.Tensor,
    kv_pre: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> list[torch.Tensor]:
    """`_qk_norm_rope` as a Dynamo-OPAQUE op, so it can live in a dense piece.

    A REGULAR custom op, not a splitting one: opacity, not a graph *split*, is
    what keeps the `attn_metadata` it reads live per forward instead of frozen
    to the warmup trace (~16k tokens). Identity marker ops ARE traced through,
    which is what faulted an earlier attempt to move work this way.
    """
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    return self._qk_norm_rope(q, kv_pre, positions).custom_op_return()


direct_register_custom_op(
    op_name="v4_qk_norm_rope",
    op_func=v4_qk_norm_rope,
    mutates_args=[],
    fake_impl=_v4_qk_norm_rope_fake,
)


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
    compress_ratios: tuple[int, ...] = field(default_factory=tuple)

    # Indexer (CSA layers only)
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 1024
    use_index_cache: bool = False
    index_topk_freq: int = 1
    index_topk_pattern: Any | None = None

    # MoE
    moe_inter_dim: int = 3072  # moe_intermediate_size
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6  # num_experts_per_tok
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 2.5  # routed_scaling_factor
    swiglu_limit: float = 10.0
    # Routed FP4 experts on MXFP8 activations, as the checkpoint's reference
    # computes them (FusedMoE ``mxfp4_fp8_activations``).
    moe_fp8_activations: bool = False

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
    expert_dtype: Literal["fp4", "fp8"] | None = None
    scale_fmt: Literal["ue8m0"] | None = None

    # V4QuantizationConfig — Linear layers auto-build the right (FP8 / FP4
    # / BF16) weight + scale params. Set by DeepseekV4ForCausalLM at init.
    quant_config: Any | None = None

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
    # A hub id is not where its files are, and this probe picks FP8 vs BF16
    # parameter shapes: answering "no" for every remote checkpoint is the
    # mismatch this docstring warns produces garbage attention output.
    model_path = local_model_dir(model_path)
    if not model_path or not os.path.isdir(model_path):
        return False
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(idx_path):
        return False
    try:
        with open(idx_path) as f:
            idx = json.load(f)
        wmap = idx.get("weight_map", {})
    except Exception:  # noqa: BLE001 -- a probe: any unreadable index means "no"
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
            # `.keys()` is not redundant: safetensors' handle has no
            # `__contains__`, so `not in h` would raise. noqa: SIM118
            if "layers.0.attn.wo_a.scale" not in h.keys():  # noqa: SIM118
                # Index lies. wo_a still FP8 but no scale → loader will fail
                # anyway; safer to fall back to no_spec, although this case is
                # unexpected.
                return True
    except Exception:  # noqa: BLE001 -- a probe: any unreadable shard means "no"
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
    w = w_fp8.unflatten(0, (-1, block)).unflatten(-1, (-1, block)).float()
    s = scale.float()
    deq = w * s[:, None, :, None]
    return deq.flatten(2, 3).flatten(0, 1).bfloat16()


# ---------------------------------------------------------------------------
# fp8 e8m0 mxscale (128x128 block-scale) batched GEMM path for the grouped
# output LoRA (`wo_a`). The kernel consumes:
#   x       [M, G, K]         fp8 activation, per-token e8m0 (GROUP_M=1)
#   wo_a    [G, N, K]         fp8 weight, batch-major
#   x_scale [M, G, K/128]     uint8 e8m0 activation scale
#   w_scale [G, N/128, K/128] uint8 e8m0 weight (128x128) block scale
# The [M, G, *] views are transposed views of contiguous batch-major [G, M, *]
# buffers (K/N contiguous).
# ---------------------------------------------------------------------------


def _wo_a_block_scale_to_e8m0(
    scale: torch.Tensor, n_groups: int
) -> torch.Tensor | None:
    """Disk FP8 128x128 block scale ``[G*N/128, K/128]`` -> uint8 e8m0
    ``[G, N/128, K/128]``, or ``None`` when the scale is not e8m0-representable.

    An e8m0 byte is a bare biased exponent, so only a power-of-two scale has an
    exact form -- true of V4 ``wo_a`` (``scale_fmt='ue8m0'``). Rounding anything
    else would alter the weights rather than restate them, so the premise is
    checked and a failure returns ``None`` to fall back to BF16.
    """
    s = scale.detach()
    if s.element_size() == 1:
        # E8M0 block scales (QuantizationConfig.blockscale_e8m0_scale) allocate
        # weight_scale as dtypes.fp8_e8m0, which aiter resolves to torch.uint8
        # when the torch build has no float8_e8m0fnu. Those bytes are already biased
        # exponents, so the float path would read 127 as a magnitude and
        # return 134; a native float8_e8m0fnu byte is the same exponent.
        e = s.view(torch.uint8)
    else:
        s = s.float()
        if not bool(torch.isfinite(s).all()) or bool((s <= 0).any()):
            return None
        exp = torch.round(torch.log2(s))
        # Round-trips exactly only if the input really was a power of two.
        if not torch.equal(torch.exp2(exp), s):
            return None
        biased = exp.to(torch.int32) + 127
        if int(biased.min()) < 0 or int(biased.max()) > 255:
            return None
        e = biased.to(torch.uint8)
    nb, kb = e.shape
    return e.reshape(n_groups, nb // n_groups, kb).contiguous()


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
) -> tuple[torch.Tensor, torch.Tensor]:
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
        key: torch.Tensor | None = None,
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

    def cos_sin_2d(self) -> tuple[torch.Tensor, torch.Tensor]:
        """2D ``[max_pos, rd//2]`` cos/sin for ops that take the flat cache."""
        return (
            self.cos_cache.squeeze(-2).squeeze(-2),
            self.sin_cache.squeeze(-2).squeeze(-2),
        )

    def inverse(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        rope_dim: int,
        prefix: str = "",
    ) -> None:
        """In-place inverse RoPE via fused Triton kernel.

        ``x`` is the whole ``[num_tokens, n_heads, head_dim]`` output; the kernel
        slices the trailing ``rope_dim`` lanes itself, so no caller hands it a
        strided view to mutate.
        """
        inverse_rope_inplace(
            x, self.cos_cache, self.sin_cache, positions, rope_dim, prefix=prefix
        )


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

        # External tensors — assigned by the owning Attention / Indexer at first forward.
        self.kv_cache: torch.Tensor | None = None
        self.rotary_emb: _V4RoPE | None = None
        # FP8 quant path only: strided fp32 view of the per-block scale region
        # of `self.kv_cache`. Bound by the V4 builder when `kv_cache.dtype` is
        # FP8 (Indexer-inner Compressor); None for BF16 cache (Main path).
        self.cache_scale: torch.Tensor | None = None
        # Compress-scatter quant mode, bound by DeepseekV4AttentionMetadataBuilder.
        # Unified with the kernel's `quant_mode` — one of:
        #   "none"        → plain BF16 Main scatter (kv_cache_rope unused)
        #   "group_fp8"   → CSA/HCA Main native 2buff (nope-fp8 + inline e8m0 into
        #                   kv_cache, bf16 rope into the parallel kv_cache_rope pool)
        #   "per_row_fp8" → Indexer-inner per-row fp8 + preshuffle
        #   "fp4"         → Indexer-inner FP4 (E2M1 + e8m0)
        self.kv_cache_rope: torch.Tensor | None = None
        self.quant_mode: str = "none"

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
        state_slot_in: torch.Tensor,  # [bs] int32
        state_slot_out: torch.Tensor,  # [bs] int32
        block_tables: torch.Tensor | None = None,  # [bs, max_blocks_per_seq] int32
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
            `self.kv_cache` (consumed by paged_decode/paged_2src via
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
            state_slot_in:  [bs] int32 — state group each seq READS its incoming
                         ring from. Differs from `state_slot_out` only on the
                         forward after a state fork (resuming from a published
                         checkpoint, or taking one), where the incoming ring
                         belongs to a group that must not be written.
            state_slot_out: [bs] int32 — state group each seq WRITES its ring to.
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
        # state_slot_out passed to fused_compress_attn are full-sequence
        # (never split in the builder), so they match the gathered `combined`.
        if _pcp_active():
            _tbo = tbo_active()
            if _tbo:
                tbo_yield_and_switch_from_compute_to_comm()
            combined = pcp_allgather_rerange(combined, get_pcp_world_size())
            if _tbo:
                tbo_switch_to_compute_sync()
        kv, score = torch.split(combined, [coff_d, coff_d], dim=-1)

        # ====== Unified fused kernel path (CSA + Indexer) ======
        # Order is critical: fused kernel reads state cache as-of-end-of-
        # PREVIOUS-fwd. `update_compressor_states` overwrites them with this
        # fwd's data for the NEXT fwd's overlap — must run AFTER the fused
        # kernel.
        cos_cache, sin_cache = self.rotary_emb.cos_cache, self.rotary_emb.sin_cache
        # Scatter mode = `self.quant_mode` (unified with the kernel's quant_mode):
        #   - "none":        plain bf16 Main scatter.
        #   - "group_fp8":   CSA/HCA Main under --kv_cache_dtype fp8 — native 2buff
        #     (nope-fp8 + inline e8m0 into `kv_cache`, bf16 rope into
        #     `kv_cache_rope`); flagged here as `main_2buff_fp8`.
        #   - "per_row_fp8": Indexer-inner per-row fp8 + preshuffle.
        #   - "fp4":         Indexer-inner FP4 (E2M1 + e8m0).
        # The kernel wrapper derives its own `quant` flag from quant_mode (Indexer
        # per_row_fp8/fp4 take the quant path; group_fp8 is driven by
        # `main_2buff_fp8`; none is plain), so the caller only passes quant_mode.
        # `self.cache_scale` is bound alongside an fp8/uint8 `kv_cache` by the V4
        # builder (indexer-inner only; group_fp8 carries scale inline, none has no
        # cache → stays None otherwise). Only CSA/HCA Main uses group_fp8, so the
        # `"indexer" not in prefix` guard is defensive.
        main_2buff_fp8 = self.quant_mode == "group_fp8" and "indexer" not in self.prefix
        # Skip the kernel's cache scatter during warmup (kv_cache/block_tables
        # not yet bound).
        if block_tables is None or self.kv_cache is None:
            scatter_kv_cache = None
            scatter_block_tables = None
            scatter_kv_cache_rope = None
        else:
            scatter_kv_cache = self.kv_cache
            scatter_block_tables = block_tables
            scatter_kv_cache_rope = self.kv_cache_rope if main_2buff_fp8 else None
        fused_compress_attn(
            kv_in=kv,
            score_in=score,
            kv_state=self.kv_state,
            score_state=self.score_state,
            plan=plan,
            state_slot_mapping=state_slot_in,
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
            cache_scale=self.cache_scale,
            use_ue8m0=(self.scale_fmt == "ue8m0"),
            preshuffle=True,
            quant_mode=self.quant_mode,
            # fp8_max only applies to float fp8 caches (per_row_fp8 / group_fp8).
            # bf16 (none) and uint8 (fp4) have no fp8_max, and torch.finfo raises
            # on non-float dtypes, so restrict to float fp8 caches.
            fp8_max=(
                torch.finfo(self.kv_cache.dtype).max
                if (
                    self.kv_cache is not None
                    and self.kv_cache.dtype not in (torch.bfloat16, torch.uint8)
                )
                else None
            ),
            main_2buff_fp8=main_2buff_fp8,
            kv_cache_rope=scatter_kv_cache_rope,
            prefix=f"{self.prefix}.fused_compress_attn",
        )
        update_compressor_states(
            kv,
            score,
            self.ape,
            self.kv_state,
            self.score_state,
            write_plan=plan.write_plan_gpu,
            state_slot_mapping=state_slot_out,
            ratio=ratio,
            overlap=overlap,
            prefix=f"{self.prefix}.update_compressor_states",
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
        self._weights_scale = self.softmax_scale * self.n_heads**-0.5
        # Index rows of the longest sequence this deployment serves: one per
        # `compress_ratio` source tokens of `max_model_len`, not of the
        # checkpoint's `max_position_embeddings`. Native scoring takes its
        # plane's width from the step's rows (`IndexRows.width`), which the
        # attention builder sizes the same way.
        self._max_model_len_idx = (
            get_current_atom_config().max_model_len // compress_ratio
        )

        # Indexer quantization mode, self-computed at construction so it is
        # correct BEFORE
        # the graphed `_attn_pre`/`forward_pre` piece is traced — the graph bakes
        # this branch, and the builder's re-assert in `build_kv_cache_tensor` does
        # NOT reliably precede that trace (verified: defaulting False here bakes
        # the FP8 branch → q_scale None → the eager FP4 `indexer_score_topk`
        # disagrees). Under the vLLM / SGLang plugins, which never call
        # `build_kv_cache_tensor`, this is also the ONLY setter.
        # Shared predicate with the builder (see `fp4_indexer_enabled`) so the two
        # cannot drift apart; `warn` is left to the builder because this runs once
        # per CSA layer and would repeat the message.
        self._indexer_fp4 = fp4_indexer_enabled(
            get_current_atom_config().index_cache_dtype
        )
        self.quant_mode = "fp4" if self._indexer_fp4 else "per_row_fp8"
        # `quant_mode` says how values are quantized; `indexer_layout` says how
        # those values are physically stored and therefore which scorer ABI
        # consumes them. Both gfx950 and gfx1250 use quant_mode="fp4", but their
        # preshuffled/natural layouts are intentionally incompatible.
        self.indexer_layout = (
            fp4_indexer_layout_for_arch(get_gfx_runtime())
            if self._indexer_fp4
            else "fp8"
        )
        self._layout = (
            NATURAL if self.indexer_layout == FP4_GFX1250_NATURAL else PRESHUFFLE
        )
        self.compressor = Compressor(
            args,
            compress_ratio,
            self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
        )
        # Set before graph tracing and before KV-cache binding. The inner
        # compressor and the Indexer scorer now share one authoritative mode.
        self.compressor.quant_mode = self.quant_mode
        # PR3-pre2c-B: Indexer.kv_cache is bound by the V4 attention builder
        # to a `[num_blocks, csa_rows_per_block, head_dim]` per-CSA-layer view
        # of the global
        # `csa_idx_kv` classical KV pool. The 1-slot register_buffer below is
        # a warmup fallback (warmup runs before allocate_kv_cache); it is
        # setattr-replaced post-binding and GC'd. Same pattern as Compressor's
        # kv_state in pre2a / Attention.swa_plane in pre2c-A.
        self.register_buffer(
            "kv_cache",
            torch.zeros(1, self._max_model_len_idx, self.head_dim),
            persistent=False,
        )
        self.rotary_emb: _V4RoPE | None = None

        # Register self in static_forward_context so the
        # `torch.ops.aiter.indexer_score_topk` dispatcher can look us up by
        # `layer_name` (= self.prefix). Same pattern as V4 MoE registration.
        get_current_atom_config().compilation_config.static_forward_context[
            prefix
        ] = self

    @mark_trace
    def forward_batched(
        self,
        x_full: torch.Tensor,  # [total_tokens, dim]
        qr_full: torch.Tensor,  # [total_tokens, q_lora_rank] — fp8 when qr_full_scale given
        positions: torch.Tensor,  # [total_tokens]
        qr_full_scale: (
            torch.Tensor | None
        ) = None,  # per_1x128 scale paired with qr_full
    ) -> torch.Tensor:
        """Q proj + RoPE + FP8-quant + weights compute (have module state),
        then dispatch to `torch.ops.aiter.indexer_score_topk`, which calls
        back into `self.indexer_score_topk(q_quant, weights, q_scale, index_topk)`.

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
        q_quant, weights, q_scale = self.forward_pre(
            x_full, qr_full, positions, qr_full_scale
        )
        return self.score_topk_from(q_quant, weights, q_scale)

    def topk(
        self,
        x_full: torch.Tensor,
        qr_full: torch.Tensor,
        positions: torch.Tensor,
        qr_full_scale: torch.Tensor | None = None,
        *,
        pre_q_quant: torch.Tensor | None = None,
        pre_weights: torch.Tensor | None = None,
        pre_q_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the top-k, reusing precomputed projections when available.

        The narrow split (TR_HYBRID) projects Q/weights in `_attn_pre` (graphed piece)
        and passes them as ``pre_q_quant``/``pre_weights``/``pre_q_scale`` so only
        the eager paged ``score_topk_from`` runs here. Otherwise (FULL/legacy)
        project inline via ``forward_batched``. Same result either way — only the
        split site differs.
        """
        if pre_q_quant is not None:
            return self.score_topk_from(pre_q_quant, pre_weights, pre_q_scale)
        return self.forward_batched(x_full, qr_full, positions, qr_full_scale)

    def forward_pre(
        self,
        x_full: torch.Tensor,  # [total_tokens, dim]
        qr_full: torch.Tensor,  # [total_tokens, q_lora_rank] fp8
        positions: torch.Tensor,  # [total_tokens]
        qr_full_scale: torch.Tensor | None = None,
    ):
        """Graphable indexer Q proj + RoPE + weights (num_tokens-shaped).

        No paged / KV-cache access, so this runs in the compiled dense piece.
        Only `score_topk_from` (paged gather + score) must stay eager.
        Returns (q_quant, weights, q_scale): q_quant is FP8 by default or packed
        FP4 (uint8) when the FP4 indexer is on; q_scale is the paired e8m0 Q scale
        on the FP4 path (None for FP8), threaded to the scorer via the op boundary.
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
        # RoPE and quantization in one launch (`quantize_query`); the bf16
        # rotated Q is never materialized. Branch on the STABLE `_indexer_fp4`
        # flag (set at __init__), NOT `kv_cache.dtype`: this runs inside the
        # graphed `_attn_pre` piece, where a kv_cache-derived branch can be
        # baked wrong (kv_cache is bound after tracing) and then disagree with
        # the eager `indexer_score_topk`.
        q_quant, q_scale = quantize_query(
            q,
            self.rotary_emb.cos_cache,
            self.rotary_emb.sin_cache,
            positions,
            rd,
            fp4=self._indexer_fp4,
            layout=self._layout,
        )
        weights = self.weights_proj(x_full)
        if self._indexer_fp4:
            # The kernel dequantizes Q itself and applies `_weights_scale`, so
            # the head weights go to it as they are.
            return q_quant, weights, q_scale
        # An FP8 query's scale is dequantized by folding it into its head's
        # weight, with `_weights_scale`.
        weights = fold_query_scale(
            weights,
            q_scale,
            self._weights_scale,
            prefix=f"{self.prefix}.scale_indexer_weights",
        )
        return q_quant, weights, None

    def score_topk_from(
        self,
        q_quant: torch.Tensor,
        weights: torch.Tensor,
        q_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score + top-k through the opaque `indexer_score_topk` op (reads the
        index cache the compressor wrote)."""
        return torch.ops.aiter.indexer_score_topk(
            q_quant, weights, q_scale, self.prefix, self.index_topk
        )  # [total_tokens, index_topk] int32

    def indexer_score_topk(
        self,
        q_quant: torch.Tensor,  # FP8 [T, H, D], or FP4 values [T, 1, H, D/2]
        weights: torch.Tensor,  # [T, H]: FP8 scale-folded fp32, FP4 bf16
        q_scale: torch.Tensor | None,  # FP4 e8m0 Q scale; None on FP8
        topk: int,
    ) -> torch.Tensor:
        """Module-side entry invoked by `torch.ops.aiter.indexer_score_topk`:
        `score_topk` against this layer's cache, on the step's rows the
        attention builder published (`attn_metadata.index_rows`).

        Returns `[T, topk] int32` row ids local to each token's request, -1
        past its visibility -- the layout `csa_translate_pack` consumes. No
        host-side short-circuit for an empty context: a row whose bound is 0
        gets all -1 from the top-k itself, so nothing is baked into a graph.
        """
        return score_topk(
            q_quant if q_scale is None else (q_quant, q_scale),
            weights,
            self.index_cache,
            get_forward_context().attn_metadata.index_rows,
            topk=topk,
            weight_scale=self._weights_scale,
        )

    @property
    def index_cache(self) -> IndexCache:
        """The keys the builder bound (`kv_cache`, and `kv_scale` under FP4) as
        the scorer reads them. Every layout pages at 64 index rows; where that
        dimension sits is the layout's."""
        if not self._indexer_fp4:
            return IndexCache(self.kv_cache, None, self.kv_cache.size(1))
        if self._layout == NATURAL:
            return IndexCache(
                self.kv_cache, self.kv_scale, self.kv_cache.size(1), NATURAL
            )
        # The row-group readers' packed-dword scale contract (N_PHYS == 1)
        # holds at 64 rows a block only: fail loudly on any other.
        assert self.kv_cache.size(3) == 64, (
            f"FP4 indexer requires 64 CSA rows per block, got "
            f"{self.kv_cache.size(3)}. Set V4 block_size=256 (CSA rows per "
            f"block = block_size // 4)."
        )
        return IndexCache(self.kv_cache, self.kv_scale, self.kv_cache.size(3))


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
        alt_stream: torch.cuda.Stream | None = None,
        indexer_stream: torch.cuda.Stream | None = None,
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
        # Cache the runtime architecture outside compiled code so _attn_post
        # can use static flags without tracing GPU detection.
        arch = get_gfx_runtime()
        self._is_gfx1250 = arch == "gfx1250"
        self._is_gfx950 = arch == "gfx950"
        self._is_preshuffle = (
            self._is_gfx1250
        )  # TODO: gfx950 will support preshuffle in the future
        # Flipped by process_weights_after_loading when wo_a is eligible for the
        # mxscale BMM; off means the BF16 grouped-LoRA path.
        self._wo_a_mxscale = False
        self._wo_a_fp8_dtype: torch.dtype | None = None
        self._wo_a_w_fp8: torch.Tensor | None = None
        self._wo_a_w_scale: torch.Tensor | None = None

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
        # Per-request sliding window: `swa_plane` is this layer's view of the
        # shared KV plane and `swa_window` (a `WindowParams`) is where in it a
        # request's rows are. Both are bound by
        # `DeepseekV4AttentionMetadataBuilder.build_kv_cache_tensor` after
        # allocate_kv_cache. The 1-row register_buffer below is a warmup
        # fallback (warmup runs before allocate_kv_cache); after binding it is
        # setattr-replaced with the plane view and the original buffer is GC'd.
        # `unified_kv` (paged_decode/paged_2src base) is NOT pre-registered
        # — V4Attention.forward short-circuits the sparse_attn dispatch on
        # `is_dummy_run` so warmup never reads it.
        self.register_buffer(
            "swa_plane",
            torch.zeros(args.window_size, self.head_dim),
            persistent=False,
        )
        self.swa_plane_rope = None
        self.swa_window = None
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
        # Frozen bool: when KV cache dtype is fp8, route writes/attention to the
        # native 2buff fp8 path (compute-only 2buff quant, op4 fp8 prefill, op5
        # asm decode). Dynamo specializes on this constant so the bf16 path
        # traces unchanged. The rope planes (swa_plane_rope / unified_kv_rope) are
        # bound onto the module by DeepseekV4AttentionMetadataBuilder.
        self.kv_fp8 = atom_config.kv_cache_dtype == "fp8"
        # Persistent HCA decode workspace: allocate at model load (before KV
        # sizing and graph capture) on every serving path, native or plugin.
        if self.compress_ratio == hca_persist.HCA_RATIO:
            hca_persist.prepare_if_usable(
                kv_fp8=self.kv_fp8,
                heads=self.n_local_heads,
                gfx=arch,
            )

    def process_weights_after_loading(self) -> None:
        """Prepare wo_a (FP8 + e8m0 block scale) for the grouped output LoRA.

        Called by ATOM's standard loader (atom.model_loader.loader.load_model)
        after all weights are filled. wo_a is allocated as FP8 ColumnParallelLinear
        so both `.weight` (FP8) and `.weight_scale` (e8m0 block scale) load
        correctly via the standard FP8 path. Two outcomes:

        * gfx950 + 128-aligned shape: keep wo_a FP8 and cache the uint8 e8m0
          [G, N/128, K/128] block scale for `batched_gemm_a8w8_mxscale`.
        * gfx1250 + same shape: preshuffle the FP8 weight for
          `batched_gemm_a8w8_mxscale_bpreshuffle`.
        * otherwise: dequant to BF16 for the grouped LoRA einsum
          (`sgd,grd->sgr`) / `batched_gemm_bf16` — aiter has no FP8 grouped
          einsum.

        Idempotent: if wo_a.weight is already BF16 (e.g. dequant was applied
        elsewhere), this is a no-op.
        """

        w = self.wo_a.weight
        if w.dtype == torch.bfloat16:
            return  # already dequanted
        scale = getattr(self.wo_a, "weight_scale", None)
        if w.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz) or scale is None:
            return  # nothing to do

        # ---- fp8 e8m0 mxscale batched-GEMM path (gfx950 / gfx1250) ---------
        # The 128x128 weight block scale and per-128 activation groups need
        # N % 128 == 0 and K % 128 == 0; anything else falls through to BF16,
        # as does a block scale with no exact e8m0 form.
        G = self.n_local_groups
        N = self.o_lora_rank
        out_dim, K = int(w.shape[0]), int(w.shape[1])
        w_scale = None
        use_mxscale = self._is_gfx950 or self._is_gfx1250
        if (
            use_mxscale
            and out_dim == G * N
            and N % 128 == 0
            and K % 128 == 0
            and scale.dim() == 2
            and scale.shape[0] == out_dim // 128
            and scale.shape[1] == K // 128
        ):
            w_scale = _wo_a_block_scale_to_e8m0(scale.data, G)
        if w_scale is not None:
            self._wo_a_fp8_dtype = w.dtype
            if self._is_preshuffle:
                # The A8W8 BMM expects a 16x16 preshuffled weight. Its
                # block scale remains in [G, N/128, K/128] e8m0 layout.
                shuffle_weights(w, layout=(16, 16))
            # Cached as module attrs so the forward skips the reshape and the
            # scale conversion on every call.
            #
            # A VIEW of the live parameter, and `shuffle_weights` now writes
            # through the storage rather than rebinding it, so this tracks the
            # bytes rather than pinning the ones it was built from. That is why
            # it is taken AFTER the shuffle above: built before, it would have
            # named the same bytes in the wrong layout. Nothing shuffles this
            # weight again -- `quant_type` is set to `No` below, which is also
            # what a weight sync reads -- so tracking is what is wanted here.
            self._wo_a_w_fp8 = w.data.view(G, N, K)
            self._wo_a_w_scale = w_scale
            self._wo_a_mxscale = True
            if self.layer_id == 0:
                logger.info(
                    "wo_a using fp8 e8m0 mxscale batched GEMM (preshuffle=%s, "
                    "G=%d, N=%d, K=%d, keeping FP8 weight); "
                    "every layer with this shape takes the same path.",
                    self._is_preshuffle,
                    G,
                    N,
                    K,
                )
            # Suppress the LinearBase CK-layout shuffle, same as the BF16 branch
            # below: gfx1250 was preshuffled above; gfx950 needs plain row-major.
            self.wo_a.quant_type = QuantType.No
            self.wo_a.need_normalize_e4m3fn_to_e4m3fnuz = False
            return

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
        self.wo_a.quant_type = QuantType.No
        self.wo_a.need_normalize_e4m3fn_to_e4m3fnuz = False

    def _compressors_async(self) -> bool:
        """Whether the compressors run on their side streams this forward: only
        under a graph, where the fork and join are recorded with it."""
        fc = get_forward_context()
        return self._use_async_compress and fc.in_hipgraph and not tbo_active()

    def _launch_compressors(self, x: torch.Tensor) -> None:
        """Fire the compressors, on side streams when `_compressors_async`, and
        return at once: main Compressor -> alt_stream (CSA + HCA), Indexer
        Compressor -> indexer_stream (CSA only). Launched ahead of the Q/KV
        projections, which they then overlap; `_sparse_attention` joins them.

        A dummy run skips them: warmup runs before allocate_kv_cache, so their
        caches are unbound."""
        fc = get_forward_context()
        if fc.context.is_dummy_run or os.environ.get("ATOM_V4_BYPASS_ATTN") == "1":
            return
        attn_md = cast("AttentionMetaData_DSV4", fc.attn_metadata)
        plan = (
            attn_md.compress_plans[self.compress_ratio] if self.compress_ratio else None
        )
        side = self._compressors_async()
        for compressor, stream in self._compressor_streams():
            if side:
                stream.wait_stream(fc.main_stream)
            with torch.cuda.stream(stream) if side else nullcontext():
                compressor(
                    x,
                    plan=plan,
                    state_slot_in=attn_md.state_slot_in,
                    state_slot_out=attn_md.state_slot_out,
                    block_tables=attn_md.block_tables,
                )

    def _compressor_streams(self):
        """Each compressor this layer runs and the side stream it runs on:
        main Compressor -> alt_stream (CSA + HCA), Indexer Compressor ->
        indexer_stream (CSA, unless this layer skips its top-k)."""
        pairs = []
        if self.compressor is not None:
            pairs.append((self.compressor, self.alt_stream))
        if self.indexer is not None and not self.skip_topk:
            pairs.append((self.indexer.compressor, self.indexer_stream))
        return pairs

    def _join_compressors(self) -> None:
        """Make the main stream wait for what `_launch_compressors` put on the
        side streams."""
        if not self._compressors_async():
            return
        main = get_forward_context().main_stream
        for _, stream in self._compressor_streams():
            main.wait_stream(stream)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        cg_mode = get_current_atom_config().compilation_config.cudagraph_mode
        if cg_mode is not None and cg_mode.uses_tr_hybrid():
            return self._forward_tr_hybrid_attention(x, positions)
        return torch.ops.aiter.v4_attention_with_output(x, positions, self.layer_name)

    def _forward_tr_hybrid_attention(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """TR_HYBRID order: compressor launch -> pre/proj+norm -> paged core
        (which joins the compressors) -> post, each a piece of its own."""
        torch.ops.aiter.v4_attn_compress_launch(x, self.layer_name)

        (
            _q,
            _kv_pre,
            _qr,
            _qr_scale,
            idx_q_quant,
            idx_weights,
            idx_q_scale,
            q_sa,
            kv,
            q_packed,
            q_rope,
            k_packed,
            k_rope,
        ) = self._attn_pre(x, positions)

        o = torch.ops.aiter.v4_sparse_attention(
            q_sa,
            kv,
            q_packed,
            q_rope,
            k_packed,
            k_rope,
            positions,
            idx_q_quant,
            idx_weights,
            idx_q_scale,
            self.layer_name,
        )
        return self._attn_post(o, positions)

    def _attn_pre(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        run_indexer_proj: bool = True,
    ):
        """Graphable projection/norm prelude shared by both attention paths.

        TR_HYBRID asks for the indexer pre-projection and immediately materializes
        QK-norm/RoPE through v4_qk_norm_rope, so the downstream paged-core op
        receives already-shaped Q/K fields. FULL keeps the legacy indexer
        forward_batched inside _sparse_attention, so it leaves q/kv_pre and
        qr/qr_scale live for the inline QK-norm and top-k path.

        The return order groups the two consumers: FULL reads q/kv_pre plus
        qr/qr_scale; TR_HYBRID reads idx_* plus the flattened QK/RoPE fields.
        """
        assert (
            x.dim() == 2 and x.shape[-1] == self.dim
        ), f"DeepseekV4Attention expects [num_tokens, {self.dim}], got {tuple(x.shape)}"
        qkv_a = self.wqkv_a(x)
        q_lora, kv_pre = torch.split(qkv_a, [self.q_lora_rank, self.head_dim], dim=-1)
        qr, qr_scale = self.q_norm(q_lora)
        q = self.wq_b(qr, x_scale=qr_scale)

        # Indexer Q/weights projection (no paged access) -> graphed piece.
        idx_q_quant = idx_weights = idx_q_scale = None
        if run_indexer_proj and self.indexer is not None and not self.skip_topk:
            idx_q_quant, idx_weights, idx_q_scale = self.indexer.forward_pre(
                x, qr, positions, qr_scale
            )

        # QK-norm/RoPE is token-shaped, so a num_tokens-keyed piece holds it.
        # (q_sa, kv, q_packed, q_rope, k_packed, k_rope)
        paged: tuple[torch.Tensor | None, ...] = (None,) * 6
        if run_indexer_proj:
            out = torch.ops.aiter.v4_qk_norm_rope(q, kv_pre, positions, self.layer_name)
            # `kv_fp8` is frozen at __init__, so Dynamo specializes this branch
            # and the list length is constant per layer.
            if self.kv_fp8:
                paged = (None, None, out[0], out[1], out[2], out[3])
            else:
                paged = (out[0], out[1], None, None, None, None)
            # Consumed. Dropping them is what keeps the captured core from
            # holding an input it no longer reads.
            q = kv_pre = None
            # `qr`/`qr_scale` are dead downstream: `indexer.topk` short-circuits
            # past them whenever `idx_q_quant` is given, and where it is not
            # there is no indexer to call.
            qr = qr_scale = None
        return (
            q,
            kv_pre,
            qr,
            qr_scale,
            idx_q_quant,
            idx_weights,
            idx_q_scale,
            *paged,
        )

    @mark_trace
    def _wo_a_grouped_lora(
        self,
        o: torch.Tensor,
        positions: torch.Tensor,
        prefix: str = "",
    ) -> torch.Tensor:
        """Output inverse RoPE + grouped output LoRA.

        `o` arrives un-inverse-RoPE'd from `_sparse_attention` on all paths.
        Owning the inverse RoPE here lets the mxscale branches fuse it into
        group quant and keeps the attention halves free of wo_a path knowledge.
        """
        num_tokens = o.size(0)
        if self._wo_a_mxscale:
            # `inverse_rope_group_quant` fuses the output inverse RoPE into the
            # per-token e8m0 group-quant, saving a bf16 round trip. The output
            # is token-major [M, G, N] with N contiguous, so the `.flatten(1)`
            # this branch ends with is a free view.
            H = self.n_local_heads
            G = self.n_local_groups
            D = H * self.head_dim // G
            o = o.view(num_tokens, H, self.head_dim)  # [S, H, head_dim] pre-rope
            x_fp8 = torch.empty(
                (num_tokens, G, D), dtype=self._wo_a_fp8_dtype, device=o.device
            )
            x_scale = torch.empty(
                (num_tokens, G, D // 128), dtype=torch.uint8, device=o.device
            )
            cos, sin = self.rotary_emb.cos_sin_2d()
            inverse_rope_group_quant(
                o,
                positions.to(torch.int64),
                cos,
                sin,
                num_groups=G,
                quant_group_size=128,
                # Row-major [S, G, Ks], which is how `x_scale` is allocated
                # above and how the mxscale GEMM below reads it. The other
                # layouts fill the same bytes in a shuffled order.
                scale_layout="row",
                x_fp8=x_fp8,
                x_scale=x_scale,
            )
            # Guarded aiter entry returns a fresh token-major [M, G, o_lora_rank]
            # (same layout as the old out= buffer); N is contiguous so the
            # flatten below is a free view.
            bmm = (
                batched_gemm_a8w8_mxscale_bpreshuffle
                if self._is_preshuffle
                else batched_gemm_a8w8_mxscale
            )
            y = bmm(
                x_fp8,
                self._wo_a_w_fp8,
                x_scale,
                self._wo_a_w_scale,
                dtype=o.dtype,
            )
            # Flattened here, like both BF16 branches below: wo_b takes
            # [M, G * o_lora_rank]. Handing it the 3-D tensor instead makes aiter
            # read (M, K) off the first two dims, so K comes out as the group
            # count and the GEMM is rejected for a shape that never existed.
            return y.flatten(1)
        self.rotary_emb.inverse(
            positions,
            o.view(num_tokens, self.n_local_heads, self.head_dim),
            self.rope_head_dim,
            prefix=f"{self.layer_name}.inverse_rope",
        )
        o = o.view(num_tokens, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        if num_tokens <= 32 or self._is_gfx1250:
            y = torch.empty(
                num_tokens,
                self.n_local_groups,
                self.o_lora_rank,
                dtype=o.dtype,
                device=o.device,
            ).transpose(0, 1)
            y = batched_gemm_bf16(o.transpose(0, 1), wo_a, YQ=y)
            return y.transpose(0, 1).flatten(1)
        return torch.einsum("sgd,grd->sgr", o, wo_a).flatten(1)

    def _attn_post(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Grouped output LoRA + wo_b (graphable, num_tokens-shaped).

        `o` is the un-inverse-RoPE'd attention output; `positions` is forwarded
        for the inverse RoPE that `_wo_a_grouped_lora` applies.

        wo_b's RowParallelLinear TP all_reduce goes through the ATOM AR layer,
        which routes it through the TBO-aware custom op on the pure-TP+TBO path
        (overlaps the partner ubatch's compute) and a plain reduce otherwise.
        """
        # The AITER BMM entry is itself compile-guarded, so only its tuned-CSV
        # lookup/kernel dispatch stays opaque; quantization and surrounding
        # tensor work remain visible to the compiled graph.
        o = self._wo_a_grouped_lora(o, positions, prefix=f"{self.layer_name}.wo_a")
        return self.wo_b(o)

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
        # FULL / legacy wide-split path. Compose the shared helpers in the
        # ORIGINAL order so behaviour is byte-equivalent to the pre-refactor
        # inline body: launch the compressor BEFORE the Q/KV projections (to
        # overlap on the side stream), run the projections (indexer projection
        # stays inline in the core via forward_batched, so run_indexer_proj=
        # False here), then the paged core, then the output LoRA.
        fc = get_forward_context()
        if fc.context.is_dummy_run or os.environ.get("ATOM_V4_BYPASS_ATTN") == "1":
            return torch.zeros_like(x)

        self._launch_compressors(x)
        q, kv_pre, qr, qr_scale, *_ = self._attn_pre(
            x, positions, run_indexer_proj=False
        )
        qkn = self._qk_norm_rope(q, kv_pre, positions)
        o = self._sparse_attention(
            qkn,
            positions,
            x=x,
            qr=qr,
            qr_scale=qr_scale,
        )
        return self._attn_post(o, positions)

    def _qk_norm_rope(
        self,
        q: torch.Tensor,
        kv_pre: torch.Tensor,
        positions: torch.Tensor,
    ) -> "QKNormRopeOut":
        """The fused QK-norm/RoPE (+ decode SWA write). Single source.

        Split out of the attention body so it can run one graph piece earlier, in
        the compiled dense piece, via the opaque `v4_qk_norm_rope` op -- its grid is
        `q.shape[0]`, `batch_id_per_q_token` is `[T]` and only gates the store, and
        `swa_dest_rows` is a whole buffer, so a num_tokens-keyed piece holds it.

        Running AHEAD of the compressor is safe: nothing here reads what the
        compressor writes, and the decode SWA write has no ordering hazard against
        the attention that reads the window.
        """
        fc = get_forward_context()
        # Same reason the other halves guard: warmup runs before allocate_kv_cache,
        # so the SWA plane this writes into is not bound yet. Called from
        # `_attn_pre` this sits UPSTREAM of that guard and needs its own.
        if fc.context.is_dummy_run or os.environ.get("ATOM_V4_BYPASS_ATTN") == "1":
            return _qkn_placeholder(self, q, q.shape[0], zeros=True)
        attn_md = cast("AttentionMetaData_DSV4", fc.attn_metadata)
        rd = self.rope_head_dim
        ratio = self.compress_ratio
        is_decode = attn_md.state is AttnState.DECODE
        swa_dest_rows = (
            attn_md.swa_dest_rows[ratio]
            if (is_decode and attn_md.swa_dest_rows is not None)
            else None
        )
        # Single kernel fuses per-head Q RMSNorm (weightless) + KV RMSNorm
        # (weighted) + GPT-J interleaved RoPE on the tail rd dims. Dispatches
        # to flydsl when the shape matches (V4-Pro is always V4-Pro shape →
        # always flydsl). Microbench shows flydsl wins at every measured T
        # from 4 (1.12×) to 32k (1.04×); used for both decode and prefill.
        # Optional FP8 quant outputs left off — downstream sparse_attn /
        # swa_write are still bf16.
        #
        # Decode FUSES the window write into this launch, dropping a separate
        # `_swa_write_kernel` per layer. Each token's post-norm/rope K row is
        # scattered to `attn_md.swa_dest_rows[ratio][t]` (batch_id<0 skips the
        # CG pad) — bf16 through the flydsl kernel over `swa_kv`, fp8
        # through the aiter group-quant launch over the two `swa_*_buff` pools.
        #
        # Decode has no ordering hazard, so writing before the attention reads
        # the window is safe. Prefill does — chunked prefix reads must see the
        # PRIOR chunk — so it passes `swa_*=None` and scatters its window tail
        # after sparse_attn below.
        #
        # bf16 → qkn.q_sa / qkn.kv populated; fp8 2buff → qkn.q_packed / qkn.q_rope
        # / qkn.k_packed / qkn.k_rope populated (the 2buff layout nope-fp8 [.,512] +
        # rope-bf16 [.,64] that op4 (prefill) / op5 (decode) consume with no
        # requant). The inactive path's fields stay None.
        qkn = qk_norm_rope_maybe_quant(
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
            fp8_2buff=self.kv_fp8,
            batch_id_per_q_token=attn_md.batch_id_per_q_token if is_decode else None,
            # Where each token's own KV row goes, built once per forward for
            # this layer's compress class. The fused write takes the row rather
            # than the slot because a window row is no longer `slot * cs +
            # pos % cs` — it interleaves by the class's layer stride, which is
            # arithmetic that belongs to `v4_pool_geometry`, not to a kernel in
            # another repository.
            swa_dest_rows=swa_dest_rows,
            # bf16 SWA fusion (flydsl kernel / Triton fallback):
            swa_kv=self.swa_plane if (is_decode and not self.kv_fp8) else None,
            # fp8 2buff SWA fusion (aiter group-quant launch):
            swa_nope_scale_buff=self.swa_plane if (is_decode and self.kv_fp8) else None,
            swa_rope_buff=self.swa_plane_rope if (is_decode and self.kv_fp8) else None,
            prefix=f"{self.layer_name}.qk_norm_rope_maybe_quant",
        )
        if _V4_USE_REF_QUANT and not self.kv_fp8:
            act_quant_inplace(qkn.kv[..., :-rd], 64, self.scale_fmt)
        return qkn

    def _sparse_attention(
        self,
        qkn: "QKNormRopeOut",
        positions: torch.Tensor,
        idx_q_quant: torch.Tensor | None = None,
        idx_weights: torch.Tensor | None = None,
        idx_q_scale: torch.Tensor | None = None,
        x: torch.Tensor | None = None,  # FULL only: inline `forward_batched`
        qr: torch.Tensor | None = None,  # FULL only, same
        qr_scale: torch.Tensor | None = None,  # FULL only, same
    ) -> torch.Tensor:
        """The TOKEN-shaped tail: indexer top-k, CSA pack, the paged attention, and
        prefill's SWA write.

        Every launch here is sized by the token count -- `csa_translate_pack`'s grid
        is `(T, ...)`, the paged kernels take `N = qo_indptr.numel()-1`, the FP8
        and gfx1250 scorers are token-shaped -- and everything it reads is
        token-shaped or a whole persistent buffer, so a token graph records it.
        The one exception is the FP4 logits, launched per band as the step
        planned it: a break of its own (`indexer.score._fp4_band_logits`).
        """
        fc = get_forward_context()
        # Row count off the Q, NOT off `positions`. The Q descends from the
        # hidden state through `wqkv_a`/`wq_b`, so it carries the same token
        # count the residual stream does. `positions` does not have to: a step
        # where any DP rank is prefilling takes the variable-length path (see
        # `running_tokens_are_unified` in `forward_context.py`) and the two can
        # differ. Sizing the attention output by `positions` then hands the mHC
        # residual a mismatched `m` -- an aiter shape assert deep inside a
        # compiled piece, far from here.
        q_rows = qkn.q_sa if qkn.q_sa is not None else qkn.q_packed
        assert (
            q_rows is not None
        ), "_sparse_attention got no Q: `_qk_norm_rope` did not run upstream."
        num_tokens = q_rows.shape[0]
        if fc.context.is_dummy_run or os.environ.get("ATOM_V4_BYPASS_ATTN") == "1":
            # warmup runs before allocate_kv_cache: `unified_kv` is unbound and
            # the paged kernels would read OOB. Same guard the core has always
            # carried; this half now runs outside it. See `_qk_norm_rope`.
            return q_rows.new_zeros(
                (num_tokens, self.n_local_heads * self.head_dim),
                dtype=torch.bfloat16,
            )
        # Before anything reads the compressed KV -- and before the FP4 logits'
        # eager region, which ends the recorded segment the fork is in.
        self._join_compressors()
        attn_md = cast("AttentionMetaData_DSV4", fc.attn_metadata)
        ratio = self.compress_ratio
        is_decode = attn_md.state is AttnState.DECODE
        state_slot_out = attn_md.state_slot_out

        # Indexer score/top-k, then translate the seq-local result -> physical
        # paged offsets in the active CSA buffer. Both run after the compressor
        # split op in the narrow (TR_HYBRID) order.
        if self.indexer is not None and not self.skip_topk:
            topk_local = self.indexer.topk(
                x,
                qr,
                positions,
                qr_scale,
                pre_q_quant=idx_q_quant,
                pre_weights=idx_weights,
                pre_q_scale=idx_q_scale,
            )
            self._fill_csa_paged_compress(attn_md, topk_local, positions, num_tokens)

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
                split_plan = attn_md.split_plan_swa
            elif ratio == 4:
                kv_indices = attn_md.kv_indices_csa
                kv_indptr = attn_md.kv_indptr_csa
                split_plan = attn_md.split_plan_csa
            else:  # ratio == 128
                kv_indices = attn_md.kv_indices_hca
                kv_indptr = attn_md.kv_indptr_hca
                split_plan = attn_md.split_plan_hca
            # Dispatch on kv-cache layout inside the wrapper: fp8 2buff
            # (unified_kv_rope set) → aiter ASM with pre-packed fp8 Q + the
            # 2buff fp8/bf16 pools read with no requant. The optional H=128
            # prefill-ASM decode route also consumes the shared empty CSR.
            o = sparse_attn_v4_paged_decode(
                qkn.q_sa,
                self.unified_kv,
                kv_indices,
                kv_indptr,
                self.attn_sink,
                self.softmax_scale,
                unified_kv_rope=self.unified_kv_rope,
                q_packed_in=qkn.q_packed,
                q_rope_in=qkn.q_rope,
                qo_indptr=attn_md.qo_indptr,
                empty_kv_indptr=attn_md.empty_kv_indptr,
                prefix=f"{self.layer_name}.sparse_attn_decode",
                split_plan=split_plan,
                compress_ratio=ratio,
            )  # [S, H, head_dim]
        else:
            # Two-source paged prefill: prefix from `unified_kv` (per-ratio
            # buffer with SWA history + compress section), extend from per-fwd
            # `kv` tensor (in-chunk SWA tail; extend buffer is layer-invariant).
            #
            # ===== PCP (full-KV) =====
            # Under PCP the model.forward entry round-robin-split x/positions to 1/W,
            # so `q_sa` and `kv` here are this rank's 1/W shard. The per-query
            # metadata (kv_indptr/indices_*, index_rows) was already reduced
            # to this rank's owned queries in the builder (_apply_pcp_reindex),
            # so `q_sa` + those indices are aligned and used as-is. The only
            # runtime fixups here are on the actual K/V data:
            #   - swa_write must write the FULL sequence SWA ring (every PCP
            #     rank keeps full KV), and
            #   - sparse_attn's extend source must be the FULL extend K so each
            #     1/W query can attend the whole in-chunk SWA window.
            # So all-gather the extend K back to full order; positions/
            # cu_seqlens_q/state_slot_out for the SWA write stay full
            # (cu_seqlens_q / state_slot_out are per-seq, never split;
            # positions_full comes from the forward context which holds the
            # pre-split copy).
            #
            # The extend K's representation depends on the kv-cache layout:
            #   - bf16 → single `qkn.kv` [T, 1, 576]; all-gather it (kv_full).
            #   - fp8 2buff → `qkn.k_packed` [T, 1, 512] fp8 (nope + inline e8m0
            #     scale) + `qkn.k_rope` [T, 1, 64] bf16; all-gather BOTH (the
            #     scale rides inside k_packed, so no separate scale gather). fp8
            #     all_gather is a pure byte movement — RCCL supports it directly.
            # Both use the SAME rerange, so the gathered tensors stay aligned
            # with the full-sequence kv_indices_extend / positions_full.
            pcp_on = _pcp_active()
            if pcp_on:
                pcp_ws = get_pcp_world_size()
                _tbo_attn = tbo_active()
                if _tbo_attn:
                    tbo_yield_and_switch_from_compute_to_comm()
                if self.kv_fp8:
                    k_packed_full = pcp_allgather_rerange(qkn.k_packed, pcp_ws)
                    k_rope_full = pcp_allgather_rerange(qkn.k_rope, pcp_ws)
                    kv_full = None
                else:
                    k_packed_full = k_rope_full = None
                    kv_full = pcp_allgather_rerange(qkn.kv, pcp_ws)
                # positions must match the full-sequence coords for the
                # swa_write ring addressing (`positions[src] % cache_size`).
                # `positions` here is this rank's 1/W shard (split in
                # ForCausalLM.forward); all-gather it back to full order with
                # the same rerange used for the extend K (NOT fc.context.
                # positions, which the builder reindexed to 1/W).
                positions_full = pcp_allgather_rerange(positions, pcp_ws)
                if _tbo_attn:
                    tbo_switch_to_compute_sync()
            else:
                k_packed_full, k_rope_full = qkn.k_packed, qkn.k_rope
                kv_full = qkn.kv
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
            # Dispatch on kv-cache layout inside the wrapper: fp8 2buff
            # (unified_kv_rope set) → aiter op4 with the 2buff fp8 prefix pool
            # (nope-fp8 + rope-bf16) + op-quantized fp8 Q and the extend K
            # (k_packed_full/k_rope_full), no dequant of the prefix and no torch
            # quant; bf16 → OPUS / Triton over q_sa and the bf16 extend kv_full.
            # Under PCP the extend K is the PCP all-gathered full sequence
            # (k_*_full for fp8, kv_full for bf16); off PCP it is this fwd's
            # tensor unchanged. On bf16 the wrapper reuses out=qkn.q_sa as the
            # attention output buffer (q_sa is not needed after this call →
            # avoids an extra empty_like); fp8 ignores both q_sa and out.
            o = sparse_attn_v4_paged_2src(
                qkn.q_sa,
                self.unified_kv,
                kv_indices_prefix,
                kv_indptr_prefix,
                kv_full,
                attn_md.kv_indices_extend,
                attn_md.kv_indptr_extend,
                self.attn_sink,
                self.softmax_scale,
                # Reuse q_sa as the attention output buffer; q_sa is not needed
                # after this call and this avoids an extra empty_like allocation.
                out=qkn.q_sa,
                unified_kv_rope=self.unified_kv_rope,
                q_packed=qkn.q_packed,
                q_rope=qkn.q_rope,
                k_packed=k_packed_full,
                k_rope=k_rope_full,
                prefix=f"{self.layer_name}.sparse_attn_prefill",
            )  # [S, H, head_dim] bf16
            # swa_write AFTER attn so chunked-prefill prefix SWA reads see the
            # prior chunk's contents (not this chunk's just-computed tail).
            # OPT (window-only prefill write): only write each seq's trailing
            # `window_size` tokens (not the whole chunk). SWA decode only ever
            # reads the last `window` tokens, so this is correct for the seq's
            # own decode; it drops cross-request prefix reuse of the middle SWA
            # blocks. Cuts prefill scatter-writes ~ chunk_len/window (e.g. 64x
            # at in8192) — kills the uncoalesced block_tables scatter that made
            # concurrent prefill slow. Correct for single- AND multi-chunk:
            # window(128) <= chunk size, so any token's window reaches back at
            # most 127 tokens = within the prior chunk's written last-128;
            # free_after_prefill_chunk keeps that trailing block until read.
            # Dispatch on kv-cache layout inside the wrapper (swa_region_rope set
            # → fp8 2buff path, which scatters the op-quantized extend K
            # (k_packed_full/k_rope_full) into both paged SWA pools; else bf16
            # single-pool write over kv_full). Both are window-only (same
            # trailing-window semantics). Under PCP every rank writes the FULL
            # sequence SWA ring, so the extend K and positions_full are the PCP
            # all-gathered full versions (k_*_full / kv_full for the data,
            # positions_full for the ring addressing); off PCP they are this
            # fwd's tensors (positions_full == positions, kv_full == qkn.kv).
            swa_write(
                kv_full,
                positions_full,
                attn_md.cu_seqlens_q,
                state_slot_out,
                self.swa_plane,
                self.swa_window,
                # Window-only: persist just the chunk's trailing `window` tokens
                # — see the OPT note above. A request's windows hold exactly its
                # last `ring_slots` writes, so persisting more would only
                # overwrite what this same call just wrote.
                # K source is the PCP all-gathered full extend K (k_*_full); off
                # PCP it falls back to qkn.k_packed/k_rope (single-rank identical).
                min(self.window_size, attn_md.max_seqlen_q),
                k_packed=k_packed_full,
                k_rope=k_rope_full,
                pool_rope=self.swa_plane_rope,
                prefix=f"{self.layer_name}.swa_write",
            )

        # `o` is returned un-inverse-RoPE'd: `_wo_a_grouped_lora` removes the
        # absolute-position contribution the value-side RoPE carried in, on every
        # path, so the positions travel downstream instead.
        return o.reshape(num_tokens, -1)  # [num_tokens, n_local_heads*head_dim]

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
          physical_block   = block_tables[batch_id_per_q_token[t], block_idx_in_seq]
          row              = physical_block * envelope_rows + slot_in_block

        Fully fused into one triton kernel — no [T, index_topk] intermediates,
        no PyTorch fancy index. CG sentinel (batch_id=-1) and OOB clamp are
        handled in-kernel. The kernel takes each token's `valid_k` from its own
        `kv_indptr_csa` delta, which the builders sized as
        `min(visible_csa(pos), index_topk)` — Indexer's per-row visibility — so
        every reserved CSA cell gets written and no `-1` pre-fill is needed.

        Args:
          topk_local_raw: [total_tokens, index_topk] int32 — RAW seq-local
            output of `Indexer.forward_batched`. The leading `valid_k[t]`
            cells are always >= 0; trailing cells are -1 sentinels never
            read by csa_translate_pack (filtered by `k_offs < valid_k`).
          positions: [total_tokens] int — global token positions; forwarded
            to csa_translate_pack so the kernel can compute per-token
            `valid_k` inline.
        """
        # csa_block_capacity = block_size // ratio = 256 // 4 = 64.
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
            attn_md.batch_id_per_q_token,
            skip_buf,
            kv_indices,
            envelope_rows=attn_md.envelope_rows,
            csa_block_capacity=csa_block_capacity,
            window_size=window_size,
            prefix=f"{self.layer_name}.csa_translate_pack",
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
        quant_config: Any | None = None,
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
        weights: torch.Tensor | None = None,  # [num_tokens, 1]  optional gate
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

    Hash routing for `layer_id < n_hash_layers` (first 3 V4 layers) is wired
    through FusedMoE via the `custom_routing_function` hook: hash layers load a
    `tid2eid` table (token-id -> expert-id) instead of `gate.bias`, and
    `select_experts` gives `custom_routing_function` precedence over the
    standard sqrtsoftplus path. Expert *selection* comes from `tid2eid[input_ids]`
    while expert *weights* still use sqrtsoftplus(gate_logits). Accuracy verified.
    """

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV4Args,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
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
        moe_cfg = SimpleNamespace(
            routed_scaling_factor=self.routed_scaling_factor,
            n_shared_experts=args.n_shared_experts,
        )
        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=self.n_activated_experts,
            hidden_size=self.dim,
            intermediate_size=args.moe_inter_dim,
            layer_id=self.layer_id,
            reduce_results=False,
            renormalize=True,
            quant_config=qc,
            use_grouped_topk=False,
            prefix=f"{prefix}.experts",
            scoring_func=args.score_func,  # "sqrtsoftplus"
            e_score_correction_bias=getattr(self.gate, "e_score_correction_bias", None),
            config=moe_cfg,
            shared_expert_prefix=f"{prefix}.shared_experts",
            # inter=3072/TP8=384 is a 128-multiple; pad to 128 (not the 256
            # default) to avoid padding the MoE intermediate up to 512.
            pad_align=128,
            enable_comm_fused=True,
            mxfp4_fp8_activations=args.moe_fp8_activations,
        )
        self.experts.swiglu_limit = args.swiglu_limit
        self._fuse_shared_into_routed = self.experts.num_fused_shared_experts > 0

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
        # Keep comm-fused token-bucket dispatch dynamic under torch.compile.
        self._use_comm_fused_dispatch = self.experts._comm_fused_moe is not None
        # Register self in static_forward_context so the custom op dispatcher
        # can look us up by `layer_name` (= self.prefix): maybe_dual_stream_forward
        # (dual-stream) and the comm-fused token-bucket dispatch.
        if self._use_dual_stream or self._use_comm_fused_dispatch:
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
        # Rows at or past the pass's real row count route to -1.
        num_valid_rows = pass_scheduled_tokens_gpu(num_tokens)

        # Fused-shared expert: the custom_routing_function path bypasses
        # select_experts' shared-expert append, so the shared expert (slot
        # n_routed_experts) would never be routed and its ~40% contribution
        # dropped. When shared is fused, write the routed result into the first
        # `topk` columns of the global topK buffer (shared cols pre-filled) and
        # return the full [N, topk + n_shared] view.
        num_fused_shared = getattr(self.experts, "num_fused_shared_experts", 0)
        if num_fused_shared > 0:
            total_topk_weights, total_topk_ids = topk_meta_rows(num_tokens)
            hash_topk_triton(
                ids,
                gating_output,
                tid2eid,
                renormalize,
                self.routed_scaling_factor,
                total_topk_ids[:, :topk],
                total_topk_weights[:, :topk],
                num_valid_rows,
            )
            return total_topk_weights, total_topk_ids

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
            num_valid_rows,
        )
        return topk_weights, topk_ids

    def routed_expert_forward(
        self,
        x: torch.Tensor,  # [num_tokens, dim]
        shared_partial: torch.Tensor | None = None,
        before_stage2_for_rows=None,
        before_shared_add=None,
        stage2_stream: torch.cuda.Stream | None = None,
    ) -> tuple[torch.Tensor, bool]:
        """Gate + FusedMoE routed-expert pass.

        For hash layers the gate's `tid2eid` lookup needs `input_ids`;
        `DeepseekV4ForCausalLM.forward` stashes it on
        `forward_context.context.input_ids` before each forward, and
        `_hash_topk` (FusedMoE's custom_routing_function) reads it there.
        """
        router_logits = self.gate(x)  # [num_tokens, n_routed_experts]
        return self.experts.forward_maybe_comm_fused(
            x,
            router_logits,
            shared_partial,
            before_stage2_for_rows=before_stage2_for_rows,
            before_shared_add=before_shared_add,
            stage2_stream=stage2_stream,
        )

    @mark_trace
    def combine_outputs(
        self,
        routed: torch.Tensor,  # [num_tokens, dim]
        shared: torch.Tensor | None,  # [num_tokens, dim] or None
        prefix: str = "",
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Add shared-expert contribution (when not fused into routed) and
        all-reduce across TP ranks.
        """
        if shared is not None:
            # Both on this rank's rows: the routed experts' PCP reduction
            # (ATOM_PCP_MOE_MERGE) happens inside FusedMoE, and the shared
            # expert, sharded by TP alone, never takes part in it.
            routed = routed + shared
        if self.tp_size > 1:
            # ATOM AR layer decides internally (via _tbo_aware_tp_reduce) whether
            # to route through the TBO-aware custom op (pure TP+TBO) or a plain
            # all_reduce (non-TBO / TBO+DP).
            routed = tensor_model_parallel_all_reduce(routed)
        return routed

    def single_stream_moe_forward(
        self,
        x: torch.Tensor,  # [num_tokens, dim]
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Sequential: shared_experts → routed_experts → combine."""
        shared = self.shared_experts(x) if self.shared_experts is not None else None
        routed, is_complete = self.routed_expert_forward(x, shared_partial=shared)
        if is_complete:
            return routed
        return self.combine_outputs(
            routed,
            shared,
            prefix=f"{self.prefix}.combine_outputs",
        )

    def dual_stream_moe_forward(
        self,
        x: torch.Tensor,  # [num_tokens, dim]
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Run shared_experts on `alt_stream` in parallel with routed_experts
        on the current stream. Mirrors V2's pattern. Both reads of `x` are
        independent; main stream waits on alt_stream's completion before
        combining.
        """
        routed_stream = torch.cuda.current_stream(x.device)
        self.alt_stream.wait_stream(routed_stream)
        shared_produced = False
        shared_waited = False

        def produce_shared(output_rows):
            nonlocal shared_produced
            with torch.cuda.stream(self.alt_stream):
                shared = self.shared_experts.forward(x)
                if shared.shape[0] != output_rows:
                    padded = shared.new_zeros((output_rows, *shared.shape[1:]))
                    padded[: shared.shape[0]].copy_(shared)
                    shared = padded
            shared.record_stream(routed_stream)
            shared_produced = True
            return shared

        def wait_for_shared():
            nonlocal shared_waited
            if not shared_produced:
                raise RuntimeError(
                    "comm-fused shared output was joined before it was produced"
                )
            if not shared_waited:
                routed_stream.wait_stream(self.alt_stream)
                shared_waited = True

        routed, is_complete = self.routed_expert_forward(
            x,
            before_stage2_for_rows=produce_shared,
            before_shared_add=wait_for_shared,
            stage2_stream=routed_stream,
        )
        if is_complete:
            if not shared_produced or not shared_waited:
                raise RuntimeError(
                    "comm-fused Stage2 completed without joining the shared-output "
                    "producer"
                )
            return routed
        with torch.cuda.stream(self.alt_stream):
            shared = self.shared_experts.forward(x)
        routed_stream.wait_stream(self.alt_stream)
        return self.combine_outputs(
            routed, shared, prefix=f"{self.prefix}.combine_outputs"
        )

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
        if self._use_dual_stream or self._use_comm_fused_dispatch:
            # Keep stream and comm-fused dispatch opaque to torch.compile.
            return torch.ops.aiter.maybe_dual_stream_forward(x, self.prefix)
        return self.single_stream_moe_forward(x)


@dataclass
class HCState:
    residual: torch.Tensor
    post_mix: torch.Tensor | None = None
    comb_mix: torch.Tensor | None = None
    x_prev: torch.Tensor | None = None
    res_preshuffle: bool = False


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
        alt_stream: torch.cuda.Stream | None = None,
        indexer_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.prefix = prefix
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
        # Load HC params in FP32. BF16 mode replaces fn storage after loading;
        # base and scale remain FP32.
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
        self.enable_hc_fn_pack_bf16 = (
            envs.ATOM_MHC_USE_BF16
            and self._mhc_pre is not None
            and hasattr(aiter, "mhc_shuffle_fn")
        )
        self.enable_fused_hc = (
            self._mhc_fused_post_pre is not None and self.layer_id != 0
        )

    # mHC `hc_post_mult_value`: V4 uses `2.0 * sigmoid(post)` for the post gate.
    HC_POST_MULT = 2.0

    def process_weights_after_loading(self):
        """Pack constant mHC fn weights into AITER's BF16 hi/lo block layout.

        AITER selects the block size for the runtime GPU architecture. Pack once
        after loading, then use w_preshuffle_bf16=1 without shuffling residuals.
        No-op unless enable_hc_fn_pack_bf16.
        """
        if not self.enable_hc_fn_pack_bf16:
            return
        for name in ("hc_attn_fn", "hc_ffn_fn"):
            fn = getattr(self, name)
            if fn.dtype == torch.int32:
                continue
            # Replace storage on the existing Parameter: retain no FP32 copy.
            fn.data = aiter.mhc_shuffle_fn(fn.data.contiguous().float())

    def hc_pre(
        self,
        residual: torch.Tensor,  # [num_tokens, hc, dim]  mHC-widened residual
        hc_fn: torch.Tensor,  # [mix_hc, hc*dim]  fp32
        hc_scale: torch.Tensor,  # [3] fp32
        hc_base: torch.Tensor,  # [mix_hc] fp32
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
        res_preshuffle: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            # mhc_pre restores the ordinary residual layout when the runtime
            # fuse policy says this input is shuffled.
            pack_kw = {"w_preshuffle_bf16": 1} if self.enable_hc_fn_pack_bf16 else {}
            if res_preshuffle:
                pack_kw["res_preshuffle"] = True
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
                **pack_kw,
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
        res_preshuffle: bool = False,
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
            post_kw = {"res_preshuffle": True} if res_preshuffle else {}
            out = torch.empty_like(residual)
            self._mhc_post(
                out,
                x,
                residual,
                post.unsqueeze(-1),
                comb,
                **post_kw,
            )
            return out

        # Torch fallback. fp32 (post, comb) × BF16 (x, residual) promotes to
        # fp32 by PyTorch promotion rules — cast back to x.dtype before return.
        # post.unsqueeze(-1) * x.unsqueeze(-2): [num_tokens, hc, dim] gating
        # comb.unsqueeze(-1) * residual.unsqueeze(-2): [num_tokens, hc, hc, dim]; sum over hc
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3
        )
        return y.type_as(x)

    @mark_trace
    def mhc_fused_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
        res_preshuffle: bool = False,
        prefix: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # BF16 weight packing is independent of the residual layout.
        pack_kw = {}
        if self.enable_hc_fn_pack_bf16:
            pack_kw["w_preshuffle_bf16"] = True
        if res_preshuffle:
            pack_kw["res_preshuffle"] = True
        return self._mhc_fused_post_pre(
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
            **pack_kw,
        )

    @mark_trace
    def mhc_post_pre(
        self,
        x: torch.Tensor | None,
        residual: torch.Tensor,
        post: torch.Tensor | None,
        comb: torch.Tensor | None,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
        res_preshuffle: bool = False,
        prefix: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x is not None:
            res = self.hc_post(x, residual, post, comb)
        else:
            res = residual
        x, post, comb = self.hc_pre(
            res,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            norm_eps,
            res_preshuffle=res_preshuffle,
        )
        return x, post, comb, res

    def fuse_hc(
        self,
        hc_state: HCState,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
    ) -> HCState:
        residual = hc_state.residual
        post = hc_state.post_mix
        comb = hc_state.comb_mix
        x = hc_state.x_prev
        if (self.enable_fused_hc or hc_state.res_preshuffle) and x is not None:
            post, comb, x, res = self.mhc_fused_post_pre(
                x,
                residual,
                post,
                comb,
                hc_fn,
                hc_scale,
                hc_base,
                norm_weight,
                norm_eps,
                res_preshuffle=hc_state.res_preshuffle,
                prefix=f"{self.prefix}.mhc_fused_post_pre",
            )
            # Match hc_pre's [tokens, hc] state for subsequent post/pre calls.
            post = post.squeeze(-1)
        else:
            x, post, comb, res = self.mhc_post_pre(
                x,
                residual,
                post,
                comb,
                hc_fn,
                hc_scale,
                hc_base,
                norm_weight,
                norm_eps,
                res_preshuffle=hc_state.res_preshuffle,
                prefix=f"{self.prefix}.mhc_post_pre",
            )
        return HCState(
            residual=res,
            post_mix=post,
            comb_mix=comb,
            x_prev=x,
            res_preshuffle=hc_state.res_preshuffle,
        )

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
        )  # [num_tokens, dim]  (input_ids from forward_context for hash MoE)
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
        self,
        x: torch.Tensor,  # [num_tokens, dim]
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
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.norm_eps,
            self.hc_eps,
            sinkhorn_repeat=0,
            **({"w_preshuffle_bf16": 1} if hc_fn.dtype == torch.int32 else {}),
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
        self._mhc_arch = get_gfx_runtime()
        self.enable_res_preshuffle = hasattr(
            aiter, "mhc_res_shuffle_enabled"
        ) and aiter.mhc_res_shuffle_enabled(1, self._mhc_arch)

        # VocabParallelEmbedding shards along vocab dim. At TP=1 weight shape
        # equals nn.Embedding's [vocab_size, dim] so dummy state_dicts load
        # directly. At TP>1 each rank holds vocab_size/tp rows.
        self.embed = VocabParallelEmbedding(
            args.vocab_size,
            args.dim,
            prefix="embed",
        )
        # alt_stream: dual-stream MoE (shared_experts // routed_experts) AND
        # Main Compressor overlap. indexer_stream: Indexer Compressor overlap.
        # Both allocated once, shared across all blocks. Attention runs before
        # MoE in each block, so attn and MoE never contend for alt_stream.
        self.alt_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self.indexer_stream: torch.cuda.Stream | None = (
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

    def process_weights_after_loading(self):
        """Pack the head's constant fn using AITER's architecture-specific layout."""
        if (
            not envs.ATOM_MHC_USE_BF16
            or not hasattr(aiter, "mhc_shuffle_fn")
            or self.hc_head_fn.dtype == torch.int32
        ):
            return
        # Replace storage instead of keeping a second packed Parameter.
        self.hc_head_fn.data = aiter.mhc_shuffle_fn(
            self.hc_head_fn.data.contiguous().float()
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # [num_tokens] int  flat ragged-batch token ids
        positions: torch.Tensor,  # [num_tokens] int  abs positions (required)
    ) -> torch.Tensor:  # [num_tokens, hc, dim]  pre-hc_head residual stream
        """Forward over `num_tokens` flat ragged-batch tokens.

        Returns the mHC residual stack `[num_tokens, hc, dim]` before hc_head
        reduction. The final layer unshuffles and uses standalone hc_post;
        `hc_head + RMSNorm + LM head` remain in `compute_logits`, and MTP keeps
        the same three-dimensional residual contract.
        """
        assert input_ids.dim() == 1, f"input_ids must be 1D, got {input_ids.shape}"
        # PCP note: under PCP, `input_ids`/`positions` arrive already round-robin-
        # split to this rank's 1/W shard (done in DeepseekV4ForCausalLM.forward,
        # OUTSIDE the torch.compile boundary — keeping comms / dynamic padding
        # out of the compiled graph). So everything here runs on the 1/W shard;
        # the K/V all-gather inside attention reconstructs full KV per layer,
        # and the final all-gather + un-pad happens back in the caller.
        h = self.embed(input_ids)  # [num_tokens, dim]
        # Keep this static: concrete-M gating is handled inside AITER custom ops.
        # Checking h.size(0) here would specialize the compiled graph to warmup M.
        res_preshuffle = self.enable_res_preshuffle and (
            self.layers[0]._mhc_fused_post_pre is not None
        )
        if hasattr(aiter, "mhc_res_repeat"):
            # Aiter writes either the ordinary repeated residual or the shuffled
            # gfx1250 layout directly, without materializing repeat + shuffle.
            h = aiter.mhc_res_repeat(h, self.hc_mult, res_preshuffle)
        else:
            # Older AITER versions only support the ordinary residual layout.
            h = h.unsqueeze(-2).repeat(1, self.hc_mult, 1)
            res_preshuffle = False
        hc_state = HCState(
            residual=h,
            post_mix=None,
            comb_mix=None,
            x_prev=None,
            res_preshuffle=res_preshuffle,
        )

        for layer in self.layers:
            hc_state = layer(hc_state, positions)

        last_layer = self.layers[-1]
        return last_layer.hc_post(
            hc_state.x_prev,
            hc_state.residual,
            hc_state.post_mix,
            hc_state.comb_mix,
            res_preshuffle=hc_state.res_preshuffle,
        )


# The recording unit, not `DeepseekV4Model`: the DP-attention ids gather below
# feeds the hash MoE, which a replay would otherwise read at its recorded address.
@token_graph
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
    weights_mapping: ClassVar[dict[str, str]] = {
        ".gate.bias": ".gate.e_score_correction_bias",
        ".scale": ".weight_scale_inv",
    }
    packed_modules_mapping: ClassVar[dict[str, tuple[str, int]]] = {
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
        # FusedMoE gathers its routed rows across the PCP group itself; the
        # MoE routes these local rows first, hash layers by these local ids.
        ctx.context.pcp_split = use_pcp
        if use_pcp and not tbo_active():
            # (PCP+TBO splits upstream in run_model: input_ids is already this
            # ubatch's 1/(2*pcp) tokens.)
            pcp_size = get_pcp_world_size()
            n_global = input_ids.shape[0]
            pad = pcp_pad_len(n_global, pcp_size) - n_global
            if pad > 0:
                input_ids = torch.cat([input_ids, input_ids.new_zeros(pad)], dim=0)
                positions = torch.cat([positions, positions.new_zeros(pad)], dim=0)
            input_ids = pcp_round_robin_split(input_ids, pcp_size)
            positions = pcp_round_robin_split(positions, pcp_size)

        ctx.context.input_ids = input_ids
        h = self.model(input_ids, positions)

        # ----- PCP: all-gather shards, restore original order, drop pad -----
        if use_pcp:
            if tbo_active():
                # PCP+TBO: skip the all-gather here. Each ubatch returns its
                # local (1/(2*pcp)) shard; run_model does a single
                # pcp_allgather_rerange after UBatchWrapper cats both shards.
                pass
            else:
                h = pcp_allgather_rerange(h, pcp_size)
                if pad > 0:
                    h = h[:n_global]
        return h

    def compute_logits(
        self,
        hidden_states: torch.Tensor,  # [num_tokens, hc, dim]  pre-hc_head residual
    ) -> torch.Tensor:  # [bs, vocab]
        # Reduce the final unshuffled residual, normalize, then project logits.
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
            if hasattr(_m, "num_fused_shared_experts"):
                num_fused_shared = getattr(_m, "num_fused_shared_experts", 0)
                break
        if num_fused_shared == 0:
            # Some plugin builds wrap/alias FusedMoE such that the exact class-name
            # probe above misses it.  If the owning MoE layer was constructed in
            # fused-shared mode, the loader will rewrite ffn.shared_experts.* to
            # ffn.experts.{n_routed_experts}.*; include that final slot here so the
            # generic expert mapping loads it into w13/w2 instead of dropping it.
            for _m in self.model.modules():
                if _m.__class__.__name__ == "MoE" and getattr(
                    _m, "_fuse_shared_into_routed", False
                ):
                    num_fused_shared = getattr(self.args, "n_shared_experts", 0)
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
        except Exception:  # noqa: BLE001 -- optional; a model without one loads fine
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
