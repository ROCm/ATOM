# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only MiniMax-M3 model support for ATOM.

This file provides the native ATOM language backbone.  vLLM plugin-specific
attention/cache integration lives under ``atom.plugin.vllm`` and must not be
imported here.
"""

import copy
import os
from typing import Optional, Union

import torch
from aiter import ActivationType
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from aiter.rotary_embedding import get_rope
from atom.config import Config, QuantizationConfig, get_current_atom_config
from atom.model_ops import module_dispatch_ops as _module_dispatch_ops  # noqa: F401
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import GemmaRMSNorm, fused_allreduce_gemma_rms_norm
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
    MinimaxM3QKVParallelLinearWithIndexer,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.minimax_m3.gemma_rmsnorm import (
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
)
from atom.model_ops.minimax_m3.index_topk import (
    minimax_m3_index_topk,
    minimax_m3_index_topk_decode,
)
from atom.model_ops.minimax_m3.sparse_attn import (
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.swiglu_oai import swiglu_oai_split
from atom.model_ops.utils import atom_parameter
from atom.utils import mark_spliting_op
from atom.utils.forward_context import AttnState, get_forward_context
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils.decorators import support_torch_compile
from torch import nn
from transformers import PretrainedConfig


class MiniMaxM3GemmaRMSNorm(nn.Module):
    """MiniMax-M3 Gemma RMSNorm matching the vLLM-ATOM path."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return gemma_rmsnorm(x, self.weight, self.variance_epsilon)
        return gemma_fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)


def _get_text_config(config: PretrainedConfig) -> PretrainedConfig:
    return config.text_config if hasattr(config, "text_config") else config


def _sparse_attention_layer_ids(config: PretrainedConfig) -> set[int]:
    cfg = getattr(config, "sparse_attention_config", None)
    if not cfg:
        return set()
    freq = cfg.get("sparse_attention_freq")
    if freq is None:
        return set()
    return {i for i, enabled in enumerate(freq) if enabled != 0}


def _is_moe_layer(config: PretrainedConfig, layer_id: int) -> bool:
    moe_layer_freq = getattr(config, "moe_layer_freq", None)
    if moe_layer_freq is None:
        return True
    return moe_layer_freq[layer_id] != 0


def _rope_theta(config: PretrainedConfig) -> float:
    return getattr(config, "rope_theta", 1000000.0)


class MiniMaxM3MLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act != "swigluoai":
            raise ValueError(
                f"Unsupported MiniMax-M3 activation {config.hidden_act!r}; "
                "expected 'swigluoai'."
            )
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_beta = config.swiglu_beta
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = swiglu_oai_split(
            gate_up,
            alpha=self.swiglu_alpha,
            beta=self.swiglu_beta,
            limit=self.swiglu_limit,
        )
        return self.down_proj(x)


class MiniMaxM3MoE(nn.Module):
    """MiniMax-M3 routed MoE.

    Uses ATOM's generic FP4 ``FusedMoE`` loader/kernels while preserving
    MiniMax-M3 routing parameters (sigmoid scores, routing bias, scaling).
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        del layer_id
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

        if getattr(config, "use_routing_bias", False):
            self.e_score_correction_bias = atom_parameter(
                torch.empty(config.num_local_experts, dtype=torch.float32)
            )
        else:
            self.register_parameter("e_score_correction_bias", None)

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        # Match vLLM: router weights are stored and computed in fp32.
        old_wlp = self.gate.weight.weight_loader_process
        self.gate.weight = atom_parameter(self.gate.weight.data.to(torch.float32))
        self.gate.weight.weight_loader_process = old_wlp

        self.shared_experts: MiniMaxM3MLP | None = None
        if getattr(config, "n_shared_experts", 0):
            self.shared_experts = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.intermediate_size * config.n_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        moe_config = copy.copy(config)
        # Keep M3 shared experts as standalone MiniMaxM3MLP modules. FusedMoE's
        # generic shared-expert fusion would otherwise append them to routed
        # experts and duplicate the standalone path.
        moe_config.n_shared_experts = 0

        # MiniMax-M3 uses SwiGLU-OAI over MXFP4 routed experts. The generic
        # non-Triton/CK MoE path is not a supported target for that activation,
        # so force the Triton MXFP4 MoE path before FusedMoE chooses its
        # quant_method implementation.
        os.environ["ATOM_USE_TRITON_MOE"] = "1"
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            reduce_results=False,
            renormalize=True,
            quant_config=quant_config,
            scoring_func=getattr(config, "scoring_func", "sigmoid"),
            e_score_correction_bias=self.e_score_correction_bias,
            activation=ActivationType.Swiglu,
            has_bias=False,
            config=moe_config,
            params_dtype=params_dtype,
            prefix=f"{prefix}.experts",
        )
        self.experts.swiglu_alpha = getattr(config, "swiglu_alpha", 1.702)
        self.experts.swiglu_beta = getattr(config, "swiglu_beta", 1.0)
        self.experts.swiglu_limit = getattr(config, "swiglu_limit", 7.0)
        if self.experts.swiglu_beta != 1.0:
            raise NotImplementedError(
                "MiniMax-M3 Triton MoE currently supports swiglu_beta=1.0 "
                f"only, got {self.experts.swiglu_beta}."
            )
        if hasattr(self.experts.quant_method, "use_triton"):
            self.experts.quant_method.use_triton = True

        # Expose this module to the FP4 MoE dispatch custom op via layer_name.
        self.layer_name = prefix
        compilation_config = get_current_atom_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, orig_shape[-1])
        router_logits = torch.nn.functional.linear(
            hidden_states.float(), self.gate.weight.float()
        )
        # Routed experts go through the FusedMoE Triton MXFP4 path, which does
        # sigmoid routing (+ bias correction, renorm, routed_scaling_factor) and
        # SwiGLU-OAI experts entirely on-device. Unlike the previous per-expert
        # torch.nonzero loop, it has no data-dependent host sync, so it is
        # CUDA-graph capturable. Expert weights were swizzled for this kernel by
        # FusedMoE's own process_weights_after_loading.
        routed_output = self.experts(hidden_states, router_logits)

        if self.shared_experts is not None:
            routed_output = routed_output + self.shared_experts(hidden_states)

        if self.tp_size > 1:
            routed_output = tensor_model_parallel_all_reduce(routed_output)
        return routed_output.view(orig_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Keep the routed MoE behind an opaque custom op (consistent with the
        # other ATOM MoE models) so the piecewise model graph stays single-piece.
        return torch.ops.aiter.minimax_m3_fp4_moe_forward(
            hidden_states, self.layer_name
        )


class MiniMaxM3Attention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        cache_config: str = "bf16",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=None,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = MiniMaxM3GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMaxM3GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary_dim = int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=_rope_theta(config),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=cache_config,
            layer_num=layer_id,
            use_mla=False,
            rotary_emb=None,
            prefix=f"{prefix}.attn",
        )

    def _qk_norm_rope(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_shape = q.shape
        k_shape = k.shape
        q = self.q_norm(q.view(*q_shape[:-1], self.num_heads, self.head_dim)).reshape(
            q_shape
        )
        k = self.k_norm(
            k.view(*k_shape[:-1], self.num_kv_heads, self.head_dim)
        ).reshape(k_shape)
        return self.rotary_emb(positions, q, k)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        # Use torch.split, not Tensor.split: the bound method traces as a
        # call_function on torch._tensor, which AOTAutograd's cache treats as
        # non-cacheable and bypasses (breaking standalone-compile artifact save).
        # torch.split (torch.functional) is on the cache allowlist, matching the
        # rest of the codebase (deepseek_v2, qwen3_next).
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._qk_norm_rope(positions, q, k)
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)


def _minimax_m3_sparse_attention_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(q)


# Mark the sparse-attention core as a splitting op so the piecewise compiler
# cuts the model graph here (mirroring the dense `unified_attention_with_output_base`
# boundary). Dynamo uses the fake impl while tracing and never inspects the
# data-dependent block top-k / Triton kernels / KV-cache mutations inside; the
# real impl (which also handles dummy/capture runs) runs eagerly at runtime.
@mark_spliting_op(
    is_custom=True, gen_fake=_minimax_m3_sparse_attention_fake, mutates_args=[]
)
def minimax_m3_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    self = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    return self._sparse_attn_impl(q, k, v, index_q, index_k)


class MiniMaxM3SparseAttention(nn.Module):
    """Native ATOM MiniMax-M3 lightning-indexer sparse attention."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        cache_config: str = "bf16",
    ) -> None:
        super().__init__()
        del quant_config
        self.is_minimax_m3_sparse_attention = True
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % self.tp_size != 0:
            raise ValueError("num_attention_heads must be divisible by TP size.")
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.tp_size:
            if self.total_num_kv_heads % self.tp_size != 0:
                raise ValueError("num_key_value_heads must divide TP size.")
        elif self.tp_size % self.total_num_kv_heads != 0:
            raise ValueError("TP size must divide num_key_value_heads replication.")
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        sparse_cfg = config.sparse_attention_config
        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        self.num_idx_heads = self.num_kv_heads
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.index_q_size = self.num_idx_heads * self.idx_head_dim
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]
        self.sparse_block_size = sparse_cfg["sparse_block_size"]
        self.init_blocks = sparse_cfg.get("sparse_init_block", 0)
        self.local_blocks = sparse_cfg.get("sparse_local_block", 1)

        self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            self.total_idx_heads,
            self.idx_head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=None,
            reduce_results=False,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = MiniMaxM3GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMaxM3GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary_dim = int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=_rope_theta(config),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.index_q_norm = MiniMaxM3GemmaRMSNorm(
            self.idx_head_dim, eps=config.rms_norm_eps
        )
        self.index_k_norm = MiniMaxM3GemmaRMSNorm(
            self.idx_head_dim, eps=config.rms_norm_eps
        )
        self.kv_cache = torch.tensor([])
        self.index_kv_cache = torch.tensor([])
        self.k_scale = self.v_scale = None
        self.kv_cache_dtype = cache_config

        # Expose this layer to the sparse-attention splitting op via layer_name.
        self.layer_name = prefix
        compilation_config = get_current_atom_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _insert_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        index_k: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        n = k.shape[0]
        slot_mapping = slot_mapping[:n]
        key_cache = self.kv_cache[:, 0].reshape(-1, self.num_kv_heads, self.head_dim)
        value_cache = self.kv_cache[:, 1].reshape(-1, self.num_kv_heads, self.head_dim)
        key_cache[slot_mapping] = k.view(-1, self.num_kv_heads, self.head_dim).to(
            key_cache.dtype
        )
        value_cache[slot_mapping] = v.view(-1, self.num_kv_heads, self.head_dim).to(
            value_cache.dtype
        )
        index_cache = self.index_kv_cache.reshape(-1, self.idx_head_dim)
        index_cache[slot_mapping] = index_k.view(-1, self.idx_head_dim).to(
            index_cache.dtype
        )

    def _run_sparse(
        self,
        q: torch.Tensor,
        index_q: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        output = torch.empty_like(q)
        q = q.view(-1, self.num_heads, self.head_dim)
        index_q = index_q.view(-1, self.num_idx_heads, self.idx_head_dim)
        if attn_metadata.state == AttnState.DECODE:
            seq_lens = attn_metadata.context_lens.to(torch.int32)
            block_table = attn_metadata.block_tables
            topk_idx = minimax_m3_index_topk_decode(
                index_q,
                self.index_kv_cache,
                block_table,
                seq_lens,
                int(attn_metadata.max_seqlen_k),
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                self.num_kv_heads,
                self.scaling,
            )
            minimax_m3_sparse_attn_decode(
                q,
                self.kv_cache,
                topk_idx,
                block_table,
                seq_lens,
                self.num_kv_heads,
                self.scaling,
                output.view(-1, self.num_heads, self.head_dim),
            )
            return output

        cu_q = attn_metadata.cu_seqlens_q.to(torch.int32)
        query_lens = cu_q[1:] - cu_q[:-1]
        seq_lens = attn_metadata.context_lens.to(torch.int32)
        prefix_lens = seq_lens - query_lens
        block_table = attn_metadata.block_tables
        topk_idx = minimax_m3_index_topk(
            index_q,
            self.index_kv_cache,
            block_table,
            cu_q,
            seq_lens,
            prefix_lens,
            int(attn_metadata.max_seqlen_q),
            int(attn_metadata.max_seqlen_k),
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            self.num_kv_heads,
            self.scaling,
        )
        minimax_m3_sparse_attn(
            q,
            self.kv_cache,
            topk_idx,
            block_table,
            cu_q,
            seq_lens,
            prefix_lens,
            int(attn_metadata.max_seqlen_q),
            self.num_kv_heads,
            self.scaling,
            output.view(-1, self.num_heads, self.head_dim),
        )
        return output

    def _sparse_attn_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
    ) -> torch.Tensor:
        # Runs eagerly inside the splitting op. Dummy/profile runs (CUDA graph
        # capture warmup) have no real KV cache populated, so skip the sparse
        # kernels and return a correctly shaped placeholder, mirroring the dense
        # MHA backend's `is_dummy_run` handling.
        fwd_ctx = get_forward_context()
        if fwd_ctx.context.is_dummy_run:
            return torch.empty_like(q)
        attn_metadata = fwd_ctx.attn_metadata
        self._insert_kv(k, v, index_k, attn_metadata.slot_mapping)
        return self._run_sparse(q, index_q, attn_metadata)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        # torch.split, not Tensor.split (see MiniMaxM3Attention.forward): the
        # bound method bypasses the AOTAutograd cache and breaks artifact save.
        q, k, v, index_q, index_k = torch.split(
            qkv,
            [
                self.q_size,
                self.kv_size,
                self.kv_size,
                self.index_q_size,
                self.idx_head_dim,
            ],
            dim=-1,
        )
        q_shape = q.shape
        k_shape = k.shape
        q = self.q_norm(q.view(*q_shape[:-1], self.num_heads, self.head_dim)).reshape(
            q_shape
        )
        k = self.k_norm(
            k.view(*k_shape[:-1], self.num_kv_heads, self.head_dim)
        ).reshape(k_shape)
        q, k = self.rotary_emb(positions, q, k)
        iq_shape = index_q.shape
        index_q = self.index_q_norm(
            index_q.view(*iq_shape[:-1], self.num_idx_heads, self.idx_head_dim)
        ).reshape(iq_shape)
        index_k = self.index_k_norm(index_k)
        index_q, index_k = self.rotary_emb(positions, index_q, index_k)

        # Splitting op: the sparse-attention core (KV insert + block top-k +
        # sparse attention) runs opaquely so the model graph stays single-piece.
        attn_output = torch.ops.aiter.minimax_m3_sparse_attention(
            q, k, v, index_q, index_k, self.layer_name
        )
        return self.o_proj(attn_output)


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: str = "bf16",
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: Optional[torch.dtype] = None,
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        attn_cls = (
            MiniMaxM3SparseAttention
            if layer_num in _sparse_attention_layer_ids(config)
            else MiniMaxM3Attention
        )
        self.self_attn = attn_cls(
            config=config,
            layer_id=layer_num,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            cache_config=cache_config,
        )

        self.is_moe_layer = _is_moe_layer(config, layer_num)
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxM3MoE(
                config=config,
                layer_id=layer_num,
                quant_config=quant_config,
                params_dtype=params_dtype,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.mlp = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.dense_intermediate_size,
                quant_config=None,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = fused_allreduce_gemma_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        hidden_states = ffn(hidden_states)
        return hidden_states, residual


@support_torch_compile
class MiniMaxM3Model(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = MiniMaxM3DecoderLayer,
    ) -> None:
        super().__init__()
        config = _get_text_config(atom_config.hf_config)
        self.config = config
        cache_config = atom_config.kv_cache_dtype
        quant_config = atom_config.quant_config

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: layer_type(
                config,
                prefix,
                cache_config=cache_config,
                quant_config=quant_config,
                layer_num=layer_num,
                params_dtype=atom_config.torch_dtype,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )

        if get_pp_group().is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.get_input_embeddings(input_ids)
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )


class MiniMaxM3SparseForCausalLM(nn.Module):
    disable_fused_shared_loading = True
    packed_modules_mapping = {
        ".index_q_proj": (".qkv_proj", "index_q"),
        ".index_k_proj": (".qkv_proj", "index_k"),
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = MiniMaxM3DecoderLayer,
    ) -> None:
        super().__init__()
        config = _get_text_config(atom_config.hf_config)
        self.config = config
        self.model = MiniMaxM3Model(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.lm_head(hidden_states)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


class MiniMaxM3SparseForConditionalGenerationTextOnly(nn.Module):
    """Native ATOM text-only view of a MiniMax-M3 VL checkpoint."""

    disable_fused_shared_loading = True
    packed_modules_mapping = MiniMaxM3SparseForCausalLM.packed_modules_mapping
    weights_mapping = {
        "model.language_model.": "language_model.",
    }
    skip_weight_prefixes = [
        "vision_tower.",
        "multi_modal_projector.",
        "patch_merge_mlp.",
    ]

    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        self.config = atom_config.hf_config
        self.language_model = MiniMaxM3SparseForCausalLM(
            atom_config=atom_config,
            prefix=prefix,
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()


# Native full VL support will be wired after the MiniMax-M3 vision tower is
# ported to ATOM.  Keep the architecture name available as a text-only fallback
# so checkpoints with the VL arch can start loading during language bring-up.
MiniMaxM3SparseForConditionalGeneration = (
    MiniMaxM3SparseForConditionalGenerationTextOnly
)
