# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only MiniMax-M3 model support for ATOM.

This file provides the native ATOM language backbone.  vLLM plugin-specific
attention/cache integration lives under ``atom.plugin.vllm`` and must not be
imported here.
"""

from typing import Optional, Union

import torch
import aiter
from aiter import ActivationType
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from aiter.rotary_embedding import get_rope
from atom.config import Config, QuantizationConfig, get_current_atom_config
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import (
    GemmaRMSNorm,
    fused_qk_norm,
    fused_allreduce_gemma_rms_norm,
)
from atom.model_ops import module_dispatch_ops as _module_dispatch_ops  # noqa: F401
from atom.model_ops.linear import (
    MinimaxM3QKVParallelLinearWithIndexer,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.minimax_m3.index_topk import (
    minimax_m3_index_topk,
    minimax_m3_index_topk_decode,
)
from atom.model_ops.minimax_m3.moe import (
    MiniMaxM3Bf16Experts,
    make_minimax_m3_expert_params_mapping,
)
from atom.model_ops.minimax_m3.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_fused_qknorm_rope_kv_insert_shuffle,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
    minimax_m3_sparse_attn_decode_asm,
    minimax_m3_sparse_attn_prefill_asm,
)
from atom.model_ops.swiglu_oai import swiglu_oai_split
from atom.model_ops.utils import atom_parameter
from atom.utils import envs
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils.decorators import support_torch_compile
from atom.utils.forward_context import get_forward_context
from torch import nn
from transformers import PretrainedConfig


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


def _can_use_fused_minimax_m3_attention_preproc(
    qkv: torch.Tensor,
    rotary_emb: nn.Module,
    *weights: torch.Tensor,
) -> bool:
    return (
        hasattr(aiter, "fused_qknorm_idxrqknorm")
        and qkv.dim() == 2
        and qkv.is_cuda
        and qkv.dtype in (torch.float16, torch.bfloat16)
        and getattr(rotary_emb, "head_size", None) == 128
        and getattr(rotary_emb, "rotary_dim", 0) > 0
        and getattr(rotary_emb, "is_neox_style", False)
        and all(weight.dtype == qkv.dtype for weight in weights)
    )


def _minimax_m3_cos_sin_cache(
    rotary_emb: nn.Module,
    query: torch.Tensor,
) -> torch.Tensor:
    cache_name = "_minimax_m3_cos_sin_cache"
    cos_cache = rotary_emb.cos_cache.squeeze(-2).squeeze(-2)
    cached = getattr(rotary_emb, cache_name, None)
    expected_shape = (*cos_cache.shape[:-1], cos_cache.shape[-1] * 2)
    if (
        cached is not None
        and cached.dtype == query.dtype
        and cached.device == query.device
        and tuple(cached.shape) == expected_shape
    ):
        return cached

    sin_cache = rotary_emb.sin_cache.squeeze(-2).squeeze(-2)
    if cos_cache.dtype != query.dtype or cos_cache.device != query.device:
        cos_cache = cos_cache.to(device=query.device, dtype=query.dtype)
        sin_cache = sin_cache.to(device=query.device, dtype=query.dtype)
    cos_sin_cache = torch.cat([cos_cache, sin_cache], dim=-1).contiguous()

    if torch.compiler.is_compiling():
        return cos_sin_cache

    if cache_name in rotary_emb._buffers:
        rotary_emb._buffers[cache_name] = cos_sin_cache
    else:
        rotary_emb.register_buffer(cache_name, cos_sin_cache, persistent=False)
    return cos_sin_cache


def _minimax_m3_gemma_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: GemmaRMSNorm,
    k_norm: GemmaRMSNorm,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k = fused_qk_norm(
        q.view(-1, num_q_heads, head_dim),
        k.view(-1, num_kv_heads, head_dim),
        q_norm.weight,
        k_norm.weight,
        q_norm.variance_epsilon,
        add_unit_offset=True,
    )
    return (
        q.view(-1, num_q_heads * head_dim),
        k.view(-1, num_kv_heads * head_dim),
    )


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
        self.swiglu_alpha = getattr(config, "swiglu_alpha", 1.702)
        self.swiglu_beta = getattr(config, "swiglu_beta", 1.0)
        self.swiglu_limit = getattr(config, "swiglu_limit", 7.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = swiglu_oai_split(
            gate_up,
            alpha=self.swiglu_alpha,
            beta=self.swiglu_beta,
            limit=self.swiglu_limit,
        )
        return self.down_proj(x)


def _minimax_m3_use_dedicated_bf16_experts(
    quant_config: Optional[QuantizationConfig],
) -> bool:
    """Whether to route routed experts through the dedicated bf16 triton MoE.

    Native (unquantized) MiniMax-M3 runs through ``MiniMaxM3Bf16Experts`` — a
    self-contained triton MoE (custom GEMM kernels + SwiGLU-OAI) that does NOT
    depend on aiter's CK ``fused_moe`` swiglu kernel, so it runs on stock aiter
    main. Quantized (FP4/MXFP4) checkpoints keep using the generic ``FusedMoE``
    path, which the dedicated experts do not implement.
    """
    if quant_config is None:
        return True
    return getattr(quant_config, "quant_method", "") in ("", None)


class MiniMaxM3MoE(nn.Module):
    """MiniMax-M3 routed MoE.

    Native bf16 checkpoints use the dedicated ``MiniMaxM3Bf16Experts`` (triton,
    no aiter-CK swiglu dependency); FP4 checkpoints use the generic ``FusedMoE``.
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
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {tp_size} is greater than "
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
        # The checkpoint stores router weights as fp32, but routing tolerates bf16
        # logits. Let the weight loader cast once instead of casting every forward.

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.use_dedicated_bf16_experts = _minimax_m3_use_dedicated_bf16_experts(
            quant_config
        )
        if self.use_dedicated_bf16_experts:
            # Dedicated triton experts: own sigmoid routing + SwiGLU-OAI; applies
            # routed_scaling_factor internally and returns per-rank (un-reduced)
            # output. Shared experts are never fused into this path.
            self.experts = MiniMaxM3Bf16Experts(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                scoring_func=getattr(config, "scoring_func", "sigmoid"),
                routed_scaling_factor=self.routed_scaling_factor,
                swiglu_alpha=getattr(config, "swiglu_alpha", 1.702),
                swiglu_beta=getattr(config, "swiglu_beta", 1.0),
                swiglu_limit=getattr(config, "swiglu_limit", 7.0),
                quant_config=quant_config,
                params_dtype=params_dtype,
                prefix=f"{prefix}.experts",
            )
            self.fuse_shared_experts = False
        else:
            self.experts = FusedMoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                params_dtype=params_dtype,
                reduce_results=False,
                renormalize=True,
                activation=ActivationType.Swiglu,
                scoring_func=getattr(config, "scoring_func", "sigmoid"),
                e_score_correction_bias=self.e_score_correction_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.experts",
                config=config,
                shared_expert_prefix=f"{prefix}.shared_experts",
            )
            self.experts.swiglu_limit = getattr(config, "swiglu_limit", 7.0)
            self.fuse_shared_experts = (
                getattr(self.experts, "num_fused_shared_experts", 0) > 0
            )

        self.shared_experts: MiniMaxM3MLP | None = None
        if getattr(config, "n_shared_experts", 0) and not self.fuse_shared_experts:
            self.shared_experts = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.intermediate_size * config.n_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, orig_shape[-1])
        router_logits = self.gate(hidden_states)

        if self.use_dedicated_bf16_experts:
            # Dedicated experts apply routed_scaling_factor internally and return
            # un-reduced output (the decoder's fused all-reduce reduces it).
            routed_output = self.experts(
                hidden_states,
                router_logits,
                self.e_score_correction_bias,
            )
        else:
            routed_output = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            if not self.fuse_shared_experts and self.routed_scaling_factor != 1.0:
                routed_output = routed_output * self.routed_scaling_factor

        if self.shared_experts is not None:
            routed_output = routed_output + self.shared_experts(hidden_states)

        return routed_output.view(orig_shape)


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
        self.layer_num = layer_id
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
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary_dim = int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=_rope_theta(config),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        _minimax_m3_cos_sin_cache(self.rotary_emb, self.q_norm.weight)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=cache_config,
            layer_num=layer_id,
            use_mla=False,
            rotary_emb=None,
            force_triton_attn=True,
            prefix=f"{prefix}.attn",
        )

    def _qk_norm_rope(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k = _minimax_m3_gemma_qk_norm(
            q,
            k,
            self.q_norm,
            self.k_norm,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        return self.rotary_emb(positions, q, k)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        if _can_use_fused_minimax_m3_attention_preproc(
            qkv, self.rotary_emb, self.q_norm.weight, self.k_norm.weight
        ):
            qkv = qkv.contiguous()
            q = torch.empty(
                (qkv.shape[0], self.q_size), dtype=qkv.dtype, device=qkv.device
            )
            cos_sin_cache = _minimax_m3_cos_sin_cache(self.rotary_emb, qkv)
            aiter.fused_qknorm_idxrqknorm(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                cos_sin_cache,
                positions,
                self.num_heads,
                self.num_kv_heads,
                self.rotary_emb.rotary_dim,
                self.q_norm.variance_epsilon,
                None,
                None,
                0,
                None,
                None,
                None,
                0,
                q,
                None,
                None,
            )
            _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            attn_output = self.attn(q, k, v)
            return self.o_proj(attn_output)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._qk_norm_rope(positions, q, k)
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)


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
        self.is_minimax_m3_sparse_attention = True
        self.hidden_size = config.hidden_size
        self.layer_num = layer_id
        self.layer_name = f"{prefix}.attn"
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
        self.kv_cache_dtype = cache_config

        sparse_cfg = config.sparse_attention_config
        sparse_block_size = sparse_cfg["sparse_block_size"]
        if sparse_block_size != SPARSE_BLOCK_SIZE:
            raise ValueError(
                "MiniMax-M3 native sparse attention requires sparse_block_size "
                f"{SPARSE_BLOCK_SIZE}, got {sparse_block_size}."
            )
        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        self.num_idx_heads = self.num_kv_heads
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.index_q_size = self.num_idx_heads * self.idx_head_dim
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]
        self.init_blocks = sparse_cfg.get("sparse_init_block", 0)
        self.local_blocks = sparse_cfg.get("sparse_local_block", 0)
        score_type = sparse_cfg.get("sparse_score_type", "max")
        if score_type != "max":
            raise ValueError(
                "MiniMax-M3 native sparse attention only supports "
                f"sparse_score_type='max', got {score_type!r}."
            )

        self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            self.total_idx_heads,
            self.idx_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary_dim = int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=_rope_theta(config),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        _minimax_m3_cos_sin_cache(self.rotary_emb, self.q_norm.weight)
        self.index_q_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
        self.index_rotary_emb = self.rotary_emb
        self.kv_cache = torch.tensor([])
        self.index_cache = torch.tensor([])
        # ASM decode path (ATOM_M3_SPARSE_USE_ASM_PA): page-16 SHUFFLE K/V *views*
        # of `self.kv_cache`, derived lazily by `_ensure_asm_shuffle_views()`.
        # The allocation/binding is unchanged from the non-ASM path (the backend
        # gives us the plain page-128 `self.kv_cache`); we only reinterpret its
        # bytes as page-16 SHUFFLE here. index cache stays the page-128
        # `index_cache` above.
        self.kv_cache_k = torch.tensor([])
        self.kv_cache_v = torch.tensor([])
        self.k_scale = self.v_scale = None
        self._use_asm_pa = bool(envs.ATOM_M3_SPARSE_USE_ASM_PA)
        compilation_config = get_current_atom_config().compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        attn_output = torch.ops.aiter.minimax_m3_sparse_attention_native(
            qkv,
            positions,
            self.layer_name,
            self.q_size,
        )
        return self.o_proj(attn_output)

    def _ensure_asm_shuffle_views(self) -> None:
        """Lazily derive the page-16 SHUFFLE K/V views from ``self.kv_cache``.

        The backend binds ``self.kv_cache`` as the plain page-128 combined cache
        (shape ``[N, 2, 128, num_kv_heads, head_dim]``) -- identical to the
        non-ASM path, no allocation change. AITER's ``pa_fwd_asm`` / the page-16
        SHUFFLE writer need a 5D page-16 view, so we reinterpret each layer's K/V
        slice (no bytes moved): one logical 128-page == 8 contiguous physical
        16-pages, so ``N`` logical blocks become ``N*8`` physical 16-pages:
            K: [N*8, num_kv_heads, head_dim//x, 16, x]
            V: [N*8, num_kv_heads, 16//x, head_dim, x]
        Both the write (reshape_and_cache asm_layout=True) and the read
        (pa_fwd_asm) go through THESE views, so the page-16 interpretation is
        self-consistent regardless of the plain layout. Idempotent: rebuilds only
        when the underlying ``self.kv_cache`` storage changes.
        """
        from atom.model_ops.minimax_m3.sparse_attn import (
            ASM_PAGE_SIZE,
            PAGES_PER_SPARSE_BLOCK,
        )

        if self.kv_cache.numel() == 0:
            return
        key_cache, value_cache = self.kv_cache.unbind(1)  # each [N, 128, h, hd]
        if (
            self.kv_cache_k.numel() != 0
            and self.kv_cache_k.data_ptr() == key_cache.data_ptr()
        ):
            return  # views already derived from this storage
        x = 16 // self.kv_cache.element_size()
        num_blocks = key_cache.shape[0]
        num_phys16 = num_blocks * PAGES_PER_SPARSE_BLOCK
        # .view (not .reshape): each unbound slice is contiguous, so this is
        # guaranteed zero-copy -- writes must land in the real pool, never a copy.
        self.kv_cache_k = key_cache.view(
            num_phys16,
            self.num_kv_heads,
            self.head_dim // x,
            ASM_PAGE_SIZE,
            x,
        )
        self.kv_cache_v = value_cache.view(
            num_phys16,
            self.num_kv_heads,
            ASM_PAGE_SIZE // x,
            self.head_dim,
            x,
        )

    def _insert_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        index_k: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.index_cache.numel() == 0:
            return
        if self._use_asm_pa:
            self._ensure_asm_shuffle_views()
            if self.kv_cache_k.numel() == 0:
                return
            # Page-16 SHUFFLE write for the ASM decode path.
            aiter.reshape_and_cache(
                k.view(-1, self.num_kv_heads, self.head_dim),
                v.view(-1, self.num_kv_heads, self.head_dim),
                self.kv_cache_k,
                self.kv_cache_v,
                slot_mapping,
                kv_cache_dtype="auto",
                k_scale=None,
                v_scale=None,
                asm_layout=True,
            )
        else:
            if self.kv_cache.numel() == 0:
                return
            key_cache, value_cache = self.kv_cache.unbind(1)
            kv_cache_dtype = (
                "auto" if self.kv_cache_dtype == "bf16" else self.kv_cache_dtype
            )
            aiter.reshape_and_cache(
                k.view(-1, self.num_kv_heads, self.head_dim),
                v.view(-1, self.num_kv_heads, self.head_dim),
                key_cache,
                value_cache,
                slot_mapping,
                kv_cache_dtype=kv_cache_dtype,
                k_scale=None,
                v_scale=None,
                asm_layout=False,
            )
        self.index_cache.view(-1, self.idx_head_dim)[slot_mapping] = index_k.to(
            self.index_cache.dtype
        )

    def _prefix_lens(self, cu_seqlens_q: torch.Tensor, seq_lens: torch.Tensor):
        query_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        return seq_lens - query_lens

    def _run_prefill_sparse(
        self,
        q: torch.Tensor,
        index_q: torch.Tensor,
        sparse_metadata,
    ) -> torch.Tensor:
        prefill_metadata = sparse_metadata.prefill
        assert prefill_metadata is not None
        cu_seqlens_q = prefill_metadata.cu_seqlens_q
        seq_lens = prefill_metadata.seq_lens
        prefix_lens = prefill_metadata.context_lens
        block_tables = prefill_metadata.block_table
        topk_idx = minimax_m3_index_topk(
            index_q,
            self.index_cache,
            block_tables,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            prefill_metadata.max_query_len,
            prefill_metadata.max_seq_len,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            self.num_kv_heads,
            self.scaling,
        )
        output = torch.empty_like(q)
        if self._use_asm_pa:
            minimax_m3_sparse_attn_prefill_asm(
                q,
                self.kv_cache_k,
                self.kv_cache_v,
                topk_idx,
                block_tables,
                prefill_metadata.query_req_id,
                prefill_metadata.query_abs_pos,
                prefill_metadata.per_token_qo_indptr,
                self.num_kv_heads,
                self.scaling,
                output,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
            )
        else:
            minimax_m3_sparse_attn(
                q,
                self.kv_cache,
                topk_idx,
                block_tables,
                cu_seqlens_q,
                seq_lens,
                prefix_lens,
                prefill_metadata.max_query_len,
                self.num_kv_heads,
                self.scaling,
                output,
            )
        return output

    def _run_decode_sparse(
        self,
        q: torch.Tensor,
        index_q: torch.Tensor,
        sparse_metadata,
    ) -> torch.Tensor:
        decode_metadata = sparse_metadata.decode
        assert decode_metadata is not None
        topk_idx = minimax_m3_index_topk_decode(
            index_q,
            self.index_cache,
            decode_metadata.block_table,
            decode_metadata.seq_lens,
            sparse_metadata.max_seq_len,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            self.num_kv_heads,
            self.scaling,
        )
        output = torch.empty_like(q)
        if self._use_asm_pa:
            if self.num_kv_heads != 1:
                raise NotImplementedError(
                    "ATOM_M3_SPARSE_USE_ASM_PA requires per-rank num_kv_heads == 1 "
                    "(tensor-parallel size >= 4); ASM PA shares one block_table "
                    f"across kv heads. Got num_kv_heads={self.num_kv_heads}."
                )
            minimax_m3_sparse_attn_decode_asm(
                q,
                self.kv_cache_k,
                self.kv_cache_v,
                topk_idx,
                decode_metadata.block_table,
                decode_metadata.seq_lens,
                self.num_kv_heads,
                self.scaling,
                output,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
            )
        else:
            minimax_m3_sparse_attn_decode(
                q,
                self.kv_cache,
                topk_idx,
                decode_metadata.block_table,
                decode_metadata.seq_lens,
                self.num_kv_heads,
                self.scaling,
                output,
            )
        return output

    def sparse_attention_forward_impl(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        fwd_ctx = get_forward_context()
        if self._use_asm_pa:
            # Derive the page-16 SHUFFLE K/V views from the (plain) self.kv_cache
            # the backend bound. No-op once derived / when cache is unbound.
            self._ensure_asm_shuffle_views()
        # self.kv_cache is the source of truth for "cache bound" in both paths;
        # the ASM views are derived from it.
        main_cache_unbound = self.kv_cache.numel() == 0
        if (
            fwd_ctx.context.is_dummy_run
            or fwd_ctx.attn_metadata is None
            or main_cache_unbound
            or self.index_cache.numel() == 0
        ):
            return torch.empty(
                (qkv.shape[0], self.q_size), dtype=qkv.dtype, device=qkv.device
            )

        attn_metadata = fwd_ctx.attn_metadata
        sparse_metadata = getattr(attn_metadata, "minimax_m3_sparse_metadata", None)
        if sparse_metadata is None:
            sparse_metadata = attn_metadata
        if _can_use_fused_minimax_m3_attention_preproc(
            qkv,
            self.rotary_emb,
            self.q_norm.weight,
            self.k_norm.weight,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
        ):
            qkv = qkv.contiguous()
            q = torch.empty(
                (qkv.shape[0], self.q_size), dtype=qkv.dtype, device=qkv.device
            )
            index_q = torch.empty(
                (qkv.shape[0], self.index_q_size), dtype=qkv.dtype, device=qkv.device
            )
            cos_sin_cache = _minimax_m3_cos_sin_cache(self.rotary_emb, qkv)
            if self._use_asm_pa:
                # Triton fallback that writes the main KV cache in page-16
                # SHUFFLE layout (the aiter fused kernel writes plain page-128).
                minimax_m3_fused_qknorm_rope_kv_insert_shuffle(
                    qkv,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    cos_sin_cache,
                    positions,
                    self.num_heads,
                    self.num_kv_heads,
                    self.rotary_emb.rotary_dim,
                    self.q_norm.variance_epsilon,
                    self.index_q_norm.weight,
                    self.index_k_norm.weight,
                    self.num_idx_heads,
                    sparse_metadata.slot_mapping,
                    self.kv_cache_k,
                    self.kv_cache_v,
                    self.index_cache,
                    q,
                    index_q,
                    self.idx_head_dim,
                )
            else:
                aiter.fused_qknorm_idxrqknorm(
                    qkv,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    cos_sin_cache,
                    positions,
                    self.num_heads,
                    self.num_kv_heads,
                    self.rotary_emb.rotary_dim,
                    self.q_norm.variance_epsilon,
                    self.index_q_norm.weight,
                    self.index_k_norm.weight,
                    self.num_idx_heads,
                    sparse_metadata.slot_mapping,
                    self.kv_cache,
                    self.index_cache,
                    self.kv_cache.shape[2],
                    q,
                    index_q,
                    sparse_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    None,
                    None,
                )
            q = q.view(-1, self.num_heads, self.head_dim)
            index_q = index_q.view(-1, self.num_idx_heads, self.idx_head_dim)
            if getattr(sparse_metadata, "num_prefills", 0) > 0:
                output = self._run_prefill_sparse(q, index_q, sparse_metadata)
            else:
                output = self._run_decode_sparse(q, index_q, sparse_metadata)
            return output.view(-1, self.q_size)

        q, k, v, index_q, index_k = qkv.split(
            [
                self.q_size,
                self.kv_size,
                self.kv_size,
                self.index_q_size,
                self.idx_head_dim,
            ],
            dim=-1,
        )
        q, k = _minimax_m3_gemma_qk_norm(
            q,
            k,
            self.q_norm,
            self.k_norm,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        q, k = self.rotary_emb(positions, q, k)

        index_q, index_k = _minimax_m3_gemma_qk_norm(
            index_q,
            index_k,
            self.index_q_norm,
            self.index_k_norm,
            self.num_idx_heads,
            1,
            self.idx_head_dim,
        )
        index_q, index_k = self.index_rotary_emb(positions, index_q, index_k)

        self._insert_kv(k, v, index_k, sparse_metadata.slot_mapping)

        q = q.view(-1, self.num_heads, self.head_dim)
        index_q = index_q.view(-1, self.num_idx_heads, self.idx_head_dim)
        if getattr(sparse_metadata, "num_prefills", 0) > 0:
            output = self._run_prefill_sparse(q, index_q, sparse_metadata)
        else:
            output = self._run_decode_sparse(q, index_q, sparse_metadata)
        return output.view(-1, self.q_size)


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
                quant_config=quant_config,
                reduce_results=False,
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
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.input_layernorm
            )

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

        hidden_states, _ = fused_allreduce_gemma_rms_norm(
            hidden_states, residual, self.norm
        )
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        num_fused_shared = getattr(self.config, "n_shared_experts", 0) or 0
        return make_minimax_m3_expert_params_mapping(
            self.config.num_local_experts + num_fused_shared
        )


class MiniMaxM3SparseForCausalLM(nn.Module):
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
        # bf16 routes routed experts through MiniMaxM3Bf16Experts with standalone
        # shared experts, so the loader must NOT fuse shared-expert weights into
        # the routed slot. FP4 keeps FusedMoE's fused-shared loading.
        self.disable_fused_shared_loading = _minimax_m3_use_dedicated_bf16_experts(
            atom_config.quant_config
        )
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

    packed_modules_mapping = MiniMaxM3SparseForCausalLM.packed_modules_mapping
    quant_exclude_name_mapping = {
        "language_model.model.": "model.",
        "language_model.lm_head": "lm_head",
    }
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
        # See MiniMaxM3SparseForCausalLM: keep shared experts standalone for bf16.
        self.disable_fused_shared_loading = _minimax_m3_use_dedicated_bf16_experts(
            atom_config.quant_config
        )
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
