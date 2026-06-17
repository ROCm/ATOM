# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM plugin wrappers for MiniMax-M3.

This module uses vLLM's generic plugin/multimodal interfaces, while MiniMax-M3
specific processor, vision, and sparse-attention helpers are implemented in
ATOM-owned modules.
"""

from collections.abc import Iterable

import aiter
import torch
from aiter import ActivationType
from aiter.dist.parallel_state import get_pp_group
from torch import nn
from transformers import PretrainedConfig
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    BlockQuantScaleParameter,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.torch_utils import is_quantized_kv_cache, kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import FullAttentionSpec, get_kv_quant_mode

from atom.config import Config, get_current_atom_config
from atom.model_loader.loader import WeightsMapper, load_model_in_plugin_mode
from atom.model_ops.layernorm import fused_allreduce_gemma_rms_norm
from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.minimax_m3.gemma_rmsnorm import (
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.swiglu_oai import swiglu_oai_split
from atom.model_ops.utils import atom_parameter
from atom.models import minimax_m3 as minimax_m3_base
from atom.models.minimax_m3 import (
    MiniMaxM3SparseForCausalLM as NativeMiniMaxM3ForCausalLM,
)
from atom.models.utils import IntermediateTensors, maybe_prefix
from atom.plugin.vllm.attention.ops import (
    minimax_m3_sparse_attention,
    minimax_m3_sparse_attention_insert_kv,
    minimax_m3_sparse_attention_preproc,
)
from atom.plugin.vllm.model_wrapper import ATOMForConditionalGeneration
from atom.plugin.vllm.models.minimax_m3_mm_preprocess import (
    MiniMaxM3VLDummyInputsBuilder,
    MiniMaxM3VLMultiModalProcessor,
    MiniMaxM3VLProcessingInfo,
)
from atom.plugin.vllm.models.minimax_m3_sparse_attention import (
    MiniMaxM3IndexerCache,
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseMetadata,
)
from atom.plugin.vllm.models.minimax_m3_vision_tower import MiniMaxVLVisionModel


def _can_use_fused_minimax_m3_attention_preproc(
    qkv: torch.Tensor,
    rotary_emb: nn.Module,
    *weights: torch.Tensor,
) -> bool:
    return (
        hasattr(aiter, "fused_minimax_m3_qknorm_rope_kv_insert")
        and qkv.dim() == 2
        and qkv.is_cuda
        and qkv.dtype in (torch.float16, torch.bfloat16)
        and getattr(rotary_emb, "head_size", None) == 128
        and getattr(rotary_emb, "rotary_dim", 0) > 0
        and getattr(rotary_emb, "is_neox_style", False)
        and all(w.dtype == qkv.dtype for w in weights)
    )


class MiniMAXGemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm from ame MiniMax-M3 AMD implementation."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
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


class MiniMaxM3MLP(nn.Module):
    """Shared-expert MLP using vLLM linear layers (matches code/ame minimax_m3)."""

    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
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
                f"Unsupported activation: {config.hidden_act}. "
                "Only swigluoai is supported."
            )
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_beta = config.swiglu_beta
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        if isinstance(gate_up, tuple):
            gate_up = gate_up[0]
        x = swiglu_oai_split(
            gate_up,
            alpha=self.swiglu_alpha,
            beta=self.swiglu_beta,
            limit=self.swiglu_limit,
        )
        x = self.down_proj(x)
        if isinstance(x, tuple):
            x = x[0]
        return x


def adjust_block_scale_shard(
    weight_block_size: tuple[int, ...] | None,
    shard_size: int,
    shard_offset: int,
) -> tuple[int, int]:
    assert weight_block_size is not None
    block_n = weight_block_size[0]
    shard_offset = (shard_offset + block_n - 1) // block_n
    shard_size = (shard_size + block_n - 1) // block_n
    return shard_size, shard_offset


class MinimaxM3QKVParallelLinearWithIndexer(QKVParallelLinear):
    """QKV projection fused with a lightning-indexer's index_q/index_k.

    NOTE: MiniMax-M3-specific. This is tailored to the M3 sparse-attention
    layers (it assumes the indexer's head count equals the KV head count and
    shares the main head_dim); it is not a general-purpose linear layer. It
    lives here only to sit alongside QKVParallelLinear, whose sharding /
    weight-loading machinery it reuses.

    A single column-parallel GEMM emits, per rank::

        [q | k | v | index_q | index_k]

    ``index_q`` must have the same head count as the KV heads
    (``total_num_index_heads == total_num_kv_heads``) and ``index_head_size ==
    head_size``, so it shards exactly like K/V -- including the KV-head
    *replication* path when ``tp_size > total_num_kv_heads`` (this is what makes
    a TP size greater than the KV-head count work). ``index_k`` is a single
    shared head, replicated to every rank.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        total_num_index_heads: int,
        index_head_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # index_q rides the KV-head sharding/replication path, so its head count
        # must match the KV heads.
        assert total_num_index_heads == total_num_kv_heads, (
            "MinimaxM3QKVParallelLinearWithIndexer requires "
            "total_num_index_heads == total_num_kv_heads"
        )
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.total_num_index_heads = total_num_index_heads
        self.index_head_size = index_head_size

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        # index_q shards identically to the KV heads.
        self.num_index_heads = self.num_kv_heads

        # Global per-group sizes (replicated groups counted x tp_size, matching
        # the QKVParallelLinear convention). index_k is a single replicated head.
        q = self.num_heads * self.head_size
        kv = self.num_kv_heads * self.head_size
        iq = self.num_index_heads * self.index_head_size
        ik = self.index_head_size
        self.output_sizes = [
            q * tp_size,  # q
            kv * tp_size,  # k
            kv * tp_size,  # v
            iq * tp_size,  # index_q
            ik * tp_size,  # index_k (replicated)
        ]

        # Skip QKVParallelLinear.__init__ (3-group layout); build the 5-group
        # column-parallel weight directly.
        ColumnParallelLinear.__init__(
            self,
            input_size=self.hidden_size,
            output_size=sum(self.output_sizes),
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def validate_shard_id(self, loaded_shard_id: str | None) -> None:
        if loaded_shard_id is None:
            return
        if loaded_shard_id not in ("q", "k", "v", "index_q", "index_k"):
            raise ValueError(
                "Shard id for MinimaxM3QKVParallelLinearWithIndexer must be one of "
                "'q', 'k', 'v', 'index_q', 'index_k'; got "
                f"{loaded_shard_id}."
            )

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int | None:
        h = self.head_size
        nq, nkv, nidx = self.num_heads, self.num_kv_heads, self.num_index_heads
        return {
            "q": 0,
            "k": nq * h,
            "v": (nq + nkv) * h,
            "index_q": (nq + 2 * nkv) * h,
            "index_k": (nq + 2 * nkv + nidx) * h,
        }.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int | None:
        h = self.head_size
        return {
            "q": self.num_heads * h,
            "k": self.num_kv_heads * h,
            "v": self.num_kv_heads * h,
            "index_q": self.num_index_heads * h,
            "index_k": self.index_head_size,
        }.get(loaded_shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        self.validate_shard_id(loaded_shard_id)
        # Index checkpoints are never pre-fused on disk; a shard id is always given.
        assert loaded_shard_id in ("q", "k", "v", "index_q", "index_k")

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        # index_k is fully replicated: num_heads == tp_size makes
        # load_qkv_weight pick shard_id_int == 0 on every rank. q/k/v/index_q ride
        # the KV-head replication factor.
        num_heads = (
            self.tp_size if loaded_shard_id == "index_k" else self.num_kv_head_replicas
        )
        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=num_heads,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        # Unquantized (bf16) path. MXFP8 checkpoints use weight_loader_v2; this
        # keeps an unquantized load correct too.
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in ("q", "k", "v", "index_q", "index_k")
        output_dim = getattr(param, "output_dim", None)
        assert output_dim is not None

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        param_data = param.data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == "q":
            shard_rank = self.tp_rank
        elif loaded_shard_id == "index_k":
            shard_rank = 0  # replicated to every rank
        else:
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        loaded_weight = loaded_weight.narrow(
            output_dim, shard_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class MiniMaxM3Attention(nn.Module):
    """Dense attention with vLLM linear/RoPE/RMSNorm, matching ame AMD path."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config=None,
    ) -> None:
        super().__init__()
        del layer_id, cache_config
        vllm_cache_config = get_current_vllm_config().cache_config

        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % tp_size != 0:
            raise ValueError("num_attention_heads must be divisible by TP size.")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError("num_key_value_heads must divide TP size.")
        elif tp_size % self.total_num_kv_heads != 0:
            raise ValueError("TP size must divide num_key_value_heads replication.")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
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
        # reduce_results=False: the attention all-reduce is fused with the
        # following post_attention_layernorm (GemmaRMSNorm) in the decoder layer
        # via fused_allreduce_gemma_rms_norm.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_cache_config,
            quant_config=None,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if _can_use_fused_minimax_m3_attention_preproc(
            qkv, self.rotary_emb, self.q_norm.weight, self.k_norm.weight
        ):
            qkv = qkv.contiguous()
            q = torch.empty(
                (qkv.shape[0], self.q_size), dtype=qkv.dtype, device=qkv.device
            )
            cos_sin_cache = self.rotary_emb._match_cos_sin_cache_dtype(qkv)
            aiter.fused_minimax_m3_qknorm_rope_kv_insert(
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
            output, _ = self.o_proj(attn_output)
            return output

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
        k = self.k_norm(k_by_head).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM3SparseAttention(nn.Module):
    """vLLM-plugin MiniMax-M3 sparse attention using vLLM's sparse backend."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
        cache_config: str = "bf16",
    ) -> None:
        super().__init__()
        del layer_id
        AttentionLayerBase.register(self.__class__)
        self.hidden_size = config.hidden_size
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

        self.q_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )
        self.index_q_norm = MiniMAXGemmaRMSNorm(
            self.idx_head_dim, eps=config.rms_norm_eps
        )
        self.index_k_norm = MiniMAXGemmaRMSNorm(
            self.idx_head_dim, eps=config.rms_norm_eps
        )
        self.index_rotary_emb = self.rotary_emb

        vllm_config = get_current_vllm_config()
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = (
            cache_config.cache_dtype
            if hasattr(cache_config, "cache_dtype")
            else cache_config
        )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        self.index_cache_torch_dtype = (
            vllm_config.model_config.dtype
            if is_quantized_kv_cache(self.kv_cache_dtype)
            else self.kv_cache_torch_dtype
        )
        self.attn_backend = MiniMaxM3SparseBackend
        self.impl = self.attn_backend.get_impl_cls()(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            sparse_block_size=sparse_cfg["sparse_block_size"],
            init_blocks=sparse_cfg.get("sparse_init_block", 0),
            local_blocks=sparse_cfg.get("sparse_local_block", 0),
            score_type=sparse_cfg.get("sparse_score_type", "max"),
            num_index_heads=self.num_idx_heads,
            index_head_dim=self.idx_head_dim,
        )

        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        self.kv_cache = torch.tensor([])
        self.index_cache = MiniMaxM3IndexerCache(
            head_dim=self.idx_head_dim,
            dtype=self.index_cache_torch_dtype,
            prefix=f"{self.layer_name}.index_cache",
            cache_config=vllm_config.cache_config,
        )

    def get_attn_backend(self):
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config):
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _insert_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        index_key: torch.Tensor,
        kv_cache: torch.Tensor,
        index_kv_cache: torch.Tensor,
        main_meta: MiniMaxM3SparseMetadata | None,
        index_meta: MiniMaxM3SparseMetadata | None,
    ) -> None:
        if main_meta is None and index_meta is None:
            return
        if not isinstance(main_meta, MiniMaxM3SparseMetadata) or not isinstance(
            index_meta, MiniMaxM3SparseMetadata
        ):
            raise TypeError("MiniMax-M3 sparse attention received wrong metadata type.")

        main_num_tokens = main_meta.num_actual_tokens
        index_num_tokens = index_meta.num_actual_tokens
        if main_num_tokens != index_num_tokens:
            raise RuntimeError(
                "MiniMax-M3 main/index metadata token count mismatch: "
                f"{main_num_tokens} vs {index_num_tokens}."
            )
        key = key[:main_num_tokens]
        value = value[:main_num_tokens]
        index_key = index_key[:index_num_tokens]
        main_slot_mapping = main_meta.slot_mapping[:main_num_tokens]
        index_slot_mapping = index_meta.slot_mapping[:index_num_tokens]

        key_cache, value_cache = kv_cache.unbind(1)
        scale = torch.ones((), device=key.device)
        ops.reshape_and_cache_flash(
            key.view(-1, self.num_kv_heads, self.head_dim),
            value.view(-1, self.num_kv_heads, self.head_dim),
            key_cache,
            value_cache,
            main_slot_mapping,
            self.kv_cache_dtype,
            scale,
            scale,
        )
        idx_cache = index_kv_cache.view(-1, self.idx_head_dim)
        idx_cache[index_slot_mapping] = index_key.to(idx_cache.dtype)

    def minimax_m3_sparse_attention_forward(
        self,
        query: torch.Tensor,
        index_query: torch.Tensor,
        kv_cache: torch.Tensor,
        index_kv_cache: torch.Tensor,
        main_meta: MiniMaxM3SparseMetadata | None,
        index_meta: MiniMaxM3SparseMetadata | None,
    ) -> torch.Tensor:
        del index_meta
        output = torch.empty_like(query)
        output = self.impl.forward(
            self,
            query,
            index_query,
            kv_cache,
            index_kv_cache,
            output,
        )
        if isinstance(main_meta, MiniMaxM3SparseMetadata):
            num_tokens = main_meta.num_actual_tokens
            if num_tokens < output.shape[0]:
                output[num_tokens:].zero_()
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        if isinstance(qkv, tuple):
            qkv = qkv[0]

        if _can_use_fused_minimax_m3_attention_preproc(
            qkv,
            self.rotary_emb,
            self.q_norm.weight,
            self.k_norm.weight,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
        ) and (
            self.kv_cache_torch_dtype == qkv.dtype
            or is_quantized_kv_cache(self.kv_cache_dtype)
        ):
            qkv = qkv.contiguous()
            q = torch.empty(
                (qkv.shape[0], self.q_size), dtype=qkv.dtype, device=qkv.device
            )
            index_q = torch.empty(
                (qkv.shape[0], self.index_q_size), dtype=qkv.dtype, device=qkv.device
            )
            kv_scale = None
            if is_quantized_kv_cache(self.kv_cache_dtype):
                kv_scale = torch.ones((), dtype=torch.float32, device=qkv.device)

            cos_sin_cache = self.rotary_emb._match_cos_sin_cache_dtype(qkv)
            torch.ops.aiter.minimax_m3_sparse_attention_preproc(
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
                self.kv_cache,
                self.index_cache.kv_cache,
                q,
                index_q,
                self.layer_name,
                self.kv_cache_dtype,
                kv_scale,
                kv_scale,
            )

            attn_output = torch.ops.aiter.minimax_m3_sparse_attention(
                q,
                index_q,
                self.kv_cache,
                self.index_cache.kv_cache,
                self.layer_name,
            )
            out = self.o_proj(attn_output)
            if isinstance(out, tuple):
                out = out[0]
            return out

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
        q = self.q_norm(q.view(*q.shape[:-1], self.num_heads, self.head_dim)).view(
            q.shape
        )
        k = self.k_norm(k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)).view(
            k.shape
        )
        q, k = self.rotary_emb(positions, q, k)

        index_q = self.index_q_norm(
            index_q.view(*index_q.shape[:-1], self.num_idx_heads, self.idx_head_dim)
        ).view(index_q.shape)
        index_k = self.index_k_norm(index_k)
        index_q, index_k = self.index_rotary_emb(positions, index_q, index_k)

        torch.ops.aiter.minimax_m3_sparse_attention_insert_kv(
            k,
            v,
            index_k,
            self.kv_cache,
            self.index_cache.kv_cache,
            self.layer_name,
        )

        attn_output = torch.ops.aiter.minimax_m3_sparse_attention(
            q,
            index_q,
            self.kv_cache,
            self.index_cache.kv_cache,
            self.layer_name,
        )
        out = self.o_proj(attn_output)
        if isinstance(out, tuple):
            out = out[0]
        return out


class MiniMaxM3MoE(nn.Module):
    """Sigmoid-routed MoE using ATOM's MiniMax-M3 expert kernels."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        expert_quant_config=None,
        params_dtype: torch.dtype | None = None,
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
        old_wlp = self.gate.weight.weight_loader_process
        self.gate.weight = atom_parameter(self.gate.weight.data.to(torch.float32))
        self.gate.weight.weight_loader_process = old_wlp

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
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
            quant_config=expert_quant_config,
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
        router_logits = torch.nn.functional.linear(
            hidden_states.float(), self.gate.weight.float()
        )
        routed_output = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        if not self.fuse_shared_experts and self.routed_scaling_factor != 1.0:
            routed_output = routed_output * self.routed_scaling_factor

        if self.shared_experts is not None:
            routed_output = routed_output + self.shared_experts(hidden_states)

        return routed_output.view(orig_shape)


class MiniMaxM3DecoderLayer(minimax_m3_base.MiniMaxM3DecoderLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: str = "bf16",
        quant_config=None,
        params_dtype: torch.dtype | None = None,
        layer_num: int = 0,
    ) -> None:
        nn.Module.__init__(self)
        atom_quant_config = quant_config
        vllm_quant_config = (
            get_current_atom_config().plugin_config.vllm_config.quant_config
        )
        attn_cls = (
            MiniMaxM3SparseAttention
            if layer_num in minimax_m3_base._sparse_attention_layer_ids(config)
            else MiniMaxM3Attention
        )
        self.self_attn = attn_cls(
            config=config,
            layer_id=layer_num,
            quant_config=vllm_quant_config,
            prefix=f"{prefix}.self_attn",
            cache_config=cache_config,
        )
        self.is_moe_layer = minimax_m3_base._is_moe_layer(config, layer_num)
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxM3MoE(
                config=config,
                layer_id=layer_num,
                quant_config=vllm_quant_config,
                expert_quant_config=atom_quant_config,
                params_dtype=params_dtype,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.mlp = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.dense_intermediate_size,
                quant_config=vllm_quant_config,
                reduce_results=False,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = MiniMAXGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MiniMAXGemmaRMSNorm(
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


class MiniMaxM3SparseForCausalLM(NativeMiniMaxM3ForCausalLM):
    weights_mapping = {
        ".scale": ".weight_scale_inv",
    }

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = MiniMaxM3DecoderLayer,
    ) -> None:
        super().__init__(
            atom_config=atom_config,
            prefix=prefix,
            layer_type=layer_type,
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        num_fused_shared = getattr(self.config, "n_shared_experts", 0) or 0
        return minimax_m3_base.make_minimax_m3_expert_params_mapping(
            self.config.num_local_experts + num_fused_shared
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor | IntermediateTensors:
        model = self.model
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else model.get_input_embeddings(input_ids)
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in model.layers[model.start_layer : model.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = fused_allreduce_gemma_rms_norm(
            hidden_states, residual, model.norm
        )
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    MiniMaxM3VLMultiModalProcessor,
    info=MiniMaxM3VLProcessingInfo,
    dummy_inputs=MiniMaxM3VLDummyInputsBuilder,
)
class MiniMaxM3SparseForConditionalGeneration_(nn.Module, SupportsMultiModal):
    supports_encoder_tp_data = True
    packed_modules_mapping = MiniMaxM3SparseForCausalLM.packed_modules_mapping
    weights_mapping = MiniMaxM3SparseForCausalLM.weights_mapping
    hf_to_atom_mapper = WeightsMapper(
        orig_to_new_prefix={
            "multi_modal_projector.": "vision_tower.multi_modal_projector.",
            "patch_merge_mlp.": "vision_tower.patch_merge_mlp.",
        },
        orig_to_new_substr={
            ".mlp.fc1.": ".fc1.",
            ".mlp.fc2.": ".fc2.",
        },
    )
    hf_to_vllm_mapper = hf_to_atom_mapper

    def __init__(self, atom_config: Config, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        self.atom_config = atom_config
        config = atom_config.hf_config
        vllm_config = atom_config.plugin_config.vllm_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        assert self.multimodal_config is not None
        self.use_data_parallel = self.multimodal_config.mm_encoder_tp_mode == "data"

        text_hidden_size = getattr(config.text_config, "hidden_size", None)
        assert text_hidden_size is not None, "text_config.hidden_size is required"
        projector_hidden_size = getattr(config, "projector_hidden_size", None)

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_tower = MiniMaxVLVisionModel(
                config=PretrainedConfig.from_dict(config.vision_config),
                text_hidden_size=text_hidden_size,
                projector_hidden_size=projector_hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = MiniMaxM3SparseForCausalLM(
                atom_config=atom_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        text_config = config.text_config
        self.vocab_size = text_config.vocab_size

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality == "image":
            return MiniMaxM3VLProcessingInfo.IMAGE_TOKEN
        if modality == "video":
            return MiniMaxM3VLProcessingInfo.VIDEO_TOKEN
        raise ValueError(f"Unsupported modality: {modality!r}")

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if pixel_values is None:
            return None
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    def _parse_and_validate_video_input(self, **kwargs: object) -> dict | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        if pixel_values_videos is None:
            return None
        return {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

    def _process_image_input(self, image_input: dict) -> tuple[torch.Tensor, ...]:
        pixel_values: torch.Tensor = image_input["pixel_values"].type(
            self.vision_tower.dtype
        )
        grid_thw: torch.Tensor = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                pixel_values,
                grid_thw.tolist(),
                rope_type="rope_3d",
            )

        image_embeds = self.vision_tower(
            pixel_values=pixel_values,
            grid_thw=grid_thw.tolist(),
        )
        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(self, video_input: dict) -> tuple[torch.Tensor, ...]:
        pixel_values: torch.Tensor = video_input["pixel_values_videos"].type(
            self.vision_tower.dtype
        )
        grid_thw: torch.Tensor = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                pixel_values,
                grid_thw.tolist(),
                rope_type="rope_3d",
            )

        video_embeds = self.vision_tower(
            pixel_values=pixel_values,
            grid_thw=grid_thw.tolist(),
        )
        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> dict[str, dict]:
        mm_input_by_modality: dict[str, dict] = {}
        for input_key in kwargs:
            if input_key == "pixel_values" and "image" not in mm_input_by_modality:
                image_input = self._parse_and_validate_image_input(**kwargs)
                if image_input is not None:
                    mm_input_by_modality["image"] = image_input
            if (
                input_key == "pixel_values_videos"
                and "video" not in mm_input_by_modality
            ):
                video_input = self._parse_and_validate_video_input(**kwargs)
                if video_input is not None:
                    mm_input_by_modality["video"] = video_input
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: list[torch.Tensor] = []
        for modality, multimodal_input in mm_input_by_modality.items():
            if modality == "image":
                multimodal_embeddings.extend(
                    self._process_image_input(multimodal_input)
                )
            if modality == "video":
                multimodal_embeddings.extend(
                    self._process_video_input(multimodal_input)
                )
        return tuple(multimodal_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.language_model.compute_logits(hidden_states)
        return None if logits is None else logits[..., : self.vocab_size]

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        return load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            prefix="model.",
            weights_mapper=self.hf_to_atom_mapper,
        )


@MULTIMODAL_REGISTRY.register_processor(
    MiniMaxM3VLMultiModalProcessor,
    info=MiniMaxM3VLProcessingInfo,
    dummy_inputs=MiniMaxM3VLDummyInputsBuilder,
)
class MiniMaxM3SparseForConditionalGeneration(ATOMForConditionalGeneration):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality == "image":
            return MiniMaxM3VLProcessingInfo.IMAGE_TOKEN
        if modality == "video":
            return MiniMaxM3VLProcessingInfo.VIDEO_TOKEN
        raise ValueError(f"Unsupported modality: {modality!r}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.model.load_weights(weights)
