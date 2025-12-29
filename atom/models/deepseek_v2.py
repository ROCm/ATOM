# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only DeepseekV2/DeepseekV3 model."""
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union

import torch
from aiter import (
    QuantType,
    concat_and_cache_mla,
    cp_gather_indexer_k_quant_cache,
    dtypes,
    flash_attn_varlen_func,
    gemm_a8w8_blockscale,
    get_hip_quant,
    indexer_k_quant_and_cache,
    top_k_per_row_prefill,
    top_k_per_row_decode,
)
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from aiter.ops.triton.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits,
    deepgemm_fp8_paged_mqa_logits_stage1,
)
from aiter.rotary_embedding import get_rope
from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant
from aiter.ops.triton.fused_mxfp4_quant import (
    fused_rms_mxfp4_quant,
    fused_reduce_rms_mxfp4_quant,
)
from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
from torch import nn
from transformers import PretrainedConfig

from atom.config import Config, QuantizationConfig, get_current_atom_config
from atom.model_ops.activation import SiluAndMul
from atom.model_ops.attention_mla import MLAModules
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.fp8_mqa_logits import fp8_mqa_logits
from atom.model_ops.layernorm import LayerNorm, RMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    MergedReplicatedLinear,
    use_triton_gemm,
)
from atom.model_ops.attention_mla import is_rocm_aiter_fp4bmm_enabled
from atom.model_ops.moe import FusedMoE
from atom.model_ops.topK import (
    is_rocm_aiter_fuse_routed_scaling_factor,
    is_rocm_aiter_fusion_shared_expert_enabled,
)
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils.forward_context import get_forward_context
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.decorators import support_torch_compile
from atom.utils import envs
from aiter.jit.utils.torch_guard import torch_compile_guard
# from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

import logging

logger = logging.getLogger("atom")

ENABLE_DS_QKNORM_QUANT_FUSION = envs.ATOM_ENABLE_DS_QKNORM_QUANT_FUSION
ENABLE_ALLREDUCE_RMSNORM_FUSION = envs.ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION
ENABLE_RMSNORM_QUANT_FUSION = envs.ATOM_ENABLE_RMSNORM_QUANT_FUSION


def _fuse_rmsnorm_fp4_quant_fake(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    shuffle: bool = True,                 
    scale_shuffle_padding: bool = True,   
    output_unquantized_inp1: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    m, n1 = x1.shape
    n2 = x2.shape[1] if x2 is not None else 0
    MXFP4_QUANT_BLOCK_SIZE = 32

    out1_quantized = torch.empty((m, n1 // 2), dtype=torch.uint8, device=x1.device)

    scale_n_valid = (n1 + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE

    scale_m = ((m + 255) // 256) * 256
    scale_n = ((scale_n_valid + 7) // 8) * 8

    out1_bs = torch.empty((scale_m, scale_n), dtype=torch.uint8, device=x1.device)

    out2 = None
    if x2 is not None:
        out2 = torch.empty((m, n2), dtype=x1.dtype, device=x1.device)

    out_res1 = None
    if res1 is not None:
        out_res1 = torch.empty((m, n1), dtype=x1.dtype, device=x1.device)

    out1_unquantized = None

    return out1_quantized, out1_bs, out1_unquantized, out2, out_res1


@torch_compile_guard(gen_fake=_fuse_rmsnorm_fp4_quant_fake)
def _fuse_rmsnorm_fp4_quant(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    shuffle: bool = True,                
    scale_shuffle_padding: bool = True,   
    output_unquantized_inp1: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    m = x1.shape[0]

    shuffle_bool = shuffle and (m >= 32)

    (out1_quantized, out1_bs), _out1_unquantized, out2, out_res1 = fused_rms_mxfp4_quant(
        x1=x1,
        x1_weight=x1_weight,
        x1_epsilon=x1_epsilon,
        x2=x2,
        x2_weight=x2_weight,
        x2_epsilon=0.0 if x2_epsilon is None else x2_epsilon,
        res1=res1,
        shuffle=shuffle_bool,
        scale_shuffle_padding=scale_shuffle_padding,
        output_unquantized_inp1=output_unquantized_inp1,
    )

    out1_unquantized = None
    return out1_quantized, out1_bs, out1_unquantized, out2, out_res1


def _fuse_rmsnorm_quant(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: float = 0.0,
    res1: Optional[torch.Tensor] = None,
    dtype_quant=dtypes.fp8,
    shuffle: Optional[bool] = True,
    scale_shuffle_padding: Optional[bool] = True,
    group_size=128,
    output_unquantized_inp1=False,
    transpose_scale=False,
):
    if dtype_quant == dtypes.fp4x2:
        out1_quantized, out1_bs, out1_unquantized, out2, out_res1 = _fuse_rmsnorm_fp4_quant(
        x1,
        x1_weight,
        x1_epsilon,
        x2,
        x2_weight,
        x2_epsilon,
        res1,
        shuffle,
        scale_shuffle_padding,
        output_unquantized_inp1,
    )
    elif dtype_quant == dtypes.fp8:
        (out1_quantized, out1_bs), out1_unquantized, out2, out_res1 = fused_rms_fp8_group_quant(
            x1,
            x1_weight,
            x1_epsilon,
            x2,
            x2_weight,
            x2_epsilon,
            group_size,
            dtype_quant,
            res1,
            output_unquantized_inp1,
            transpose_scale,
        )
    else:
        raise ValueError(f"No fused rmsnorm quant kernel availble for quant dtype: {dtype_quant}.")
    return (out1_quantized, out1_bs), out1_unquantized, out2, out_res1


def _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4_fake(
    hidden_states_quant: torch.Tensor,
    hidden_states_quant_scale: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    shuffle: bool = True,
    scale_shuffle_padding: bool = True,
    output_unquantized_inp1: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]:
    M = hidden_states_quant.shape[0]
    MXFP4_QUANT_BLOCK_SIZE = 32
    device = hidden_states_quant.device
    q_c = torch.empty((M, q_lora_rank // 2), dtype=torch.uint8, device=device)
    scale_n_valid = (q_lora_rank + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    scale_m = ((M + 255) // 256) * 256
    scale_n = ((scale_n_valid + 7) // 8) * 8
    q_c_scale = torch.empty((scale_m, scale_n), dtype=torch.uint8, device=device)
    kv_c_normed = torch.empty((M, kv_lora_rank), dtype=torch.bfloat16, device=device)
    k_pe = torch.empty((M, q_lora_rank + kv_lora_rank + qk_rope_head_dim), dtype=torch.bfloat16, device=device)[..., :qk_rope_head_dim]
    return q_c, q_c_scale, kv_c_normed, k_pe


@torch_compile_guard(gen_fake=_fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4_fake, mutates_args=[])
def _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4(
    hidden_states_quant: torch.Tensor,
    hidden_states_quant_scale: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    shuffle: bool = True,
    scale_shuffle_padding: bool = True,
    output_unquantized_inp1: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]:
    M = hidden_states_quant.shape[0]

    if M >= 32:
        hidden_states_quant_scale = hidden_states_quant_scale.view(torch.uint8).view(hidden_states_quant_scale.shape[0] // 32, -1)
    else:
        hidden_states_quant_scale = hidden_states_quant_scale[:M, ...].view(torch.uint8)

    qkv_lora = gemm_afp4wfp4_preshuffle(
        hidden_states_quant.view(torch.uint8), 
        weight_qkv_a_proj.view(torch.uint8).view(weight_qkv_a_proj.shape[0] // 16, -1),
        hidden_states_quant_scale, 
        weight_scale_qkv_a_proj.view(torch.uint8).view(weight_scale_qkv_a_proj.shape[0] // 32, -1), 
        skip_reduce=True)

    q_c, kv_c, k_pe = torch.split(
                    qkv_lora,
                    [q_lora_rank, kv_lora_rank, qk_rope_head_dim],
                    dim=-1,
                )
    
    shuffle_bool = shuffle and (M >= 32)

    k_pe_reduced = None
    k_pe_reduced_out = None
    if k_pe.dim() == 3:
        device = hidden_states_quant.device
        k_pe_reduced = k_pe
        k_pe_reduced_out = torch.empty((M, q_lora_rank + kv_lora_rank + qk_rope_head_dim), dtype=torch.bfloat16, device=device)[..., :qk_rope_head_dim]
    (q_c, q_c_scale), _, kv_c_normed, _, k_pe_reduced_out = fused_reduce_rms_mxfp4_quant(
        q_c,
        q_a_layernorm_weight,
        q_a_layernorm_variance_epsilon, 
        kv_c,
        kv_a_layernorm_weight,
        kv_a_layernorm_variance_epsilon,
        k_pe_reduced,
        res1=None,
        shuffle=shuffle_bool,
        scale_shuffle_padding=scale_shuffle_padding,
        output_unquantized_inp1=output_unquantized_inp1,
        dtype=torch.bfloat16,
        out3=k_pe_reduced_out,
    )

    if k_pe_reduced_out is not None:
        k_pe = k_pe_reduced_out
    return q_c, q_c_scale, kv_c_normed, k_pe

def _fuse_qkv_a_proj_reduce_rmsnorm_quant(
    hidden_states_quant: torch.Tensor,
    hidden_states_quant_scale: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dtype_quant=dtypes.fp8,
    shuffle: Optional[bool] = False,
    scale_shuffle_padding: Optional[bool] = False,
    group_size: Optional[int] = 128,
    output_unquantized_inp1: Optional[bool] = False,
    transpose_scale: Optional[bool] = False,
):
    if dtype_quant == dtypes.fp4x2:
        q_c, q_c_scale, kv_c_normed, k_pe = _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4(
            hidden_states_quant,
            hidden_states_quant_scale,
            weight_qkv_a_proj,
            weight_scale_qkv_a_proj,
            q_a_layernorm_weight,
            q_a_layernorm_variance_epsilon,
            kv_a_layernorm_weight,
            kv_a_layernorm_variance_epsilon,
            q_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
            shuffle,
            scale_shuffle_padding,
            output_unquantized_inp1,
        )
    elif dtype_quant == dtypes.fp8:
        # TODO add
        raise ValueError(f"No fused rmsnorm quant kernel availble for quant dtype: {dtype_quant}.")
    else:
        raise ValueError(f"No fused rmsnorm quant kernel availble for quant dtype: {dtype_quant}.")

    
    # logger.info(f"{q_c.shape=}, {q_c_scale.shape=}, {kv_c_normed.shape=}, {k_pe.shape=}, {q_c.stride()=}, {q_c_scale.stride()=}, {kv_c_normed.stride()=}, {k_pe.stride()=}")
    return q_c, q_c_scale, kv_c_normed, k_pe

class DeepseekV2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class DeepseekV2MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.reduce_results = reduce_results

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts))
        else:
            self.gate.e_score_correction_bias = None

        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            config=config)

        if config.n_shared_experts is not None and not is_rocm_aiter_fusion_shared_expert_enabled():
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = None
        if self.n_shared_experts is not None and not is_rocm_aiter_fusion_shared_expert_enabled():
            shared_output = self.shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        if hidden_states.dtype != torch.float16:
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits)
            if not is_rocm_aiter_fuse_routed_scaling_factor():
                final_hidden_states = final_hidden_states * self.routed_scaling_factor
        else:
            # Fix FP16 overflow
            # See DeepseekV2DecoderLayer for more details.
            final_hidden_states = self.experts(hidden_states=hidden_states,
                                               router_logits=router_logits)
        if shared_output is not None:
            if hidden_states.dtype != torch.float16:
                final_hidden_states = final_hidden_states + shared_output
            else:
                # Fix FP16 overflow
                # See DeepseekV2DecoderLayer for more details.
                final_hidden_states = final_hidden_states + shared_output \
                    * (1. / self.routed_scaling_factor)
        if self.tp_size > 1 and self.reduce_results:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV32IndexerCache(nn.Module):

    def __init__(self, head_dim: int, dtype: torch.dtype, prefix: str,
                 cache_config: str):
        super().__init__()
        self.kv_cache = [torch.tensor([])]
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    context = forward_context.context
    slot_mapping = attn_metadata.slot_mapping
    if kv_cache.numel() == 0:
        # dummy runner
        return weights
    num_decode_tokens = context.batch_size if not context.is_prefill else 0
    indexer_k_quant_and_cache(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )
    if context.is_prefill:
        prefill_metadata = attn_metadata
        num_prefills = context.batch_size
        total_seq_lens = hidden_states.shape[0]
        k_fp8 = torch.empty(
            [total_seq_lens, head_dim],
            device=k.device,
            dtype=dtypes.fp8)
        k_scale = torch.empty(
            [total_seq_lens, 1],
            device=k.device,
            dtype=torch.float32)
        if prefill_metadata.block_tables.shape[0] < num_prefills:
            new_shape = (num_prefills, prefill_metadata.block_tables.shape[1])
            prefill_metadata.block_tables = torch.full(new_shape, -1, dtype=torch.long, device=prefill_metadata.block_tables.device)
        cp_gather_indexer_k_quant_cache(
            kv_cache,
            k_fp8,
            k_scale.view(dtypes.fp8),
            prefill_metadata.block_tables,
            prefill_metadata.cu_seqlens_q,
            # num_prefills,
        )
        cu_seqlen_ks = prefill_metadata.cu_seqlen_ks
        cu_seqlen_ke = prefill_metadata.cu_seqlen_ke
        num_tokens = hidden_states.shape[0]
        logits = fp8_mqa_logits(Q=q_fp8[num_decode_tokens:num_tokens], KV=k_fp8, kv_scales=k_scale, weights=weights[num_decode_tokens:num_tokens],
                                cu_starts=cu_seqlen_ks, cu_ends=cu_seqlen_ke)
        num_rows = logits.shape[0]
        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
        topk_indices = topk_indices_buffer[
            num_decode_tokens:num_tokens, :topk_tokens
        ]
        top_k_per_row_prefill(
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
        )
    else:
        decode_metadata = attn_metadata
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(context.batch_size, -1, *q_fp8.shape[1:])
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == context.batch_size
        num_padded_tokens = batch_size * next_n
        batch_size, next_n, heads, _ = padded_q_fp8_decode_tokens.shape
        logits = torch.empty([batch_size * next_n, max_model_len], dtype=torch.float32, device="cuda")
        deepgemm_fp8_paged_mqa_logits(padded_q_fp8_decode_tokens, kv_cache, weights[:num_padded_tokens], logits, decode_metadata.context_lens, attn_metadata.block_tables, max_model_len)
        num_rows = logits.shape[0]
        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
        topk_indices = topk_indices_buffer[
            :num_decode_tokens, :topk_tokens
        ]
        top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.context_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
        )
    return weights

def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # profile run
    # NOTE(Chen): create the max possible flattened_kv. So that
    # profile_run can get correct memory usage.
    _flattened_kv = torch.empty([total_seq_lens, head_dim + 4],
                                device=k.device,
                                dtype=torch.uint8)
    _k_fp8 = _flattened_kv[..., :head_dim].view(
        torch.float8_e4m3fn).contiguous()
    _k_scale = _flattened_kv[..., head_dim:].view(torch.float32).contiguous()
    return weights

direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
)

class Indexer(nn.Module):

    def __init__(self,
                 atom_config: Config,
                 config: PretrainedConfig,
                 hidden_size: int,
                 q_lora_rank: int,
                 quant_config: Optional[QuantizationConfig],
                 cache_config: str,
                 topk_indices_buffer: Optional[torch.Tensor],
                 prefix: str = ""):
        super().__init__()
        self.atom_config = atom_config
        self.config = config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(self.q_lora_rank,
                                     self.head_dim * self.n_head,
                                     bias=False,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.wq_b")
        self.wk = ReplicatedLinear(hidden_size,
                                   self.head_dim,
                                   bias=False,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.wk")
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(hidden_size,
                                             self.n_head,
                                             quant_config=None,
                                             prefix=f"{prefix}.weights_proj")
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_func = get_hip_quant(QuantType.per_1x128)
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        # TODO (zyongye) change dim to fp8 later to (self.head_dim + 4)
        self.k_cache = DeepseekV32IndexerCache(head_dim=self.head_dim + 4,
                                               dtype=torch.uint8,
                                               prefix=f"{prefix}.k_cache",
                                               cache_config=cache_config)
        self.max_model_len = atom_config.max_model_len
        self.prefix = prefix
        self.max_total_seq_len = atom_config.max_num_seqs * self.max_model_len
        # register_metadata_builder("indexer_attn_metadata", self.k_cache.get_attn_backend().get_builder_cls())

    def forward(self, hidden_states: torch.Tensor, qr: torch.Tensor, positions,
                rotary_emb) -> torch.Tensor:
        q = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)

        k = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)

        # we only quant q here since k quant is fused with cache insertion
        q = q.view(-1, self.head_dim)

        q_fp8, q_scale = self.quant_func(q, quant_dtype=dtypes.fp8)
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights = self.weights_proj(hidden_states)
        weights = weights.unsqueeze(
            -1) * q_scale * self.softmax_scale * self.n_head**-0.5
        weights = weights.squeeze(-1)

        return torch.ops.aiter.sparse_attn_indexer(
            hidden_states, self.k_cache.prefix, self.k_cache.kv_cache[0],
            q_fp8, k, weights, self.quant_block_size, self.scale_fmt,
            self.topk_tokens, self.head_dim, self.max_model_len,
            self.max_total_seq_len, self.topk_indices_buffer)


class DeepseekV2MLAAttention(nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

    For more info see MLACommonImpl in: vllm/attention/backends/mla/utils.py
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: str = "bf16",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_num: int = 0,
        topk_indices_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_num = layer_num

        # For FP4 and use_triton_gemm(), fused_qkv_a_proj and q_b_proj are AITER-Triton FP4 GEMMs but o_proj remains BF16 GEMM,
        # Otherwise, if use_triton_gemm() is off, all projections are BF16 GEMMs
        # For FP8, use_triton_gemm() is ignored and AITER FP8 GEMM is used
        if quant_config["quant_dtype"] == dtypes.fp4x2:
            if not use_triton_gemm(): 
                # TODO use ignore layer for mxfp4 attention
                source_quant_dtype = None
                quant_config = None
                base_quant_config = None
            else:
                source_quant_dtype = torch.bfloat16
                base_quant_config = None
        else:
            source_quant_dtype = None
            base_quant_config = quant_config
        
        if self.q_lora_rank is not None:
            # self.q_a_proj = ReplicatedLinear(self.hidden_size,
            #                                  self.q_lora_rank,
            #                                  bias=False,
            #                                  quant_config=quant_config,
            #                                  prefix=f"{prefix}.q_a_proj")
            self.fused_qkv_a_proj = MergedReplicatedLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                source_quant_dtype=source_quant_dtype)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj",
                                                 source_quant_dtype=source_quant_dtype)
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj",
                                               source_quant_dtype=source_quant_dtype)

            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
                source_quant_dtype=source_quant_dtype)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config if is_rocm_aiter_fp4bmm_enabled() else base_quant_config,
            prefix=f"{prefix}.kv_b_proj",
            source_quant_dtype=source_quant_dtype if is_rocm_aiter_fp4bmm_enabled() else None)
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=base_quant_config,
                                        reduce_results=not ENABLE_ALLREDUCE_RMSNORM_FUSION,
                                        prefix=f"{prefix}.o_proj",
                                        source_quant_dtype=None)

        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.is_v32 = hasattr(config, "index_topk")

        if self.is_v32:
            self.indexer = Indexer(
                get_current_atom_config(),
                config,
                hidden_size,
                q_lora_rank,
                base_quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer = None
        # In the MLA backend, kv_cache includes both k_c and
        # pe (i.e. decoupled position embeddings). In particular,
        # the concat_and_cache_mla op requires
        #     k_c.size(1) + k_pe.size(1) == kv_cache.size(2)
        # i.e.
        #     kv_lora_rank + qk_rope_head_dim == head_size

        mla_modules = MLAModules(
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_proj if self.q_lora_rank is None else self.q_b_proj,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
            indexer=self.indexer,
        )

        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scaling,
            num_kv_heads=1,
            kv_cache_dtype=cache_config,
            layer_num=layer_num,
            use_mla=True,
            mla_modules=mla_modules,
            prefix=prefix,
        )

        # self.fuse_qknorm_quant is turned on only if FP8 or (FP4 and use_triton_gemm()),
        # because for (FP4 and not use_triton_gemm()) case, BF16 GEMMs are used
        self.prefix = prefix
        self.quant_dtype = None
        self.fuse_qknorm_quant = False
        if quant_config is not None and ENABLE_DS_QKNORM_QUANT_FUSION:
            if quant_config["quant_dtype"] == dtypes.fp8 or (quant_config["quant_dtype"] == dtypes.fp4x2 and use_triton_gemm()):
                self.quant_dtype = quant_config["quant_dtype"]
                self.fuse_qknorm_quant = True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_scale = None
        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states

        if self.q_lora_rank is not None:
            if self.fuse_qknorm_quant and self.quant_dtype != dtypes.fp8:
                q_c, q_c_scale, kv_c_normed, k_pe = _fuse_qkv_a_proj_reduce_rmsnorm_quant(
                    hidden_states,
                    hidden_states_scale,
                    self.fused_qkv_a_proj.weight,
                    self.fused_qkv_a_proj.weight_scale,
                    self.q_a_layernorm.weight,
                    self.q_a_layernorm.eps,
                    self.kv_a_layernorm.weight,
                    self.kv_a_layernorm.eps,
                    self.q_lora_rank,
                    self.kv_lora_rank,
                    self.qk_rope_head_dim,
                    dtype_quant=self.quant_dtype,
                    shuffle=True,
                    scale_shuffle_padding=True,
                    group_size=128,
                    output_unquantized_inp1=False,
                    transpose_scale=False,
                )
                hidden_states_or_q_c = q_c
                hidden_states_or_q_c_scale = q_c_scale
            else:
                qkv_lora = self.fused_qkv_a_proj(hidden_states, hidden_states_scale)
                # ckq = self.q_a_proj(hidden_states)
                q_c, kv_c, k_pe = torch.split(
                    qkv_lora,
                    [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim],
                    dim=-1,
                )
                # fuse q_c norm + kv_c norm + quant of hidden_states_or_q_c
                if self.fuse_qknorm_quant:
                    (hidden_states_or_q_c, 
                    hidden_states_or_q_c_scale), _, kv_c_normed, _ = _fuse_rmsnorm_quant(
                        q_c,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.eps,
                        kv_c,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.eps,
                        None,
                        dtype_quant=self.quant_dtype,
                        shuffle=True,
                        scale_shuffle_padding=True,
                        group_size=128,
                        output_unquantized_inp1=False,
                        transpose_scale=False,
                    )
                else:
                    hidden_states_or_q_c = self.q_a_layernorm(q_c)
        else:
            hidden_states_or_q_c = hidden_states
            kv_c, k_pe = torch.split(self.kv_a_proj_with_mqa(hidden_states, hidden_states_scale),
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        if not self.fuse_qknorm_quant:
            kv_c_normed = self.kv_a_layernorm(kv_c)
        if self.is_v32 and self.indexer is not None:
            _topk_indices = self.indexer(hidden_states, hidden_states_or_q_c, positions, self.rotary_emb)

        return self.mla_attn(hidden_states_or_q_c,
                             kv_c_normed,
                             k_pe,
                             positions,
                             None if not self.fuse_qknorm_quant else hidden_states_or_q_c_scale,)



class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        topk_indices_buffer: Optional[torch.Tensor] = None,
        cache_config: str = "bf16",
        quant_config: Optional[QuantizationConfig] = None,
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep='.')[-1])
        self.layer_idx = layer_idx

        self.self_attn = DeepseekV2MLAAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            layer_num=layer_num,
            topk_indices_buffer=topk_indices_buffer,
        )

        # self.fuse_input_norm_quant is turned on only if FP8 or (FP4 and use_triton_gemm()),
        # because for (FP4 and not use_triton_gemm()) case, BF16 GEMMs are used
        # note that ENABLE_ALLREDUCE_RMSNORM_FUSION will be turned off if it was on
        self.quant_dtype = None
        self.fuse_input_norm_quant = False
        self.fuse_ar_input_norm = ENABLE_ALLREDUCE_RMSNORM_FUSION
        if quant_config is not None and ENABLE_RMSNORM_QUANT_FUSION:
            if quant_config["quant_dtype"] == dtypes.fp8 or (quant_config["quant_dtype"] == dtypes.fp4x2 and use_triton_gemm()):
                self.quant_dtype = quant_config["quant_dtype"]
                self.fuse_input_norm_quant = True
                if self.fuse_ar_input_norm:
                    self.fuse_ar_input_norm = False
                    logger.info("Warning: Because ENABLE_RMSNORM_QUANT_FUSION is turned on, AR + RMS fusion is turned off for input_layernorm and reduce_results is re-enabled for first k dense layer down_proj")
        
        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                reduce_results=not self.fuse_ar_input_norm,
                prefix=f"{prefix}.mlp",
            )
        else:
            # next_layer_dense = True
            # if (config.n_routed_experts is not None
            #     and (layer_idx + 1) >= config.first_k_dense_replace
            #     and (layer_idx + 1) % config.moe_layer_freq == 0):
            #     next_layer_dense = False
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=not self.fuse_ar_input_norm,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       fused_allreduce=self.fuse_ar_input_norm and self.layer_idx > 0)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION)
        self.routed_scaling_factor = config.routed_scaling_factor


    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if self.fuse_input_norm_quant:
            assert self.quant_dtype is not None
            weight = self.input_layernorm.weight
            eps = self.input_layernorm.eps
            if residual is None:
                residual = hidden_states
                (hidden_states_quant, hidden_states_quant_scale), _, _, _ = _fuse_rmsnorm_quant(
                    hidden_states,
                    weight,
                    eps,
                    None,
                    None,
                    None,
                    None,
                    dtype_quant=self.quant_dtype,
                    shuffle=True,
                    scale_shuffle_padding=True,
                    group_size=128,
                    output_unquantized_inp1=False,
                    transpose_scale=False,
                )
            else:
                (hidden_states_quant, hidden_states_quant_scale), _, _, residual = _fuse_rmsnorm_quant(
                    hidden_states,
                    weight,
                    eps,
                    None,
                    None,
                    None,
                    residual,
                    dtype_quant=self.quant_dtype,
                    shuffle=True,
                    scale_shuffle_padding=True,
                    group_size=128,
                    output_unquantized_inp1=False,
                    transpose_scale=False,
                )

            hidden_states = (hidden_states_quant, hidden_states_quant_scale)

        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        if hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1. / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1. / self.routed_scaling_factor

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if isinstance(self.mlp,
                      DeepseekV2MLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1. / self.routed_scaling_factor

        return hidden_states, residual

@support_torch_compile
class DeepseekV2Model(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = DeepseekV2DecoderLayer,
    ):
        super().__init__()

        config = atom_config.hf_config
        cache_config = atom_config.kv_cache_dtype
        quant_config = atom_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(
            config, "index_topk"
        )
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(atom_config.max_num_batched_tokens,
                                              topk_tokens,
                                              dtype=torch.int32,
                                              device="cuda")
        else:
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: DeepseekV2DecoderLayer(
                config,
                prefix,
                topk_indices_buffer=topk_indices_buffer,
                cache_config=cache_config,
                quant_config=quant_config,
                layer_num=layer_num,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size,
                                eps=config.rms_norm_eps,
                                fused_allreduce=self.layers[self.end_layer - 1].fuse_ar_input_norm,)
                                # fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION,)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + (self.config.n_shared_experts if is_rocm_aiter_fusion_shared_expert_enabled() else 0))


class DeepseekV2ForCausalLM(nn.Module):
    # packed_modules_mapping = {
    #     "q_a_proj" : ("fused_qkv_a_proj", 0),
    #     "kv_a_proj_with_mqa":  ("fused_qkv_a_proj", 1),
    #     "gate_proj": ("gate_up_proj", 0),
    #     "up_proj": ("gate_up_proj", 1),
    # }

    def __init__(
            self,
            atom_config: Config,
            prefix: str = "",
            layer_type: type[nn.Module] = DeepseekV2DecoderLayer
    ):
        super().__init__()
        config = atom_config.hf_config
        quant_config = atom_config.quant_config
        self.config = config
        self.quant_config = quant_config

        if hasattr(config, 'q_lora_rank') and config.q_lora_rank is not None:
            self.packed_modules_mapping = {
                "q_a_proj": ("fused_qkv_a_proj", 0),
                "kv_a_proj_with_mqa": ("fused_qkv_a_proj", 1),
                "gate_proj": ("gate_up_proj", 0),
                "up_proj": ("gate_up_proj", 1),
            }
        else:
            self.packed_modules_mapping = {
                "gate_proj": ("gate_up_proj", 0),
                "up_proj": ("gate_up_proj", 1),
            }

        self.model = DeepseekV2Model(atom_config=atom_config,
                                     prefix=maybe_prefix(prefix, "model"),
                                     layer_type = layer_type)
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          org_num_embeddings=config.vocab_size,
                                          prefix=maybe_prefix(prefix, "lm_head"),
                                        )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.lm_head(hidden_states)
        return logits


    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass
