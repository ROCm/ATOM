# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Plugin mode extensions for MLAAttention.
This module provides additional methods for MLAAttention when running in plugin mode.
"""

import torch
import aiter
from aiter import dtypes
from aiter.ops.triton.batched_gemm_a16wfp4 import batched_gemm_a16wfp4

from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501 # isort: skip
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as _aiter_triton_fp8_bmm,
)
from aiter.mla import mla_decode_fwd

from functools import partial as functools_partial
from atom.config import get_current_atom_config
from atom.model_ops.linear import use_triton_gemm

import logging

logger = logging.getLogger("atom")


if use_triton_gemm():
    try:
        from aiter.ops.triton.fused_gemm_a8w8_blockscale_split_cat import (
            fused_gemm_a8w8_blockscale_preshuffle_split_cat,
        )
        from aiter.ops.triton.fused_gemm_afp4wfp4_split_cat import (
            fused_gemm_afp4wfp4_preshuffle_split_cat,
        )
    except ImportError as e:
        logger.warning(f"Triton fused GEMM split_cat not available: {e}")
        fused_gemm_afp4wfp4_preshuffle_split_cat = None
        fused_gemm_a8w8_blockscale_preshuffle_split_cat = None


def reorg_kvcache(
    allgatered_kv_c_normed: torch.Tensor,
    allgatered_k_pe: torch.Tensor,
    padded_local_chunk_seq_lens_lst: list[int],
    local_context_lens_allranks: list[list[int]],
    sum_seq_len: int,
    max_seq_len: int,
    chunk_size: int,
    chunk_idx: int,
    toks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    reorg and unpad kvcache after cp local gather to tp layout for attn kernel.
    e.g.
    allgatered_kv_c_normed = [T0_0, T0_1, T0_2, T0_3, T1_0, T1_1, ...,
                              T0_4, T0_5, pad, pad, T1_2, pad, ...]
    -> reorganized_kv_c_normed = [T0_0, T0_1, T0_2, T0_3, T0_4, T0_5,
                                  T1_0, T1_1, T1_2, ...]
    Args:
        padded_local_chunk_seq_lens_lst: local chunk context lengths
            under current CP rank.
        local_context_lens_allranks: local context lengths on each CP rank.
        sum_seq_len: the sum of cp_chunk_seq_lens_lst.
        max_seq_len: the max value of cp_chunk_seq_lens_lst.
        chunk_size: the local padded max context chunk from
            chunked_context_metadata building.
        chunk_idx: chunk idx of chunked_prefill.
        toks: the number of tokens for local gather cache.
    """
    kv_c_segments = []
    k_pe_segments = []
    src_token_idx = 0
    max_seq_len_check = 0
    for padded_local_chunk_seq_len, local_context_lens in zip(
        padded_local_chunk_seq_lens_lst, local_context_lens_allranks
    ):
        cur_seq_len = 0
        for rank, local_context_len in enumerate(local_context_lens):
            # Note(qcs): We split the context into multiple chunks,
            # depending on the size of the workspace.
            # local_context in dcp0:   |-----------------|
            # local_context in dcp1:   |--------------|
            # n*padded_local_chunk:    |-----|-----|-----|
            # local_chunk_len in dcp1: |-----|-----|--|
            # so we need update the last chunk length in dcp1.
            local_chunk_len = min(
                max(0, local_context_len - chunk_idx * chunk_size),
                padded_local_chunk_seq_len,
            )
            if local_chunk_len != 0:
                kv_c_segment = allgatered_kv_c_normed[
                    rank * toks
                    + src_token_idx : rank * toks
                    + src_token_idx
                    + local_chunk_len
                ]
                k_pe_segment = allgatered_k_pe[
                    rank * toks
                    + src_token_idx : rank * toks
                    + src_token_idx
                    + local_chunk_len
                ]
                kv_c_segments.append(kv_c_segment)
                k_pe_segments.append(k_pe_segment)
                cur_seq_len += local_chunk_len
        max_seq_len_check = max(max_seq_len_check, cur_seq_len)
        src_token_idx += padded_local_chunk_seq_len
    reorganized_kv_c_normed = torch.cat(kv_c_segments, dim=0)
    reorganized_k_pe = torch.cat(k_pe_segments, dim=0)
    assert reorganized_kv_c_normed.shape[0] == sum_seq_len
    assert reorganized_k_pe.shape[0] == sum_seq_len
    assert max_seq_len_check == max_seq_len
    return reorganized_kv_c_normed, reorganized_k_pe


class MLAAttentionImplPluginModeMethods:
    """
    Container class for plugin mode methods.
    This class cannot be instantiated - it only serves as a namespace for methods
    that will be added to PagedAttentionImpl via decorator.
    """

    def __init__(self):
        raise TypeError(
            "MLAAttentionImplPluginModeMethods cannot be instantiated. "
            "It is only used as a method container for the decorator."
        )

    def _concat_k_nope_k_pe_plugin_mode(
        self, k_nope: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficiently concatenate k_nope and k_pe tensors along the last dimension.

        This function avoids the performance penalty of torch.cat with expanded
        non-contiguous tensors by pre-allocating the output and using direct copies.

        Args:
            k_nope: Tensor of shape [..., nope_dim]
            k_pe: Tensor to broadcast and concatenate, typically shape [..., 1, pe_dim]
                or [..., pe_dim]

        Returns:
            Tensor of shape [..., nope_dim + pe_dim]
        """
        k = torch.empty(
            (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
            dtype=k_nope.dtype,
            device=k_nope.device,
        )
        # Direct copies with efficient broadcasting
        k[..., : k_nope.shape[-1]] = k_nope
        k[..., k_nope.shape[-1] :] = k_pe
        return k

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        output = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            return_lse=return_softmax_lse,
            **kwargs,
        )

        return output

    def _run_prefill_new_tokens_plugin_mode(self, prefill, q, k, v, return_softmax_lse):
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=prefill.query_start_loc,
            cu_seqlens_k=prefill.query_start_loc,
            max_seqlen_q=prefill.max_query_len,
            max_seqlen_k=prefill.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )

    def _run_prefill_context_chunk_plugin_mode(self, prefill, chunk_idx, q, k, v):
        assert prefill.chunked_context is not None
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=prefill.query_start_loc,
            cu_seqlens_k=prefill.chunked_context.cu_seq_lens[chunk_idx],
            max_seqlen_q=prefill.max_query_len,
            max_seqlen_k=prefill.chunked_context.max_seq_lens[chunk_idx],
            softmax_scale=self.scale,
            causal=False,  # Context is unmasked
            return_softmax_lse=True,
        )

    def _context_parallel_compute_prefill_context_plugin_mode(
        self,
        q,
        kv_c_and_k_pe_cache,
        attn_metadata,
        k_scale,
        dcp_world_size,
    ):
        assert k_scale is None, "DCP not support scaled kvcache now."
        assert attn_metadata.plugin_metadata.prefill is not None
        prefill_metadata = attn_metadata.plugin_metadata.prefill
        assert prefill_metadata.chunked_context is not None
        assert prefill_metadata.chunked_context.padded_local_chunk_seq_lens is not None
        assert prefill_metadata.chunked_context.local_context_lens_allranks is not None
        assert prefill_metadata.chunked_context.padded_local_cu_seq_lens is not None
        assert prefill_metadata.chunked_context.cu_seq_lens_lst is not None
        assert prefill_metadata.chunked_context.chunk_size is not None

        output = None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace

        from vllm import _custom_ops as ops
        from vllm.distributed.parallel_state import get_dcp_group
        from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            ops.cp_gather_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_table,
                cu_seq_lens=prefill_metadata.chunked_context.padded_local_cu_seq_lens[
                    i
                ],
                batch_size=attn_metadata.plugin_metadata.num_prefills,
                seq_starts=prefill_metadata.chunked_context.starts[i],
            )
            # workspace
            # |------- N tokens --------|--------- N*dcp_size tokens ----------|
            # |<- use for loca_gather ->|<--------- use for allgather -------->|
            allgather_offset = workspace.shape[0] // (dcp_world_size + 1)
            assert allgather_offset * (dcp_world_size + 1) == workspace.shape[0]
            assert toks <= allgather_offset
            local_gathered_kvcache = workspace[:toks]
            cur_allgather_workspace = workspace[
                allgather_offset : allgather_offset * (1 + dcp_world_size)
            ]
            assert toks * dcp_world_size <= cur_allgather_workspace.shape[0]
            cur_allgather_kvcache = cur_allgather_workspace[: toks * dcp_world_size]
            cur_allgather_kvcache.copy_(
                get_dcp_group().all_gather(local_gathered_kvcache, dim=0)
            )
            assert (
                cur_allgather_kvcache.shape[-1]
                == self.kv_lora_rank + self.qk_rope_head_dim
            )
            allgatered_kv_c_normed, allgatered_k_pe = cur_allgather_kvcache.unsqueeze(
                1
            ).split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

            kv_c_normed, k_pe = reorg_kvcache(
                allgatered_kv_c_normed,
                allgatered_k_pe,
                padded_local_chunk_seq_lens_lst=prefill_metadata.chunked_context.padded_local_chunk_seq_lens[
                    i
                ],
                local_context_lens_allranks=prefill_metadata.chunked_context.local_context_lens_allranks,
                sum_seq_len=prefill_metadata.chunked_context.cu_seq_lens_lst[i][-1],
                max_seq_len=prefill_metadata.chunked_context.max_seq_lens[i],
                chunk_size=prefill_metadata.chunked_context.chunk_size,
                chunk_idx=i,
                toks=toks,
            )

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = self._concat_k_nope_k_pe_plugin_mode(k_nope, k_pe)

            attn_output, attn_softmax_lse = self._run_prefill_context_chunk_plugin_mode(
                prefill=prefill_metadata,
                chunk_idx=i,
                q=q,
                k=k,
                v=v,
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                merge_attn_states(
                    output=output_tmp,
                    output_lse=output_lse_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse

    def _compute_prefill_context_plugin_mode(
        self,
        q,
        kv_c_and_k_pe_cache,
        attn_metadata,
        k_scale,
    ):
        assert attn_metadata.plugin_metadata.prefill is not None
        prefill_metadata = attn_metadata.plugin_metadata.prefill
        assert prefill_metadata.chunked_context is not None

        output = None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace

        from vllm import _custom_ops as ops
        from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            ops.gather_and_maybe_dequant_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_table,
                cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                token_to_seq=prefill_metadata.chunked_context.token_to_seq[i],
                num_tokens=prefill_metadata.chunked_context.chunk_total_token[i],
                kv_cache_dtype=self.kv_cache_dtype,
                scale=k_scale,
                seq_starts=prefill_metadata.chunked_context.starts[i],
            )
            kv_c_normed = workspace[:toks][..., : self.kv_lora_rank]
            k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = self._concat_k_nope_k_pe_plugin_mode(k_nope, k_pe)

            attn_output, attn_softmax_lse = self._run_prefill_context_chunk_plugin_mode(
                prefill=prefill_metadata,
                chunk_idx=i,
                q=q,
                k=k,
                v=v,
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                merge_attn_states(
                    output=output_tmp,
                    output_lse=output_lse_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse

    def _forward_prefill_plugin_mode(
        self,
        q,
        kv_c_normed,
        k_pe,
        kv_c_and_k_pe_cache,
        attn_metadata,
        k_scale,
        output,
    ):
        # TODO (zyongye): Prefill function hereplugin_metadata
        assert attn_metadata.plugin_metadata.prefill is not None
        assert self.dcp_world_size != -1

        has_context = attn_metadata.plugin_metadata.prefill.chunked_context is not None

        if use_triton_gemm():
            weight = self.kv_b_proj.weight
            weight_scale = self.kv_b_proj.weight_scale
            if (
                fused_gemm_afp4wfp4_preshuffle_split_cat is not None
                and weight.dtype == dtypes.fp4x2
            ):  # FP4 GEMM + split + cat
                m = kv_c_normed.shape[0]
                # from aiter.ops.triton.quant import dynamic_mxfp4_quant
                # input = kv_c_normed
                # input_2d = input.view(-1, input.shape[-1])
                output_dtype = kv_c_normed.dtype

                # q_input, x_scale = dynamic_mxfp4_quant(input_2d)
                quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
                q_input, x_scale = quant_func(
                    kv_c_normed,
                    quant_dtype=dtypes.fp4x2,
                    shuffle=(m >= 32),
                )

                if m >= 32:
                    x_scale = x_scale.view(torch.uint8).view(x_scale.shape[0] // 32, -1)
                else:
                    x_scale = x_scale[:m, ...].view(torch.uint8)

                k, v = fused_gemm_afp4wfp4_preshuffle_split_cat(
                    q_input.view(torch.uint8),
                    weight.view(torch.uint8).view(weight.shape[0] // 16, -1),
                    k_pe.expand((-1, self.num_heads, -1)),
                    x_scale,
                    weight_scale.view(torch.uint8).view(
                        weight_scale.shape[0] // 32, -1
                    ),
                    self.qk_nope_head_dim,
                    self.v_head_dim,
                    output_dtype,
                )
            elif (
                fused_gemm_a8w8_blockscale_preshuffle_split_cat is not None
                and weight.dtype == dtypes.fp8
            ):  # FP8 GEMM + split + cat
                weight_shuffled = weight.reshape(
                    weight.shape[0] // 16, weight.shape[1] * 16
                )

                output_dtype = kv_c_normed.dtype

                quant_func = functools_partial(
                    aiter.get_hip_quant(aiter.QuantType.per_1x128), transpose_scale=True
                )
                q_input, x_scale = quant_func(
                    kv_c_normed,
                    quant_dtype=dtypes.fp8,
                    scale=getattr(self.kv_b_proj, "input_scale", None),
                )

                k, v = fused_gemm_a8w8_blockscale_preshuffle_split_cat(
                    q_input,
                    weight_shuffled,
                    k_pe.expand((-1, self.num_heads, -1)),
                    x_scale,
                    weight_scale,
                    self.qk_nope_head_dim,
                    self.v_head_dim,
                    output_dtype,
                )
            else:
                kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                    -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
                )
                k_nope, v = kv_nope.split(
                    [self.qk_nope_head_dim, self.v_head_dim], dim=-1
                )

                k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)
        else:
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # k = self._concat_k_nope_k_pe_plugin_mode(k_nope, k_pe)
            k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        output_prefill = self._run_prefill_new_tokens_plugin_mode(
            prefill=attn_metadata.plugin_metadata.prefill,
            q=q,
            k=k,
            v=v,
            return_softmax_lse=has_context,
        )

        from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

        if has_context:
            suffix_output, suffix_lse = output_prefill
            if self.dcp_world_size > 1:
                context_output, context_lse = (
                    self._context_parallel_compute_prefill_context_plugin_mode(
                        q,
                        kv_c_and_k_pe_cache,
                        attn_metadata,
                        k_scale=None,
                        dcp_world_size=self.dcp_world_size,
                    )
                )
            else:
                context_output, context_lse = self._compute_prefill_context_plugin_mode(
                    q, kv_c_and_k_pe_cache, attn_metadata, k_scale
                )

            # unpad if necessary
            if self._pad_v:
                context_output = context_output[..., : v.shape[-1]]
                suffix_output = suffix_output[..., : v.shape[-1]]

            output = output.view(-1, self.num_heads, self.v_head_dim)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )
        else:
            output_prefill = output_prefill[..., : v.shape[-1]].flatten(start_dim=-2)
            output.copy_(output_prefill)

    def _forward_decode_plugin_mode(
        self,
        q,
        kv_c_and_k_pe_cache,
        attn_metadata,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.plugin_metadata.decode is not None
        assert attn_metadata.plugin_metadata.decode.max_qo_len is not None

        # if type(q) is tuple:
        #     q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        o = torch.zeros(
            B,
            self.num_heads,
            self.kv_lora_rank,
            dtype=attn_metadata.plugin_metadata.decode.attn_out_dtype,
            device=q.device,
        )

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)
        use_persistent_mode = not (
            self.dcp_world_size > 1 and self.kv_cache_dtype == "fp8"
        )
        if not use_persistent_mode:
            # DP : disable persistent mode to avoid overflow
            work_meta_data = None
            work_indptr = None
            work_info_set = None
            reduce_indptr = None
            reduce_final_map = None
            reduce_partial_map = None
        else:
            work_meta_data = attn_metadata.work_meta_data
            work_indptr = attn_metadata.work_indptr
            work_info_set = attn_metadata.work_info_set
            reduce_indptr = attn_metadata.reduce_indptr
            reduce_final_map = attn_metadata.reduce_final_map
            reduce_partial_map = attn_metadata.reduce_partial_map

        paged_kv_indptr = attn_metadata.plugin_metadata.decode.paged_kv_indptr
        paged_kv_indices = attn_metadata.plugin_metadata.decode.paged_kv_indices

        mla_decode_fwd(
            q,
            kv_buffer.view(-1, 1, 1, q.shape[-1]),
            o,
            attn_metadata.plugin_metadata.decode.qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            attn_metadata.plugin_metadata.decode.paged_kv_last_page_len,
            attn_metadata.plugin_metadata.decode.max_qo_len,
            sm_scale=self.scale,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            q_scale=layer._q_scale,
            kv_scale=layer._k_scale,
        )
        return o, None

    def forward_impl_plugin_mode(
        self,
        layer,
        q,
        k_c_normed,
        k_pe,
        kv_cache,
        attn_metadata=None,
        output=None,
    ):
        assert output is not None, "Output tensor must be provided."
        from vllm.distributed.parallel_state import get_dcp_group
        from vllm import _custom_ops as ops
        from vllm.platforms import current_platform
        from vllm.v1.attention.ops.common import cp_lse_ag_out_rs

        # create the output here, it use query shape
        if attn_metadata is None:
            # During the profile run try to simulate to worse case output size
            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
            # since this can be large
            _ = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )

            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        if self.dcp_world_size == -1:
            self.dcp_world_size = get_dcp_group().world_size

        fp8_attention = self.kv_cache_dtype.startswith("fp8")

        num_actual_toks = attn_metadata.plugin_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        assert (
            attn_metadata.plugin_metadata.num_decodes is not None
            and attn_metadata.plugin_metadata.num_prefills is not None
            and attn_metadata.plugin_metadata.num_decode_tokens is not None
        )

        has_decode = attn_metadata.plugin_metadata.num_decodes > 0
        has_prefill = attn_metadata.plugin_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.plugin_metadata.num_decode_tokens

        atom_config = get_current_atom_config()
        positions = atom_config.compilation_config.static_forward_context["positions"][
            :num_actual_toks
        ]
        k_pe = k_pe.unsqueeze(1)
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        decode_q = q[:num_decode_tokens]
        prefill_q = q[num_decode_tokens:]
        prefill_k_pe = k_pe[num_decode_tokens:]
        prefill_k_c_normed = k_c_normed[num_decode_tokens:]

        decode_only = has_decode and not has_prefill
        if not decode_only:
            if self.rotary_emb is not None:
                self.rotary_emb(positions, q[..., self.qk_nope_head_dim :], k_pe)

            # write the latent and rope to kv cache
            if kv_cache.numel() > 0:
                aiter.concat_and_cache_mla(
                    k_c_normed,
                    k_pe.squeeze(1),
                    kv_cache,
                    attn_metadata.plugin_metadata.slot_mapping.flatten(),
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=layer._k_scale,
                )

        if fp8_attention:
            kv_cache = kv_cache.view(current_platform.fp8_dtype())

        if has_prefill:
            self._forward_prefill_plugin_mode(
                prefill_q,
                prefill_k_c_normed,
                prefill_k_pe,
                kv_cache,
                attn_metadata,
                layer._k_scale,
                output=output[num_decode_tokens:],
            )

        if has_decode:
            assert attn_metadata.plugin_metadata.decode is not None

            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )

            # Convert from (B, N, P) to (N, B, P)
            decode_q_nope = decode_q_nope.transpose(0, 1)

            if self.q_pad_num_heads is not None:
                B, N, L = decode_q_pe.shape
                decode_pe_padded = decode_q_pe.new_empty((B, self.q_pad_num_heads, L))
                decode_pe_padded.resize_((B, N, L))
                decode_pe_padded.copy_(decode_q_pe)
                decode_q_pe = decode_pe_padded

            if self.is_aiter_triton_fp4_bmm_enabled:
                decode_ql_nope = batched_gemm_a16wfp4(
                    decode_q_nope,
                    self.W_K,
                    self.W_K_scale,
                    transpose_bm=True,
                    prequant=True,
                    y_scale=layer._q_scale if fp8_attention else None,
                )
            # elif self.is_aiter_triton_fp8_bmm_enabled:
            else:
                # Multiply+Transpose (N, B, P)x(N, P, L)->(N, B, L)->(B, N, L)
                decode_ql_nope = _aiter_triton_fp8_bmm(
                    decode_q_nope,
                    self.W_K,
                    self.W_K_scale,
                    group_size=128,
                    transpose_bm=True,
                )

            if decode_only:
                decode_q = torch.empty(
                    (
                        decode_ql_nope.shape[0],
                        self.num_heads,
                        self.kv_lora_rank + self.qk_rope_head_dim,
                    ),
                    dtype=(
                        dtypes.fp8
                        if self.kv_cache_dtype.startswith("fp8")
                        else self.dtype
                    ),
                    device=decode_ql_nope.device,
                )
                aiter.fused_qk_rope_concat_and_cache_mla(
                    decode_ql_nope,
                    decode_q_pe,
                    k_c_normed,
                    k_pe.squeeze(1),
                    kv_cache.view(
                        kv_cache.shape[0], -1, self.kv_lora_rank + self.qk_rope_head_dim
                    ),
                    decode_q,
                    attn_metadata.plugin_metadata.slot_mapping,
                    layer._k_scale,
                    layer._q_scale,
                    positions,
                    self.rotary_emb.cos_cache,
                    self.rotary_emb.sin_cache,
                    is_neox=self.rotary_emb.is_neox_style,
                    is_nope_first=True,
                )
            else:
                if fp8_attention:
                    ql_nope_shape = decode_ql_nope.shape
                    q_pe_shape = decode_q_pe.shape
                    assert decode_ql_nope.shape[0] == decode_q_pe.shape[0]
                    assert decode_ql_nope.shape[1] == decode_q_pe.shape[1]
                    decode_q_shape = (
                        ql_nope_shape[0],
                        ql_nope_shape[1],
                        ql_nope_shape[2] + q_pe_shape[2],
                    )
                    # Using empty and copy since torch.cat introduces significant overhead.
                    decode_q0 = torch.empty(
                        decode_q_shape,
                        device=decode_ql_nope.device,
                        dtype=decode_ql_nope.dtype,
                    )
                    decode_q0[..., : ql_nope_shape[2]].copy_(decode_ql_nope)
                    decode_q0[..., ql_nope_shape[2] :].copy_(decode_q_pe)

                    decode_q, _ = ops.scaled_fp8_quant(
                        decode_q0.view(decode_q_shape[0], -1),
                        layer._q_scale,
                    )
                    decode_q = decode_q.view(decode_q_shape)
                else:
                    decode_q = (decode_ql_nope, decode_q_pe)
                    decode_q = torch.cat(decode_q, dim=-1)
            if self.dcp_world_size > 1:
                assert not fp8_attention, "DCP not support fp8 kvcache now."
                # decode_q do allgather in head dim.
                decode_q = get_dcp_group().all_gather(decode_q, dim=1)

            # call decode attn
            attn_out, lse = self._forward_decode_plugin_mode(
                decode_q, kv_cache, attn_metadata, layer
            )

            # correct dcp attn_out with lse.
            if self.dcp_world_size > 1:
                attn_out = cp_lse_ag_out_rs(
                    attn_out,
                    lse,
                    get_dcp_group(),
                    is_lse_base_on_e=not getattr(self, "_use_fi_prefill", False),
                )

            # v_up projection
            # self._v_up_proj(attn_out, out=output[:num_decode_tokens])
            # TODO: remove this copy.
            out_up_proj = self._v_up_proj(attn_out)
            output[:num_decode_tokens] = out_up_proj

        return output_padded


def MLAAttentionImplDecoratorForPluginMode(cls):
    method_names = [
        "_concat_k_nope_k_pe_plugin_mode",
        "_flash_attn_varlen_diff_headdims",
        "_run_prefill_new_tokens_plugin_mode",
        "_run_prefill_context_chunk_plugin_mode",
        "_context_parallel_compute_prefill_context_plugin_mode",
        "_compute_prefill_context_plugin_mode",
        "_forward_prefill_plugin_mode",
        "_forward_decode_plugin_mode",
        "forward_impl_plugin_mode",
    ]

    logger.info("Use MLAAttentionImplDecoratorForPluginMode to decorate MLAAttention")

    # Add all methods to the target class
    for method_name in method_names:
        method = getattr(MLAAttentionImplPluginModeMethods, method_name)
        setattr(cls, method_name, method)

    return cls
