# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Plugin mode extensions for MLAAttention.
This module provides additional methods for MLAAttention when running in plugin mode.
"""

import torch
from aiter.ops.triton.batched_gemm_a16wfp4 import batched_gemm_a16wfp4

from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501 # isort: skip
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as _aiter_triton_fp8_bmm,
)

from functools import partial as functools_partial
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
