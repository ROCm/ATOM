# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import itertools
from dataclasses import dataclass
from typing import Optional, Type

import aiter
import numpy as np
import torch
from aiter import dtypes
from aiter.dist.parallel_state import get_tp_group
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mha import Attention
from atom.utils import CpuGpuBuffer
from atom.utils.block_convert import block_table_convert_triton
from atom.utils.forward_context import AttentionMetaData, Context

from .aiter_attention import AiterBackend, AiterAttentionMetadataBuilder


class GDNAttentionBackend(AiterBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_GDN_ATTENTION"

    @staticmethod
    def get_builder_cls() -> Type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["Attention"]:
        return Attention

@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNAttentionMetadataBuilder(AiterAttentionMetadataBuilder):

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        model_runner,
    ):
        super().__init__(model_runner)
        config = model_runner.config
        hf_config = config.hf_config
        # self.vllm_config = vllm_config
        # self.compilation_config = vllm_config.compilation_config
        # self.speculative_config = vllm_config.speculative_config
        # self.kv_cache_spec = kv_cache_spec

        # if self.speculative_config:
        #     assert self.speculative_config.num_speculative_tokens is not None
        #     self.num_spec: int = self.speculative_config.num_speculative_tokens
        # else:
        #     self.num_spec = 0
        # self.use_spec_decode = self.num_spec > 0
        
        self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0
        # self._init_reorder_batch_threshold(1, self.use_spec_decode)

        # self.use_full_cuda_graph = (
        #     self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        # )

        # self.max_bs = (
        #     self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        # )
        # if self.compilation_config.max_cudagraph_capture_size is not None:
        #     self.max_bs = min(
        #         self.max_bs,
        #         self.compilation_config.max_cudagraph_capture_size,
        #     )

        self.spec_state_indices_tensor = torch.empty(
            (self.max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=self.device,
        )
        self.non_spec_state_indices_tensor = torch.empty(
            (self.max_bs,),
            dtype=torch.int32,
            device=self.device,
        )
        self.spec_sequence_masks = torch.empty(
            (self.max_bs,),
            dtype=torch.bool,
            device=self.device,
        )
        self.spec_token_indx = torch.empty(
            (self.max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=self.device,
        )
        self.non_spec_token_indx = torch.empty(
            (self.max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=self.device,
        )
        self.spec_query_start_loc = torch.empty(
            (self.max_bs + 1,),
            dtype=torch.int32,
            device=self.device,
        )
        self.non_spec_query_start_loc = torch.empty(
            (self.max_bs + 1,),
            dtype=torch.int32,
            device=self.device,
        )
        self.num_accepted_tokens = torch.empty(
            (self.max_bs,),
            dtype=torch.int32,
            device=self.device,
        )

    def prepare_prefill(  # type: ignore[override]
        self,
        batch: ScheduledBatch,
    ) -> GDNAttentionMetadata:
        
        attn_metadata, positions = super().prepare_prefill(batch)

        if batch.block_tables==[]:
            attn_metadata.gdn_metadata=None
            return attn_metadata, positions
        query_start_loc = attn_metadata.cu_seqlens_q
        context_lens_tensor = torch.zeros((batch.total_seqs_num_prefill)).cuda()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        
        sum_scheduled_tokens = batch.total_tokens_num_prefill

        spec_sequence_masks = None
        num_spec_decodes = 0

        # num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        #     split_decodes_and_prefills(m, decode_threshold=1)
        # )
        num_decodes = batch.total_seqs_num_decode
        num_prefills = batch.total_seqs_num_prefill
        num_decode_tokens = batch.total_tokens_num_decode
        num_prefill_tokens = batch.total_tokens_num_prefill
        num_spec_decode_tokens = 0
        spec_token_indx = None
        non_spec_token_indx = None
        spec_state_indices_tensor = None
        # non_spec_state_indices_tensor = torch.Tensor(batch.block_tables)[:,0]
        non_spec_state_indices_tensor = torch.Tensor([1]).cuda().int()
        # non_spec_state_indices_tensor = torch.tensor([1], dtype=torch.int).cuda()
        spec_query_start_loc = None
        non_spec_query_start_loc = query_start_loc
        num_accepted_tokens = None

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(non_spec_query_start_loc)
            )
        else:
            has_initial_state = None

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        # batch_size = m.num_actual_tokens
        batch_size = batch.total_seqs_num_prefill

        if (
            False #self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.max_bs
            and num_spec_decode_tokens <= self.max_bs
        ):
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks, non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            False #self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        gdn_attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=batch.total_tokens_num,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        
        attn_metadata.gdn_metadata = gdn_attn_metadata
        return attn_metadata, positions

    def prepare_decode(  # type: ignore[override]
        self,
        batch: ScheduledBatch,
        bs: int,
    ) -> GDNAttentionMetadata:
        
        attn_metadata, positions = super().prepare_decode(batch, bs)

        cu_seqlens = torch.arange(0, bs + 1, 
                             dtype=torch.int32).cuda()
        query_start_loc = cu_seqlens
        context_lens_tensor = attn_metadata.context_lens
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        
        sum_scheduled_tokens = batch.total_tokens_num_decode

        spec_sequence_masks = None
        num_spec_decodes = 0

        # num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        #     split_decodes_and_prefills(m, decode_threshold=1)
        # )
        num_decodes = batch.total_seqs_num_decode
        num_prefills = batch.total_seqs_num_prefill
        num_decode_tokens = batch.total_tokens_num_decode
        num_prefill_tokens = batch.total_tokens_num_prefill
        num_spec_decode_tokens = 0
        spec_token_indx = None
        non_spec_token_indx = None
        spec_state_indices_tensor = None
        non_spec_state_indices_tensor = torch.Tensor(batch.block_tables).int().cuda()[:,0]
        
        non_spec_state_indices_tensor = torch.Tensor([1]).cuda().int()
        spec_query_start_loc = None
        non_spec_query_start_loc = query_start_loc
        num_accepted_tokens = None

        # assert num_accepted_tokens is not None
        # num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(non_spec_query_start_loc)
            )
        else:
            has_initial_state = None

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        # batch_size = m.num_actual_tokens
        batch_size = batch.total_seqs_num_decode

        if (
            False and self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.max_bs
            and num_spec_decode_tokens <= self.max_bs
        ):
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks, non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            False and self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        gdn_attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=batch.total_tokens_num,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        
        attn_metadata.gdn_metadata = gdn_attn_metadata
        return attn_metadata, positions

    def build_for_cudagraph_capture(
        self, bs: int
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.max_bs
            and m.num_actual_tokens <= self.max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)

PAD_SLOT_ID = -1
def compute_causal_conv1d_metadata(query_start_loc_p: torch.Tensor):
    # Needed for causal_conv1d
    seqlens = query_start_loc_p.diff().to("cpu")
    nums_dict = {}  # type: ignore
    batch_ptr = None
    token_chunk_offset_ptr = None
    device = query_start_loc_p.device
    for BLOCK_M in [8]:  # cover all BLOCK_M values
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]["nums"] = nums
        nums_dict[BLOCK_M]["tot"] = nums.sum().item()
        mlist = torch.from_numpy(np.repeat(np.arange(len(nums)), nums))
        nums_dict[BLOCK_M]["mlist"] = mlist
        mlist_len = len(nums_dict[BLOCK_M]["mlist"])
        nums_dict[BLOCK_M]["mlist_len"] = mlist_len
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2
        offsetlist = []  # type: ignore
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32)
        nums_dict[BLOCK_M]["offsetlist"] = offsetlist

        if batch_ptr is None:
            # Update default value after class definition
            batch_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
            token_chunk_offset_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
        else:
            if batch_ptr.nelement() < MAX_NUM_PROGRAMS:
                batch_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)
                token_chunk_offset_ptr.resize_(  # type: ignore
                    MAX_NUM_PROGRAMS
                ).fill_(PAD_SLOT_ID)

        batch_ptr[0:mlist_len].copy_(mlist)
        token_chunk_offset_ptr[  # type: ignore
            0:mlist_len
        ].copy_(offsetlist)
        nums_dict[BLOCK_M]["batch_ptr"] = batch_ptr
        nums_dict[BLOCK_M]["token_chunk_offset_ptr"] = token_chunk_offset_ptr  # type: ignore

    return nums_dict, batch_ptr, token_chunk_offset_ptr