# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Type

import numpy as np
import torch
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_gdn import GatedDetlaNet
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
    def get_impl_cls() -> Type["GatedDetlaNet"]:
        return GatedDetlaNet


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
        self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0

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

        gdn_metadata = {
            "spec_state_indices": self.spec_state_indices_tensor,
            "non_spec_state_indices": self.non_spec_state_indices_tensor,
            "spec_sequence_masks": self.spec_sequence_masks,
            "spec_token_indx": self.spec_token_indx,
            "non_spec_token_indx": self.non_spec_token_indx,
            "spec_query_start_loc": self.spec_query_start_loc,
            "non_spec_query_start_loc": self.non_spec_query_start_loc,
            "num_accepted_tokens": self.num_accepted_tokens,
        }
        self.model_runner.forward_vars.update(gdn_metadata)

    def prepare_gdn_metadata(
        self,
        batch: ScheduledBatch,
        attn_metadata: AttentionMetaData,
    ) -> GDNAttentionMetadata:

        num_decodes = batch.total_seqs_num_decode
        num_prefills = batch.total_seqs_num_prefill
        num_decode_tokens = batch.total_tokens_num_decode
        num_prefill_tokens = batch.total_tokens_num_prefill
        num_reqs = batch.total_seqs_num
        self.prepare_block_tables(batch)

        block_tables = self.model_runner.forward_vars["block_tables"].copy_to_gpu(
            num_reqs
        )

        context_lens_tensor = attn_metadata.context_lens
        query_start_loc = attn_metadata.cu_seqlens_q
        context_lens_tensor = torch.zeros((batch.total_seqs_num_prefill)).cuda()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        if not self.use_spec_decode:
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_tables[:num_reqs, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = attn_metadata.cu_seqlens_q
            num_accepted_tokens = None
            spec_sequence_masks = None
            num_spec_decodes = 0
            num_spec_decode_tokens = 0
        else:
            pass

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if self.use_spec_decode:
                pass
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(non_spec_query_start_loc)
            )
        else:
            has_initial_state = None

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
        # print("gdn attn metadata: ", gdn_attn_metadata, flush=True)
        return gdn_attn_metadata

    def prepare_prefill(  # type: ignore[override]
        self,
        batch: ScheduledBatch,
    ) -> GDNAttentionMetadata:

        attn_metadata, positions = super().prepare_prefill(batch)
        if batch.block_tables == []:
            attn_metadata.gdn_metadata = None
            return attn_metadata, positions
        gdn_metadata = self.prepare_gdn_metadata(batch, attn_metadata)

        attn_metadata.gdn_metadata = gdn_metadata
        return attn_metadata, positions

    def prepare_decode(  # type: ignore[override]
        self,
        batch: ScheduledBatch,
        bs: int,
    ) -> GDNAttentionMetadata:

        num_decodes = batch.total_seqs_num_decode
        attn_metadata, positions = super().prepare_decode(batch, bs)
        gdn_metadata = self.prepare_gdn_metadata(batch, attn_metadata)
        # transfer data to ps buffer
        self.model_runner.forward_vars["cu_seqlens_q"].cpu[
            bs:
        ] = batch.total_tokens_num_decode
        gdn_metadata.non_spec_query_start_loc = self.model_runner.forward_vars[
            "cu_seqlens_q"
        ].copy_to_gpu(bs + 1)
        self.non_spec_state_indices_tensor[:num_decodes].copy_(
            gdn_metadata.non_spec_state_indices_tensor, non_blocking=True
        )
        self.non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)
        gdn_metadata.non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
            :num_decodes
        ]
        # print("gdn metadata decode: ", gdn_metadata, flush=True)
        attn_metadata.gdn_metadata = gdn_metadata
        return attn_metadata, positions

    def build_for_cudagraph_capture(self, bs: int):
        var = self.model_runner.forward_vars
        if self.block_size == 1024:
            ctx_pa_ps = self.set_aiter_persistent_worker_buffers(bs)
        else:
            ctx_pa_ps = {}
        attn_metadata = AttentionMetaData(
            slot_mapping=var["slot_mapping"].gpu[:bs],
            context_lens=var["context_lens"].gpu[:bs],
            block_tables=var["block_tables"].gpu[:bs],
            max_seqlen_q=var["max_qlen"],
            cu_seqlens_q=var["cu_seqlens_q"].gpu[: bs + 1],
            kv_indptr=var["kv_indptr"].gpu[: bs + 1],
            kv_indices=var["kv_indices"].gpu[:],
            max_seqlen_k=self.model_runner.config.max_model_len,
            block_tables_converted=(
                var["block_tables_converted"].gpu[:bs]
                if "block_tables_converted" in var
                else None
            ),
            **ctx_pa_ps,
        )
        gdn_metadata = GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=bs,
            num_decode_tokens=bs,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=bs,
            has_initial_state=None,
            spec_query_start_loc=None,
            non_spec_query_start_loc=var["cu_seqlens_q"].gpu[: bs + 1],
            spec_state_indices_tensor=None,
            non_spec_state_indices_tensor=var["non_spec_state_indices"],
            spec_sequence_masks=None,
            spec_token_indx=None,
            non_spec_token_indx=None,
            num_accepted_tokens=None,
            nums_dict=None,
            batch_ptr=None,
            token_chunk_offset_ptr=None,
        )
        attn_metadata.gdn_metadata = gdn_metadata

        positions = var["positions"].copy_to_gpu(bs)
        context = Context(
            positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs
        )
        return attn_metadata, context


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
                token_chunk_offset_ptr.resize_(MAX_NUM_PROGRAMS).fill_(  # type: ignore
                    PAD_SLOT_ID
                )

        batch_ptr[0:mlist_len].copy_(mlist)
        token_chunk_offset_ptr[0:mlist_len].copy_(offsetlist)  # type: ignore
        nums_dict[BLOCK_M]["batch_ptr"] = batch_ptr
        nums_dict[BLOCK_M]["token_chunk_offset_ptr"] = token_chunk_offset_ptr  # type: ignore

    return nums_dict, batch_ptr, token_chunk_offset_ptr
