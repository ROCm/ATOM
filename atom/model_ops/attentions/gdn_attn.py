# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Type

import numpy as np
import torch
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_gdn import GatedDetlaNet
from atom.utils import CpuGpuBuffer
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
        self.num_spec = 0
        if hasattr(model_runner, "drafter"):
            self.num_spec = model_runner.drafter.mtp_k
        self.use_spec_decode = self.num_spec > 0

        # self.spec_state_indices_tensor = torch.empty(
        #     (self.max_bs, self.num_spec + 1),
        #     dtype=torch.int32,
        #     device=self.device,
        # )
        # self.non_spec_state_indices_tensor = torch.empty(
        #     (self.max_bs,),
        #     dtype=torch.int32,
        #     device=self.device,
        # )
        self.spec_state_indices_tensor = CpuGpuBuffer(
            (self.max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=self.device,
        )
        self.non_spec_state_indices_tensor = CpuGpuBuffer(
            (self.max_bs,),
            dtype=torch.int32,
            device=self.device,
        )
        self.spec_sequence_masks = torch.ones(
            (self.max_bs,),
            dtype=torch.bool,
            device=self.device,
        )
        self.spec_token_indx = torch.arange(
            (self.max_bs * (self.num_spec + 1)),
            dtype=torch.int32,
            device=self.device,
        )
        self.non_spec_token_indx = torch.empty(
            (self.max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=self.device,
        )
        self.spec_query_start_loc = torch.arange(
            start=0,
            end=(self.max_bs + 1) * (self.num_spec + 1),
            step=(self.num_spec + 1),
            dtype=torch.int32,
            device=self.device,
        )
        self.non_spec_query_start_loc = torch.empty(
            (self.max_bs + 1,),
            dtype=torch.int32,
            device=self.device,
        )
        self.num_accepted_tokens = torch.ones(
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

    def prepare_state_indices(self, batch: ScheduledBatch, with_spec: bool = False):
        non_spec_state_indices = self.non_spec_state_indices_tensor.np
        spec_state_indices = self.spec_state_indices_tensor.np
        for idx, mamba_block_table in enumerate(batch.mamba_block_tables):
            non_spec_state_indices[idx] = 0
            spec_state_indices[idx] = 0

            if not with_spec:
                non_spec_state_indices[idx] = mamba_block_table[0]
            else:
                spec_state_indices[idx, : 1 + self.num_spec] = mamba_block_table[
                    : 1 + self.num_spec
                ]

    def prepare_num_accepted_tokens(self, batch: ScheduledBatch):
        self.num_accepted_tokens.fill_(1)

        if self.model_runner.tokenID_processor.num_bonus is None:
            return
        for idx, num_bonus in enumerate(
            self.model_runner.tokenID_processor.num_bonus
        ):
            self.num_accepted_tokens[idx] = num_bonus + 1

    def prepare_gdn_metadata(
        self,
        batch: ScheduledBatch,
        attn_metadata: AttentionMetaData,
        is_prefill: bool = False,
    ) -> GDNAttentionMetadata:

        num_decodes = batch.total_seqs_num_decode
        num_prefills = batch.total_seqs_num_prefill
        num_decode_tokens = batch.total_tokens_num_decode
        num_prefill_tokens = batch.total_tokens_num_prefill
        num_reqs = batch.total_seqs_num
        self.prepare_block_tables(batch)

        context_lens_tensor = attn_metadata.context_lens
        query_start_loc = attn_metadata.cu_seqlens_q
        context_lens_tensor = torch.zeros((batch.total_seqs_num_prefill)).cuda()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        if not self.use_spec_decode or is_prefill:
            # print("not use spec decode or is prefill", flush=True)
            self.prepare_state_indices(batch, with_spec=False)
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = (
                self.non_spec_state_indices_tensor.copy_to_gpu(num_reqs)
            )
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            num_accepted_tokens = None
            spec_sequence_masks = None
            num_spec_decodes = 0
            num_spec_decode_tokens = 0
        else:
            self.prepare_state_indices(batch, with_spec=True)
            self.prepare_num_accepted_tokens(batch)
            spec_token_size = min(
                num_decodes * (self.num_spec + 1), query_start_loc[-1].item()
            )
            spec_token_indx = torch.arange(
                spec_token_size, dtype=torch.int32, device=self.device
            )
            non_spec_token_indx = torch.empty(
                0, dtype=torch.int32, device=query_start_loc.device
            )
            spec_sequence_masks = torch.ones(
                num_reqs, dtype=torch.bool, device=self.device
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor.copy_to_gpu(
                num_reqs
            )
            non_spec_state_indices_tensor = None
            spec_query_start_loc = query_start_loc
            non_spec_query_start_loc = None
            num_accepted_tokens = self.num_accepted_tokens[:num_reqs]
            # if hasattr(self.model_runner.tokenID_processor, "num_bonus_tokens"):
            #     num_accepted_tokens = self.model_runner.tokenID_processor.num_bonus_tokens + 1
            # else:
            #     num_accepted_tokens = self.num_accepted_tokens[:num_reqs]
            # num_accepted_tokens = batch.num_accepted_tokens
            # num_accepted_tokens = self.model_runner.tokenID_processor.num_bonus_tokens
            # print("num acceptted tokens: ", num_accepted_tokens, flush=True)
            # if num_accepted_tokens is None:
            #     num_accepted_tokens = self.num_accepted_tokens
            num_spec_decodes = num_decodes
            num_prefills = 0
            num_decodes = 0
            num_spec_decode_tokens = num_decode_tokens
            num_decode_tokens = 0
            num_prefill_tokens = 0
            # print("num_accepted tokens : ", num_accepted_tokens, flush=True)

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
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
        return gdn_attn_metadata

    def prepare_prefill(  # type: ignore[override]
        self,
        batch: ScheduledBatch,
    ) -> GDNAttentionMetadata:
        # print("prepare prefill", flush=True)
        attn_metadata, positions = super().prepare_prefill(batch)
        if batch.block_tables == []:
            attn_metadata.gdn_metadata = None
            return attn_metadata, positions
        gdn_metadata = self.prepare_gdn_metadata(batch, attn_metadata, is_prefill=True)

        attn_metadata.gdn_metadata = gdn_metadata
        # print("gdn attn metadata: ", gdn_metadata, flush=True)
        # print("positions: ", positions, flush=True)
        return attn_metadata, positions

    def prepare_decode(  # type: ignore[override]
        self,
        batch: ScheduledBatch,
        bs: int,
    ) -> GDNAttentionMetadata:
        # print("prepare decode", flush=True)
        num_decodes = batch.total_seqs_num_decode
        # print("num decodes: ", num_decodes, flush=True)
        attn_metadata, positions = super().prepare_decode(batch, bs)
        self.model_runner.forward_vars["cu_seqlens_q"].cpu[
            bs:
        ] = batch.total_tokens_num_decode
        # we fill the attn_metadata cu_seqlens_q here since aiter attn won't calc it for decode
        attn_metadata.cu_seqlens_q = self.model_runner.forward_vars[
            "cu_seqlens_q"
        ].copy_to_gpu(bs + 1)

        gdn_metadata = self.prepare_gdn_metadata(batch, attn_metadata)

        # transfer data to ps buffer
        # gdn_metadata.non_spec_query_start_loc = self.model_runner.forward_vars["cu_seqlens_q"].copy_to_gpu(bs + 1)
        if self.use_spec_decode:
            # self.spec_state_indices_tensor.gpu[:num_decodes, : self.num_spec + 1].copy_(
            #     gdn_metadata.spec_state_indices_tensor, non_blocking=True
            # )
            self.spec_state_indices_tensor.gpu[num_decodes:, :].fill_(PAD_SLOT_ID)
            # gdn_metadata.spec_state_indices_tensor = self.spec_state_indices_tensor[:num_decodes]

            self.spec_sequence_masks[:num_decodes].copy_(
                gdn_metadata.spec_sequence_masks, non_blocking=True
            )
            self.spec_sequence_masks[num_decodes:].fill_(False)
            gdn_metadata.spec_sequence_masks = self.spec_sequence_masks[:num_decodes]

            self.spec_token_indx[: gdn_metadata.spec_token_indx.size(0)].copy_(
                gdn_metadata.spec_token_indx, non_blocking=True
            )
            gdn_metadata.spec_token_indx = self.spec_token_indx[
                : gdn_metadata.spec_token_indx.size(0)
            ]

            self.spec_query_start_loc[: num_decodes + 1].copy_(
                gdn_metadata.spec_query_start_loc[: num_decodes + 1], non_blocking=True
            )
            spec_num_query_tokens = self.spec_query_start_loc[num_decodes]
            self.spec_query_start_loc[num_decodes + 1 :].fill_(spec_num_query_tokens)
            gdn_metadata.spec_query_start_loc = self.spec_query_start_loc[
                : num_decodes + 1
            ]

            self.num_accepted_tokens[:num_decodes].copy_(
                gdn_metadata.num_accepted_tokens[:num_decodes], non_blocking=True
            )
            self.num_accepted_tokens[num_decodes:].fill_(1)
            gdn_metadata.num_accepted_tokens = self.num_accepted_tokens[:num_decodes]
        else:
            # self.non_spec_state_indices_tensor[:num_decodes].copy_(gdn_metadata.non_spec_state_indices_tensor, non_blocking=True)
            self.non_spec_state_indices_tensor.gpu[num_decodes:].fill_(PAD_SLOT_ID)
            # gdn_metadata.non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[:num_decodes]

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                gdn_metadata.non_spec_query_start_loc[: num_decodes + 1],
                non_blocking=True,
            )
            self.non_spec_query_start_loc[num_decodes + 1 :].fill_(
                gdn_metadata.non_spec_query_start_loc[num_decodes]
            )
            gdn_metadata.non_spec_query_start_loc = self.non_spec_query_start_loc[
                : num_decodes + 1
            ]

        attn_metadata.gdn_metadata = gdn_metadata
        # print("gdn attn meta: ", gdn_metadata, flush=True)
        # print("attn meta: ", attn_metadata, flush=True)
        # print("positions: ", positions, flush=True)
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
        # num_accepted_tokens = torch.diff(attn_metadata.cu_seqlen_q)
        # self.num_accepted_tokens[:bs].copy_(1, non_blocking=True)
        if self.use_spec_decode:
            gdn_metadata = GDNAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=0,
                num_decode_tokens=0,
                num_spec_decodes=bs,
                num_spec_decode_tokens=bs * (self.num_spec + 1),
                num_actual_tokens=bs * (self.num_spec + 1),
                has_initial_state=None,
                spec_query_start_loc=self.spec_query_start_loc[: bs + 1],
                non_spec_query_start_loc=None,
                spec_state_indices_tensor=self.spec_state_indices_tensor.gpu[:bs],
                non_spec_state_indices_tensor=None,
                spec_sequence_masks=self.spec_sequence_masks[:bs],
                spec_token_indx=self.spec_token_indx[: bs * (self.num_spec + 1)],
                non_spec_token_indx=self.non_spec_token_indx[:0],
                num_accepted_tokens=self.num_accepted_tokens[:bs],
                nums_dict=None,
                batch_ptr=None,
                token_chunk_offset_ptr=None,
            )
        else:
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
                non_spec_query_start_loc=self.non_spec_query_start_loc[: bs + 1],
                spec_state_indices_tensor=None,
                non_spec_state_indices_tensor=self.non_spec_state_indices_tensor.gpu[
                    :bs
                ],
                spec_sequence_masks=None,
                spec_token_indx=None,
                non_spec_token_indx=None,
                num_accepted_tokens=None,
                nums_dict=None,
                batch_ptr=None,
                token_chunk_offset_ptr=None,
            )
        attn_metadata.gdn_metadata = gdn_metadata
        # print("gdn attn metadata for cudagraph capture: ", gdn_metadata, flush=True)
        # print("attn metadata for cudagraph capture: ", attn_metadata, flush=True)

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
