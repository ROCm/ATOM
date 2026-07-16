# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import inspect
import logging
from dataclasses import dataclass
from typing import List, Optional, Type

import numpy as np
import torch
import triton
from atom.utils import envs
from aiter import (
    decode_update_mla_metadata_v1,
    dtypes,
    get_mla_metadata_info_v1,
    get_mla_metadata_v1,
)
from atom.distributed.pcp_utils import (
    get_pcp_world_size,
    pcp_is_enabled,
    pcp_pad_dense,
    pcp_pad_len,
    pcp_round_robin_query_indices,
)
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mla import _MLA_MIN_HEADS, MLAAttention
from atom.utils import CpuGpuBuffer
from atom.utils.block_convert import (
    kv_indices_generate_triton,
    mtp_prepare_decode_mla_kernel,
)
from atom.utils.forward_context import AttentionMetaData, Context

from .backends import AttentionBackend, CommonAttentionBuilder

logger = logging.getLogger("atom")

# `max_split_per_batch` is only needed (and only exists in newer aiter builds)
# for the segmented page_size>1 MLA path. Detect support once so the default
# page_size=1 path never passes an unsupported kwarg.
try:
    _MLA_META_SUPPORTS_MAX_SPLIT = (
        "max_split_per_batch" in inspect.signature(get_mla_metadata_info_v1).parameters
    )
except (TypeError, ValueError):
    _MLA_META_SUPPORTS_MAX_SPLIT = False


def _mla_seg_meta_kwargs() -> dict:
    """Extra kwargs for ``get_mla_metadata_info_v1`` on the seg (page_size>1)
    path. Empty on the original page_size=1 path so behavior is unchanged."""
    if envs.ATOM_MLA_PAGE_SIZE > 1 and _MLA_META_SUPPORTS_MAX_SPLIT:
        return {"max_split_per_batch": 16}
    return {}


@dataclass
class MLAChunkContextMetadata:
    """Per-chunk slices of the cached prefix for chunked MLA prefill.

    Built host-side in `AiterMLAMetadataBuilder.prepare_prefill` when the
    cached prefix exceeds `config.attn_prefill_chunk_size`. The forward iterates
    these chunks instead of materializing the full `total_kv × heads × dim`
    k/v tensors (which OOM on long contexts).

    Each list entry [c] holds the chunk-c data:
      kv_indptr[c]:   [bs+1] cumulative chunk-local block range per seq
      kv_indices[c]:  [sum_chunk_blocks] physical block ids for this chunk
      cu_seqlens_k[c]: [bs+1] cumulative chunk-local token counts per seq
      total_tokens[c]: int — sum of per-seq chunk lengths
      max_seqlen_k[c]: int — max per-seq chunk length

    `k_workspace` / `v_workspace` are shared across chunks (overwritten each
    iteration); only `[:total_tokens[c]]` is valid for chunk c.
    """

    kv_indptr: List[torch.Tensor]
    kv_indices: List[torch.Tensor]
    cu_seqlens_k: List[torch.Tensor]
    total_tokens: List[int]
    max_seqlen_k: List[int]
    num_chunks: int
    k_workspace: torch.Tensor
    v_workspace: torch.Tensor
    # Block-granular CSR per chunk for the shuffled-KV gather (block_size=64
    # blocks instead of token slots). None for the plain token-slot layout.
    shuffle_kv_block_indptr: Optional[List[torch.Tensor]] = None
    shuffle_kv_block_indices: Optional[List[torch.Tensor]] = None


def cdiv(a, b):
    return (a + b - 1) // b


class AiterMLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA"

    @staticmethod
    def get_builder_cls() -> Type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["MLAAttention"]:
        return MLAAttention


class AiterMLAMetadataBuilder(CommonAttentionBuilder):
    # EagleProposer folds the per-draft-step position/context bump into
    # prepare_mtp_decode's fused kernel when this is set (matches the MHA
    # backend). The fused kernel handles both sparse and dense MLA.
    fuse_mtp_decode_position_update = True

    def __init__(self, model_runner):
        if envs.ATOM_MLA_PAGE_SIZE > 1:
            self.block_size = envs.ATOM_MLA_PAGE_SIZE
        else:
            self.block_size = 1
        if envs.ATOM_USE_TRITON_MLA and envs.ATOM_USE_TRITON_MLA_SHUFFLE_KV:
            assert model_runner.block_size == 64, (
                f"ATOM_USE_TRITON_MLA=1 and ATOM_USE_TRITON_MLA_SHUFFLE_KV=1 expects --block-size 64 "
                f"for {model_runner.kv_cache_dtype} KV cache, "
                f"got --block-size {model_runner.block_size}"
            )
        CommonAttentionBuilder.__init__(self, model_runner)
        # Single-program block for the fused MTP-decode metadata kernel. Sized
        # to the max batch (runtime bs <= max_bs) so one tl.cumsum spans the
        # whole batch in a single launch.
        self._mtp_fuse_block = triton.next_power_of_2(self.max_bs + 1)
        config = model_runner.config
        hf_config = config.hf_config
        # `self.num_attention_heads` set by CommonAttentionBuilder.__init__.
        self.padded_num_attention_heads = max(self.num_attention_heads, _MLA_MIN_HEADS)
        self.is_sparse = model_runner.is_deepseek_v32
        self.index_topk = hf_config.index_topk if self.is_sparse else -1
        self.dtype_kv = dtypes.d_dtypes[config.kv_cache_dtype]
        self.dtype_q = self.dtype_kv

        max_seqlen_qo = getattr(model_runner, "num_spec_tokens", 0) + 1
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            self.max_bs,
            max_seqlen_qo,
            self.padded_num_attention_heads,
            self.dtype_q,
            self.dtype_kv,
            is_sparse=self.is_sparse,
            fast_mode=True,
            **_mla_seg_meta_kwargs(),
        )
        i32_kwargs = {"dtype": torch.int32, "device": self.device}

        mla_metadata = {
            # AITER MLA specific persistent buffers
            "work_meta_data": torch.empty(
                work_meta_data_size, dtype=work_meta_data_type, device=self.device
            ),
            "work_indptr": torch.empty(
                work_indptr_size, dtype=work_indptr_type, device=self.device
            ),
            "work_info_set": torch.empty(
                work_info_set_size, dtype=work_info_set_type, device=self.device
            ),
            "reduce_indptr": torch.empty(
                reduce_indptr_size, dtype=reduce_indptr_type, device=self.device
            ),
            "reduce_final_map": torch.empty(
                reduce_final_map_size, dtype=reduce_final_map_type, device=self.device
            ),
            "reduce_partial_map": torch.empty(
                reduce_partial_map_size,
                dtype=reduce_partial_map_type,
                device=self.device,
            ),
            "kv_indptr": CpuGpuBuffer(self.max_bs + 1, **i32_kwargs),
            "kv_indices": CpuGpuBuffer(
                self.max_bs * self.max_num_blocks_per_seq,
                **i32_kwargs,
            ),
            "kv_last_page_lens": CpuGpuBuffer(self.max_bs, **i32_kwargs),
        }
        mla_metadata["kv_last_page_lens"].cpu.fill_(1)
        mla_metadata["kv_last_page_lens"].copy_to_gpu()
        if self.is_sparse:
            mla_metadata["cu_seqlen_ke"] = CpuGpuBuffer(
                self.max_num_batched_tokens, **i32_kwargs
            )
            mla_metadata["cu_seqlen_ks"] = CpuGpuBuffer(
                self.max_num_batched_tokens, **i32_kwargs
            )
            mla_metadata["sparse_kv_indptr"] = CpuGpuBuffer(
                self.max_num_batched_tokens + 1, **i32_kwargs
            )
            mla_metadata["sparse_cu_seqlens_q"] = CpuGpuBuffer(
                self.max_num_batched_tokens + 1, **i32_kwargs
            )
            mla_metadata["sparse_cu_seqlens_q"].np[:] = np.arange(
                self.max_num_batched_tokens + 1, dtype=np.int32
            )
            mla_metadata["sparse_cu_seqlens_q"].copy_to_gpu()
            mla_metadata["sparse_kv_last_page_lens"] = CpuGpuBuffer(
                self.max_num_batched_tokens, **i32_kwargs
            )
            mla_metadata["sparse_kv_last_page_lens"].np[:] = 1
            mla_metadata["sparse_kv_last_page_lens"].copy_to_gpu()
            self._sparse_kv_indices_gpu = torch.empty(
                self.max_num_batched_tokens * self.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
            (
                (spp_wmd_size, spp_wmd_type),
                (spp_wi_size, spp_wi_type),
                (spp_wis_size, spp_wis_type),
                (spp_ri_size, spp_ri_type),
                (spp_rfm_size, spp_rfm_type),
                (spp_rpm_size, spp_rpm_type),
            ) = get_mla_metadata_info_v1(
                self.max_num_batched_tokens,
                1,  # sparse prefill treats each query token as q_len=1
                self.padded_num_attention_heads,
                self.dtype_q,
                self.dtype_kv,
                is_sparse=True,
                fast_mode=True,
            )
            mla_metadata["sparse_prefill_work_meta_data"] = torch.empty(
                spp_wmd_size, dtype=spp_wmd_type, device=self.device
            )
            mla_metadata["sparse_prefill_work_indptr"] = torch.empty(
                spp_wi_size, dtype=spp_wi_type, device=self.device
            )
            mla_metadata["sparse_prefill_work_info_set"] = torch.empty(
                spp_wis_size, dtype=spp_wis_type, device=self.device
            )
            mla_metadata["sparse_prefill_reduce_indptr"] = torch.empty(
                spp_ri_size, dtype=spp_ri_type, device=self.device
            )
            mla_metadata["sparse_prefill_reduce_final_map"] = torch.empty(
                spp_rfm_size, dtype=spp_rfm_type, device=self.device
            )
            mla_metadata["sparse_prefill_reduce_partial_map"] = torch.empty(
                spp_rpm_size, dtype=spp_rpm_type, device=self.device
            )

        if self.is_sparse and max_seqlen_qo > 1:
            # Allocate a second set of persistent work buffers for sparse MTP
            # per-token layout: max_bs*max_seqlen_qo virtual seqs, each q_len=1.
            smt_max_bs = self.max_bs * max_seqlen_qo
            (
                (smt_wmd_size, smt_wmd_type),
                (smt_wi_size, smt_wi_type),
                (smt_wis_size, smt_wis_type),
                (smt_ri_size, smt_ri_type),
                (smt_rfm_size, smt_rfm_type),
                (smt_rpm_size, smt_rpm_type),
            ) = get_mla_metadata_info_v1(
                smt_max_bs,
                1,  # max_seqlen_qo=1 for per-token
                self.padded_num_attention_heads,
                self.dtype_q,
                self.dtype_kv,
                is_sparse=True,
                fast_mode=True,
                **_mla_seg_meta_kwargs(),
            )
            mla_metadata["sparse_mtp_work_meta_data"] = torch.empty(
                smt_wmd_size, dtype=smt_wmd_type, device=self.device
            )
            mla_metadata["sparse_mtp_work_indptr"] = torch.empty(
                smt_wi_size, dtype=smt_wi_type, device=self.device
            )
            mla_metadata["sparse_mtp_work_info_set"] = torch.empty(
                smt_wis_size, dtype=smt_wis_type, device=self.device
            )
            mla_metadata["sparse_mtp_reduce_indptr"] = torch.empty(
                smt_ri_size, dtype=smt_ri_type, device=self.device
            )
            mla_metadata["sparse_mtp_reduce_final_map"] = torch.empty(
                smt_rfm_size, dtype=smt_rfm_type, device=self.device
            )
            mla_metadata["sparse_mtp_reduce_partial_map"] = torch.empty(
                smt_rpm_size, dtype=smt_rpm_type, device=self.device
            )

        self.model_runner.forward_vars.update(mla_metadata)

        # Chunked-context workspaces for the prefill has_cached path. Sized
        # to config.attn_prefill_chunk_size (defaults to max_num_batched_tokens)
        # so peak memory is bounded regardless of total context length.
        # Allocated outside any per-step scope so a single buffer is shared
        # across all chunks and layers.
        self.attn_prefill_chunk_size = config.attn_prefill_chunk_size
        self.k_chunk_workspace: Optional[torch.Tensor] = None
        self.v_chunk_workspace: Optional[torch.Tensor] = None
        if self.attn_prefill_chunk_size > 0:
            qk_head_dim = hf_config.qk_nope_head_dim + hf_config.qk_rope_head_dim
            v_head_dim = hf_config.v_head_dim
            model_dtype = config.torch_dtype
            self.k_chunk_workspace = torch.empty(
                (
                    self.attn_prefill_chunk_size,
                    self.num_attention_heads,
                    qk_head_dim,
                ),
                dtype=model_dtype,
                device=self.device,
            )
            self.v_chunk_workspace = torch.empty(
                (
                    self.attn_prefill_chunk_size,
                    self.num_attention_heads,
                    v_head_dim,
                ),
                dtype=model_dtype,
                device=self.device,
            )
            mib = (
                self.k_chunk_workspace.numel() * self.k_chunk_workspace.element_size()
                + self.v_chunk_workspace.numel() * self.v_chunk_workspace.element_size()
            ) / (1024 * 1024)
            logger.info(
                "Allocated MLA chunked-prefill workspaces: "
                "k%s v%s (%.1f MiB total, dtype=%s)",
                tuple(self.k_chunk_workspace.shape),
                tuple(self.v_chunk_workspace.shape),
                mib,
                model_dtype,
            )

        if self.is_sparse:
            sfc = config.compilation_config.static_forward_context
            for module in sfc.values():
                if hasattr(module, "sparse_kv_indices_buffer"):
                    module.sparse_kv_indices_buffer = self._sparse_kv_indices_gpu
                impl = getattr(module, "impl", None)
                if impl is not None and hasattr(impl, "sparse_kv_indices_buffer"):
                    impl.sparse_kv_indices_buffer = self._sparse_kv_indices_gpu
            self._token_to_seq_idxs_gpu = torch.zeros(
                self.max_num_batched_tokens,
                dtype=torch.int32,
                device=self.device,
            )

        # Per-ubatch buffers for CUDAGraph TBO
        if config.enable_tbo:
            self._allocate_ubatch_buffers(
                max_seqlen_qo,
                work_meta_data_size,
                work_meta_data_type,
                work_indptr_size,
                work_indptr_type,
                work_info_set_size,
                work_info_set_type,
                reduce_indptr_size,
                reduce_indptr_type,
                reduce_final_map_size,
                reduce_final_map_type,
                reduce_partial_map_size,
                reduce_partial_map_type,
            )

    _NUM_TBO_UBATCHES = 2

    def _allocate_ubatch_buffers(
        self,
        max_seqlen_qo,
        work_meta_data_size,
        work_meta_data_type,
        work_indptr_size,
        work_indptr_type,
        work_info_set_size,
        work_info_set_type,
        reduce_indptr_size,
        reduce_indptr_type,
        reduce_final_map_size,
        reduce_final_map_type,
        reduce_partial_map_size,
        reduce_partial_map_type,
    ):
        """Allocate per-ubatch CpuGpuBuffers for CUDAGraph TBO."""
        i32_kwargs = {"dtype": torch.int32, "device": self.device}
        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        var = self.model_runner.forward_vars
        ub_max_bs = self.max_bs  # allocate full size for safety

        for ub_idx in range(self._NUM_TBO_UBATCHES):
            p = f"ub{ub_idx}_"
            var[f"{p}kv_indptr"] = CpuGpuBuffer(ub_max_bs + 1, **i32_kwargs)
            var[f"{p}kv_indices"] = CpuGpuBuffer(
                self.max_bs * self.max_num_blocks_per_seq,
                **i32_kwargs,
            )
            var[f"{p}context_lens"] = CpuGpuBuffer(ub_max_bs, **i32_kwargs)
            var[f"{p}kv_last_page_lens"] = CpuGpuBuffer(ub_max_bs, **i32_kwargs)
            var[f"{p}kv_last_page_lens"].cpu.fill_(0)
            var[f"{p}kv_last_page_lens"].copy_to_gpu()
            var[f"{p}slot_mapping"] = CpuGpuBuffer(
                ub_max_bs * max_seqlen_qo,
                **i64_kwargs,
            )
            var[f"{p}block_tables"] = CpuGpuBuffer(
                ub_max_bs,
                self.max_num_blocks_per_seq // self.block_ratio,
                **i32_kwargs,
            )
            var[f"{p}cu_seqlens_q"] = CpuGpuBuffer(ub_max_bs + 1, **i32_kwargs)
            var[f"{p}cu_seqlens_q"].cpu.copy_(
                torch.arange(
                    0,
                    (ub_max_bs + 1) * max_seqlen_qo,
                    step=max_seqlen_qo,
                    dtype=torch.int32,
                )
            )
            var[f"{p}cu_seqlens_q"].copy_to_gpu()

            if self.is_sparse:
                var[f"{p}sparse_kv_indptr"] = CpuGpuBuffer(
                    ub_max_bs + 1,
                    **i32_kwargs,
                )

            # MLA work buffers per ubatch (GPU only)
            var[f"{p}work_meta_data"] = torch.empty(
                work_meta_data_size,
                dtype=work_meta_data_type,
                device=self.device,
            )
            var[f"{p}work_indptr"] = torch.empty(
                work_indptr_size,
                dtype=work_indptr_type,
                device=self.device,
            )
            var[f"{p}work_info_set"] = torch.empty(
                work_info_set_size,
                dtype=work_info_set_type,
                device=self.device,
            )
            var[f"{p}reduce_indptr"] = torch.empty(
                reduce_indptr_size,
                dtype=reduce_indptr_type,
                device=self.device,
            )
            var[f"{p}reduce_final_map"] = torch.empty(
                reduce_final_map_size,
                dtype=reduce_final_map_type,
                device=self.device,
            )
            var[f"{p}reduce_partial_map"] = torch.empty(
                reduce_partial_map_size,
                dtype=reduce_partial_map_type,
                device=self.device,
            )

    @property
    def prep_stream(self):
        # return self.model_runner.tokenID_processor.async_copy_stream
        return self.model_runner.async_execute_stream

    def _set_mla_persistent_worker_buffers_sparse_mtp(
        self,
        num_tokens: int,
    ):
        """Compute persistent metadata for sparse MTP per-token layout.

        B = batch_size * max_seqlen_q tokens are treated as B independent
        virtual sequences each with q_len=1.  cu_seqlens_q = [0,1,...,B],
        kv_indptr = per-token sparse_kv_indptr, kv_last_page_lens = all 1s.

        Uses separate sparse_mtp_* buffers so dense layers can keep
        their own persistent metadata (max_seqlen_qo=2) intact.
        """
        var = self.model_runner.forward_vars
        split_params = {
            "kv_granularity": max(self.block_size, 16),
            "max_seqlen_qo": 1,
            "uni_seqlen_qo": 1,
            "fast_mode": 1,
            "max_split_per_batch": 16,
        }
        work_meta_data = var["sparse_mtp_work_meta_data"]
        work_info_set = var["sparse_mtp_work_info_set"]
        work_indptr = var["sparse_mtp_work_indptr"]
        reduce_indptr = var["sparse_mtp_reduce_indptr"]
        reduce_final_map = var["sparse_mtp_reduce_final_map"]
        reduce_partial_map = var["sparse_mtp_reduce_partial_map"]
        get_mla_metadata_v1(
            var["sparse_cu_seqlens_q"].gpu[: num_tokens + 1],
            var["sparse_kv_indptr"].gpu[: num_tokens + 1],
            var["sparse_kv_last_page_lens"].gpu[:num_tokens],
            self.padded_num_attention_heads,
            1,  # nhead_kv
            True,
            work_meta_data,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            page_size=self.block_size,
            dtype_q=self.dtype_q,
            dtype_kv=self.dtype_kv,
            **split_params,
        )
        return {
            "sparse_mtp_work_meta_data": work_meta_data,
            "sparse_mtp_work_info_set": work_info_set,
            "sparse_mtp_work_indptr": work_indptr,
            "sparse_mtp_reduce_indptr": reduce_indptr,
            "sparse_mtp_reduce_final_map": reduce_final_map,
            "sparse_mtp_reduce_partial_map": reduce_partial_map,
        }

    def set_mla_persistent_worker_buffers(
        self,
        bs: int,
        max_q_len: int,
        only_update: bool = False,
        num_reject_tokens: torch.Tensor = None,
        sparse_decode: bool = False,
    ):
        split_params = {
            "kv_granularity": max(self.block_size, 16),
            "max_seqlen_qo": max_q_len,
            "uni_seqlen_qo": max_q_len,
            "fast_mode": 1,
            "max_split_per_batch": 16,
        }
        var = self.model_runner.forward_vars
        work_meta_data = var["work_meta_data"]
        work_info_set = var["work_info_set"]
        work_indptr = var["work_indptr"]
        reduce_indptr = var["reduce_indptr"]
        reduce_final_map = var["reduce_final_map"]
        reduce_partial_map = var["reduce_partial_map"]
        # This work metadata feeds sparse (DSA) attention when either:
        #   - max_q_len == 1: the plain single-token sparse decode, or
        #   - sparse_decode=True: the MTP draft (EagleProposer) whose single
        #     sparse block reuses these buffers but passes the target's original
        #     max_seqlen_qo (>1) through prepare_mtp_decode, so the max_q_len==1
        #     test alone misses it.
        # In both cases the KV is the per-token top-k selection, so the metadata
        # must be built from sparse_kv_indptr; using the dense kv_indptr makes the
        # asm kernel's kv_end run past sparse_kv_indptr[-1] into the stale region
        # of the persistent sparse-index buffer once the context exceeds
        # index_topk (dense >> sparse) -> illegal KV-cache access.
        use_sparse_meta = self.is_sparse and (max_q_len == 1 or sparse_decode)
        kv_indptr_for_metadata = (
            var["sparse_kv_indptr"].gpu[: bs + 1]
            if use_sparse_meta
            else var["kv_indptr"].gpu[: bs + 1]
        )
        # Sparse decode packs KV per query token at page_size=1, so every "page"
        # is exactly one token -> last_page_len must be 1. The dense
        # var["kv_last_page_lens"] holds the real last-BLOCK fill (1..block_size);
        # feeding it here makes get_mla_metadata_v1 compute a per-seq KV extent of
        # (sparse_count - 1 + dense_last_page_len), i.e. up to block_size-1 pages
        # PAST the written sparse-index region -> stale-index over-read. Mirror
        # kv_indptr_for_metadata (and the prefill/MTP-verify paths, which already
        # use the all-1s sparse buffer).
        kv_last_page_lens_for_metadata = (
            var["sparse_kv_last_page_lens"].gpu[:bs]
            if use_sparse_meta
            else var["kv_last_page_lens"].gpu[:bs]
        )
        if only_update:
            decode_update_mla_metadata_v1(
                var["cu_seqlens_q"].gpu[: bs + 1],
                kv_indptr_for_metadata,
                kv_last_page_lens_for_metadata,
                self.padded_num_attention_heads,
                1,  # nhead_kv,
                True,
                work_meta_data,
                work_info_set,
                work_indptr,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                page_size=self.block_size,
                kv_granularity=max(self.block_size, 16),
                max_seqlen_qo=max_q_len,
                dtype_q=self.dtype_q,
                dtype_kv=self.dtype_kv,
                num_reject_tokens=num_reject_tokens,
            )
        else:
            get_mla_metadata_v1(
                var["cu_seqlens_q"].gpu[: bs + 1],
                kv_indptr_for_metadata,
                kv_last_page_lens_for_metadata,
                self.padded_num_attention_heads,
                1,  # nhead_kv,
                True,
                work_meta_data,
                work_info_set,
                work_indptr,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                page_size=self.block_size,
                dtype_q=self.dtype_q,
                dtype_kv=self.dtype_kv,
                **split_params,
            )
        return {
            "work_meta_data": work_meta_data,
            "work_info_set": work_info_set,
            "work_indptr": work_indptr,
            "reduce_indptr": reduce_indptr,
            "reduce_final_map": reduce_final_map,
            "reduce_partial_map": reduce_partial_map,
        }

    def prepare_mtp_decode(
        self,
        bs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        positions: torch.Tensor,  # [total_tokens] int32
        only_update: bool = False,
        num_reject_tokens: torch.Tensor = None,
        *,
        update_context_lens: bool = False,
        positions_out: torch.Tensor | None = None,
        last_token_indices: torch.Tensor | None = None,
    ):
        """Per-draft-step MLA metadata update, fused into a single kernel.

        One ``_mtp_prepare_decode_mla_kernel`` launch performs, in place:
          - ``kv_indptr += cu_seqlens_q`` (needed by kv_indices + slot_mapping),
          - (sparse) per-seq ``min(kv_count, index_topk)`` cumsum ->
            ``sparse_kv_indptr``,
          - (fused position update) ``positions += 1`` when ``positions_out`` is
            given, and ``context_lens += 1`` when ``update_context_lens`` is set.

        ``fuse_mtp_decode_position_update`` makes EagleProposer route the
        per-step position/context bumps through here instead of launching them
        as separate kernels. ``last_token_indices`` is accepted for signature
        parity with the MHA backend but unused (MLA's ``positions`` is already
        one entry per sequence at this point).
        """
        del last_token_indices  # MLA positions are already per-seq (1 per token)
        var = self.model_runner.forward_vars
        kv_indptr = var["kv_indptr"].gpu[: bs + 1]
        cu_seqlens_q = var["cu_seqlens_q"].gpu[: bs + 1]
        if self.is_sparse:
            sparse_kv_indptr = var["sparse_kv_indptr"].gpu[: bs + 1]
        else:
            assert self.block_size == 1
            sparse_kv_indptr = None

        update_positions = positions_out is not None
        context_lens = var["context_lens"].gpu[:bs] if update_context_lens else None

        mtp_prepare_decode_mla_kernel[(1,)](
            kv_indptr,
            cu_seqlens_q,
            sparse_kv_indptr if self.is_sparse else kv_indptr,
            positions_out if update_positions else kv_indptr,
            context_lens if update_context_lens else kv_indptr,
            bs,
            self.index_topk if self.is_sparse else 0,
            positions_out.stride(0) if update_positions else 1,
            IS_SPARSE=self.is_sparse,
            UPDATE_POSITIONS=update_positions,
            UPDATE_CONTEXT_LENS=update_context_lens,
            BLOCK=self._mtp_fuse_block,
        )

        kv_indices_generate_triton(
            var["block_tables"].gpu[:bs],
            var["kv_indices"].gpu,
            kv_indptr,
            self.block_ratio,
            max_seqlen_k,
        )
        if self.is_sparse:
            # The MTP draft's single sparse block reads sparse_kv_indptr, but it
            # reuses the TARGET's work_info buffer, which was built dense. The
            # incremental decode_update path cannot convert that dense work_info
            # to sparse: it rebases each item's (dense) work_kv_len onto the new
            # sparse seq_kv_end, driving kv_start negative and kv_end past the
            # written sparse-index region -> illegal access. So do a FRESH sparse
            # build (only_update=False) from sparse_kv_indptr instead. The draft
            # emits exactly one query token per seq (cu_seqlens_q is an arange),
            # so max_seqlen_qo must be 1 — passing the caller's max_seqlen_q (the
            # target's verify width, e.g. 4) sets uni_seqlen_qo>1 while
            # cu_seqlens_q says 1, which makes get_mla_metadata_v1 emit q ranges
            # that run past the actual query rows. sparse_kv_indptr already
            # reflects the reject-adjusted KV lengths, so num_reject_tokens is
            # not needed here.
            result = self.set_mla_persistent_worker_buffers(
                bs,
                1,
                only_update=False,
                num_reject_tokens=None,
                sparse_decode=True,
            )
            result["sparse_kv_indptr"] = sparse_kv_indptr
        else:
            result = self.set_mla_persistent_worker_buffers(
                bs, max_seqlen_q, only_update, num_reject_tokens
            )
        return result

    def compute_block_bytes(self) -> int:
        """MLA per-block bytes: single 576-dim packed tensor per layer
        (k_c + k_pe; V is absorbed into latent compression — no separate
        V cache or kv_scale).

        DeepSeek-V3.2 sparse variant adds an indexer cache contribution
        for every bound layer, including draft/MTP layers.
        """
        runner = self.model_runner
        config = runner.config
        hf_config = config.hf_config
        total_num_layers = runner._get_total_num_layers()
        kv_dtype_size = dtypes.d_dtypes[config.kv_cache_dtype].itemsize

        block_bytes = total_num_layers * runner.block_size * 576 * kv_dtype_size
        if runner.is_deepseek_v32:
            index_dim = hf_config.index_head_dim + 4
            aligned_index_dim = ((index_dim + 15) // 16) * 16
            block_bytes += (
                total_num_layers
                * runner.block_size
                * aligned_index_dim
                * dtypes.fp8.itemsize
            )
        return block_bytes

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict:
        """MLA: single 576-dim paged tensor per layer (k_c + k_pe packed,
        no separate V cache — MLA absorbs V into the latent compression).

        DeepSeek-V3.2 sparse variant additionally allocates an `index_cache`
        for the indexer module; the aligned dimension is also returned so
        build_kv_cache_tensor can reslice without recomputing it.
        """
        runner = self.model_runner
        config = runner.config
        hf_config = config.hf_config
        total_num_layers = hf_config.num_hidden_layers + num_draft_layers
        out: dict = {
            "kv_cache": torch.zeros(
                total_num_layers,
                runner.num_physical_kvcache_blocks,
                runner.physical_block_size,
                576,
                dtype=dtypes.d_dtypes[config.kv_cache_dtype],
                device="cuda",
            ),
        }
        if runner.is_deepseek_v32:
            # Align last dimension to 16 bytes for fp8 (1 byte per element)
            # to avoid unaligned memory access in torch inductor.
            index_dim = hf_config.index_head_dim + 4
            aligned = ((index_dim + 15) // 16) * 16
            out["aligned_index_dim"] = aligned
            out["index_cache"] = torch.zeros(
                total_num_layers,
                runner.num_physical_kvcache_blocks,
                runner.physical_block_size,
                aligned,
                dtype=dtypes.fp8,
                device="cuda",
            )
        return out

    def build_kv_cache_tensor(self, layer_id: int, module):
        """Bind one MLA attention module to its KV slice.

        Handles standard MLA (single 576-dim KV cache per layer) and the
        DeepSeek-V3.2 sparse variant (additional indexer cache hooked via
        `module.indexer.k_cache.kv_cache[0]`). Returns the KVCacheTensor or
        None if the module is not an MLA attention this builder owns.
        Side effects: sets module `kv_cache`, `max_model_len`, and (V3.2)
        the indexer's k_cache slot.
        """
        from atom.config import KVCacheTensor

        if not (
            hasattr(module, "base_attention")
            and hasattr(module, "use_mla")
            and module.use_mla
        ):
            return None

        runner = self.model_runner
        kv_cache = runner.kv_cache[layer_id].view(
            runner.num_physical_kvcache_blocks * runner.physical_block_size,
            1,
            576,
        )
        module.max_model_len = runner.config.max_model_len
        if runner.is_deepseek_v32 and module.indexer is not None:
            # Use aligned dimension to avoid memory copy in torch inductor
            module.indexer.k_cache.kv_cache[0] = runner.index_cache[layer_id].view(
                runner.num_physical_kvcache_blocks * runner.physical_block_size,
                1,
                runner.aligned_index_dim,
            )
        module.kv_cache = kv_cache
        return KVCacheTensor(
            layer_num=layer_id,
            k_cache=kv_cache,
            v_cache=None,
            k_scale=None,
            v_scale=None,
        )

    def get_kv_transfer_tensors(self):
        from atom.kv_transfer.disaggregation.types import (
            KVTransferRegion,
            KVTransferTensors,
        )

        runner = self.model_runner
        if not hasattr(runner, "kv_cache"):
            return None

        block_regions: list[KVTransferRegion] = []
        num_layers = runner.kv_cache.shape[0]
        for layer_id in range(num_layers):
            t = runner.kv_cache[layer_id]
            bpb = t.stride(0) * t.element_size() * self.block_ratio
            block_regions.append(
                KVTransferRegion(
                    base_addr=t.data_ptr(),
                    total_bytes=t.numel() * t.element_size(),
                    unit_bytes=bpb,
                )
            )

        if hasattr(runner, "index_cache"):
            for layer_id in range(runner.index_cache.shape[0]):
                t = runner.index_cache[layer_id]
                bpb = t.stride(0) * t.element_size() * self.block_ratio
                block_regions.append(
                    KVTransferRegion(
                        base_addr=t.data_ptr(),
                        total_bytes=t.numel() * t.element_size(),
                        unit_bytes=bpb,
                    )
                )

        return KVTransferTensors(
            block_regions=block_regions,
            slot_regions=[],
            num_blocks=runner.config.num_kvcache_blocks,
        )

    def prepare_prefill(self, batch: ScheduledBatch):
        attn_metadata, positions = CommonAttentionBuilder.prepare_prefill(self, batch)
        bs = batch.total_seqs_num_prefill
        sum_scheduled_tokens = batch.total_tokens_num_prefill
        var = self.model_runner.forward_vars
        if self.is_sparse and attn_metadata.max_seqlen_k > self.index_topk:
            if attn_metadata.block_tables is None:
                self.prepare_block_tables(batch)
                attn_metadata.block_tables = var["block_tables"].copy_to_gpu(bs)
            counts = var["cu_seqlens_q"].np[1 : bs + 1] - var["cu_seqlens_q"].np[:bs]
            local_offsets = np.concatenate(
                [np.arange(s, dtype=np.int32) for s in counts]
            )
            if attn_metadata.has_cached:
                # Full context (cached + new): each query token can see the cached
                # prefix plus previous query tokens in this chunk, not future chunk
                # tokens.
                seq_starts = var["cu_seqlens_k"].np[:bs]
                seq_lens = var["cu_seqlens_k"].np[1 : bs + 1] - seq_starts
                cached_lens = seq_lens - counts
                repeated_seq_starts = np.repeat(seq_starts, counts)
                repeated_cached_lens = np.repeat(cached_lens, counts)
                var["cu_seqlen_ks"].np[:sum_scheduled_tokens] = np.repeat(
                    seq_starts, counts
                )
                var["cu_seqlen_ke"].np[:sum_scheduled_tokens] = (
                    repeated_seq_starts + repeated_cached_lens + local_offsets + 1
                )
                sparse_counts = repeated_cached_lens + local_offsets + 1
            else:
                var["cu_seqlen_ke"].np[:sum_scheduled_tokens] = (
                    np.arange(sum_scheduled_tokens, dtype=np.int32) + 1
                )
                var["cu_seqlen_ks"].np[:sum_scheduled_tokens] = np.repeat(
                    var["cu_seqlens_q"].np[:bs], counts
                )
                sparse_counts = local_offsets + 1
            attn_metadata.cu_seqlen_ks = var["cu_seqlen_ks"].copy_to_gpu(
                sum_scheduled_tokens
            )
            attn_metadata.cu_seqlen_ke = var["cu_seqlen_ke"].copy_to_gpu(
                sum_scheduled_tokens
            )
            attn_metadata.sparse_cu_seqlens_q = var["sparse_cu_seqlens_q"].gpu[
                : sum_scheduled_tokens + 1
            ]
            # Sparse (DSA) attention: one last-page len per query token (all 1s,
            # page_size=1). Lives only on sparse_kv_last_page_lens; kv_last_page_lens
            # stays the dense per-seq buffer set by the has_cached block below.
            attn_metadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:sum_scheduled_tokens]

            # Per-query req_id: token_id 0..sum_scheduled_tokens-1 maps to batch id.
            # Use counts (new tokens per batch), not context_lens (full seq len).
            attn_metadata.token_to_seq_idxs = torch.repeat_interleave(
                torch.arange(bs, dtype=torch.int32, device=self.device),
                torch.tensor(counts, dtype=torch.int64, device=self.device),
            )
            var["sparse_kv_indptr"].np[0] = 0
            var["sparse_kv_indptr"].np[1 : sum_scheduled_tokens + 1] = np.cumsum(
                np.minimum(sparse_counts, self.index_topk),
                dtype=np.int32,
            )
            attn_metadata.sparse_kv_indptr = var["sparse_kv_indptr"].copy_to_gpu(
                sum_scheduled_tokens + 1
            )
            get_mla_metadata_v1(
                attn_metadata.sparse_cu_seqlens_q,
                attn_metadata.sparse_kv_indptr,
                attn_metadata.sparse_kv_last_page_lens,
                self.padded_num_attention_heads,
                1,  # nhead_kv
                True,
                var["sparse_prefill_work_meta_data"],
                var["sparse_prefill_work_info_set"],
                var["sparse_prefill_work_indptr"],
                var["sparse_prefill_reduce_indptr"],
                var["sparse_prefill_reduce_final_map"],
                var["sparse_prefill_reduce_partial_map"],
                page_size=self.block_size,
                dtype_q=self.dtype_q,
                dtype_kv=self.dtype_kv,
                kv_granularity=max(self.block_size, 16),
                max_seqlen_qo=1,
                uni_seqlen_qo=1,
                fast_mode=1,
                max_split_per_batch=16,
            )
            attn_metadata.sparse_prefill_work_meta_data = var[
                "sparse_prefill_work_meta_data"
            ]
            attn_metadata.sparse_prefill_work_info_set = var[
                "sparse_prefill_work_info_set"
            ]
            attn_metadata.sparse_prefill_work_indptr = var["sparse_prefill_work_indptr"]
            attn_metadata.sparse_prefill_reduce_indptr = var[
                "sparse_prefill_reduce_indptr"
            ]
            attn_metadata.sparse_prefill_reduce_final_map = var[
                "sparse_prefill_reduce_final_map"
            ]
            attn_metadata.sparse_prefill_reduce_partial_map = var[
                "sparse_prefill_reduce_partial_map"
            ]

            # ---- Prefill Context Parallel: shrink per-query sparse metadata --
            # to this rank's 1/pcp round-robin queries. Gate on
            # `not batch.is_dummy_run` so the reindex stays in lock-step with the
            # model's round-robin token split (ForCausalLM._pcp_active() also
            # skips dummy/warmup). Per-sequence + KV-write fields (slot_mapping,
            # block_tables, cu_seqlens_q/k) stay FULL — every rank keeps full KV.
            if pcp_is_enabled() and not batch.is_dummy_run:
                self._apply_pcp_reindex(
                    attn_metadata, sum_scheduled_tokens, sparse_counts
                )

        if hasattr(self.model_runner, "drafter") or attn_metadata.has_cached:
            # Populate kv_last_page_lens for full sequence (needed for MLA prefill with
            # prefix cache; decode does the same)
            if self.model_runner.block_size != 1:
                var["kv_last_page_lens"].np[:bs] = np.asarray(
                    batch.last_block_num_tokens[:bs], dtype=np.int32
                )
            else:
                var["kv_last_page_lens"].np[:bs] = 1
            var["kv_last_page_lens"].copy_to_gpu()

            attn_metadata.kv_indices = var["kv_indices"].gpu
            attn_metadata.kv_indptr = var["kv_indptr"].gpu[: bs + 1]
            attn_metadata.kv_indptr[0] = 0
            attn_metadata.kv_indptr[1 : bs + 1] = torch.cumsum(
                attn_metadata.context_lens, 0
            )
            attn_metadata.kv_last_page_lens = var["kv_last_page_lens"].gpu[:bs]

            # kv_indices_generate_triton expects logical block_tables (one entry
            # per block_ratio tokens). Re-copy from var to get a fresh logical
            # snapshot independent of attn_metadata.block_tables sharing.
            self.prepare_block_tables(batch)
            block_tables_for_kv = var["block_tables"].copy_to_gpu(bs)
            kv_indices_generate_triton(
                block_tables_for_kv,
                attn_metadata.kv_indices,
                attn_metadata.kv_indptr,
                self.block_ratio,
                attn_metadata.max_seqlen_k,
            )

            # Build chunked-context metadata when enabled AND the cached
            # prefix is large enough to risk OOM in the single-pass path.
            # The non-cached new-tokens portion is handled separately by the
            # forward (self-attention via kv_b_proj), so chunks span only the
            # cached prefix.
            if (
                self.attn_prefill_chunk_size > 0
                and attn_metadata.has_cached
                and attn_metadata.total_kv > self.attn_prefill_chunk_size
            ):
                attn_metadata.mla_chunk_meta = self._build_mla_chunk_meta(batch, bs)

        attn_metadata.dtype_q = self.dtype_q
        return attn_metadata, positions

    def _build_mla_chunk_meta(
        self, batch: ScheduledBatch, bs: int
    ) -> Optional[MLAChunkContextMetadata]:
        """Build per-chunk slices of the cached prefix.

        Chunks the cached-prefix tokens along the GLOBAL token axis (not the
        per-seq axis). Per-chunk total token count ≤ `attn_prefill_chunk_size`,
        which is what the k/v workspace is sized for. Each chunk c contains a
        contiguous slice of the concatenated per-seq slot list; per-seq
        contributions to chunk c are the intersection of seq i's slot range
        with [c*K, (c+1)*K).

        Seqs with 0 contribution to a chunk emit empty k for that seq —
        flash_attn returns lse=-inf which merge_attn_states handles correctly
        (the prefix output for that seq is preserved unchanged).
        """
        chunk_size = self.attn_prefill_chunk_size
        runner_bs = self.model_runner.block_size

        cached_lens = np.asarray(batch.num_cached_tokens[:bs], dtype=np.int64)
        total_cached = int(cached_lens.sum())
        if total_cached == 0:
            return None
        num_chunks = (total_cached + chunk_size - 1) // chunk_size

        # Per-seq absolute slot id for every cached token, in seq order, then
        # concatenated into a single global slot array of length total_cached.
        per_seq_slots: List[np.ndarray] = []
        for i in range(bs):
            cached_len = int(cached_lens[i])
            if cached_len == 0:
                per_seq_slots.append(np.empty(0, dtype=np.int32))
                continue
            block_ids = np.asarray(batch.block_tables[i], dtype=np.int64)
            needed_blocks = (cached_len + runner_bs - 1) // runner_bs
            block_ids = block_ids[:needed_blocks]
            base = block_ids[:, None] * runner_bs
            offsets = np.arange(runner_bs, dtype=np.int64)[None, :]
            slots = (base + offsets).reshape(-1)[:cached_len].astype(np.int32)
            per_seq_slots.append(slots)
        global_slots = (
            np.concatenate(per_seq_slots) if bs > 0 else np.empty(0, np.int32)
        )
        seq_offsets = np.zeros(bs + 1, dtype=np.int64)
        np.cumsum(cached_lens, out=seq_offsets[1:])

        kv_indptr_list: List[torch.Tensor] = []
        kv_indices_list: List[torch.Tensor] = []
        cu_seqlens_k_list: List[torch.Tensor] = []
        total_tokens_list: List[int] = []
        max_seqlen_k_list: List[int] = []

        for c in range(num_chunks):
            g_start = c * chunk_size
            g_end = min(g_start + chunk_size, total_cached)
            # Per-seq contribution: intersect [seq_offsets[i], seq_offsets[i+1])
            # with [g_start, g_end).
            seq_lo = np.maximum(seq_offsets[:bs], g_start)
            seq_hi = np.minimum(seq_offsets[1 : bs + 1], g_end)
            per_seq_chunk_lens = np.maximum(seq_hi - seq_lo, 0).astype(np.int32)
            chunk_indices = global_slots[g_start:g_end].astype(np.int32, copy=False)
            cu = np.zeros(bs + 1, dtype=np.int32)
            np.cumsum(per_seq_chunk_lens, out=cu[1:])
            total_tokens = int(cu[-1])
            # cu doubles as gather_kv_b_proj kv_indptr (block_size=1 → block
            # indptr == token indptr) and flash_attn cu_seqlens_k.
            kv_indptr_list.append(
                torch.from_numpy(cu).pin_memory().to(self.device, non_blocking=True)
            )
            kv_indices_list.append(
                torch.from_numpy(chunk_indices)
                .pin_memory()
                .to(self.device, non_blocking=True)
            )
            cu_seqlens_k_list.append(kv_indptr_list[-1])  # same tensor
            total_tokens_list.append(total_tokens)
            max_seqlen_k_list.append(int(per_seq_chunk_lens.max(initial=0)))

        return MLAChunkContextMetadata(
            kv_indptr=kv_indptr_list,
            kv_indices=kv_indices_list,
            cu_seqlens_k=cu_seqlens_k_list,
            total_tokens=total_tokens_list,
            max_seqlen_k=max_seqlen_k_list,
            num_chunks=num_chunks,
            k_workspace=self.k_chunk_workspace,
            v_workspace=self.v_chunk_workspace,
        )

    def _apply_pcp_reindex(
        self,
        attn_metadata: AttentionMetaData,
        sum_scheduled_tokens: int,
        sparse_counts: np.ndarray,
    ) -> None:
        """Reduce the per-query sparse-prefill metadata to this PCP rank's
        1/pcp round-robin queries.

        Prefill Context Parallel round-robin splits the token sequence so each
        rank runs the model on 1/pcp of the query tokens while still keeping the
        FULL KV. Only *query-indexed* metadata shrinks here; *per-sequence* and
        *KV-write* fields (slot_mapping, block_tables, cu_seqlens_q/k) stay full
        so the full k-cache is still written and gathered.

        The global token count is padded to a multiple of pcp_size; the extra
        (dummy) queries get zero-length KV (they attend nothing and their hidden
        output is dropped after the model's final all-gather + unpad).
        """
        device = self.device
        pcp_ws = get_pcp_world_size()
        s_real = int(sum_scheduled_tokens)
        padded_total = pcp_pad_len(s_real, pcp_ws)
        n_pad = padded_total - s_real
        owned_q = pcp_round_robin_query_indices(padded_total, pcp_ws).to(device)
        n_owned = int(owned_q.shape[0])

        # --- dense per-query fields: pad with zeros (dummy query -> 0), select.
        #     cu_seqlen_ks/ke become 0/0 for dummies == empty logits row.
        ks_padded = pcp_pad_dense(attn_metadata.cu_seqlen_ks, n_pad)
        attn_metadata.cu_seqlen_ks = ks_padded[owned_q].contiguous()
        ke_padded = pcp_pad_dense(attn_metadata.cu_seqlen_ke, n_pad)
        attn_metadata.cu_seqlen_ke = ke_padded[owned_q].contiguous()
        t2s_padded = pcp_pad_dense(attn_metadata.token_to_seq_idxs, n_pad)
        attn_metadata.token_to_seq_idxs = t2s_padded[owned_q].contiguous()

        # --- one query per row (incl dummies) -> sparse_cu_seqlens_q = arange.
        attn_metadata.sparse_cu_seqlens_q = torch.arange(
            n_owned + 1, dtype=torch.int32, device=device
        )

        # --- sparse_kv_indptr: cumsum of min(sparse_counts, topk); dummy -> 0.
        sparse_counts_t = torch.as_tensor(sparse_counts, device=device)
        owned_counts = pcp_pad_dense(sparse_counts_t, n_pad)[owned_q].to(torch.int64)
        owned_counts = torch.clamp(owned_counts, max=self.index_topk)
        indptr_owned = torch.zeros(n_owned + 1, dtype=torch.int32, device=device)
        indptr_owned[1:] = torch.cumsum(owned_counts, 0).to(torch.int32)
        attn_metadata.sparse_kv_indptr = indptr_owned

        # --- sparse kv_last_page_lens: one page per owned query (all 1s).
        attn_metadata.kv_last_page_lens = torch.ones(
            n_owned, dtype=torch.int32, device=device
        )

        # --- rebuild the sparse-prefill work buffers for the owned queries.
        var = self.model_runner.forward_vars
        get_mla_metadata_v1(
            attn_metadata.sparse_cu_seqlens_q,
            attn_metadata.sparse_kv_indptr,
            attn_metadata.kv_last_page_lens,
            self.padded_num_attention_heads,
            1,  # nhead_kv
            True,
            var["sparse_prefill_work_meta_data"],
            var["sparse_prefill_work_info_set"],
            var["sparse_prefill_work_indptr"],
            var["sparse_prefill_reduce_indptr"],
            var["sparse_prefill_reduce_final_map"],
            var["sparse_prefill_reduce_partial_map"],
            page_size=self.block_size,
            dtype_q=self.dtype_q,
            dtype_kv=self.dtype_kv,
            kv_granularity=max(self.block_size, 16),
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=1,
            max_split_per_batch=16,
        )
        attn_metadata.sparse_prefill_work_meta_data = var[
            "sparse_prefill_work_meta_data"
        ]
        attn_metadata.sparse_prefill_work_info_set = var["sparse_prefill_work_info_set"]
        attn_metadata.sparse_prefill_work_indptr = var["sparse_prefill_work_indptr"]
        attn_metadata.sparse_prefill_reduce_indptr = var["sparse_prefill_reduce_indptr"]
        attn_metadata.sparse_prefill_reduce_final_map = var[
            "sparse_prefill_reduce_final_map"
        ]
        attn_metadata.sparse_prefill_reduce_partial_map = var[
            "sparse_prefill_reduce_partial_map"
        ]

        # --- owned slot_mapping for the fused q_out kernel in MLAAttention. The
        #     fused MLA kernel that produces q_out also writes k to these slots;
        #     that write is throwaway (the full-KV completion write in
        #     MLAAttention overwrites every real slot). Dummy queries clamp to
        #     the last real slot so they can never touch an unrelated slot.
        owned_clamped = torch.clamp(owned_q, max=max(s_real - 1, 0))
        attn_metadata.slot_mapping_owned = attn_metadata.slot_mapping[
            owned_clamped
        ].contiguous()

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        scheduled_bs = batch.total_seqs_num_decode
        dropout_p = 0.0
        max_seqlen_q = 1
        if hasattr(self.model_runner, "drafter"):
            max_seqlen_q = self.model_runner.drafter.mtp_k + 1

        var = self.model_runner.forward_vars
        context_lens = np.asarray(batch.context_lens, dtype=np.int32)
        block_tables = batch.block_tables
        if not batch.is_dummy_run:
            if max_seqlen_q > 1:
                # Get num_rejected (already mapped to current batch order in prepare_input_ids)
                num_rejected = self.model_runner.tokenID_processor.num_rejected
                if num_rejected is not None:
                    context_lens -= num_rejected
                    num_blocks = cdiv(context_lens, self.model_runner.block_size)
                    block_tables = [bt[:n] for bt, n in zip(block_tables, num_blocks)]

                slot_mapping = [
                    block_table[pos // self.model_runner.block_size]
                    * self.model_runner.block_size
                    + (pos % self.model_runner.block_size)
                    for block_table, seq_len in zip(block_tables, context_lens)
                    for pos in range(seq_len - max_seqlen_q, seq_len)
                ]
            else:
                slot_mapping = [
                    block_table[-1] * self.model_runner.block_size + last_block_num - 1
                    for block_table, last_block_num in zip(
                        block_tables, batch.last_block_num_tokens
                    )
                ]
        positions = np.tile(
            np.arange(max_seqlen_q, dtype=np.int32), scheduled_bs
        ) + np.repeat(context_lens - max_seqlen_q, max_seqlen_q)

        # Use scheduled_bs since in dummy run, total_seqs_num_decode is 1.
        sum_scheduled_tokens = scheduled_bs * max_seqlen_q
        var["slot_mapping"].np[: bs * max_seqlen_q] = -1
        if not batch.is_dummy_run:
            var["slot_mapping"].np[:sum_scheduled_tokens] = slot_mapping
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens
        var["context_lens"].np[scheduled_bs:bs] = 0

        if any(batch.is_first_decode_without_local_prefill):
            num_blocks_per_seq = [
                (
                    len(batch.block_tables[i])
                    if is_first
                    else cdiv(ctx_len, self.block_size)
                )
                for i, (ctx_len, is_first) in enumerate(
                    zip(
                        batch.context_lens,
                        batch.is_first_decode_without_local_prefill,
                    )
                )
            ]
        else:
            num_blocks_per_seq = cdiv(context_lens, self.block_size)
        kv_indptr = np.cumsum(num_blocks_per_seq)
        sum_blocks = kv_indptr[-1]

        self.prepare_block_tables(batch)
        var["kv_indptr"].np[1 : scheduled_bs + 1] = kv_indptr
        var["kv_indptr"].np[scheduled_bs + 1 : bs + 1] = sum_blocks
        var["kv_last_page_lens"].np[:scheduled_bs] = (
            batch.last_block_num_tokens if self.block_size != 1 else 1
        )
        var["kv_last_page_lens"].np[scheduled_bs:bs] = 0
        vars_used = [
            ("slot_mapping", bs * max_seqlen_q),
            ("context_lens", bs),
            ("cu_seqlens_q", bs + 1),
            # ("kv_indptr", bs + 1),
            ("kv_last_page_lens", bs),
            ("block_tables", bs),
        ]
        metadata_deps = {
            "cu_seqlens_q",
            "kv_last_page_lens",
        }

        if self.is_sparse:
            index_topk = self.index_topk
            if max_seqlen_q > 1:
                # MTP verify: per-token sparse metadata
                # Each token at offset j in seq s sees (context_lens[s] - max_seqlen_q + j + 1) KV entries
                per_token_kv_lens = (
                    np.repeat(context_lens[:scheduled_bs], max_seqlen_q)
                    - max_seqlen_q
                    + np.tile(
                        np.arange(1, max_seqlen_q + 1, dtype=np.int32), scheduled_bs
                    )
                )
                sparse_per_token_lens = np.clip(per_token_kv_lens, 0, index_topk)
                var["sparse_kv_indptr"].np[1 : sum_scheduled_tokens + 1] = np.cumsum(
                    sparse_per_token_lens, dtype=np.int32
                )
                sum_tokens = bs * max_seqlen_q
                var["sparse_kv_indptr"].np[
                    sum_scheduled_tokens + 1 : sum_tokens + 1
                ] = var["sparse_kv_indptr"].np[sum_scheduled_tokens]
                vars_used.append(("sparse_kv_indptr", sum_tokens + 1))
                vars_used.append(("sparse_cu_seqlens_q", sum_tokens + 1))
                metadata_deps.add("sparse_kv_indptr")
            else:
                sparse_context_lens = np.clip(
                    var["context_lens"].np[:bs], None, index_topk
                )
                var["sparse_kv_indptr"].np[1 : bs + 1] = np.cumsum(
                    sparse_context_lens, dtype=np.int32
                )
                var["sparse_kv_indptr"].np[scheduled_bs : bs + 1] = var[
                    "sparse_kv_indptr"
                ].np[scheduled_bs]
                vars_used.append(("sparse_kv_indptr", bs + 1))
                metadata_deps.add("sparse_kv_indptr")

        prep_stream = self.prep_stream
        vars_for_metadata = [(el, num) for el, num in vars_used if el in metadata_deps]
        vars_remaining = [(el, num) for el, num in vars_used if el not in metadata_deps]
        max_seqlen_k = context_lens.max()

        ctx = {}
        ctx["kv_indptr"] = var["kv_indptr"].copy_to_gpu(bs + 1)
        # prep_stream does remaining copies + kv_indices
        current_stream = torch.cuda.current_stream()
        prep_stream.wait_stream(current_stream)
        with torch.cuda.stream(prep_stream):
            ctx_rest = {el: var[el].copy_to_gpu(num) for el, num in vars_remaining}
            ctx.update(ctx_rest)
            ctx["kv_indices"] = var["kv_indices"].gpu
            kv_indices_generate_triton(
                ctx["block_tables"],
                ctx["kv_indices"],
                ctx["kv_indptr"],
                self.block_ratio,
                max_seqlen_k,
            )

        is_sparse_mtp = self.is_sparse and max_seqlen_q > 1
        # metadata copies on main_stream
        positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        ctx.update({el: var[el].copy_to_gpu(num) for el, num in vars_for_metadata})

        if is_sparse_mtp:
            sum_tokens = bs * max_seqlen_q
            ctx_mla_ps = self.set_mla_persistent_worker_buffers(bs, max_seqlen_q)
            ctx_mla_ps_sparse = self._set_mla_persistent_worker_buffers_sparse_mtp(
                sum_tokens
            )
        else:
            ctx_mla_ps = self.set_mla_persistent_worker_buffers(bs, max_seqlen_q)
            ctx_mla_ps_sparse = None
        ctx.update(ctx_mla_ps)
        current_stream.wait_stream(prep_stream)
        attn_metadata = AttentionMetaData(
            dropout_p=dropout_p,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            **ctx,
        )
        attn_metadata.dtype_q = self.dtype_q

        if ctx_mla_ps_sparse is not None:
            for k, v in ctx_mla_ps_sparse.items():
                setattr(attn_metadata, k, v)

        if is_sparse_mtp:
            sum_tokens = bs * max_seqlen_q
            attn_metadata.sparse_cu_seqlens_q = var["sparse_cu_seqlens_q"].gpu[
                : sum_tokens + 1
            ]
            attn_metadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:sum_tokens]
            self._token_to_seq_idxs_gpu[:sum_scheduled_tokens] = torch.arange(
                scheduled_bs, dtype=torch.int32, device=self.device
            ).repeat_interleave(max_seqlen_q)
            self._token_to_seq_idxs_gpu[sum_scheduled_tokens:sum_tokens] = 0
            attn_metadata.token_to_seq_idxs = self._token_to_seq_idxs_gpu[:sum_tokens]
        elif self.is_sparse:
            # Non-MTP sparse decode (single token per seq): the sparse KV is
            # packed at page_size=1, so last_page_len is 1 for every seq. Expose
            # the all-1s buffer so _forward_decode passes it to mla_decode_fwd
            # instead of the dense per-block kv_last_page_lens (which would make
            # the kernel over-read past the written sparse-index region).
            attn_metadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:bs]

        # Use bs (graph_bs) >= 2 instead of scheduled_bs >= 2 to avoid accuracy issue:
        if self.model_runner.config.enable_tbo_decode and bs >= 2:
            self._prepare_ubatch_decode(
                scheduled_bs,
                bs,
                max_seqlen_q,
                context_lens,
            )

        return attn_metadata, positions

    def _prepare_ubatch_decode(
        self,
        scheduled_bs: int,
        bs: int,
        max_seqlen_q: int,
        context_lens: np.ndarray,
    ):
        """
        Splits the full-batch data into per-ubatch .
        """
        var = self.model_runner.forward_vars
        N = self._NUM_TBO_UBATCHES
        half = bs // N

        ub_ranges = [
            (0, half),
            (half, bs),
        ]
        padded_bs_list = [half, bs - half]

        for ub_idx, ((req_start, req_end), padded_bs) in enumerate(
            zip(ub_ranges, padded_bs_list)
        ):
            p = f"ub{ub_idx}_"
            # How many real requests fall in this ubatch's range
            ub_real_reqs = max(0, min(scheduled_bs, req_end) - req_start)

            var[f"{p}context_lens"].np[:ub_real_reqs] = var["context_lens"].np[
                req_start : req_start + ub_real_reqs
            ]
            var[f"{p}context_lens"].np[ub_real_reqs:padded_bs] = 0

            var[f"{p}kv_last_page_lens"].np[:ub_real_reqs] = var[
                "kv_last_page_lens"
            ].np[req_start : req_start + ub_real_reqs]
            var[f"{p}kv_last_page_lens"].np[ub_real_reqs:padded_bs] = 0

            tok_start = req_start * max_seqlen_q
            ub_real_tokens = ub_real_reqs * max_seqlen_q
            padded_tok_count = padded_bs * max_seqlen_q
            var[f"{p}slot_mapping"].np[:ub_real_tokens] = var["slot_mapping"].np[
                tok_start : tok_start + ub_real_tokens
            ]
            var[f"{p}slot_mapping"].np[ub_real_tokens:padded_tok_count] = -1

            var[f"{p}block_tables"].np[:ub_real_reqs] = var["block_tables"].np[
                req_start : req_start + ub_real_reqs
            ]
            var[f"{p}block_tables"].np[ub_real_reqs:padded_bs] = 0

            full_kv_indptr = var["kv_indptr"].np
            base = full_kv_indptr[req_start]
            var[f"{p}kv_indptr"].np[0] = 0
            if ub_real_reqs > 0:
                var[f"{p}kv_indptr"].np[1 : ub_real_reqs + 1] = (
                    full_kv_indptr[req_start + 1 : req_start + ub_real_reqs + 1] - base
                )
            last_val = var[f"{p}kv_indptr"].np[ub_real_reqs] if ub_real_reqs > 0 else 0
            var[f"{p}kv_indptr"].np[ub_real_reqs + 1 : padded_bs + 1] = last_val

            if self.is_sparse:
                full_sparse = var["sparse_kv_indptr"].np
                sparse_base = full_sparse[req_start]
                var[f"{p}sparse_kv_indptr"].np[0] = 0
                if ub_real_reqs > 0:
                    var[f"{p}sparse_kv_indptr"].np[1 : ub_real_reqs + 1] = (
                        full_sparse[req_start + 1 : req_start + ub_real_reqs + 1]
                        - sparse_base
                    )
                sparse_last = (
                    var[f"{p}sparse_kv_indptr"].np[ub_real_reqs]
                    if ub_real_reqs > 0
                    else 0
                )
                var[f"{p}sparse_kv_indptr"].np[
                    ub_real_reqs + 1 : padded_bs + 1
                ] = sparse_last

            last_cu = ub_real_reqs * max_seqlen_q
            var[f"{p}cu_seqlens_q"].np[: ub_real_reqs + 1] = np.arange(
                0,
                (ub_real_reqs + 1) * max_seqlen_q,
                max_seqlen_q,
                dtype=np.int32,
            )
            var[f"{p}cu_seqlens_q"].np[ub_real_reqs + 1 : padded_bs + 1] = last_cu

            vars_used = [
                (f"{p}context_lens", padded_bs),
                (f"{p}kv_last_page_lens", padded_bs),
                (f"{p}slot_mapping", padded_tok_count),
                (f"{p}block_tables", padded_bs),
                (f"{p}kv_indptr", padded_bs + 1),
                (f"{p}cu_seqlens_q", padded_bs + 1),
            ]
            if self.is_sparse:
                vars_used.append((f"{p}sparse_kv_indptr", padded_bs + 1))

            for el, num in vars_used:
                var[el].copy_to_gpu(num)

            ub_max_seqlen_k = (
                int(context_lens[req_start : req_start + ub_real_reqs].max())
                if ub_real_reqs > 0
                else 0
            )
            kv_indices_generate_triton(
                var[f"{p}block_tables"].gpu[:padded_bs],
                var[f"{p}kv_indices"].gpu,
                var[f"{p}kv_indptr"].gpu[: padded_bs + 1],
                self.block_ratio,
                ub_max_seqlen_k,
            )

            self._set_ubatch_mla_buffers(padded_bs, max_seqlen_q, ub_idx)

    def _set_ubatch_mla_buffers(self, padded_bs, max_q_len, ubatch_idx):
        """Compute MLA work buffers for a per-ubatch forward_vars set."""
        p = f"ub{ubatch_idx}_"
        var = self.model_runner.forward_vars

        kv_indptr_for_mla = var[f"{p}kv_indptr"].gpu[: padded_bs + 1]
        kv_last_page_lens_for_mla = var[f"{p}kv_last_page_lens"].gpu[:padded_bs]
        if self.is_sparse:
            kv_indptr_for_mla = var[f"{p}sparse_kv_indptr"].gpu[: padded_bs + 1]
            # Sparse KV is packed per token at page_size=1 -> last_page_len is 1.
            # The dense per-block buffer would over-read past the sparse indices
            # (see set_mla_persistent_worker_buffers). The all-1s sparse buffer is
            # batch-independent, so the shared (non-ubatch) copy is safe here.
            kv_last_page_lens_for_mla = var["sparse_kv_last_page_lens"].gpu[:padded_bs]

        get_mla_metadata_v1(
            var[f"{p}cu_seqlens_q"].gpu[: padded_bs + 1],
            kv_indptr_for_mla,
            kv_last_page_lens_for_mla,
            self.padded_num_attention_heads,
            1,  # nhead_kv
            True,
            var[f"{p}work_meta_data"],
            var[f"{p}work_info_set"],
            var[f"{p}work_indptr"],
            var[f"{p}reduce_indptr"],
            var[f"{p}reduce_final_map"],
            var[f"{p}reduce_partial_map"],
            page_size=self.block_size,
            dtype_q=self.dtype_q,
            dtype_kv=self.dtype_kv,
            kv_granularity=max(self.block_size, 16),
            max_seqlen_qo=max_q_len,
            uni_seqlen_qo=max_q_len,
            fast_mode=1,
            max_split_per_batch=16,
        )

    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        var = self.model_runner.forward_vars
        # Self-consistent minimal KV metadata for capture: give every sequence
        # exactly 1 page (kv_indptr = [0,1,...,bs]) pointing at block 0, with a
        # 1-token last page. The split-KV stage1 asm kernel computes per batch
        # full_pages = page_count - (tail_len != 0). With model_runner's default
        # zeroed kv_indptr (page_count == 0) but kv_last_page_lens == 1, that
        # subtraction underflows (0 - 1 -> 0xFFFFFFFF), inflating the kv loop
        # count to ~2^32 so the kernel never exits and cudagraph capture hangs
        # (only hit when num_kv_splits > 1; passes==1 takes the bf16 fast path).
        # Replay overwrites these buffers with real values, so this only affects
        # capture-time loop termination, not inference correctness.
        if self.block_size > 1:
            kv_indptr_buf = var["kv_indptr"]
            kv_indptr_buf.np[: bs + 1] = np.arange(bs + 1, dtype=np.int32)
            kv_indptr_buf.copy_to_gpu(bs + 1)
            var["kv_indices"].gpu[:bs].zero_()
            var["kv_last_page_lens"].gpu[:bs].fill_(1)
        sparse_kv_indptr = var["sparse_kv_indptr"].gpu if self.is_sparse else None
        max_q_len = var["mtp_k"] + 1 if "mtp_k" in var else 1
        sum_tokens = bs * max_q_len
        is_sparse_mtp = self.is_sparse and max_q_len > 1
        if is_sparse_mtp:
            # Two sets: normal for dense layers, sparse_mtp for sparse layers
            ctx_mla_ps = self.set_mla_persistent_worker_buffers(bs, max_q_len)
            ctx_mla_ps_sparse = self._set_mla_persistent_worker_buffers_sparse_mtp(
                sum_tokens
            )
        else:
            ctx_mla_ps = self.set_mla_persistent_worker_buffers(bs, max_q_len)
            ctx_mla_ps_sparse = None
        attn_matadata = AttentionMetaData(
            slot_mapping=var["slot_mapping"].gpu[:sum_tokens],
            context_lens=var["context_lens"].gpu[:bs],
            block_tables=var["block_tables"].gpu[:bs],
            max_seqlen_q=max_q_len,
            cu_seqlens_q=var["cu_seqlens_q"].gpu[: bs + 1],
            kv_indptr=var["kv_indptr"].gpu[: bs + 1],
            kv_indices=var["kv_indices"].gpu,
            kv_last_page_lens=var["kv_last_page_lens"].gpu[:bs],
            sparse_kv_indptr=sparse_kv_indptr,
            **ctx_mla_ps,
        )
        attn_matadata.dtype_q = self.dtype_q
        if ctx_mla_ps_sparse is not None:
            for k, v in ctx_mla_ps_sparse.items():
                setattr(attn_matadata, k, v)
        if is_sparse_mtp:
            attn_matadata.sparse_cu_seqlens_q = var["sparse_cu_seqlens_q"].gpu[
                : sum_tokens + 1
            ]
            attn_matadata.sparse_kv_indptr = var["sparse_kv_indptr"].gpu[
                : sum_tokens + 1
            ]
            attn_matadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:sum_tokens]
            self._token_to_seq_idxs_gpu[:sum_tokens] = torch.arange(
                bs, dtype=torch.int32, device=self.device
            ).repeat_interleave(max_q_len)
            attn_matadata.token_to_seq_idxs = self._token_to_seq_idxs_gpu[:sum_tokens]
        elif self.is_sparse:
            # Non-MTP sparse decode capture: all-1s per-token last-page lens,
            # matching prepare_decode so _forward_decode reads the sparse buffer.
            attn_matadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:bs]
        positions = var["positions"].copy_to_gpu(sum_tokens)
        context = Context(
            positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs
        )
        return attn_matadata, context

    def build_ubatch_metadata(
        self,
        ubatch_idx: int,
        padded_bs: int,
    ) -> AttentionMetaData:
        """Create per-ubatch AttentionMetaData from pre-allocated forward_vars."""
        var = self.model_runner.forward_vars
        p = f"ub{ubatch_idx}_"
        max_q_len = var["mtp_k"] + 1 if "mtp_k" in var else 1

        # Compute MLA work buffers for this ubatch
        self._set_ubatch_mla_buffers(padded_bs, max_q_len, ubatch_idx)

        attn = AttentionMetaData(
            slot_mapping=var[f"{p}slot_mapping"].gpu[: padded_bs * max_q_len],
            context_lens=var[f"{p}context_lens"].gpu[:padded_bs],
            block_tables=var[f"{p}block_tables"].gpu[:padded_bs],
            max_seqlen_q=max_q_len,
            cu_seqlens_q=var[f"{p}cu_seqlens_q"].gpu[: padded_bs + 1],
            kv_indptr=var[f"{p}kv_indptr"].gpu[: padded_bs + 1],
            kv_indices=var[f"{p}kv_indices"].gpu,
            kv_last_page_lens=var[f"{p}kv_last_page_lens"].gpu[:padded_bs],
            sparse_kv_indptr=(
                var[f"{p}sparse_kv_indptr"].gpu[: padded_bs + 1]
                if self.is_sparse
                else None
            ),
            sparse_kv_last_page_lens=(
                var["sparse_kv_last_page_lens"].gpu[:padded_bs]
                if self.is_sparse
                else None
            ),
            work_meta_data=var[f"{p}work_meta_data"],
            work_info_set=var[f"{p}work_info_set"],
            work_indptr=var[f"{p}work_indptr"],
            reduce_indptr=var[f"{p}reduce_indptr"],
            reduce_final_map=var[f"{p}reduce_final_map"],
            reduce_partial_map=var[f"{p}reduce_partial_map"],
        )
        attn.dtype_q = self.dtype_q
        return attn

    def build_ubatch_prefill_metadata(
        self,
        attn_metadata: AttentionMetaData,
        ub_slice,
        padded_bs: int,
        ubatch_idx: int = 0,
    ) -> AttentionMetaData:
        """
        Split prefill AttentionMetaData for MLA.
        """
        del ubatch_idx  # MLA has no per-ubatch pooled buffers to disambiguate
        from atom.utils.tbo.ubatch_splitting import split_attn_metadata

        ub_attn = split_attn_metadata(attn_metadata, ub_slice, padded_bs)

        ts = ub_slice.token_slice
        rs = ub_slice.request_slice
        req_start = rs.start

        if (
            hasattr(attn_metadata, "cu_seqlen_ks")
            and attn_metadata.cu_seqlen_ks is not None
        ):
            ub_attn.cu_seqlen_ks = attn_metadata.cu_seqlen_ks[ts]

        if (
            hasattr(attn_metadata, "cu_seqlen_ke")
            and attn_metadata.cu_seqlen_ke is not None
        ):
            ub_attn.cu_seqlen_ke = attn_metadata.cu_seqlen_ke[ts]

        if (
            hasattr(attn_metadata, "sparse_cu_seqlens_q")
            and attn_metadata.sparse_cu_seqlens_q is not None
        ):
            base = attn_metadata.sparse_cu_seqlens_q[ts.start]
            ub_attn.sparse_cu_seqlens_q = (
                attn_metadata.sparse_cu_seqlens_q[ts.start : ts.stop + 1] - base
            )

        if (
            hasattr(attn_metadata, "token_to_seq_idxs")
            and attn_metadata.token_to_seq_idxs is not None
        ):
            ub_attn.token_to_seq_idxs = attn_metadata.token_to_seq_idxs[ts] - req_start

        total_tokens = (
            attn_metadata.slot_mapping.shape[0]
            if attn_metadata.slot_mapping is not None
            else 0
        )
        # Sparse prefill: sparse_kv_last_page_lens is per query TOKEN, so slice it
        # by the token slice. (The dense kv_last_page_lens is per-seq and is sliced
        # by request in split_attn_metadata.)
        if (
            attn_metadata.sparse_kv_last_page_lens is not None
            and attn_metadata.sparse_kv_last_page_lens.shape[0] == total_tokens
        ):
            ub_attn.sparse_kv_last_page_lens = attn_metadata.sparse_kv_last_page_lens[
                ts
            ]

        if (
            attn_metadata.sparse_kv_indptr is not None
            and attn_metadata.sparse_kv_indptr.shape[0] == total_tokens + 1
        ):
            base = attn_metadata.sparse_kv_indptr[ts.start]
            ub_attn.sparse_kv_indptr = (
                attn_metadata.sparse_kv_indptr[ts.start : ts.stop + 1] - base
            )

        # ── Token-midpoint split straddle handling ──────────────────────
        self._attach_tbo_token_split_straddle_prefix(attn_metadata, ub_attn, ub_slice)

        return ub_attn

    # ================================================================
    # TBO PREFILL TOKEN-SPLIT (ATOM_TBO_PREFILL_TOKEN_SPLIT) — MLA path
    # ================================================================

    def _attach_tbo_token_split_straddle_prefix(self, attn_metadata, ub_attn, ub_slice):
        """If this ubatch's first request is cut from a previous ubatch, attach
        the prior portion's KV-cache slots as chunked cached prefixes so dense
        MLA attention can see it (token-midpoint split correctness). No-op when
        not straddling."""
        from atom.utils.tbo import compute_straddle_split_info

        if self.k_chunk_workspace is None:
            return  # chunked workspace disabled → cannot serve a prefix

        cu_np = self.model_runner.forward_vars["cu_seqlens_q"].np
        info = compute_straddle_split_info(cu_np, ub_slice)
        if not info.is_straddling:
            return  # not straddling — first request starts at the slice edge

        ts = ub_slice.token_slice
        req_global_start = info.req_global_start
        prefix_len = info.prefix_len
        ub_num_reqs = info.ub_num_reqs

        slot_mapping = attn_metadata.slot_mapping
        if slot_mapping is None:
            return
        # Physical KV-cache slots of the straddled request's first half
        # (written by the previous ubatch). MLA block_size==1, so slot ids are
        # the gather kv_indices directly.
        prefix_slots = slot_mapping[req_global_start : ts.start].to(torch.int32)

        device = prefix_slots.device
        # Only the first (straddled) request has a cached prefix; all other
        # requests in this ubatch contribute 0 cached tokens. Chunk the prefix
        # along the token axis so each chunk fits the k/v workspace
        # (attn_prefill_chunk_size), mirroring _build_mla_chunk_meta.
        chunk_size = self.attn_prefill_chunk_size
        num_chunks = max(1, cdiv(prefix_len, chunk_size))
        kv_indptr_list = []
        kv_indices_list = []
        total_tokens_list = []
        max_seqlen_k_list = []
        for c in range(num_chunks):
            c_lo = c * chunk_size
            c_hi = min(c_lo + chunk_size, prefix_len)
            c_len = c_hi - c_lo
            cu = np.full(ub_num_reqs + 1, c_len, dtype=np.int32)
            cu[0] = 0
            kv_indptr_list.append(
                torch.from_numpy(cu).pin_memory().to(device, non_blocking=True)
            )
            kv_indices_list.append(prefix_slots[c_lo:c_hi])
            total_tokens_list.append(c_len)
            max_seqlen_k_list.append(c_len)

        ub_attn.has_cached = True
        # total_kv = this ubatch's new tokens + the straddle prefix it now reads
        # from cache. Only referenced by the chunked-prefill debug log, but keep
        # it consistent to avoid a None in "%d" formatting.
        ub_attn.total_kv = int(info.ub_num_tokens + prefix_len)
        ub_attn.mla_chunk_meta = MLAChunkContextMetadata(
            kv_indptr=kv_indptr_list,
            kv_indices=kv_indices_list,
            cu_seqlens_k=kv_indptr_list,
            total_tokens=total_tokens_list,
            max_seqlen_k=max_seqlen_k_list,
            num_chunks=num_chunks,
            k_workspace=self.k_chunk_workspace,
            v_workspace=self.v_chunk_workspace,
        )
