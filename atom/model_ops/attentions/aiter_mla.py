# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import inspect
import logging
from dataclasses import dataclass

import numpy as np
import torch
import triton
from aiter import (
    decode_update_mla_metadata_v1,
    dtypes,
    get_mla_metadata_info_v1,
    get_mla_metadata_v1,
)

from atom.distributed.dcp_utils import (
    dcp_persistent_supported,
    get_dcp_rank,
    get_dcp_world_size,
)
from atom.distributed.pcp_utils import (
    get_pcp_world_size,
    pcp_is_enabled,
    pcp_pad_dense,
    pcp_pad_len,
    pcp_round_robin_query_indices,
)
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mla import (
    _MLA_MIN_HEADS,
    _MLA_SPLIT_BUDGET_AUTO,
    MLAAttention,
    mla_dcp_decode_is_persistent,
    mla_dcp_kernel_num_heads,
)
from atom.model_ops.glm5_next.geometry import (
    effective_kpool_size,
    topk_output_width,
)
from atom.utils import CpuGpuBuffer, envs, upload_numpy
from atom.utils.block_convert import (
    kv_indices_generate_triton,
    mtp_prepare_decode_mla_kernel,
)
from atom.utils.forward_context import AttentionMetaData, Context

from .backends import AttentionBackend, CommonAttentionBuilder
from .pool_layout.sub_pool_spec import SubPoolSpec, page_pool
from .token_layout.decode import decode_positions
from .token_layout.slots import slot_mapping

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


def mla_kv_entry_dim(hf_config) -> int:
    """Width of one MLA KV cache entry.

    Normally ``kv_lora_rank + qk_rope_head_dim``. A NoPE model (GLM-5.3-Flash,
    ``qk_rope_head_dim == 0``) materializes the rope block at a padded width and
    holds it at zero so the standard 576-wide MLA kernels apply unchanged; it
    declares that padded width as ``mla_kv_entry_dim``. Sizing the cache from
    the raw config instead would allocate 512-wide rows under a 576-wide write.
    """
    declared = getattr(hf_config, "mla_kv_entry_dim", None)
    if declared:
        return int(declared)
    return hf_config.kv_lora_rank + hf_config.qk_rope_head_dim


def mla_qk_head_dim(hf_config) -> int:
    """Per-head q/k width the MLA kernels and their workspaces are built for.

    The rope block's width has to come from `mla_kv_entry_dim`, not from the
    raw config. A NoPE model leaves ``qk_rope_head_dim`` at its true 0 so the
    INDEXER stays NoPE, and widens the block to a zero pad on the MLA side
    only, so the raw sum understates the MLA width by exactly the pad.

    Getting this wrong is not a size warning, it is a compile error one kernel
    deep: `gather_kv_b_proj` takes the rope width from the destination buffer
    (``qk_nope_pe_dim = k_prefix.shape[-1]``) and the nope width from the
    kv_b_proj weight, so a buffer sized 256 against a 256-wide nope half makes
    ``KV_PeDim = 0`` and Triton rejects the resulting ``tl.arange(0, 0)``.
    """
    rope_dim = mla_kv_entry_dim(hf_config) - hf_config.kv_lora_rank
    return hf_config.qk_nope_head_dim + rope_dim


def aligned_index_cache_dim(hf_config) -> int:
    """Indexer key plus fp32 scale, padded to a 16-byte row."""
    index_dim = hf_config.index_head_dim + 4
    return ((index_dim + 15) // 16) * 16


def _pad_prefill_mla_draft_tail(
    kv_indptr: torch.Tensor,
    kv_last_page_lens: np.ndarray,
    block_tables: np.ndarray,
    scheduled_bs: int,
    running_bs: int,
) -> None:
    """Give widened Eagle rows an empty, fully initialized MLA KV range."""
    assert 0 <= scheduled_bs <= running_bs
    if scheduled_bs == running_bs:
        return
    kv_indptr[scheduled_bs + 1 : running_bs + 1] = kv_indptr[scheduled_bs]
    kv_last_page_lens[scheduled_bs:running_bs] = 0
    block_tables[scheduled_bs:running_bs] = 0


def _global_index_cache_layer_ids(
    indexer_types,
    num_hidden_layers: int,
    num_draft_layers: int,
    layer_types=None,
) -> tuple[int, ...]:
    """Return global layers that own an index-key cache slice.

    GLM-5.2 ``shared`` layers reuse a preceding full layer's temporary top-k
    positions and do not construct an indexer, so their index-key cache slices
    are dead. Other sparse MLA models have no ``indexer_types`` schedule and
    retain the existing one-slice-per-layer layout.
    """
    target_layer_ids = range(num_hidden_layers)
    if layer_types is not None:
        # Hybrid models (GLM-5.3-Flash) interleave linear-attention layers that
        # have no MLA and therefore no indexer; a slice for them would be dead
        # allocation. `indexer_types` does not encode this -- GLM-5.3 marks
        # every layer "full" -- so the attention layout is the authority.
        target_layer_ids = [
            layer_id
            for layer_id in target_layer_ids
            if layer_id >= len(layer_types)
            or layer_types[layer_id] != "linear_attention"
        ]
    if indexer_types is not None:
        target_layer_ids = (
            layer_id
            for layer_id in target_layer_ids
            # MTP layers are not included in indexer_types. Only the GLM
            # "shared" value means no indexer module/cache owner; DeepSeek's
            # index_topk_pattern "S" has different semantics and keeps a cache.
            if layer_id >= len(indexer_types) or indexer_types[layer_id] != "shared"
        )
    return tuple(target_layer_ids) + tuple(
        range(num_hidden_layers, num_hidden_layers + num_draft_layers)
    )


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

    kv_indptr: list[torch.Tensor]
    kv_indices: list[torch.Tensor]
    cu_seqlens_k: list[torch.Tensor]
    total_tokens: list[int]
    max_seqlen_k: list[int]
    num_chunks: int
    k_workspace: torch.Tensor
    v_workspace: torch.Tensor
    # Block-granular CSR per chunk for the shuffled-KV gather (block_size=64
    # blocks instead of token slots). None for the plain token-slot layout.
    shuffle_kv_block_indptr: list[torch.Tensor] | None = None
    shuffle_kv_block_indices: list[torch.Tensor] | None = None
    # --- DCP (Decode Context Parallel) prefill fields ---
    # Present only when dcp_world_size > 1. The cached context KV is sharded
    # (interleaved) across DCP ranks, so per chunk each rank gathers its local
    # compressed KV via `local_slot_ids`, AllGathers, and `ag_row_indices`
    # rebuilds the per-sequence contiguous layout. `cu_seqlens_k` above then
    # holds the GLOBAL per-seq chunk lengths (post-reorg) for flash attention.
    is_dcp: bool = False
    # Per chunk: absolute slot ids of this rank's local cached tokens, laid out
    # per-seq contiguous and padded so every rank's AllGather block has
    # identical `seq_tot` tokens.
    local_slot_ids: list[torch.Tensor] | None = None
    # Per chunk: the reorg as a row map into the `[seq_tot * dcp, d]` AllGather
    # buffer, per-seq contiguous and rank-major within a seq (length ==
    # `total_tokens[c]`). Doubles as the fused gather's `kv_indices`, which is
    # what lets the reorg, dequant, kv_b_proj and k_pe concat collapse into one
    # kernel -- see `dcp_reorg_row_indices`.
    ag_row_indices: list[torch.Tensor] | None = None
    # Per chunk: number of local tokens per rank in the AllGather block.
    seq_tot: list[int] | None = None


def cdiv(a, b):
    return (a + b - 1) // b


class AiterMLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA"

    @staticmethod
    def get_builder_cls() -> type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["MLAAttention"]:
        return MLAAttention


class AiterMLAMetadataBuilder(CommonAttentionBuilder):
    # EagleProposer folds the per-draft-step position/context bump into
    # prepare_mtp_decode's fused kernel when this is set (matches the MHA
    # backend). The fused kernel handles both sparse and dense MLA.
    fuse_mtp_decode_position_update = True

    def _global_num_draft_layers(self) -> int:
        """Return draft layers in the target MLA pool across all PP stages."""
        runner = self.model_runner
        spec_config = getattr(runner.config, "speculative_config", None)
        # Eagle3 draft layers are owned by eagle3_draft_builder and use a
        # separate KV pool. Only MTP-style draft layers share the target MLA
        # pool and therefore belong in this pool's global KV/index-cache layout.
        if spec_config is None or hasattr(runner, "eagle3_draft_builder"):
            return 0
        draft_hf_config = spec_config.draft_model_hf_config
        # Mirror ModelRunner._get_total_num_layers(), which is authoritative for
        # the rows actually allocated in this target MLA pool.
        return getattr(draft_hf_config, "num_nextn_predict_layers", 1)

    def _index_cache_layout(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return (local, global) global-layer IDs owning index cache slices."""
        from aiter.dist.parallel_state import get_pp_group

        from atom.models.utils import get_pp_indices

        runner = self.model_runner
        hf_config = runner.config.hf_config
        num_hidden_layers = hf_config.num_hidden_layers
        pp_group = get_pp_group()
        start_layer, end_layer = get_pp_indices(
            num_hidden_layers, pp_group.rank_in_group, pp_group.world_size
        )
        num_local_target_layers = end_layer - start_layer
        num_local_draft_layers = (
            runner._get_total_num_layers() - num_local_target_layers
        )
        global_layer_ids = _global_index_cache_layer_ids(
            getattr(hf_config, "indexer_types", None),
            num_hidden_layers,
            self._global_num_draft_layers(),
            getattr(hf_config, "layer_types", None),
        )
        local_layer_ids = tuple(
            layer_id
            for layer_id in global_layer_ids
            if start_layer <= layer_id < end_layer
            # start_layer and end_layer DONOT contain draft layers
            or (
                num_hidden_layers
                <= layer_id
                < num_hidden_layers + num_local_draft_layers
            )
        )
        return local_layer_ids, global_layer_ids

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
        # GLM-5.3's pooled indexer selects `index_topk // index_kpool` POOLS --
        # index_topk tokens -- and then appends the trailing incomplete pool,
        # which is never scored (`index_kpool_always_select_tail`). So a row can
        # carry up to `index_topk + index_kpool - 1` entries, not index_topk.
        # The per-row width also has to divide the conversion kernels' BLOCK_N,
        # hence the round up to 128; the extra columns are never read, because
        # `sparse_kv_indptr` caps each row at its true count.
        #
        # `index_topk` itself stays the sparse-vs-dense THRESHOLD: at or below
        # it every pool is selected and the dense path is exactly equivalent.
        configured_kpool = (
            int(getattr(hf_config, "index_kpool", 1) or 1) if self.is_sparse else 1
        )
        self.index_kpool = effective_kpool_size(configured_kpool)
        self.index_topk_out = topk_output_width(self.index_topk, configured_kpool)
        self.dtype_kv = dtypes.d_dtypes[config.kv_cache_dtype]
        self.dtype_q = self.dtype_kv

        self.dcp_world_size = get_dcp_world_size()
        self.dcp_rank = get_dcp_rank()
        self._publishes_dcp_local_lens = self.is_sparse and self.dcp_world_size > 1
        self._tbo_full_running_bs = 0

        # DCP decode all-gathers Q on the head dim, so the head count reaching
        # mla_decode_fwd (and thus the persistent decode metadata) is the padded
        # gathered width, not the per-rank one. Pad it the same way the module
        # does (mla_dcp_decode_is_persistent picks the width set), so these
        # descriptors always describe the kernel that will actually run.
        # Only gfx950 runs DCP in persistent mode (gfx942 lacks the lse persistent
        # kernel and stays non-persistent, where this metadata is unused); scale
        # by dcp only there so gfx942 keeps the original per-rank head sizing.
        dcp_persistent = dcp_persistent_supported()
        self.sparse_dcp_metadata_rebuild = (
            self.is_sparse
            and self.dcp_world_size > 1
            and dcp_persistent
            and self.block_size == 1
        )
        if self.dcp_world_size > 1 and dcp_persistent:
            self.persistent_num_heads = mla_dcp_kernel_num_heads(
                self.num_attention_heads,
                self.dcp_world_size,
                kv_cache_dtype=config.kv_cache_dtype,
                persistent=mla_dcp_decode_is_persistent(
                    self.is_sparse,
                    self.dcp_world_size,
                    dcp_persistent,
                    sparse_metadata_rebuild=self.sparse_dcp_metadata_rebuild,
                ),
            )
        else:
            self.persistent_num_heads = self.padded_num_attention_heads

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
            self.persistent_num_heads,
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
            # Global (un-sharded) per-request KV indptr for round-robin CP: cumsum
            # of the GLOBAL context_lens (token-level, page_size=1). Only filled
            # when dcp_world_size > 1; consumed by the cprr kernel via
            # mla_decode_fwd(g_kv_indptr=...) to apply the global-position causal
            # mask for MTP (max_q_len > 1).
            "g_kv_indptr": CpuGpuBuffer(self.max_bs + 1, **i32_kwargs),
            "kv_indices": CpuGpuBuffer(
                self.max_bs * self.max_num_blocks_per_seq,
                **i32_kwargs,
            ),
            "kv_last_page_lens": CpuGpuBuffer(self.max_bs, **i32_kwargs),
        }
        if self._publishes_dcp_local_lens:
            # Layer-invariant sparse-DSA indexer metadata: one row per query
            # token, derived once per step and reused by every full layer.
            mla_metadata["dcp_local_context_lens"] = CpuGpuBuffer(
                self.max_bs * max_seqlen_qo, **i32_kwargs
            )
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
                self.max_num_batched_tokens * self.index_topk_out,
                dtype=torch.int32,
                device=self.device,
            )
            # DCP sparse decode compacts each rank's owned top-k slots to the
            # front (no -1 holes), so the per-request region length becomes data-
            # AND layer-dependent.
            self._dcp_sparse_kv_indptr_gpu = torch.zeros(
                self.max_num_batched_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            )
            self._dcp_owned_counts_gpu = torch.zeros(
                self.max_num_batched_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            # One block-table row per query token; only MTP verify needs a
            # copy. Built once per step, not in the indexer, where every
            # full-index layer would allocate one into the CUDAGraph pool.
            self._dcp_token_block_tables_gpu = (
                torch.empty(
                    self.max_bs * max_seqlen_qo,
                    self.block_table_cols,
                    **i32_kwargs,
                )
                if self.dcp_world_size > 1 and max_seqlen_qo > 1
                else None
            )
            sparse_prefill_num_heads = (
                self.persistent_num_heads
                if self.sparse_dcp_metadata_rebuild
                else self.padded_num_attention_heads
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
                sparse_prefill_num_heads,
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
            # Same widening as sparse prefill: the DCP rebuild regenerates these
            # for the gathered query width, not for a single rank's heads.
            sparse_mtp_num_heads = (
                self.persistent_num_heads
                if self.sparse_dcp_metadata_rebuild
                else self.padded_num_attention_heads
            )
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
                sparse_mtp_num_heads,
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
        self.k_chunk_workspace: torch.Tensor | None = None
        self.v_chunk_workspace: torch.Tensor | None = None
        if self.attn_prefill_chunk_size > 0:
            # Not the raw config sum: this buffer is the gather's destination
            # and its width sets KV_PeDim. See `mla_qk_head_dim`.
            qk_head_dim = mla_qk_head_dim(hf_config)
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
                impl = getattr(module, "impl", None)
                # DCP compact buffers ride along with the indices buffer: the
                # indexer writes all three, the attention impl reads the indices
                # and the offsets in the same layer.
                for tgt in (module, impl):
                    if tgt is None or not hasattr(tgt, "sparse_kv_indices_buffer"):
                        continue
                    tgt.sparse_kv_indices_buffer = self._sparse_kv_indices_gpu
                    tgt.dcp_sparse_kv_indptr_buffer = self._dcp_sparse_kv_indptr_gpu
                    tgt.dcp_owned_counts_buffer = self._dcp_owned_counts_gpu
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
            # Per-ubatch global (un-sharded) kv_indptr for round-robin CP (see the
            # shared "g_kv_indptr" buffer). Filled in _build_ubatch when dcp>1.
            var[f"{p}g_kv_indptr"] = CpuGpuBuffer(ub_max_bs + 1, **i32_kwargs)
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
                ub_max_bs, self.block_table_cols, **i32_kwargs
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
            "max_split_per_batch": _MLA_SPLIT_BUDGET_AUTO,
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
        is_cp_round_robin: bool = False,
    ):
        assert num_reject_tokens is None or num_reject_tokens.shape[0] >= bs, (
            f"num_reject_tokens covers {num_reject_tokens.shape[0]} sequences "
            f"but this asks for {bs}; the update kernel loads one per work item "
            f"below `cu_num`, unconditionally"
        )
        split_params = {
            "kv_granularity": max(self.block_size, 16),
            "max_seqlen_qo": max_q_len,
            "uni_seqlen_qo": max_q_len,
            "fast_mode": 1,
            "max_split_per_batch": _MLA_SPLIT_BUDGET_AUTO,
        }
        # round-robin CP only lands on the full-build path: decode_update_mla_
        # metadata_v1 has no is_cp_round_robin arg and collapses qlen>1 to 1.
        assert not (only_update and is_cp_round_robin), (
            "is_cp_round_robin requires the full get_mla_metadata_v1 build "
            "(only_update path does not support round-robin CP / qlen>1)"
        )
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
                self.persistent_num_heads,
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
                self.persistent_num_heads,
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
                is_cp_round_robin=is_cp_round_robin,
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
        running_bs = int(positions.shape[-1])  # rows; see the base contract
        del last_token_indices  # MLA positions are already per-seq (1 per token)
        var = self.model_runner.forward_vars
        kv_indptr = var["kv_indptr"].gpu[: running_bs + 1]
        cu_seqlens_q = var["cu_seqlens_q"].gpu[: running_bs + 1]
        if self.is_sparse:
            sparse_kv_indptr = var["sparse_kv_indptr"].gpu[: running_bs + 1]
        else:
            assert self.block_size == 1
            sparse_kv_indptr = None

        update_positions = positions_out is not None
        context_lens = (
            var["context_lens"].gpu[:running_bs] if update_context_lens else None
        )

        mtp_prepare_decode_mla_kernel[(1,)](
            kv_indptr,
            cu_seqlens_q,
            sparse_kv_indptr if self.is_sparse else kv_indptr,
            positions_out if update_positions else kv_indptr,
            context_lens if update_context_lens else kv_indptr,
            running_bs,
            self.index_topk if self.is_sparse else 0,
            positions_out.stride(0) if update_positions else 1,
            IS_SPARSE=self.is_sparse,
            UPDATE_POSITIONS=update_positions,
            UPDATE_CONTEXT_LENS=update_context_lens,
            BLOCK=self._mtp_fuse_block,
        )

        # DCP + MTP draft decode: each prepare_mtp_decode call advances every
        # sequence's KV by exactly 1 token (mtp_k controls how many times this
        # runs, not tokens-per-call), so rebuild for LOCAL qlen=1 regardless of
        # the max_seqlen_q param above (which the non-DCP path may pass as
        # i0_max_seqlen_q for its incremental-update fast path — DCP always
        # needs a full local-shard rebuild, no incremental variant exists).
        dcp_local_rebuild = self.dcp_world_size > 1 and not self.is_sparse
        if dcp_local_rebuild or self._publishes_dcp_local_lens:
            W = self.dcp_world_size
            r = self.dcp_rank
            ctx_g = var["context_lens"].gpu[:running_bs].to(torch.int64)
            base = ctx_g // W
            remainder = (ctx_g - base * W - r).clamp_(0, 1)
            local_ctx = (base + remainder).to(torch.int32)  # local KV tokens/blocks
        if self._publishes_dcp_local_lens:
            # A draft step runs one query per sequence, so this is already the
            # per-token map. The verify step's copy is stale after
            # context_lens += 1, and a short length drops the newest tokens.
            var["dcp_local_context_lens"].gpu[:running_bs] = local_ctx
        if dcp_local_rebuild:
            assert self.block_size == 1
            kv_indptr[0] = 0
            kv_indptr[1 : running_bs + 1] = torch.cumsum(
                local_ctx, dim=0, dtype=torch.int32
            )
            var["kv_last_page_lens"].gpu[:running_bs] = (local_ctx > 0).to(
                var["kv_last_page_lens"].gpu.dtype
            )
            # Host upper bound for the index generator's loop (safe overestimate,
            # avoids a device->host sync); kv_indptr caps the real per-seq count.
            local_max_k = max_seqlen_k // W + 1

        kv_indices_generate_triton(
            var["block_tables"].gpu[:running_bs],
            var["kv_indices"].gpu,
            kv_indptr,
            self.block_ratio,
            local_max_k if dcp_local_rebuild else max_seqlen_k,
        )
        if dcp_local_rebuild:
            # qlen==1 local decode: full build (not cprr; is_cp_round_robin=False),
            # over the just-rebuilt LOCAL kv_indptr / kv_last_page_lens.
            return self.set_mla_persistent_worker_buffers(
                running_bs, 1, only_update=False, num_reject_tokens=None
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
                running_bs,
                1,
                only_update=False,
                num_reject_tokens=None,
                sparse_decode=True,
            )
            result["sparse_kv_indptr"] = sparse_kv_indptr
        else:
            # `bs`, not `running_bs`, and paired with `num_reject_tokens`: this count
            # becomes `cu_num`, and the update kernel loads
            # `num_reject_tokens[batch_id]` for every work item below it, before
            # any length test. That tensor is the sampler's, so it is `bs` long.
            # The two branches above pass no counts and so take the row count.
            result = self.set_mla_persistent_worker_buffers(
                bs, max_seqlen_q, only_update, num_reject_tokens
            )
        return result

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        """One paged KV pool. Per-block bytes = a single 576-dim packed
        tensor per layer (k_c + k_pe; V is absorbed into latent compression —
        no separate V cache or kv_scale).

        DeepSeek-V3.2 sparse variants add an indexer cache contribution
        for every indexer-owning layer, including draft/MTP layers. GLM-5.2
        shared layers do not own an indexer and are excluded.
        """
        runner = self.model_runner
        config = runner.config
        hf_config = config.hf_config
        total_num_layers = runner._get_total_num_layers()
        kv_dtype_size = dtypes.d_dtypes[config.kv_cache_dtype].itemsize

        block_bytes = total_num_layers * runner.block_size * 576 * kv_dtype_size
        if runner.is_deepseek_v32:
            aligned_index_dim = aligned_index_cache_dim(hf_config)
            index_cache_layer_ids, _ = self._index_cache_layout()
            block_bytes += (
                len(index_cache_layer_ids)
                * runner.block_size
                * aligned_index_dim
                * dtypes.fp8.itemsize
            )
        return [page_pool(block_bytes)]

    def _index_cache_block_bytes(self, index_cache_layer: torch.Tensor) -> int:
        """Bytes one SCHEDULER block owns in one layer of the index cache.

        Here dim 0 counts PHYSICAL blocks and there is one row per token, so a
        scheduler block spans `block_ratio` of them. A builder whose index
        cache is indexed by scheduler block, or whose indexer compresses
        several tokens into one row, overrides this -- applying `block_ratio`
        to such a cache would over-report by exactly the compression ratio.
        """
        t = index_cache_layer
        return t.stride(0) * t.element_size() * self.block_ratio

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict:
        """MLA: single 576-dim paged tensor per layer (k_c + k_pe packed,
        no separate V cache — MLA absorbs V into the latent compression).

        DeepSeek-V3.2 sparse variants additionally allocate an `index_cache`
        for indexer-owning layers; the aligned dimension and compact layer map
        are returned so build_kv_cache_tensor can bind the correct slice.
        """
        runner = self.model_runner
        config = runner.config
        hf_config = config.hf_config
        total_num_layers = runner._get_total_num_layers()
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
            aligned = aligned_index_cache_dim(hf_config)
            index_cache_layer_ids, _ = self._index_cache_layout()
            out["aligned_index_dim"] = aligned
            out["index_cache_layer_ids"] = index_cache_layer_ids
            out["index_cache_layer_map"] = {
                global_layer_id: compact_layer_id
                for compact_layer_id, global_layer_id in enumerate(
                    index_cache_layer_ids
                )
            }
            out["index_cache"] = torch.zeros(
                len(index_cache_layer_ids),
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
        num_slots = runner.num_physical_kvcache_blocks * runner.physical_block_size
        kv_cache = runner.kv_cache[layer_id].view(num_slots, 1, 576)
        module.max_model_len = runner.config.max_model_len
        index_cache = None
        if runner.is_deepseek_v32 and module.indexer is not None:
            # `layer_id` is a PP-local cache-row counter, while the compact map
            # is keyed by global model layer IDs. On a non-first PP stage they
            # differ (for example local 0 may be global 39), so use layer_num
            # to avoid binding this indexer to another stage's compact row.
            global_layer_id = getattr(module, "layer_num", None)
            if global_layer_id not in runner.index_cache_layer_map:
                raise RuntimeError(
                    "Sparse MLA indexer layer is missing from the compact index "
                    f"cache layout: layer_num={global_layer_id}"
                )
            index_cache_layer_id = runner.index_cache_layer_map[global_layer_id]
            index_cache = runner.index_cache[index_cache_layer_id]
            # Use aligned dimension to avoid memory copy in torch inductor
            module.indexer.k_cache.kv_cache[0] = index_cache.view(
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
            index_cache=index_cache if runner.is_deepseek_v32 else None,
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
                bpb = self._index_cache_block_bytes(t)
                block_regions.append(
                    KVTransferRegion(
                        base_addr=t.data_ptr(),
                        total_bytes=t.numel() * t.element_size(),
                        unit_bytes=bpb,
                    )
                )

        block_region_consumer_indices = None
        index_cache_layer_ids = getattr(runner, "index_cache_layer_ids", ())
        if index_cache_layer_ids:
            local_index_layer_ids, global_index_layer_ids = self._index_cache_layout()
            if tuple(index_cache_layer_ids) != local_index_layer_ids:
                raise RuntimeError(
                    "Allocated and transfer-time index cache layouts disagree"
                )
            num_hidden_layers = runner.config.hf_config.num_hidden_layers
            # A hybrid model (GLM-5.3-Flash) allocates an MLA row only for its
            # full-attention layers, so the KV rows are NOT one-per-layer.
            # `full_attention_layers` is set by the GDN state mixin for those
            # models; its absence means the dense one-row-per-layer layout.
            # `or None`: the GDN state mixin derives this from
            # `linear_attn_config["full_attn_layers"]` and leaves it EMPTY when the
            # config omits that key, and an empty list is not None -- it would take
            # the hybrid branch with zero MLA layers, so every index-cache region
            # gets numbered on top of the KV regions instead of after them and a
            # P/D transfer writes index bytes into KV rows. The length check below
            # cannot catch it because both sides derive from the same empty list.
            hybrid_mla_layers = getattr(runner, "full_attention_layers", None) or None
            num_global_draft_layers = sum(
                layer_id >= num_hidden_layers for layer_id in global_index_layer_ids
            )
            # Index-cache regions are numbered after the KV regions, so this
            # offset must count the KV rows that actually exist -- the MLA
            # layers for a hybrid, every layer otherwise.
            num_global_mla_layers = (
                len(hybrid_mla_layers)
                if hybrid_mla_layers is not None
                else num_hidden_layers
            )
            num_global_kv_layers = num_global_mla_layers + num_global_draft_layers
            # Unlike index_cache_layer_map (PP-local allocated rows), this map
            # numbers compact index-cache rows in the consumer's global region
            # list. It is used only to translate local P/D regions.
            global_compact_index_slot_by_layer = {
                global_layer_id: compact_layer_id
                for compact_layer_id, global_layer_id in enumerate(
                    global_index_layer_ids
                )
            }
            from aiter.dist.parallel_state import get_pp_group

            from atom.models.utils import get_pp_indices

            pp_group = get_pp_group()
            start_layer, end_layer = get_pp_indices(
                num_hidden_layers, pp_group.rank_in_group, pp_group.world_size
            )
            if hybrid_mla_layers is not None:
                local_target_layer_ids = tuple(
                    layer_id
                    for layer_id in hybrid_mla_layers
                    if start_layer <= layer_id < end_layer
                )
                compact_mla_slot_by_layer = {
                    global_layer_id: compact_layer_id
                    for compact_layer_id, global_layer_id in enumerate(
                        hybrid_mla_layers
                    )
                }
                local_target_consumer_indices = tuple(
                    compact_mla_slot_by_layer[layer_id]
                    for layer_id in local_target_layer_ids
                )
            else:
                local_target_layer_ids = tuple(range(start_layer, end_layer))
                local_target_consumer_indices = local_target_layer_ids
            num_local_target_layers = len(local_target_layer_ids)
            num_local_draft_layers = num_layers - num_local_target_layers
            local_kv_consumer_indices = local_target_consumer_indices + tuple(
                range(
                    num_global_mla_layers,
                    num_global_mla_layers + num_local_draft_layers,
                )
            )
            if len(local_kv_consumer_indices) != num_layers:
                raise RuntimeError(
                    "MLA KV cache layer count does not match the PP-local layout: "
                    f"cache={num_layers}, layout={len(local_kv_consumer_indices)}, "
                    f"target={num_local_target_layers}, "
                    f"draft={num_local_draft_layers}"
                )
            block_region_consumer_indices = list(local_kv_consumer_indices) + [
                num_global_kv_layers + global_compact_index_slot_by_layer[layer_id]
                for layer_id in local_index_layer_ids
            ]

        return KVTransferTensors(
            block_regions=block_regions,
            slot_regions=[],
            num_blocks=runner.config.num_kvcache_blocks,
            block_region_consumer_indices=block_region_consumer_indices,
        )

    def _build_dcp_indexer_prefill_meta(self, attn_metadata, bs: int, counts, var):
        """Metadata for the DCP sparse-prefill indexer gather.

        The indexer scores against the WHOLE sequence, but under DCP each rank's
        index cache holds only the round-robin 1/W shard. The fix mirrors the
        decode side: gather the local shard with *local* cu_seqlens, all-gather
        it, then de-interleave back to global order.

        Two products, both rank-independent in shape so the all-gather is a plain
        concat:

        ``dcp_indexer_local_cu_seqlens``
            cumsum of ``Lpad[b] = ceil(g_b / (S*W)) * S`` -- the per-sequence local
            length PADDED to the max over ranks (interleave S hands out S tokens
            per rank per S*W super-block; S=1 -> ``ceil(g/W)``), so every rank
            gathers the same count. Reading past a rank's real local length stays
            inside its allocated blocks (``ceil(g/(bs*W))*bs >= ceil(g/(S*W))*S``
            when ``bs % S == 0``), so the padding slots read uninitialized cache
            rather than out of bounds; they are dropped by the de-interleave below.

        ``dcp_indexer_gather_index``
            output-position -> source index into the flattened all-gathered
            ``[W, sum(Lpad)]`` buffer. Global sequence-local position ``p`` lives on
            rank ``(p//S) % W`` at local index ``(p//(S*W))*S + p%S`` (S=1 -> the
            round-robin ``p%W`` / ``p//W``), hence
            ``src = owner(p) * sum(Lpad) + cu_pad[b] + local_index(p)``.
        """
        from atom.model_ops.dcp_ops import dcp_local_index, dcp_owner_rank

        W = self.dcp_world_size
        S = self.cp_kv_cache_interleave_size
        if attn_metadata.has_cached:
            g_cu = var["cu_seqlens_k"].np[: bs + 1].astype(np.int64)
        else:
            g_cu = var["cu_seqlens_q"].np[: bs + 1].astype(np.int64)
        g_lens = g_cu[1:] - g_cu[:bs]
        del counts  # kept for signature symmetry with the caller's other helpers

        lpad = ((g_lens + S * W - 1) // (S * W)) * S
        cu_pad = np.zeros(bs + 1, dtype=np.int64)
        np.cumsum(lpad, out=cu_pad[1:])
        local_total = int(cu_pad[bs])
        total_kv = int(g_cu[bs])

        # Position within its sequence for every global KV token.
        pos = np.arange(total_kv, dtype=np.int64) - np.repeat(g_cu[:bs], g_lens)
        src = (
            dcp_owner_rank(pos, W, S) * local_total
            + np.repeat(cu_pad[:bs], g_lens)
            + dcp_local_index(pos, W, S)
        )

        dev = self.device
        attn_metadata.dcp_indexer_local_total = local_total
        attn_metadata.dcp_indexer_local_cu_seqlens = torch.from_numpy(
            cu_pad.astype(np.int32)
        ).to(dev, non_blocking=True)
        attn_metadata.dcp_indexer_gather_index = torch.from_numpy(
            src.astype(np.int32)
        ).to(dev, non_blocking=True)

    def _sparse_selected_counts(self, seq_lens):
        """How many KV entries the indexer actually selects for each row.

        Token-granular: ``min(seq_len, index_topk)``.

        Pooled (kpool): ``min(pools, index_topk // kpool) * kpool`` history
        tokens plus the ``seq_len % kpool`` tail tokens, which are appended
        unscored. Note this collapses to exactly ``seq_len`` whenever
        ``seq_len <= index_topk`` -- the same value the token-granular
        expression gives -- so the two paths agree below the threshold by
        construction rather than by coincidence.
        """
        if self.index_kpool <= 1:
            return np.minimum(seq_lens, self.index_topk)
        kpool = self.index_kpool
        pools = seq_lens // kpool
        history = np.minimum(pools, self.index_topk // kpool) * kpool
        return (history + seq_lens % kpool).astype(np.int32)

    def prepare_prefill(self, batch: ScheduledBatch, running_bs: int):
        attn_metadata, positions = CommonAttentionBuilder.prepare_prefill(
            self, batch, running_bs
        )
        bs = batch.total_seqs_num_prefill
        scheduled_tokens = batch.total_tokens_num_prefill
        var = self.model_runner.forward_vars
        # kpool writes ONE compressed key per index_kpool tokens into the paged
        # index cache on every prefill, short or long, so it needs the block
        # table even below the sparse threshold -- where the token-granular
        # path never asks for one.
        if (
            self.is_sparse
            and self.index_kpool > 1
            and attn_metadata.block_tables is None
        ):
            self.prepare_block_tables(batch)
            attn_metadata.block_tables = var["block_tables"].copy_to_gpu(bs)
        if self.is_sparse and attn_metadata.max_seqlen_k > self.index_topk:
            if attn_metadata.block_tables is None:
                # Already marshalled by the base builder; only the upload is
                # gated on `has_cached`.
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
                full_seq_lens = var["cu_seqlens_k"].np[1 : bs + 1] - seq_starts
                cached_lens = full_seq_lens - counts
                repeated_seq_starts = np.repeat(seq_starts, counts)
                repeated_cached_lens = np.repeat(cached_lens, counts)
                var["cu_seqlen_ks"].np[:scheduled_tokens] = repeated_seq_starts
                var["cu_seqlen_ke"].np[:scheduled_tokens] = (
                    repeated_seq_starts + repeated_cached_lens + local_offsets + 1
                )
                sparse_counts = repeated_cached_lens + local_offsets + 1
            else:
                var["cu_seqlen_ke"].np[:scheduled_tokens] = (
                    np.arange(scheduled_tokens, dtype=np.int32) + 1
                )
                var["cu_seqlen_ks"].np[:scheduled_tokens] = np.repeat(
                    var["cu_seqlens_q"].np[:bs], counts
                )
                sparse_counts = local_offsets + 1
                full_seq_lens = counts
            if self.index_kpool > 1:
                # The pooled gather output must have exactly this many rows.
                # Compute it from host metadata already owned by the builder,
                # instead of synchronizing on pool_cu[-1] in every indexer layer.
                attn_metadata.kpool_total_pools = int(
                    np.sum(full_seq_lens // self.index_kpool)
                )
            attn_metadata.cu_seqlen_ks = var["cu_seqlen_ks"].copy_to_gpu(
                scheduled_tokens
            )
            attn_metadata.cu_seqlen_ke = var["cu_seqlen_ke"].copy_to_gpu(
                scheduled_tokens
            )
            attn_metadata.sparse_cu_seqlens_q = var["sparse_cu_seqlens_q"].gpu[
                : scheduled_tokens + 1
            ]
            # Sparse (DSA) attention: one last-page len per query token (all 1s,
            # page_size=1). Lives only on sparse_kv_last_page_lens; kv_last_page_lens
            # stays the dense per-seq buffer set by the has_cached block below.
            attn_metadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:scheduled_tokens]

            # Per-query req_id: token_id 0..scheduled_tokens-1 maps to batch id.
            # Use counts (new tokens per batch), not context_lens (full seq len).
            attn_metadata.token_to_seq_idxs = torch.repeat_interleave(
                torch.arange(bs, dtype=torch.int32, device=self.device),
                torch.tensor(counts, dtype=torch.int64, device=self.device),
            )
            var["sparse_kv_indptr"].np[0] = 0
            var["sparse_kv_indptr"].np[1 : scheduled_tokens + 1] = np.cumsum(
                self._sparse_selected_counts(sparse_counts),
                dtype=np.int32,
            )
            attn_metadata.sparse_kv_indptr = var["sparse_kv_indptr"].copy_to_gpu(
                scheduled_tokens + 1
            )
            if self.dcp_world_size > 1:
                self._build_dcp_indexer_prefill_meta(attn_metadata, bs, counts, var)
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
                max_split_per_batch=_MLA_SPLIT_BUDGET_AUTO,
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
                self._apply_pcp_reindex(attn_metadata, scheduled_tokens, sparse_counts)

        if hasattr(self.model_runner, "drafter") or attn_metadata.has_cached:
            # Populate kv_last_page_lens for full sequence (needed for MLA prefill with
            # prefix cache; decode does the same)
            if self.model_runner.block_size != 1:
                var["kv_last_page_lens"].np[:bs] = np.asarray(
                    batch.last_block_num_tokens[:bs], dtype=np.int32
                )
            else:
                var["kv_last_page_lens"].np[:bs] = 1

            attn_metadata.kv_indices = var["kv_indices"].gpu
            kv_indptr = var["kv_indptr"].gpu[: running_bs + 1]
            attn_metadata.kv_indptr = kv_indptr[: bs + 1]
            attn_metadata.kv_indptr[0] = 0
            # `context_lens` is padded past the requests this indptr counts.
            attn_metadata.kv_indptr[1 : bs + 1] = torch.cumsum(
                attn_metadata.context_lens[:bs], 0
            )

            # kv_indices_generate_triton expects logical block_tables (one
            # entry per block_ratio tokens). The parent packed exactly that
            # this step, and the only write to the mirror in between is the
            # tail zeroing below, which starts at `bs`.
            _pad_prefill_mla_draft_tail(
                kv_indptr,
                var["kv_last_page_lens"].np,
                var["block_tables"].np,
                bs,
                running_bs,
            )
            var["kv_last_page_lens"].copy_to_gpu(running_bs)
            attn_metadata.kv_last_page_lens = var["kv_last_page_lens"].gpu[:bs]
            block_tables_for_kv = var["block_tables"].copy_to_gpu(running_bs)[:bs]
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
            if self.dcp_world_size > 1 and attn_metadata.has_cached:
                # DCP always routes the cached-prefix prefill through the chunked
                # context path (even single-chunk) because the cross-rank
                # AllGather + reorg live there. attn_prefill_chunk_size still
                # bounds per-chunk memory.
                attn_metadata.mla_chunk_meta = self._build_mla_chunk_meta_dcp(batch, bs)
            elif (
                self.attn_prefill_chunk_size > 0
                and attn_metadata.has_cached
                and sum(batch.num_cached_tokens[:bs]) > self.attn_prefill_chunk_size
            ):
                attn_metadata.mla_chunk_meta = self._build_mla_chunk_meta(batch, bs)

        attn_metadata.dtype_q = self.dtype_q
        # TBO: publish CPU length copies for zero-sync ubatch splits.
        # Attached last, once all metadata is finalized. No-op unless TBO is on.
        self._attach_tbo_prefill_cpu_lens(attn_metadata, bs)
        return attn_metadata, positions

    def _build_mla_chunk_meta(
        self, batch: ScheduledBatch, bs: int
    ) -> MLAChunkContextMetadata | None:
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
        per_seq_slots: list[np.ndarray] = []
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

        kv_indptr_list: list[torch.Tensor] = []
        kv_indices_list: list[torch.Tensor] = []
        cu_seqlens_k_list: list[torch.Tensor] = []
        total_tokens_list: list[int] = []
        max_seqlen_k_list: list[int] = []

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
            kv_indptr_list.append(upload_numpy(cu, self.device))
            kv_indices_list.append(upload_numpy(chunk_indices, self.device))
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

    def _build_mla_chunk_meta_dcp(
        self, batch: ScheduledBatch, bs: int
    ) -> MLAChunkContextMetadata | None:
        """DCP variant of `_build_mla_chunk_meta` (compressed-KV AllGather).

        The cached context is interleaved across DCP ranks: global token `g`
        lives on rank `(g // S) % dcp_world_size` at local index
        `(g // (S*W)) * S + g % S` (S = cp_kv_cache_interleave_size; S=1 = the
        original round-robin `g % W` / `g // W`). This builder produces, per
        chunk, the metadata to (1) index_select this rank's local cached tokens,
        (2) AllGather them, and (3) reorg them into per-sequence contiguous
        layout. The reorg needs only the per-rank *counts* (from
        `get_dcp_local_seq_lens(..., S)`), not S-aware ordering: the cached
        context is fully visible (no causal mask) so attention is permutation-
        invariant over it, and RoPE travels with each key — the S=1 reorg
        already emits a non-global order and is correct, so any S is too.

        Chunking is per-seq (plugin-style, not the global-axis chunking of the
        non-DCP builder): each chunk covers a `chunk_size`-token local window
        [c*chunk_size, (c+1)*chunk_size) of every sequence, which is what makes
        every rank's AllGather block the same size.

        Local token `p` of a sequence maps to physical slot
        `block_table[p // block_size] * block_size + (p % block_size)` — the
        same contiguous local packing used by the Phase-1 slot_mapping writes.
        """
        from atom.model_ops.dcp_ops import (
            dcp_reorg_row_indices,
            get_dcp_local_seq_lens,
        )

        dcp = self.dcp_world_size
        bsz = self.model_runner.block_size
        vbs = bsz * dcp  # virtual (global) block size

        cached_lens = np.asarray(batch.num_cached_tokens[:bs], dtype=np.int64)
        total_cached = int(cached_lens.sum())
        if total_cached == 0:
            return None
        num_with_context = int((cached_lens > 0).sum())

        # Real local context length per (seq, rank). Shared across chunks; the
        # reorg row map uses it to drop the per-seq padding.
        s_itl = self.cp_kv_cache_interleave_size
        local_lens_allranks = np.stack(
            [get_dcp_local_seq_lens(cached_lens, dcp, r, s_itl) for r in range(dcp)],
            axis=1,
        ).astype(
            np.int64
        )  # [bs, dcp]

        # Number of local blocks per seq is identical on every rank
        # (= ceil(global_len / vbs)), so the padded local length is uniform and
        # the AllGather block is uniform across ranks.
        num_local_blocks = cdiv(cached_lens, vbs)  # [bs]
        padded_local_len = num_local_blocks * bsz  # [bs]
        max_padded_local = int(padded_local_len.max(initial=0))

        # Local per-seq chunk budget (in local tokens), block-aligned so each
        # chunk window starts on a block boundary. The post-reorg GLOBAL context
        # for one chunk has up to `num_with_context * dcp * chunk_size` tokens
        # (dcp ranks combined), so divide the token budget by both to keep the
        # decompressed k/v tensors bounded by attn_prefill_chunk_size, matching
        # the non-DCP path's memory footprint.
        budget = self.attn_prefill_chunk_size if self.attn_prefill_chunk_size > 0 else 1
        chunk_size = max(bsz, (budget // max(num_with_context * dcp, 1)) // bsz * bsz)
        num_chunks = max(1, int(cdiv(max_padded_local, chunk_size)))

        block_tables = batch.block_tables

        local_slot_ids_list: list[torch.Tensor] = []
        ag_row_indices_list: list[torch.Tensor] = []
        cu_seqlens_k_list: list[torch.Tensor] = []
        total_tokens_list: list[int] = []
        seq_tot_list: list[int] = []
        max_seqlen_k_list: list[int] = []

        for c in range(num_chunks):
            c_lo = c * chunk_size
            c_hi = c_lo + chunk_size

            # Per-seq padded local chunk length (uniform across ranks).
            plc = np.clip(np.minimum(padded_local_len, c_hi) - c_lo, 0, None).astype(
                np.int64
            )  # [bs]
            # Per-(seq, rank) REAL local chunk length in this window.
            real_local_chunk = np.clip(
                np.minimum(local_lens_allranks, c_hi) - c_lo, 0, None
            )  # [bs, dcp]
            # GLOBAL per-seq chunk length after reorg == sum over ranks.
            global_chunk_len = real_local_chunk.sum(axis=1).astype(np.int32)  # [bs]

            cu = np.zeros(bs + 1, dtype=np.int32)
            np.cumsum(global_chunk_len, out=cu[1:])
            cu_seqlens_k_list.append(upload_numpy(cu, self.device))
            total_tokens_list.append(int(cu[-1]))
            max_seqlen_k_list.append(int(global_chunk_len.max(initial=0)))
            seq_tot_list.append(int(plc.sum()))
            ag_row_indices_list.append(
                upload_numpy(dcp_reorg_row_indices(plc, real_local_chunk), self.device)
            )

            # This rank's local slot ids for the chunk, per-seq padded to plc[i].
            slot_segments: list[np.ndarray] = []
            for i in range(bs):
                n = int(plc[i])
                if n == 0:
                    continue
                block_ids = np.asarray(block_tables[i], dtype=np.int64)
                p = c_lo + np.arange(n, dtype=np.int64)  # local positions
                slots = block_ids[p // bsz] * bsz + (p % bsz)
                slot_segments.append(slots.astype(np.int32))
            slot_ids = (
                np.concatenate(slot_segments)
                if slot_segments
                else np.empty(0, np.int32)
            )
            local_slot_ids_list.append(upload_numpy(slot_ids, self.device))

        return MLAChunkContextMetadata(
            kv_indptr=[],
            kv_indices=[],
            cu_seqlens_k=cu_seqlens_k_list,
            total_tokens=total_tokens_list,
            max_seqlen_k=max_seqlen_k_list,
            num_chunks=int(num_chunks),
            k_workspace=self.k_chunk_workspace,
            v_workspace=self.v_chunk_workspace,
            is_dcp=True,
            local_slot_ids=local_slot_ids_list,
            ag_row_indices=ag_row_indices_list,
            seq_tot=seq_tot_list,
        )

    def _apply_pcp_reindex(
        self,
        attn_metadata: AttentionMetaData,
        scheduled_tokens: int,
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
        s_real = int(scheduled_tokens)
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
            max_split_per_batch=_MLA_SPLIT_BUDGET_AUTO,
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

    def _dcp_round_robin_slot(self, block_table, pos: int) -> int:
        """DCP write slot for global position ``pos`` on this rank, or -1 if
        another rank owns it. KV is interleave-sharded: token pos lives on rank
        (pos//S)%W at local index (pos//(S*W))*S + pos%S (S = cp_kv_cache_interleave_size;
        S=1 = token-level round-robin). Shared by the qlen==1 and MTP qlen>1
        decode paths."""
        from atom.model_ops.dcp_ops import dcp_local_index, dcp_owner_rank

        W = self.dcp_world_size
        S = self.cp_kv_cache_interleave_size
        if dcp_owner_rank(pos, W, S) != self.dcp_rank:
            return -1
        block_size = self.model_runner.block_size
        return (
            block_table[pos // (block_size * W)] * block_size
            + dcp_local_index(pos, W, S) % block_size
        )

    def _publish_dcp_token_block_tables(
        self, attn_metadata, running_bs: int, max_seqlen_q: int
    ) -> None:
        """Give the DCP sparse indexer one block-table row per query token.

        Both aiter ops it drives address the table by row. A step with one query
        per sequence already has that table, so only MTP verify copies.
        """
        if not (self.is_sparse and self.dcp_world_size > 1):
            return
        block_tables = attn_metadata.block_tables
        if max_seqlen_q == 1:
            attn_metadata.dcp_token_block_tables = block_tables
            return
        rows = self._dcp_token_block_tables_gpu[: running_bs * max_seqlen_q]
        rows.view(running_bs, max_seqlen_q, -1).copy_(block_tables.unsqueeze(1))
        attn_metadata.dcp_token_block_tables = rows

    def prepare_decode(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
    ):
        scheduled_bs = batch.total_seqs_num_decode
        dropout_p = 0.0

        var = self.model_runner.forward_vars
        context_lens = np.asarray(batch.context_lens, dtype=np.int32)
        block_tables = batch.block_tables
        if not batch.is_dummy_run and max_seqlen_q > 1:
            # Get num_rejected (already mapped to current batch order in prepare_input_ids)
            num_rejected = self.model_runner.tokenID_processor.num_rejected
            if num_rejected is not None:
                context_lens -= num_rejected
                num_blocks = cdiv(context_lens, self.model_runner.block_size)
                block_tables = [bt[:n] for bt, n in zip(block_tables, num_blocks)]
        positions = decode_positions(context_lens, max_seqlen_q)

        # Before the slots, not after: `slot_mapping` reads this packed table.
        # DCP still walks the trimmed ragged rows -- its slot is a per-rank
        # filter, not an address this table can answer.
        self.prepare_block_tables(batch)

        if not batch.is_dummy_run:
            if max_seqlen_q > 1:
                if self.dcp_world_size > 1:
                    # DCP round-robin + MTP: each of the max_seqlen_q new (draft)
                    # tokens is written only on the rank owning its global pos.
                    slots = [
                        self._dcp_round_robin_slot(block_table, pos)
                        for block_table, seq_len in zip(block_tables, context_lens)
                        for pos in range(seq_len - max_seqlen_q, seq_len)
                    ]
                else:
                    slots = slot_mapping(
                        positions,
                        np.full(len(context_lens), max_seqlen_q, dtype=np.int32),
                        var["block_tables"].np,
                        self.model_runner.block_size,
                        scratch=self.token_axis_scratch,
                    )
            else:
                if self.dcp_world_size > 1:
                    slots = [
                        self._dcp_round_robin_slot(block_table, seq_len - 1)
                        for block_table, seq_len in zip(block_tables, context_lens)
                    ]
                else:
                    # One token per sequence reduces the gather to its last row,
                    # and `last_block_num_tokens` names the offset directly.
                    slots = [
                        block_table[-1] * self.model_runner.block_size
                        + last_block_num
                        - 1
                        for block_table, last_block_num in zip(
                            block_tables, batch.last_block_num_tokens
                        )
                    ]

        # Use scheduled_bs since in dummy run, total_seqs_num_decode is 1.
        scheduled_tokens = scheduled_bs * max_seqlen_q
        var["slot_mapping"].np[:running_tokens] = -1
        if not batch.is_dummy_run:
            var["slot_mapping"].np[:scheduled_tokens] = slots
        var["positions"].np[:scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens
        var["context_lens"].np[scheduled_bs:running_bs] = 0

        if self.dcp_world_size > 1:
            from atom.model_ops.dcp_ops import (
                get_dcp_local_seq_lens,
                get_dcp_local_window_lens,
            )

            local_context_lens = get_dcp_local_seq_lens(
                context_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            if self._publishes_dcp_local_lens:
                # Publish once per step instead of launching 7 elementwise
                # kernels in every full sparse-indexer layer. One row per query
                # token: a draft position's extra token lands on a single rank,
                # so a per-request length is only right for the last position.
                var["dcp_local_context_lens"].np[:scheduled_tokens] = (
                    get_dcp_local_window_lens(
                        context_lens,
                        max_seqlen_q,
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    )
                )
                var["dcp_local_context_lens"].np[scheduled_tokens:running_tokens] = 0
            num_blocks_per_seq = cdiv(local_context_lens, self.block_size)
        elif any(batch.is_first_decode_without_local_prefill):
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

        var["kv_indptr"].np[1 : scheduled_bs + 1] = kv_indptr
        var["kv_indptr"].np[scheduled_bs + 1 : running_bs + 1] = sum_blocks
        if self.dcp_world_size > 1:
            # Global (un-sharded) token-level kv_indptr for the round-robin CP
            # causal mask (page_size=1 -> token granularity). context_lens is the
            # GLOBAL per-request KV length (local shard = get_dcp_local_seq_lens).
            g_kv_indptr = np.cumsum(context_lens[:scheduled_bs], dtype=np.int32)
            var["g_kv_indptr"].np[0] = 0
            var["g_kv_indptr"].np[1 : scheduled_bs + 1] = g_kv_indptr
            var["g_kv_indptr"].np[scheduled_bs + 1 : running_bs + 1] = (
                g_kv_indptr[-1] if scheduled_bs > 0 else 0
            )
        if self.dcp_world_size > 1 and self.block_size != 1:
            local_last = local_context_lens % self.block_size
            var["kv_last_page_lens"].np[:scheduled_bs] = np.where(
                local_last == 0,
                np.where(local_context_lens > 0, self.block_size, 0),
                local_last,
            )
        else:
            var["kv_last_page_lens"].np[:scheduled_bs] = (
                batch.last_block_num_tokens if self.block_size != 1 else 1
            )
        var["kv_last_page_lens"].np[scheduled_bs:running_bs] = 0
        vars_used = [
            ("slot_mapping", running_tokens),
            ("context_lens", running_bs),
            # ("kv_indptr", running_bs + 1),
            ("kv_last_page_lens", running_bs),
            ("block_tables", running_bs),
        ]
        if self._publishes_dcp_local_lens:
            vars_used.append(("dcp_local_context_lens", running_tokens))
        metadata_deps = {
            "kv_last_page_lens",
        }

        if self.is_sparse:
            if max_seqlen_q > 1:
                # A token sees every KV entry up to its own, so the count is one
                # past its position -- worked out above, not a second walk.
                per_token_kv_lens = positions + 1
                sparse_per_token_lens = self._sparse_selected_counts(
                    np.maximum(per_token_kv_lens, 0)
                )
                var["sparse_kv_indptr"].np[1 : scheduled_tokens + 1] = np.cumsum(
                    sparse_per_token_lens, dtype=np.int32
                )
                var["sparse_kv_indptr"].np[
                    scheduled_tokens + 1 : running_tokens + 1
                ] = var["sparse_kv_indptr"].np[scheduled_tokens]
                vars_used.append(("sparse_kv_indptr", running_tokens + 1))
                vars_used.append(("sparse_cu_seqlens_q", running_tokens + 1))
                metadata_deps.add("sparse_kv_indptr")
            else:
                sparse_context_lens = self._sparse_selected_counts(
                    var["context_lens"].np[:running_bs]
                )
                var["sparse_kv_indptr"].np[1 : running_bs + 1] = np.cumsum(
                    sparse_context_lens, dtype=np.int32
                )
                var["sparse_kv_indptr"].np[scheduled_bs : running_bs + 1] = var[
                    "sparse_kv_indptr"
                ].np[scheduled_bs]
                vars_used.append(("sparse_kv_indptr", running_bs + 1))
                metadata_deps.add("sparse_kv_indptr")

        vars_for_metadata = [(el, num) for el, num in vars_used if el in metadata_deps]
        vars_remaining = [(el, num) for el, num in vars_used if el not in metadata_deps]
        max_seqlen_k = context_lens.max()

        # The side prep_stream overlaps the metadata H2D copies + kv_indices
        # generation with the main stream. Under intra-GPU disagg the decode runs
        # on a CU-masked stream, and the prep_stream's wait_stream barriers
        # serialize against it, adding per-step decode latency. So in disagg mode
        # do the copies synchronously on the current stream; otherwise keep the
        # async overlap.
        disagg = self.model_runner.config.enable_rapidserve
        ctx = {}
        ctx["kv_indptr"] = var["kv_indptr"].copy_to_gpu(running_bs + 1)
        if disagg:
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
        else:
            prep_stream = self.prep_stream
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
        # metadata copies on main stream
        positions = var["positions"].copy_to_gpu(scheduled_tokens)
        ctx.update({el: var[el].copy_to_gpu(num) for el, num in vars_for_metadata})
        # A view: `publish_cu_seqlens_q` already uploaded it this step, and
        # nothing here writes the host copy.
        ctx["cu_seqlens_q"] = var["cu_seqlens_q"].gpu[: running_bs + 1]

        if is_sparse_mtp:
            ctx_mla_ps = self.set_mla_persistent_worker_buffers(
                running_bs, max_seqlen_q
            )
            ctx_mla_ps_sparse = self._set_mla_persistent_worker_buffers_sparse_mtp(
                running_tokens
            )
        else:
            # DCP + MTP (max_q_len>1) needs the round-robin global-position causal
            # mask: build persistent metadata with is_cp_round_robin so the kernel
            # masks on global positions g(j)=j*W+r instead of local-causal trim.
            cp_round_robin = self.dcp_world_size > 1 and max_seqlen_q > 1
            # For cprr, build metadata over the REAL requests (scheduled_bs), not
            # the cudagraph-padded running_bs. The padded tail is 0-query, which the cprr
            # metadata kernel mishandles (num_qo_tiles=0 -> div-by-zero; and its
            # is_cp_round_robin "no trim" is overridden by the nhead=128 M-fold
            # qk_batch_ratio branch). The persistent kernel is work_indptr-driven,
            # so scheduled_bs metadata naturally skips the padding rows. A step
            # that padded nothing reaches here with the two equal, and then this
            # is a no-op -- but eager no longer implies that, since a DP-unified
            # batch pads off the graph path too.
            meta_bs = scheduled_bs if cp_round_robin else running_bs
            ctx_mla_ps = self.set_mla_persistent_worker_buffers(
                meta_bs, max_seqlen_q, is_cp_round_robin=cp_round_robin
            )
            ctx_mla_ps_sparse = None
        ctx.update(ctx_mla_ps)
        if not disagg:
            current_stream.wait_stream(prep_stream)
        attn_metadata = AttentionMetaData(
            dropout_p=dropout_p,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            **ctx,
        )
        attn_metadata.dtype_q = self.dtype_q

        # Round-robin CP global kv_indptr (only under DCP; None otherwise so the
        # non-DCP / qlen=1 paths keep the plain kernel). Consumed by
        # _forward_decode -> mla_decode_fwd for the MTP (max_q_len>1) cprr mask.
        attn_metadata.g_kv_indptr = (
            var["g_kv_indptr"].copy_to_gpu(running_bs + 1)
            if self.dcp_world_size > 1
            else None
        )

        if ctx_mla_ps_sparse is not None:
            for k, v in ctx_mla_ps_sparse.items():
                setattr(attn_metadata, k, v)

        if is_sparse_mtp:
            attn_metadata.sparse_cu_seqlens_q = var["sparse_cu_seqlens_q"].gpu[
                : running_tokens + 1
            ]
            attn_metadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:running_tokens]
            self._token_to_seq_idxs_gpu[:scheduled_tokens] = torch.arange(
                scheduled_bs, dtype=torch.int32, device=self.device
            ).repeat_interleave(max_seqlen_q)
            self._token_to_seq_idxs_gpu[scheduled_tokens:running_tokens] = 0
            attn_metadata.token_to_seq_idxs = self._token_to_seq_idxs_gpu[
                :running_tokens
            ]
        elif self.is_sparse:
            # Non-MTP sparse decode (single token per seq): the sparse KV is
            # packed at page_size=1, so last_page_len is 1 for every seq. Expose
            # the all-1s buffer so _forward_decode passes it to mla_decode_fwd
            # instead of the dense per-block kv_last_page_lens (which would make
            # the kernel over-read past the written sparse-index region).
            attn_metadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:running_bs]
        self._publish_dcp_token_block_tables(attn_metadata, running_bs, max_seqlen_q)

        # running_bs, not scheduled_bs: the padded rows have to be split into the
        # ubatches too, or accuracy drifts.
        if self.model_runner.config.enable_tbo_decode and running_bs >= 2:
            self._prepare_ubatch_decode(
                scheduled_bs,
                running_bs,
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
        self._tbo_full_running_bs = bs
        N = self._NUM_TBO_UBATCHES
        half = bs // N

        ub_ranges = [
            (0, half),
            (half, bs),
        ]
        running_bs_list = [half, bs - half]

        for ub_idx, ((req_start, req_end), running_bs) in enumerate(
            zip(ub_ranges, running_bs_list)
        ):
            p = f"ub{ub_idx}_"
            # How many real requests fall in this ubatch's range
            ub_real_reqs = max(0, min(scheduled_bs, req_end) - req_start)

            var[f"{p}context_lens"].np[:ub_real_reqs] = var["context_lens"].np[
                req_start : req_start + ub_real_reqs
            ]
            var[f"{p}context_lens"].np[ub_real_reqs:running_bs] = 0

            var[f"{p}kv_last_page_lens"].np[:ub_real_reqs] = var[
                "kv_last_page_lens"
            ].np[req_start : req_start + ub_real_reqs]
            var[f"{p}kv_last_page_lens"].np[ub_real_reqs:running_bs] = 0

            tok_start = req_start * max_seqlen_q
            ub_real_tokens = ub_real_reqs * max_seqlen_q
            ub_running_tokens = running_bs * max_seqlen_q
            var[f"{p}slot_mapping"].np[:ub_real_tokens] = var["slot_mapping"].np[
                tok_start : tok_start + ub_real_tokens
            ]
            var[f"{p}slot_mapping"].np[ub_real_tokens:ub_running_tokens] = -1

            var[f"{p}block_tables"].np[:ub_real_reqs] = var["block_tables"].np[
                req_start : req_start + ub_real_reqs
            ]
            var[f"{p}block_tables"].np[ub_real_reqs:running_bs] = 0

            full_kv_indptr = var["kv_indptr"].np
            base = full_kv_indptr[req_start]
            var[f"{p}kv_indptr"].np[0] = 0
            if ub_real_reqs > 0:
                var[f"{p}kv_indptr"].np[1 : ub_real_reqs + 1] = (
                    full_kv_indptr[req_start + 1 : req_start + ub_real_reqs + 1] - base
                )
            last_val = var[f"{p}kv_indptr"].np[ub_real_reqs] if ub_real_reqs > 0 else 0
            var[f"{p}kv_indptr"].np[ub_real_reqs + 1 : running_bs + 1] = last_val

            if self.dcp_world_size > 1:
                # Per-ubatch slice of the global kv_indptr (rebased); diffs (per-req
                # global KV length) are preserved, which is what the cprr mask uses.
                full_g_kv_indptr = var["g_kv_indptr"].np
                g_base = full_g_kv_indptr[req_start]
                var[f"{p}g_kv_indptr"].np[0] = 0
                if ub_real_reqs > 0:
                    var[f"{p}g_kv_indptr"].np[1 : ub_real_reqs + 1] = (
                        full_g_kv_indptr[req_start + 1 : req_start + ub_real_reqs + 1]
                        - g_base
                    )
                g_last = (
                    var[f"{p}g_kv_indptr"].np[ub_real_reqs] if ub_real_reqs > 0 else 0
                )
                var[f"{p}g_kv_indptr"].np[ub_real_reqs + 1 : running_bs + 1] = g_last

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
                    ub_real_reqs + 1 : running_bs + 1
                ] = sparse_last

            var[f"{p}cu_seqlens_q"].np[: ub_real_reqs + 1] = np.arange(
                0,
                (ub_real_reqs + 1) * max_seqlen_q,
                max_seqlen_q,
                dtype=np.int32,
            )
            # The flat cumsum past the real requests is where the real ones end.
            var[f"{p}cu_seqlens_q"].np[
                ub_real_reqs + 1 : running_bs + 1
            ] = ub_real_tokens

            vars_used = [
                (f"{p}context_lens", running_bs),
                (f"{p}kv_last_page_lens", running_bs),
                (f"{p}slot_mapping", ub_running_tokens),
                (f"{p}block_tables", running_bs),
                (f"{p}kv_indptr", running_bs + 1),
                (f"{p}cu_seqlens_q", running_bs + 1),
            ]
            if self.dcp_world_size > 1:
                vars_used.append((f"{p}g_kv_indptr", running_bs + 1))
            if self.is_sparse:
                vars_used.append((f"{p}sparse_kv_indptr", running_bs + 1))

            for el, num in vars_used:
                var[el].copy_to_gpu(num)

            ub_max_seqlen_k = (
                int(context_lens[req_start : req_start + ub_real_reqs].max())
                if ub_real_reqs > 0
                else 0
            )
            kv_indices_generate_triton(
                var[f"{p}block_tables"].gpu[:running_bs],
                var[f"{p}kv_indices"].gpu,
                var[f"{p}kv_indptr"].gpu[: running_bs + 1],
                self.block_ratio,
                ub_max_seqlen_k,
            )

            self._set_ubatch_mla_buffers(
                running_bs,
                max_seqlen_q,
                ub_idx,
                is_cp_round_robin=self.dcp_world_size > 1 and max_seqlen_q > 1,
            )

    def _set_ubatch_mla_buffers(
        self, running_bs, max_q_len, ubatch_idx, is_cp_round_robin=False
    ):
        """Compute MLA work buffers for a per-ubatch forward_vars set."""
        p = f"ub{ubatch_idx}_"
        var = self.model_runner.forward_vars

        kv_indptr_for_mla = var[f"{p}kv_indptr"].gpu[: running_bs + 1]
        kv_last_page_lens_for_mla = var[f"{p}kv_last_page_lens"].gpu[:running_bs]
        if self.is_sparse:
            kv_indptr_for_mla = var[f"{p}sparse_kv_indptr"].gpu[: running_bs + 1]
            # Sparse KV is packed per token at page_size=1 -> last_page_len is 1.
            # The dense per-block buffer would over-read past the sparse indices
            # (see set_mla_persistent_worker_buffers). The all-1s sparse buffer is
            # batch-independent, so the shared (non-ubatch) copy is safe here.
            kv_last_page_lens_for_mla = var["sparse_kv_last_page_lens"].gpu[:running_bs]

        get_mla_metadata_v1(
            var[f"{p}cu_seqlens_q"].gpu[: running_bs + 1],
            kv_indptr_for_mla,
            kv_last_page_lens_for_mla,
            self.persistent_num_heads,
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
            is_cp_round_robin=is_cp_round_robin,
            kv_granularity=max(self.block_size, 16),
            max_seqlen_qo=max_q_len,
            uni_seqlen_qo=max_q_len,
            fast_mode=1,
            max_split_per_batch=_MLA_SPLIT_BUDGET_AUTO,
        )

    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        var = self.model_runner.forward_vars
        self._tbo_full_running_bs = bs
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
        scheduled_tokens = bs * max_q_len
        is_sparse_mtp = self.is_sparse and max_q_len > 1
        # DCP + MTP (max_q_len>1) capture: the cprr kernel masks on GLOBAL
        # positions, so capture needs a self-consistent (local, global) KV layout
        # or the graph shape won't match replay. Give every seq 1 local token per
        # rank -> global length = dcp_world_size. Replay overwrites these buffers
        # with real values.
        cp_round_robin = self.dcp_world_size > 1 and max_q_len > 1
        if cp_round_robin:
            if self.block_size == 1:
                var["kv_indptr"].np[: bs + 1] = np.arange(bs + 1, dtype=np.int32)
                var["kv_indptr"].copy_to_gpu(bs + 1)
                var["kv_indices"].gpu[:bs].zero_()
                var["kv_last_page_lens"].gpu[:bs].fill_(1)
            # g_kv_indptr is the only thing telling the cprr kernel how long each
            # sequence is GLOBALLY, and nothing else initializes it -- capturing
            # with it left at its allocation value walks the kernel off the KV
            # list (illegal access). The round-robin is token-level whatever the
            # block size, so the one-local-token-per-rank layout set up here (or
            # by the block_size > 1 branch above) is a global length of
            # dcp_world_size in both cases.
            var["g_kv_indptr"].np[: bs + 1] = (
                np.arange(bs + 1, dtype=np.int32) * self.dcp_world_size
            )
            var["g_kv_indptr"].copy_to_gpu(bs + 1)
        dcp_local_context_lens = None
        if self._publishes_dcp_local_lens:
            # The warmup forward reads this buffer before replay overwrites it.
            # One local token matches the synthetic capture KV metadata above;
            # one row per query token, as prepare_decode publishes it.
            var["dcp_local_context_lens"].np[:scheduled_tokens] = 1
            dcp_local_context_lens = var["dcp_local_context_lens"].copy_to_gpu(
                scheduled_tokens
            )
        if is_sparse_mtp:
            # Two sets: normal for dense layers, sparse_mtp for sparse layers
            ctx_mla_ps = self.set_mla_persistent_worker_buffers(bs, max_q_len)
            ctx_mla_ps_sparse = self._set_mla_persistent_worker_buffers_sparse_mtp(
                scheduled_tokens
            )
        else:
            ctx_mla_ps = self.set_mla_persistent_worker_buffers(
                bs, max_q_len, is_cp_round_robin=cp_round_robin
            )
            ctx_mla_ps_sparse = None
        attn_matadata = AttentionMetaData(
            slot_mapping=var["slot_mapping"].gpu[:scheduled_tokens],
            context_lens=var["context_lens"].gpu[:bs],
            block_tables=var["block_tables"].gpu[:bs],
            max_seqlen_q=max_q_len,
            cu_seqlens_q=var["cu_seqlens_q"].gpu[: bs + 1],
            kv_indptr=var["kv_indptr"].gpu[: bs + 1],
            kv_indices=var["kv_indices"].gpu,
            kv_last_page_lens=var["kv_last_page_lens"].gpu[:bs],
            sparse_kv_indptr=sparse_kv_indptr,
            dcp_local_context_lens=dcp_local_context_lens,
            **ctx_mla_ps,
        )
        attn_matadata.dtype_q = self.dtype_q
        # Attach the round-robin CP global kv_indptr for the captured graph so
        # replay (which overwrites the buffer with real values) matches. Only
        # consumed by _forward_decode when dcp>1 and max_q_len>1.
        attn_matadata.g_kv_indptr = (
            var["g_kv_indptr"].gpu[: bs + 1] if self.dcp_world_size > 1 else None
        )
        if ctx_mla_ps_sparse is not None:
            for k, v in ctx_mla_ps_sparse.items():
                setattr(attn_matadata, k, v)
        if is_sparse_mtp:
            attn_matadata.sparse_cu_seqlens_q = var["sparse_cu_seqlens_q"].gpu[
                : scheduled_tokens + 1
            ]
            attn_matadata.sparse_kv_indptr = var["sparse_kv_indptr"].gpu[
                : scheduled_tokens + 1
            ]
            attn_matadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:scheduled_tokens]
            self._token_to_seq_idxs_gpu[:scheduled_tokens] = torch.arange(
                bs, dtype=torch.int32, device=self.device
            ).repeat_interleave(max_q_len)
            attn_matadata.token_to_seq_idxs = self._token_to_seq_idxs_gpu[
                :scheduled_tokens
            ]
        elif self.is_sparse:
            # Non-MTP sparse decode capture: all-1s per-token last-page lens,
            # matching prepare_decode so _forward_decode reads the sparse buffer.
            attn_matadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:bs]
        self._publish_dcp_token_block_tables(attn_matadata, bs, max_q_len)
        positions = var["positions"].copy_to_gpu(scheduled_tokens)
        context = Context(
            positions=positions,
            is_prefill=False,
            scheduled_bs=bs,
            running_bs=bs,
            # A capture runs a full synthetic batch: nothing is padded.
            scheduled_tokens=scheduled_tokens,
            running_tokens=scheduled_tokens,
        )
        return attn_matadata, context

    def build_ubatch_metadata(
        self,
        ubatch_idx: int,
        running_bs: int,
    ) -> AttentionMetaData:
        """Create per-ubatch AttentionMetaData from pre-allocated forward_vars."""
        var = self.model_runner.forward_vars
        p = f"ub{ubatch_idx}_"
        max_q_len = var["mtp_k"] + 1 if "mtp_k" in var else 1
        dcp_local_context_lens = None
        if self._publishes_dcp_local_lens:
            requests_per_ubatch = self._tbo_full_running_bs // self._NUM_TBO_UBATCHES
            request_start = ubatch_idx * requests_per_ubatch
            dcp_local_context_lens = var["dcp_local_context_lens"].gpu[
                request_start : request_start + running_bs
            ]

        # Compute MLA work buffers for this ubatch
        self._set_ubatch_mla_buffers(
            running_bs,
            max_q_len,
            ubatch_idx,
            is_cp_round_robin=self.dcp_world_size > 1 and max_q_len > 1,
        )

        attn = AttentionMetaData(
            slot_mapping=var[f"{p}slot_mapping"].gpu[: running_bs * max_q_len],
            context_lens=var[f"{p}context_lens"].gpu[:running_bs],
            block_tables=var[f"{p}block_tables"].gpu[:running_bs],
            max_seqlen_q=max_q_len,
            cu_seqlens_q=var[f"{p}cu_seqlens_q"].gpu[: running_bs + 1],
            kv_indptr=var[f"{p}kv_indptr"].gpu[: running_bs + 1],
            kv_indices=var[f"{p}kv_indices"].gpu,
            kv_last_page_lens=var[f"{p}kv_last_page_lens"].gpu[:running_bs],
            dcp_local_context_lens=dcp_local_context_lens,
            sparse_kv_indptr=(
                var[f"{p}sparse_kv_indptr"].gpu[: running_bs + 1]
                if self.is_sparse
                else None
            ),
            sparse_kv_last_page_lens=(
                var["sparse_kv_last_page_lens"].gpu[:running_bs]
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
        # Per-ubatch round-robin CP global kv_indptr (None when non-DCP). Consumed
        # by _forward_decode when dcp>1 and max_q_len>1 (MTP).
        attn.g_kv_indptr = (
            var[f"{p}g_kv_indptr"].gpu[: running_bs + 1]
            if self.dcp_world_size > 1
            else None
        )
        if self.is_sparse and self.dcp_world_size > 1:
            # Config refuses decode TBO together with speculative decode under
            # DCP, so a ubatch always runs one query per sequence.
            assert max_q_len == 1
            attn.dcp_token_block_tables = attn.block_tables
        return attn

    def build_ubatch_prefill_metadata(
        self,
        attn_metadata: AttentionMetaData,
        ub_slice,
        running_bs: int,
        ubatch_idx: int = 0,
    ) -> AttentionMetaData:
        """
        Split prefill AttentionMetaData for MLA.
        """
        del ubatch_idx  # MLA has no per-ubatch pooled buffers to disambiguate
        from atom.utils.tbo.ubatch_splitting import split_attn_metadata

        ub_attn = split_attn_metadata(attn_metadata, ub_slice, running_bs)

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
            kv_indptr_list.append(upload_numpy(cu, device))
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
