# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import functools
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
    NONPS_MAX_Q_ROWS,
    dcp_persistent_supported,
    get_dcp_rank,
    get_dcp_world_size,
    mla_dcp_decode_is_persistent,
    mla_dcp_sparse_prefill_is_persistent,
    mla_dcp_sparse_prefill_uses_nonps,
)
from atom.distributed.pcp_utils import (
    get_pcp_rank,
    get_pcp_world_size,
    pcp_is_enabled,
    pcp_owned_rows_by_request,
    pcp_pad_dense,
    pcp_pad_len,
    pcp_round_robin_query_indices,
)
from atom.kv_transfer.disaggregation.index_staging import (
    gather_dcp_preshuffled_index_pages,
    prepare_dcp_index_gather_indices,
)
from atom.kv_transfer.disaggregation.pd_producer import (
    index_staging_pool_size as _index_staging_pool_size,
)
from atom.kv_transfer.disaggregation.pd_producer import mooncake_pd_producer_configured
from atom.kv_transfer.disaggregation.sharded_transfer import DCPShardPlan
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mla import (
    _MLA_MIN_HEADS,
    _MLA_SPLIT_BUDGET_AUTO,
    MLAAttention,
    mla_dcp_kernel_num_heads,
    mla_dcp_sparse_prefill_num_heads,
    mla_kernel_num_heads,
)
from atom.model_ops.attentions.selection_pack import (
    pack_row_selections,
)
from atom.model_ops.attentions.sparse_mla_head_groups import (
    HEADS as HEAD_GROUP_HEADS,
)
from atom.model_ops.attentions.sparse_mla_head_groups import (
    SparsePrefillHeadGroups,
)
from atom.model_ops.fp4_mqa_ragged_metadata import Fp4MqaRaggedMetadata
from atom.model_ops.glm5_next.geometry import (
    effective_kpool_size,
    pool_plane_width,
    topk_output_width,
)
from atom.model_ops.indexer import (
    FP8_MQA_LOGITS_TAKES_OUT,
    ConcatKeys,
    Fp4LaunchStorages,
    IndexRows,
    RowSplit,
    concat_rows,
    fp4_rows,
    fp8_rows,
)
from atom.model_ops.indexer.schedule import any_layer_skips_index_topk
from atom.model_ops.sparse_indexer_chunk import (
    ConcatKeyWorkspace,
    LogitsWorkspace,
    concat_plane_width,
    logits_rows,
)
from atom.model_ops.sparse_indexer_fp4 import (
    fp4_index_scale_rows,
    sparse_indexer_fp4_enabled,
)
from atom.utils import CpuGpuBuffer, envs, upload_numpy
from atom.utils.block_convert import (
    decompose_slots_triton,
    kv_indices_generate_triton,
    mtp_prepare_decode_mla_kernel,
)
from atom.utils.block_tables import block_table_state
from atom.utils.forward_context import (
    AttentionMetaData,
    Context,
    get_forward_context,
)
from atom.utils.tbo import NUM_TBO_UBATCHES, UBatchSlice

from .backends import (
    PERSISTENT_WORK_BUFFERS,
    AttentionBackend,
    CommonAttentionBuilder,
)
from .mla_kv_pool import MlaKvPool
from .pool_layout.sub_pool_spec import SubPoolSpec, page_pool
from .token_layout.decode import (
    decode_positions,
    query_is_uniform,
    ragged_decode_refusal,
)
from .token_layout.prefill import PrefillBatch
from .token_layout.slots import slot_mapping

logger = logging.getLogger("atom")

# The step's per-token index buffers a TBO decode micro-batch reads its own
# copy of (`_ubatch_decode_token_rows`), where the step allocated them.
_UBATCH_INDEX_TOKEN_ROWS = (
    "dcp_local_context_lens",
    "index_row_ends",
    "index_mtp_row_ends",
)


@functools.cache
def _warn_legacy_shuffle_kv() -> None:
    if envs.ATOM_USE_TRITON_MLA:
        logger.warning(
            "ATOM_USE_TRITON_MLA_SHUFFLE_KV is deprecated; use "
            "ATOM_MLA_KV_SHUFFLED=1 (keep ATOM_USE_TRITON_MLA=1 for the "
            "Triton decode)"
        )
    else:
        logger.warning(
            "ATOM_USE_TRITON_MLA_SHUFFLE_KV has no effect without "
            "ATOM_USE_TRITON_MLA=1 and is deprecated"
        )


# The MLA pool's row space. Named rather than derived from a geometry: every
# MLA layer packs into one row of the same width, so there is only ever one.
MLA_ROWS = "mla"

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
    block_tables: np.ndarray | None,
    scheduled_bs: int,
    running_bs: int,
) -> None:
    """Give widened Eagle rows an empty, fully initialized MLA KV range."""
    assert 0 <= scheduled_bs <= running_bs
    if scheduled_bs == running_bs:
        return
    kv_indptr[scheduled_bs + 1 : running_bs + 1] = kv_indptr[scheduled_bs]
    kv_last_page_lens[scheduled_bs:running_bs] = 0
    if block_tables is not None:
        block_tables[scheduled_bs:running_bs] = 0


def _global_index_cache_layer_ids(
    indexer_types,
    num_hidden_layers: int,
    num_draft_layers: int,
    layer_types=None,
    dsa_layer_indices=None,
) -> tuple[int, ...]:
    """Return global layers that own an index-key cache slice.

    GLM-5.2 ``shared`` layers reuse a preceding full layer's temporary top-k
    positions and do not construct an indexer, so their index-key cache slices
    are dead. A model that names its DSA layers (``dsa_layer_indices``) owns
    slices on those alone. Other sparse MLA models have no such schedule and
    retain the existing one-slice-per-layer layout.
    """
    target_layer_ids = (
        range(num_hidden_layers) if dsa_layer_indices is None else dsa_layer_indices
    )
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


@dataclass(frozen=True)
class StagedIndexRows:
    """A prefill's index rows as staged on the host, for the GPU build after
    the upload: `rows` query rows, the widest `width`; FP4 under PCP groups
    them by its own `query_starts` entries; a pooled indexer's `pools`."""

    rows: int
    width: int
    query_starts: int | None = None
    pools: int | None = None


@dataclass(frozen=True)
class StagedSparsePrefill:
    """What a sparse prefill's GPU build reads from its host staging: each
    query row's visible keys, the batch's keys, whether PCP splits the rows,
    and the staged index rows."""

    sparse_counts: np.ndarray
    total_kv: int
    pcp_split: bool
    index_rows: StagedIndexRows


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


def decode_row_ends(
    positions: np.ndarray, kpool: int, max_model_len: int
) -> np.ndarray:
    """Each decode row's index bound off its position: the keys it sees, or
    the pools they close (`kpool`), a draft row past the model length capped
    at the last pool the plane holds."""
    ends = positions + 1
    if kpool > 1:
        return np.minimum(ends // kpool, pool_plane_width(max_model_len, kpool))
    return ends


def indexes_by_token_table(is_sparse: bool, dcp_world_size: int, kpool: int) -> bool:
    """Whether the sparse indexer reads its keys off a block-table row per
    query token: the DCP decode exchange, and a pooled indexer (kpool), whose
    rows each see their own count of pools."""
    return is_sparse and (dcp_world_size > 1 or kpool > 1)


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

    # `__init__` decides these; the defaults serve builders assembled field by
    # field, as tests do.
    _indexer_fp4 = False
    mtp_keeps_selection = False

    def _global_num_draft_layers(self) -> int:
        """Return draft layers in the target MLA pool across all PP stages."""
        runner = self.model_runner
        spec_config = getattr(runner.config, "speculative_config", None)
        # A draft with a pool of its own is owned by `draft_kv_builder`. Only
        # draft layers that share the target MLA pool belong in this pool's
        # global KV/index-cache layout.
        if spec_config is None or hasattr(runner, "draft_kv_builder"):
            return 0
        draft_hf_config = spec_config.draft_model_hf_config
        # Every PP stage sees the whole draft stack: a draft is not split, so
        # `get_pp_indices` never covers it and this count is the same on each.
        # A standalone DSpark draft (Kimi-K3-DSpark) is a model of its own
        # layers; an MTP draft names its predict layers.
        if spec_config.use_dspark_with_draft():
            return draft_hf_config.num_hidden_layers
        return getattr(draft_hf_config, "num_nextn_predict_layers", 1)

    def _global_num_draft_index_layers(self) -> int:
        """Draft layers that own an index-key cache slice: all of them when the
        draft declares an indexer (an MTP layer of a sparse target), none when
        it does not (Kimi-K3-DSpark's dense MLA layers)."""
        spec_config = getattr(self.model_runner.config, "speculative_config", None)
        if spec_config is None or not getattr(
            spec_config.draft_model_hf_config, "index_topk", None
        ):
            return 0
        return self._global_num_draft_layers()

    def _local_draft_rows(self) -> int:
        """Rows of this stage's pool that a shared draft's stack holds.

        Counted from the draft's own modules under the same predicate the walk
        uses. Not `pool rows - this stage's layer span`: those are different
        units on a hybrid -- the span counts every layer of the stage, the pool
        only the MLA ones -- and the difference goes negative, which empties the
        draft's half of the index-cache layout.

        The asserts below are what "a draft under PP" means, stated once so
        nothing re-derives it from the rank.
        """
        from aiter.dist.parallel_state import get_pp_group

        runner = self.model_runner
        if not runner.draft_shares_kv_pool():
            return 0
        rows = sum(
            1
            for module in runner.drafter.model.modules()
            if MLA_ROWS in self._module_kinds(module)
        )
        # An earlier stage reaching here would be counting rows nobody
        # allocated.
        assert get_pp_group().is_last_rank, (
            "a draft's KV rows reached a PP stage that is not the last; the "
            "drafter is built whole on the last stage and split across none"
        )
        # The walk against the config. Where they differ the pool is sized off
        # one and addressed by the other.
        declared = self._global_num_draft_layers()
        assert rows == declared, (
            f"the draft has {rows} MLA modules on this stage but declares "
            f"{declared} layers; an unsplit draft's rows are all of them"
        )
        return rows

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
        num_local_draft_layers = (
            self._local_draft_rows() if self._global_num_draft_index_layers() else 0
        )
        global_layer_ids = _global_index_cache_layer_ids(
            getattr(hf_config, "indexer_types", None),
            num_hidden_layers,
            self._global_num_draft_index_layers(),
            getattr(hf_config, "layer_types", None),
            getattr(hf_config, "dsa_layer_indices", None),
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
        if envs.ATOM_USE_TRITON_MLA_SHUFFLE_KV:
            _warn_legacy_shuffle_kv()
        self.kv_shuffled = bool(envs.ATOM_MLA_KV_SHUFFLED)
        if self.kv_shuffled:
            assert model_runner.block_size == 64, (
                f"ATOM_MLA_KV_SHUFFLED=1 expects --block-size 64 "
                f"for {model_runner.kv_cache_dtype} KV cache, "
                f"got --block-size {model_runner.block_size}"
            )
        CommonAttentionBuilder.__init__(self, model_runner)
        # Set by `allocate_kv_cache_tensors`, which runs long after
        # construction -- on the P/D decode side too, over the imported pool.
        self.kv_pool: MlaKvPool | None = None
        # Each prefill batch's host-side sparse plan, by buffer prefix, from
        # `_stage_prefill_batch` to `_finish_prefill_batch`; None for a dense
        # one.
        self._sparse_prefill_staged: dict[str, StagedSparsePrefill | None] = {}
        # An MTP draft's steps 1+ keep step 0's selection
        # (`compact_topk_indices`).
        self.mtp_keeps_selection = (
            hasattr(model_runner, "drafter") and model_runner.drafter.share_mtp_indices
        )
        # Single-program block for the fused MTP-decode metadata kernel. Sized
        # to the max batch (runtime bs <= max_bs) so one tl.cumsum spans the
        # whole batch in a single launch.
        self._mtp_fuse_block = triton.next_power_of_2(self.max_bs + 1)
        config = model_runner.config
        hf_config = config.hf_config
        # `self.num_attention_heads` set by CommonAttentionBuilder.__init__.
        # Must equal `MLAAttention.padded_num_heads`: that pads the query, this
        # sizes the kernel's work descriptors for it.
        self.padded_num_attention_heads = mla_kernel_num_heads(
            max(self.num_attention_heads, _MLA_MIN_HEADS)
        )
        self.is_sparse = model_runner.has_mla_indexer
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
        self._indexer_fp4 = self.is_sparse and sparse_indexer_fp4_enabled(
            config.index_cache_dtype, hf_config, warn=True
        )
        self.index_n_heads = hf_config.index_n_heads if self.is_sparse else 0
        self.index_logits_workspace: LogitsWorkspace | None = None
        # The FP8 prefill indexer's gathered keys (`_concat_prefill_scored`).
        self.index_prefill_key_workspace: ConcatKeyWorkspace | None = None
        self._fp4_launches = Fp4LaunchStorages(
            self.device, self.max_num_batched_tokens, self.max_bs
        )
        self.dtype_kv = dtypes.d_dtypes[config.kv_cache_dtype]
        self.dtype_q = self.dtype_kv

        self.dcp_world_size = get_dcp_world_size()
        self.dcp_rank = get_dcp_rank()
        self._publishes_dcp_local_lens = self.is_sparse and self.dcp_world_size > 1
        self.indexes_by_token_table = indexes_by_token_table(
            self.is_sparse, self.dcp_world_size, self.index_kpool
        )
        # e8m0 row-swizzle table for decompose_slots_triton. Compile the kernel
        # now: warmup never reaches a long DCP prefill.
        self._fp4_scale_row_lut = None
        if self._indexer_fp4 and self.dcp_world_size > 1:
            block = model_runner.block_size
            rows = torch.arange(block, dtype=torch.int32, device=self.device)
            self._fp4_scale_row_lut = fp4_index_scale_rows(rows, block)
            decompose_slots_triton(rows[:1], 1, block, self._fp4_scale_row_lut)

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

        nonps_args = (
            self.dcp_world_size,
            self.num_attention_heads * self.dcp_world_size,
            self.dtype_q == dtypes.fp8 and self.dtype_kv == dtypes.fp8,
            dcp_persistent,
        )
        self.dcp_sparse_prefill_nonps = self.is_sparse and (
            mla_dcp_sparse_prefill_uses_nonps(
                *nonps_args, config.max_num_batched_tokens
            )
        )
        # Warn only when the step budget is what turned the flag off.
        if (
            self.is_sparse
            and not self.dcp_sparse_prefill_nonps
            and mla_dcp_sparse_prefill_uses_nonps(*nonps_args, NONPS_MAX_Q_ROWS)
        ):
            logger.warning(
                "ATOM_DCP_SPARSE_PREFILL_NONPS ignored: max_num_batched_tokens=%d "
                "exceeds the %d q rows the non-persistent gqa64 kernel takes; "
                "DCP sparse prefill keeps its default kernel.",
                config.max_num_batched_tokens,
                NONPS_MAX_Q_ROWS,
            )
        # Decode and sparse prefill round the gathered head count with different
        # width tables, so the two can disagree. Their work/reduce buffers are
        # separate, so size each for its own consumer.
        self.sparse_prefill_num_heads = self.padded_num_attention_heads
        if self.sparse_dcp_metadata_rebuild:
            self.sparse_prefill_num_heads = mla_dcp_sparse_prefill_num_heads(
                self.num_attention_heads,
                self.dcp_world_size,
                persistent=mla_dcp_sparse_prefill_is_persistent(
                    self.dcp_world_size,
                    dcp_persistent,
                    sparse_metadata_rebuild=self.sparse_dcp_metadata_rebuild,
                ),
            )

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
            "kv_indptr": CpuGpuBuffer(
                self.max_bs + 1, publication_group="mla_csr", **i32_kwargs
            ),
            # Global (un-sharded) per-request KV indptr for round-robin CP: cumsum
            # of the GLOBAL context_lens (token-level, page_size=1). Only filled
            # when dcp_world_size > 1; consumed by the cprr kernel via
            # mla_decode_fwd(g_kv_indptr=...) to apply the global-position causal
            # mask for MTP (max_q_len > 1).
            "g_kv_indptr": CpuGpuBuffer(
                self.max_bs + 1, publication_group="mla_csr", **i32_kwargs
            ),
            "kv_indices": CpuGpuBuffer(
                self.max_bs * self.max_num_blocks_per_seq,
                **i32_kwargs,
            ),
            "kv_last_page_lens": CpuGpuBuffer(
                self.max_bs, publication_group="mla_csr", **i32_kwargs
            ),
        }
        if self._publishes_dcp_local_lens:
            # Layer-invariant sparse-DSA indexer metadata: one row per query
            # token, derived once per step and reused by every full layer.
            mla_metadata["dcp_local_context_lens"] = CpuGpuBuffer(
                self.max_bs * max_seqlen_qo, publication_group="mla_csr", **i32_kwargs
            )
        if self.is_sparse:
            # The indexer's key bound for each query row, local to its request,
            # derived once per step and read by every full layer.
            mla_metadata["index_row_ends"] = CpuGpuBuffer(
                max(self.max_bs * max_seqlen_qo, self.max_num_batched_tokens),
                publication_group="mla_csr",
                **i32_kwargs,
            )
        if self._indexer_fp4:
            # The FP4 prefill's rows grouped by request -- under PCP the rows
            # this rank owns, the padding rows one more group.
            mla_metadata["index_query_start_loc"] = CpuGpuBuffer(
                self.max_bs + 2, publication_group="mla_sparse", **i32_kwargs
            )
        if self.index_kpool > 1:
            # A pooled prefill's rows in one upload: where each request's run
            # of closed pools starts, gathered end to end, then where each
            # row's does, then each row's pools, an MTP layer's last.
            mla_metadata["index_pool_rows"] = CpuGpuBuffer(
                self.max_bs + 1 + 3 * self.max_num_batched_tokens,
                publication_group="mla_sparse",
                **i32_kwargs,
            )
            # A decode's MTP-layer row bounds (`mtp_index_rows`).
            mla_metadata["index_mtp_row_ends"] = CpuGpuBuffer(
                self.max_bs * max_seqlen_qo, publication_group="mla_csr", **i32_kwargs
            )
        mla_metadata["kv_last_page_lens"].cpu.fill_(1)
        mla_metadata["kv_last_page_lens"].copy_to_gpu()
        if self.is_sparse:
            mla_metadata["cu_seqlen_ks"] = CpuGpuBuffer(
                self.max_num_batched_tokens,
                publication_group="mla_sparse",
                **i32_kwargs,
            )
            mla_metadata["sparse_kv_indptr"] = CpuGpuBuffer(
                self.max_num_batched_tokens + 1,
                publication_group="mla_sparse",
                **i32_kwargs,
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
            # Where `compact_topk_indices` packs before copying back to the
            # front of the buffer and indptr it reads from.
            self._packed_indices = torch.empty(
                self.max_bs * self.index_topk_out, dtype=torch.int32, device=self.device
            )
            self._packed_indptr = torch.empty(
                self.max_bs + 1, dtype=torch.int32, device=self.device
            )
            # DCP sparse decode compacts each rank's owned top-k slots to the
            # front (no -1 holes), so the per-request region length becomes data-
            # AND layer-dependent.
            # One closing entry a micro-batch (`_attach_selection`).
            self._dcp_sparse_kv_indptr_gpu = torch.zeros(
                self.max_num_batched_tokens + NUM_TBO_UBATCHES,
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
            self._token_block_tables_gpu = (
                torch.empty(
                    self.max_bs * max_seqlen_qo,
                    self.block_table_cols,
                    **i32_kwargs,
                )
                if self.indexes_by_token_table and max_seqlen_qo > 1
                else None
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
                self.sparse_prefill_num_heads,
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
            # The indexers' decode logits plane, every layer's in turn, which
            # the step's rows are cut to. Allocated before the memory profile
            # so the KV budget counts it. DCP decode scores in its fused
            # exchange.
            # The FP8 prefill (concat) scorer bands into the same plane: a
            # forward publishes prefill or decode index rows, never both, so
            # the plane is as large as the larger of the two needs.
            capacity = 0
            if self._publishes_decode_index_rows():
                width = self._index_decode_width()
                capacity = logits_rows(self.max_bs * max_seqlen_qo, width) * width
            if self._concat_prefill_scored():
                self._alloc_prefill_concat_keys()
                capacity = max(capacity, self._prefill_concat_plane_capacity())
            if capacity:
                self.index_logits_workspace = LogitsWorkspace(capacity, self.device)
                logger.info(
                    "Sparse indexer logits workspace: %.2f GiB",
                    self.index_logits_workspace.nbytes / (1 << 30),
                )
            for module in sfc.values():
                impl = getattr(module, "impl", None)
                # The indexer writes all three; an attention impl holds the
                # selection only for a mono layer to find the one shared buffer.
                if hasattr(module, "sparse_kv_indices_buffer"):
                    module.sparse_kv_indices_buffer = self._sparse_kv_indices_gpu
                    module.dcp_sparse_kv_indptr_buffer = self._dcp_sparse_kv_indptr_gpu
                    module.dcp_owned_counts_buffer = self._dcp_owned_counts_gpu
                if hasattr(impl, "sparse_kv_indices_buffer"):
                    impl.sparse_kv_indices_buffer = self._sparse_kv_indices_gpu
            # One set of head-group buffers serves every owning layer in turn.
            # Not under top-k sharing: a grouped owner leaves the shared
            # sparse-prefill work plan unbuilt, and a skipping layer reads it.
            if (
                self.sparse_dcp_metadata_rebuild
                and self.sparse_prefill_num_heads == HEAD_GROUP_HEADS
                and self.dtype_kv == torch.bfloat16
                and not any_layer_skips_index_topk(hf_config)
            ):
                head_groups = SparsePrefillHeadGroups(self.device)
                for module in sfc.values():
                    impl = getattr(module, "impl", None)
                    if getattr(impl, "owns_sparse_indexer", False):
                        impl.sparse_prefill_head_groups = head_groups

        if config.enable_tbo:
            self._allocate_ubatch_buffers(max_seqlen_qo)

    def h2d_group_members(self) -> dict[str, tuple[str, ...]]:
        # Ready together before `kv_indices_generate_triton`, the step's first
        # consumer, sharing prefill destinations.
        var = self.model_runner.forward_vars
        optional = (
            "dcp_local_context_lens",
            "sparse_kv_indptr",
            "index_row_ends",
            "index_mtp_row_ends",
        )
        sparse_prefill = (
            "cu_seqlen_ks",
            "sparse_kv_indptr",
            "index_row_ends",
            "index_query_start_loc",
            "index_pool_rows",
        )
        members = super().h2d_group_members()
        members["decode_metadata"] = (
            members.get("decode_metadata", ())
            + (
                "kv_indptr",
                "g_kv_indptr",
                "slot_mapping",
                "context_lens",
                "kv_last_page_lens",
                "block_tables",
                "positions",
            )
            + tuple(name for name in optional if name in var)
        )
        # Joins the common prefill mirrors: a drafter's or cached prefix's
        # token CSR, and a sparse prefill's per-row bounds.
        members["prefill_metadata"] = (
            members.get("prefill_metadata", ())
            + ("kv_indptr", "kv_last_page_lens")
            + tuple(name for name in sparse_prefill if name in var)
        )
        return members

    def _ubatch_buffer_shapes(self) -> dict[str, tuple[int, ...]]:
        var = self.model_runner.forward_vars
        bs, tokens = self.max_bs, self._ubatch_tokens()
        shapes = super()._ubatch_buffer_shapes()
        shapes["kv_last_page_lens"] = (bs,)
        for name in _UBATCH_INDEX_TOKEN_ROWS:
            if name in var:
                shapes[name] = (self._ubatch_share(var[name].cpu.shape[0]),)
        if "index_query_start_loc" in var:
            shapes["index_query_start_loc"] = (bs + 2,)
        if "index_pool_rows" in var:
            shapes["index_pool_rows"] = (bs + 1 + 3 * tokens,)
        if self.is_sparse:
            # A decode's per request, a prefill's per token.
            shapes["sparse_kv_indptr"] = (tokens + 1,)
            shapes["cu_seqlen_ks"] = (tokens,)
            work = get_mla_metadata_info_v1(
                tokens,
                1,  # sparse prefill treats each query token as q_len=1
                self.sparse_prefill_num_heads,
                self.dtype_q,
                self.dtype_kv,
                is_sparse=True,
                fast_mode=True,
            )
            for name, (size, _) in zip(PERSISTENT_WORK_BUFFERS, work):
                shapes[f"sparse_prefill_{name}"] = size
        return shapes

    @property
    def prep_stream(self):
        # return self.model_runner.tokenID_processor.async_copy_stream
        return self.model_runner.async_execute_stream

    def _set_mla_persistent_worker_buffers_sparse_mtp(
        self,
        num_tokens: int,
    ):
        """Compute persistent metadata for sparse MTP per-token layout.

        The ``num_tokens`` tokens are treated as as many independent virtual
        sequences each with q_len=1.  cu_seqlens_q = [0,1,...,num_tokens],
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
            # Rows the plan addresses; every consumer's q / o must hold at
            # least this many. MLAAttention._forward_decode asserts it without
            # DCP, on eager steps and at capture (a host check: not on replay).
            "sparse_mtp_num_tokens": num_tokens,
        }

    def set_mla_persistent_worker_buffers(
        self,
        bs: int,
        max_q_len: int,
        only_update: bool = False,
        num_reject_tokens: torch.Tensor = None,
        sparse_decode: bool = False,
        is_cp_round_robin: bool = False,
        prefix: str = "",
    ):
        """Plan the persistent MLA kernel's work over the ``prefix`` buffers
        (``ub{i}_`` for a micro-batch); return the plan's buffers by name."""
        if self.kv_shuffled:
            return {}
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
        work_meta_data = var[f"{prefix}work_meta_data"]
        work_info_set = var[f"{prefix}work_info_set"]
        work_indptr = var[f"{prefix}work_indptr"]
        reduce_indptr = var[f"{prefix}reduce_indptr"]
        reduce_final_map = var[f"{prefix}reduce_final_map"]
        reduce_partial_map = var[f"{prefix}reduce_partial_map"]
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
            var[f"{prefix}sparse_kv_indptr"].gpu[: bs + 1]
            if use_sparse_meta
            else var[f"{prefix}kv_indptr"].gpu[: bs + 1]
        )
        # Sparse decode packs KV per query token at page_size=1, so every "page"
        # is exactly one token -> last_page_len must be 1. The dense
        # var["kv_last_page_lens"] holds the real last-BLOCK fill (1..block_size);
        # feeding it here makes get_mla_metadata_v1 compute a per-seq KV extent of
        # (sparse_count - 1 + dense_last_page_len), i.e. up to block_size-1 pages
        # PAST the written sparse-index region -> stale-index over-read. Mirror
        # kv_indptr_for_metadata (and the prefill/MTP-verify paths, which already
        # use the all-1s sparse buffer). That buffer is batch-independent, so a
        # micro-batch reads the shared one.
        kv_last_page_lens_for_metadata = (
            var["sparse_kv_last_page_lens"].gpu[:bs]
            if use_sparse_meta
            else var[f"{prefix}kv_last_page_lens"].gpu[:bs]
        )
        cu_seqlens_q = var[f"{prefix}cu_seqlens_q"].gpu[: bs + 1]
        if only_update:
            decode_update_mla_metadata_v1(
                cu_seqlens_q,
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
                cu_seqlens_q,
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

    def compact_topk_indices(self, rows: torch.Tensor) -> None:
        """Steps 1+ of an MTP draft reuse step 0's top-k
        (`index_share_for_mtp_iteration`): move each sequence's selection --
        the one of its token ``rows`` under step 0's per-token
        ``sparse_kv_indptr`` -- to the front of the shared sparse buffer,
        packed in sequence order, and point ``sparse_kv_indptr`` at it, which
        `prepare_mtp_decode` then keeps (`mtp_keeps_selection`).

        A prefill run dense (`run_sparse` off) selected every key: its
        selection is each sequence's whole context, laid out from the block
        tables."""
        fc = get_forward_context()
        attn_metadata = fc.attn_metadata
        indptr = self.model_runner.forward_vars["sparse_kv_indptr"].gpu
        buf = self._sparse_kv_indices_gpu
        num_seqs = rows.numel()
        if fc.context.is_prefill and not attn_metadata.run_sparse:
            indptr[:1].zero_()
            torch.cumsum(
                attn_metadata.context_lens[:num_seqs].clamp_max(self.index_topk),
                0,
                dtype=torch.int32,
                out=indptr[1 : num_seqs + 1],
            )
            kv_indices_generate_triton(
                attn_metadata.block_tables[:num_seqs],
                buf,
                indptr[: num_seqs + 1],
                self.block_ratio,
                attn_metadata.max_seqlen_k,
            )
            return
        pack_row_selections(
            rows,
            indptr,
            buf,
            self._packed_indptr,
            self._packed_indices,
            self.index_topk_out,
        )
        indptr[: num_seqs + 1].copy_(self._packed_indptr[: num_seqs + 1])
        packed = num_seqs * self.index_topk_out
        buf[:packed].copy_(self._packed_indices[:packed])

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
            ``sparse_kv_indptr``, unless ``mtp_keeps_selection``: the step
            reads the selection `compact_topk_indices` packed, at its indptr,
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

        # A padded row's `kv_indptr` entry is a range from whatever batch last
        # occupied it, and `_enter_decode_metadata` rebases only the real rows,
        # so the tail no longer continues them. Once the real rows total more
        # tokens than that stale value the array runs BACKWARDS and the pad row
        # comes out with a NEGATIVE length -- which the kernel below turns into
        # a negative sparse count, and sparse decode into a wild kv_start/kv_end
        # that faults. A long request landing on a bucket a shorter batch left
        # behind is enough. Close the tail the way a draft PREFILL's widened
        # rows already are (`_pad_prefill_mla_draft_tail`) -- repeat the last
        # real end. This has to precede the kernel, which reads the diffs; the
        # `+= cu_seqlens_q` it applies then leaves each pad row holding the same
        # single token it leaves every real one.
        if running_bs > bs:
            kv_indptr[bs + 1 : running_bs + 1] = kv_indptr[bs]

        mtp_prepare_decode_mla_kernel[(1,)](
            kv_indptr,
            cu_seqlens_q,
            sparse_kv_indptr if self.is_sparse else kv_indptr,
            positions_out if update_positions else kv_indptr,
            context_lens if update_context_lens else kv_indptr,
            running_bs,
            self.index_topk if self.is_sparse else 0,
            positions_out.stride(0) if update_positions else 1,
            IS_SPARSE=self.is_sparse and not self.mtp_keeps_selection,
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
            # One query row a sequence: its key bound is its context, or the
            # pools that closes -- off `context_lens` as this step bumped it.
            row_ends = var["context_lens"].gpu[:running_bs]
            if self.index_kpool > 1:
                row_ends = self._closed_pools(
                    row_ends, var["index_row_ends"].gpu[:running_bs]
                )
            # The draft's tables and lengths, as `EagleProposer` re-pointed the
            # step's metadata at them before this.
            rows = self._decode_index_rows(
                get_forward_context().attn_metadata, bs, 1, bs, row_ends=row_ends
            )
            result["index_rows"] = rows
            if self.index_kpool > 1:
                # An MTP layer verifying off its own positions, a token behind,
                # sees the pools a context one shorter closed.
                mtp_ends = var["index_mtp_row_ends"].gpu[:running_bs]
                torch.sub(var["context_lens"].gpu[:running_bs], 1, out=mtp_ends)
                self._closed_pools(mtp_ends, mtp_ends)
                result["mtp_index_rows"] = self._mtp_index_rows(rows)
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
        """One paged KV pool: a single packed tensor per layer (k_c + k_pe; V
        is absorbed into latent compression — no separate V cache or kv_scale).

        DeepSeek-V3.2 sparse variants add an indexer cache contribution
        for every indexer-owning layer, including draft/MTP layers. GLM-5.2
        shared layers do not own an indexer and are excluded.
        """
        return [page_pool(self._declare_kv_pool().entry_bytes)]

    def paged_pool_bytes(self, blocks: int) -> int:
        """The same declaration `sub_pool_specs` prices, times the blocks the
        budget bought."""
        return self._declare_kv_pool().pool_bytes(blocks)

    def release_kv_pools(self) -> None:
        if self.kv_pool is not None:
            self.kv_pool.release()

    def _module_kinds(self, module) -> tuple:
        """One row space: the MLA layers, target and shared draft alike."""
        is_mla = (
            hasattr(module, "base_attention")
            and hasattr(module, "use_mla")
            and module.use_mla
        )
        return (MLA_ROWS,) if is_mla else ()

    def _kv_pool_layers(self) -> int:
        """Rows the paged pool holds, one per layer that caches KV.

        The modules, counted -- so sizing, allocation and the P/D adopt path
        cannot disagree, and a hybrid's linear layers or a draft's stack are
        simply not in the walk rather than something to subtract.
        """
        return self.row_counts().get(MLA_ROWS, 0)

    def _index_rows_per_block(self) -> int:
        """Indexer rows one scheduler block owns — one per token by default.

        An indexer that pools several tokens into a row overrides this and
        keeps fewer.
        """
        return self.model_runner.block_size

    def _declare_kv_pool(self) -> MlaKvPool:
        """This model's MLA layers as a pool, declared but not allocated.

        Sizing asks before a block count exists, so both steps come from here
        and the pool that is charged for is the pool that gets built.
        """
        runner = self.model_runner
        hf_config = runner.config.hf_config
        # `aligned_index_cache_dim` reads indexer config a dense model has no
        # reason to carry, so it is asked only when there are indexer layers.
        indexer = (
            {
                "index_layers": len(self._index_cache_layout()[0]),
                "index_rows_per_block": self._index_rows_per_block(),
                "index_dim": aligned_index_cache_dim(hf_config),
                "index_dtype": dtypes.fp8,
                "index_head_dim": hf_config.index_head_dim,
                "index_fp4": self._indexer_fp4,
            }
            if runner.has_mla_indexer
            else {}
        )
        return MlaKvPool(
            layers=self._kv_pool_layers(),
            block_size=runner.block_size,
            entry_dim=mla_kv_entry_dim(hf_config),
            kv_dtype=dtypes.d_dtypes[runner.config.kv_cache_dtype],
            **indexer,
        )

    def _supports_dcp_index_staging(self) -> bool:
        """Whether producer index pages can be gathered into DCP consumer pages.

        The gather reconstructs scheduler-block MFMA tiles from token-granular
        physical pages. Hybrid KDA builders whose index cache is already
        scheduler-block indexed (and may compress tokens) override this.
        """
        return True

    def allocate_kv_cache_tensors(self, *, blocks: int, buf) -> dict:
        """Allocate this model's MLA pool inside the runner's paged region.

        The KV rows and the indexer keys are regions of the one buffer the
        runner holds, so neither goes back by name. Only the aligned dimension
        and the compact layer map do, so `build_kv_cache_tensor` can pick the
        right indexer slice.
        """
        self.num_blocks = blocks * self.block_ratio
        runner = self.model_runner
        hf_config = runner.config.hf_config
        self.kv_pool = self._declare_kv_pool()
        self.kv_pool.allocate(blocks, runner.device, buf=buf)
        out: dict = {}
        if runner.has_mla_indexer:
            index_cache_layer_ids, _ = self._index_cache_layout()
            out["aligned_index_dim"] = aligned_index_cache_dim(hf_config)
            out["index_cache_layer_ids"] = index_cache_layer_ids
            out["index_cache_layer_map"] = {
                global_layer_id: compact_layer_id
                for compact_layer_id, global_layer_id in enumerate(
                    index_cache_layer_ids
                )
            }
        return out

    def build_kv_cache_tensor(self, module):
        """Bind one MLA attention module to its KV slice.

        Handles standard MLA (one packed KV row per token) and the
        DeepSeek-V3.2 sparse variant (additional indexer cache hooked via
        `module.indexer.k_cache.kv_cache[0]`). Returns the KVCacheTensor or
        None if the module is not an MLA attention this builder owns.
        Side effects: sets module `kv_cache`, `max_model_len`, and (V3.2)
        the indexer's k_cache slot.
        """
        from atom.config import KVCacheTensor

        if MLA_ROWS not in self._module_kinds(module):
            return None

        runner = self.model_runner
        # The row the sizing walk gave this module. Its `layer_num` would
        # agree only while every layer of the model is MLA and this is stage 0.
        row = self.pool_rows[MLA_ROWS][module]
        # The pool's slice is `[blocks, rows, dim]`; MLA addresses by row, so
        # the block axis folds away. Contiguous, so this is a view.
        kv_cache = self.kv_pool.layer("kv", row).view(-1, 1, self.kv_pool.entry_dim)
        module.max_model_len = runner.config.max_model_len
        index_cache = None
        index_scale = None
        if runner.has_mla_indexer and module.indexer is not None:
            # The compact map is keyed by GLOBAL model layer ids, which the
            # pool row above is not -- on a non-first PP stage row 0 may be
            # global layer 39.
            global_layer_id = module.layer_num
            if global_layer_id not in runner.index_cache_layer_map:
                raise RuntimeError(
                    "Sparse MLA indexer layer is missing from the compact index "
                    f"cache layout: layer_num={global_layer_id}"
                )
            index_cache_layer_id = runner.index_cache_layer_map[global_layer_id]
            index_cache = self.kv_pool.layer("index", index_cache_layer_id)
            # `is_sparse` is this branch's own condition, so both sides
            # reduce to the same `sparse_indexer_fp4_enabled` call on the same
            # two inputs and cannot legitimately disagree. Compared rather than
            # assigned: the Indexer built `k_cache` from its own answer before
            # this runs, so writing the builder's over it would leave the flag
            # describing an object it no longer matches.
            assert module.indexer._indexer_fp4 == self._indexer_fp4, (
                "FP4 sparse indexer verdict diverged: metadata builder says "
                f"{self._indexer_fp4}, layer {global_layer_id}'s Indexer says "
                f"{module.indexer._indexer_fp4}"
            )
            if self._indexer_fp4:
                # Held in a local as well as on the indexer: the index
                # region is these two block-major planes, not one, so whatever
                # moves `index_cache` has to be handed this too (see
                # `KVCacheTensor.index_scale`). It is the smaller of the two --
                # one e8m0 byte per 16 packed keys -- but it is not optional.
                index_scale = self.kv_pool.layer("index_scale", index_cache_layer_id)
                module.indexer.k_cache.kv_cache[0] = index_cache
                module.indexer.k_cache.kv_cache_scale = index_scale
            else:
                # Use aligned dimension to avoid memory copy in torch inductor
                module.indexer.k_cache.kv_cache[0] = index_cache.view(
                    -1, 1, runner.aligned_index_dim
                )
        module.kv_cache = kv_cache
        return KVCacheTensor(
            layer_num=module.layer_num,
            k_cache=kv_cache,
            v_cache=None,
            k_scale=None,
            v_scale=None,
            index_cache=index_cache if runner.has_mla_indexer else None,
            index_scale=index_scale if runner.has_mla_indexer else None,
        )

    def get_kv_transfer_tensors(self):
        from atom.kv_transfer.disaggregation.page_region import page_region
        from atom.kv_transfer.disaggregation.types import (
            INDEX_CACHE_ROLE,
            MLA_KV_ROLE,
            KVTransferRegion,
            KVTransferTensors,
            PageRegion,
        )

        runner = self.model_runner
        if self.kv_pool is None:
            return None
        from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

        if (
            self._indexer_fp4
            and not KVConnectorFactory.topology_region_readers_copy_whole_blocks(
                runner.config
            )
        ):
            # A P/D connector is handed one `INDEX_CACHE_ROLE` region per layer
            # -- the whole vocabulary it and its DCP shard plan have for the
            # indexer -- and the FP4 cache is two planes neither parses.
            # A transport registered with `copies_whole_block_regions`
            # (`lmcache_mp`) is the exception, admitted above: it copies whole
            # blocks of every published region, so the FP4 data and its e8m0
            # scale plane travel as two regions per layer (see below).
            #
            # That is a statement about the REGION MAP, so it only binds the
            # transports that read one. Dense offload does not: it never looks
            # at the return value here, building its codec from the
            # `KVCacheTensor`s instead, where an FP4 layer now carries both
            # planes (`index_cache` plus `index_scale`). Its unit is a whole
            # block, so the e8m0 row swizzle -- which lives inside one block --
            # rides along untouched. Refusing it too would cost FP4 the offload
            # path for a reason that does not apply to it.
            #
            # `topology_uses_pd_staging` is NOT the predicate for this: it
            # answers whether a backend needs compressor P/D staging, and
            # `lmcache_mp` declares that False while still requiring these
            # regions. Ask the question actually being asked.
            if KVConnectorFactory.topology_reads_block_regions(runner.config):
                raise NotImplementedError(
                    "KV transfer that addresses the cache through the PAGE "
                    "region map is unsupported with the FP4 sparse indexer "
                    "unless lmcache_mp is the only transport reading the map: "
                    "the P/D backends and the hybrid/m3/kimi_k3 offload layouts "
                    "parse one index region per layer and cannot place its "
                    "separate e8m0 scale plane. Pass --index_cache_dtype fp8 to "
                    "use one; lmcache_mp and dense CPU/NVMe offload "
                    "(lmcache_offload) carry both planes and need no change."
                )
            return None
        # What the pool was built with, not what the hook would recompute: a
        # hybrid caches for fewer layers than the model has, and the consumer
        # indices below are positions in the allocated rows.
        num_layers = self.kv_pool.layers
        index_tensors: list[torch.Tensor] = []

        # A row of each is one scheduler block, so `stride(0)` is already the
        # bytes a transfer moves per block -- no `block_ratio` after the fact,
        # and no per-field override to keep in step with the pooling ones.
        # DCP PD still dispatches on the two collapsed roles (`mla.kv` /
        # `dsa.index_cache`) rather than the pool's per-layer names.
        pages: list[PageRegion] = []
        for role, t in self.kv_pool.region_tensors():
            if role.startswith("index."):
                pages.append(page_region(t, semantic_role=INDEX_CACHE_ROLE))
                index_tensors.append(t)
                continue
            pages.append(
                page_region(
                    t,
                    semantic_role=(
                        MLA_KV_ROLE if role.startswith("kv.") else f"mla.{role}"
                    ),
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
            # the draft's KV rows, not its index slices: a dense draft
            # (Kimi-K3-DSpark) has rows and no slices
            num_global_draft_layers = self._global_num_draft_layers()
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
            # From the draft's modules, not `num_layers - target`: derived that
            # way the count below adds back up to `num_layers` by construction
            # and the check cannot fail. Two independent walks make it a check.
            num_local_draft_layers = self._local_draft_rows()
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
            if self._indexer_fp4:
                # `region_tensors` publishes the FP4 scale planes after the
                # index data planes, one per index layer; number them after
                # every consumer's index regions so the list still names one
                # consumer region per published PAGE region.
                block_region_consumer_indices += [
                    num_global_kv_layers
                    + len(global_index_layer_ids)
                    + global_compact_index_slot_by_layer[layer_id]
                    for layer_id in local_index_layer_ids
                ]

        index_staging_region = None
        index_staging_pool_size = 0
        index_staging_chunk_pages = 0
        prepare_sharded_index = None
        gather_sharded_index = None
        if (
            index_tensors
            and getattr(self, "dcp_world_size", None) == 1
            and mooncake_pd_producer_configured(runner.config)
            and self._supports_dcp_index_staging()
        ):
            # A Mooncake P/D producer can receive requests from a DCP
            # consumer whose index cache is sharded below one MFMA tile. Keep a
            # small per-send-thread pool that repacks one index layer at a time;
            # latent MLA pages continue to transfer directly.
            scheduler_block_size = runner.config.kv_cache_block_size
            if scheduler_block_size % 16:
                raise RuntimeError(
                    "Preshuffled DSA index P/D staging requires "
                    "kv_cache_block_size divisible by 16, got "
                    f"{scheduler_block_size}"
                )
            index_staging_pool_size = _index_staging_pool_size(runner.config)
            index_staging_chunk_pages = 256
            first_index_page = index_tensors[0]
            index_head_dim = getattr(runner.config.hf_config, "index_head_dim", None)
            page_bytes = first_index_page.stride(0) * first_index_page.element_size()
            staging = torch.empty(
                (
                    index_staging_pool_size,
                    index_staging_chunk_pages,
                    page_bytes,
                ),
                dtype=torch.uint8,
                device=first_index_page.device,
            )
            index_staging_region = KVTransferRegion(
                base_addr=staging.data_ptr(),
                total_bytes=staging.numel() * staging.element_size(),
                unit_bytes=index_staging_chunk_pages * page_bytes,
                semantic_role="dsa.index_staging",
            )

            def prepare_sharded_index(plan: DCPShardPlan):
                return prepare_dcp_index_gather_indices(plan, first_index_page.device)

            def gather_sharded_index(
                region_idx,
                indices,
                pool_idx,
            ):
                index_region_idx = region_idx - num_layers
                if not 0 <= index_region_idx < len(index_tensors):
                    raise IndexError(
                        f"Index region {region_idx} maps to invalid cache row "
                        f"{index_region_idx}"
                    )
                slot = staging[pool_idx]
                pages = gather_dcp_preshuffled_index_pages(
                    index_tensors[index_region_idx],
                    slot,
                    indices,
                    index_head_dim,
                    runner.config.kv_cache_block_size,
                    1,
                )
                return slot.data_ptr(), pages

        return KVTransferTensors(
            pages=pages,
            block_region_consumer_indices=block_region_consumer_indices,
            index_staging_region=index_staging_region,
            index_staging_pool_size=index_staging_pool_size,
            index_staging_chunk_pages=index_staging_chunk_pages,
            prepare_sharded_index=prepare_sharded_index,
            gather_sharded_index=gather_sharded_index,
            # MLA's latent projection is replicated across TP. Sparse MLA's
            # index-key projection/cache is replicated as well; only the query
            # heads and absorbed KV-B/output projections are TP-sharded.
            # Consequently every PAGE byte published above is identical on
            # every TP worker (DCP/PCP are separate axes and are rejected by
            # the current LMCache MP connector).
            tp_replication_factor=int(
                getattr(runner.config, "tensor_parallel_size", 1) or 1
            ),
        )

    def _build_dcp_indexer_prefill_meta(self, attn_metadata, bs: int, prefix: str):
        """Metadata for the DCP sparse-prefill indexer gather, off the
        ``prefix`` set's host mirrors.

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
        var = self.model_runner.forward_vars
        committed = "cu_seqlens_k" if attn_metadata.has_cached else "cu_seqlens_q"
        g_cu = var[prefix + committed].np[: bs + 1].astype(np.int64)
        g_lens = g_cu[1:] - g_cu[:bs]

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
        if self._indexer_fp4:
            self._build_dcp_indexer_fp4_prefill_meta(
                attn_metadata,
                bs,
                lpad,
                cu_pad,
                g_cu,
                g_lens,
                var[f"{prefix}block_tables"].np[:bs],
            )

    def _build_dcp_indexer_fp4_prefill_meta(
        self, attn_metadata, bs: int, lpad, cu_pad, g_cu, g_lens, block_tables
    ):
        """Where FP4 staging reads this rank's shard and where it lays the
        gathered keys out: each request's from a page of its own, so a
        request's row of `dcp_indexer_fp4_block_tables` pages its keys as its
        own block table does and the scorer's columns are the request's own.

        The read formula must track `cp_gather_indexer_k_quant_cache`, the FP8
        plane's gather, which the FP4 planes have no equivalent of. Its padded
        indices stay inside the sequence's own blocks, so the staging read
        cannot leave the allocation.
        """
        block = self.model_runner.block_size
        seq_of = np.repeat(np.arange(bs, dtype=np.int64), lpad)
        j = np.arange(int(cu_pad[bs]), dtype=np.int64) - np.repeat(cu_pad[:bs], lpad)
        table = block_tables.astype(np.int64)
        read = table[seq_of, j // block] * block + (j % block)
        pages = (g_lens + block - 1) // block
        first_page = np.zeros(bs + 1, dtype=np.int64)
        np.cumsum(pages, out=first_page[1:])
        stage = np.repeat(first_page[:bs] * block - g_cu[:bs], g_lens) + np.arange(
            int(g_cu[bs]), dtype=np.int64
        )
        # Page / row / e8m0 row of both slot lists, built once per forward, not
        # per layer: one upload, one launch.
        slots = upload_numpy(
            np.concatenate([read, stage]).astype(np.int32), self.device
        )
        lut = getattr(self, "_fp4_scale_row_lut", None)
        if lut is not None:
            parts = decompose_slots_triton(slots, 0, block, lut)[:3]
        else:
            # Torch fallback (CPU / test builders); also the kernel's reference.
            page, row = slots // block, slots % block
            parts = (page, row, fp4_index_scale_rows(row, block))
        for side, cut in (
            ("read", slice(0, len(read))),
            ("stage", slice(len(read), None)),
        ):
            for part, tensor in zip(("page", "row", "scale_row"), parts):
                setattr(attn_metadata, f"dcp_indexer_fp4_{side}_{part}", tensor[cut])
        attn_metadata.dcp_indexer_fp4_stage_pages = int(first_page[bs])

        # The block tables' own width: the scorer specializes on this table's
        # stride, so the DCP and non-DCP prefills share one variant.
        cols = np.arange(self.block_table_cols, dtype=np.int64)
        staged_tables = np.where(
            cols < pages[:, None], first_page[:bs, None] + cols, 0
        ).astype(np.int32)
        attn_metadata.dcp_indexer_fp4_block_tables = upload_numpy(
            staged_tables, self.device
        )

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

    def _prefill_block_table_rows(self, batch, running_bs, has_cached):
        # kpool writes ONE compressed key per index_kpool tokens into the paged
        # index cache on every prefill, short or long, so it needs the block
        # table even below the sparse threshold; a sparse prefill reads it too.
        if self.is_sparse and (
            self.index_kpool > 1
            or self._run_sparse(int(batch.context_lens.max(initial=0)))
        ):
            return running_bs
        return super()._prefill_block_table_rows(batch, running_bs, has_cached)

    def _run_sparse(self, max_seqlen_k: int) -> bool:
        """A prefill runs sparse MLA once its keys outnumber the top-k: at or
        below it top-k selects every key, and dense attention is the same."""
        return self.is_sparse and max_seqlen_k > self.index_topk

    def stage_prefill(self, batch, running_bs, running_tokens, group):
        self._check_metadata_writable("mla_csr", "mla_sparse")
        attn_metadata, positions = super().stage_prefill(
            batch, running_bs, running_tokens, group
        )
        self._attach_selection(attn_metadata)
        return attn_metadata, positions

    def stage_ubatch_prefill(self, batch, attn_metadata, ubatches, pcp_groups):
        groups = super().stage_ubatch_prefill(
            batch, attn_metadata, ubatches, pcp_groups
        )
        for ub_idx, ((ub, _), ub_attn) in enumerate(
            zip(ubatches, attn_metadata.ubatches)
        ):
            self._attach_selection(ub_attn, ub.token_slice.start, ub_idx)
        return groups

    def _stage_prefill_batch(
        self, batch, positions, running_bs, running_tokens, group, prefix=""
    ):
        attn_metadata = super()._stage_prefill_batch(
            batch, positions, running_bs, running_tokens, group, prefix
        )
        bs = batch.scheduled_bs
        scheduled_tokens = batch.scheduled_tokens
        var = self.model_runner.forward_vars
        attn_metadata.run_sparse = self._run_sparse(attn_metadata.max_seqlen_k)
        self._sparse_prefill_staged[prefix] = None
        if attn_metadata.run_sparse:
            counts = batch.seqlens_q
            local_offsets = np.concatenate(
                [np.arange(s, dtype=np.int32) for s in counts]
            )
            cu_seqlen_ks = var[f"{prefix}cu_seqlen_ks"]
            if attn_metadata.has_cached:
                # Full context (cached + new): each query token can see the cached
                # prefix plus previous query tokens in this chunk, not future chunk
                # tokens.
                seq_starts = var[f"{prefix}cu_seqlens_k"].np[:bs]
                full_seq_lens = batch.context_lens
                cu_seqlen_ks.np[:scheduled_tokens] = np.repeat(seq_starts, counts)
                sparse_counts = np.repeat(batch.cached_lens, counts) + local_offsets + 1
            else:
                cu_seqlen_ks.np[:scheduled_tokens] = np.repeat(
                    var[f"{prefix}cu_seqlens_q"].np[:bs], counts
                )
                sparse_counts = local_offsets + 1
                full_seq_lens = counts
            attn_metadata.cu_seqlen_ks = cu_seqlen_ks.gpu[:scheduled_tokens]
            attn_metadata.sparse_cu_seqlens_q = var["sparse_cu_seqlens_q"].gpu[
                : scheduled_tokens + 1
            ]
            # Sparse (DSA) attention: one last-page len per query token (all 1s,
            # page_size=1). Lives only on sparse_kv_last_page_lens; kv_last_page_lens
            # stays the dense per-seq buffer set by the has_cached block below.
            attn_metadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:scheduled_tokens]

            sparse_kv_indptr = var[f"{prefix}sparse_kv_indptr"]
            sparse_kv_indptr.np[0] = 0
            sparse_kv_indptr.np[1 : scheduled_tokens + 1] = np.cumsum(
                self._sparse_selected_counts(sparse_counts),
                dtype=np.int32,
            )
            attn_metadata.sparse_kv_indptr = sparse_kv_indptr.gpu[
                : scheduled_tokens + 1
            ]
            group.set_counts(
                {
                    f"{prefix}cu_seqlen_ks": scheduled_tokens,
                    f"{prefix}sparse_kv_indptr": scheduled_tokens + 1,
                }
            )
            # ---- Prefill Context Parallel: shrink per-query sparse metadata --
            # to this rank's 1/pcp round-robin queries. Gate on
            # `not batch.is_dummy_run` so the reindex stays in lock-step with the
            # model's round-robin token split (ForCausalLM._pcp_active() also
            # skips dummy/warmup). Per-sequence + KV-write fields (slot_mapping,
            # block_tables, cu_seqlens_q/k) stay FULL — every rank keeps full KV.
            pcp_split = pcp_is_enabled() and not batch.is_dummy_run
            self._sparse_prefill_staged[prefix] = StagedSparsePrefill(
                sparse_counts=sparse_counts,
                total_kv=int(full_seq_lens.sum()),
                pcp_split=pcp_split,
                index_rows=self._stage_prefill_index_rows(
                    bs,
                    running_bs,
                    sparse_counts,
                    pcp_split,
                    attn_metadata.has_cached,
                    group,
                    prefix,
                ),
            )

        # The token-granular paged CSR (kv_indptr / kv_indices /
        # kv_last_page_lens) is read by the drafter and by the plain-layout
        # cached-prefix gather. The shuffled gather reads only block-granular
        # CSR (built below), so a shuffled cached prefill without a drafter
        # skips it.
        if self._prefill_reads_token_csr(attn_metadata):
            # The tokens in each request's last page, cached + new: needed for
            # MLA prefill with prefix cache; decode does the same.
            block_size = self.model_runner.block_size
            last_page_lens = var[f"{prefix}kv_last_page_lens"].np
            last_page_lens[:bs] = (batch.context_lens - 1) % block_size + 1

            attn_metadata.kv_indices = var[f"{prefix}kv_indices"].gpu
            # Built on the host mirror like the decode path, then one upload.
            # `context_lens` is padded past the requests this indptr counts.
            kv_indptr = var[f"{prefix}kv_indptr"].np
            kv_indptr[0] = 0
            np.cumsum(batch.context_lens, out=kv_indptr[1 : bs + 1])

            # kv_indices_generate_triton expects logical block_tables (one
            # entry per block_ratio tokens). The final `running_bs`-row table
            # was prepared before the upload; its host mirror remains borrowed.
            _pad_prefill_mla_draft_tail(
                kv_indptr,
                last_page_lens,
                None,  # block-table padding was prepared before its only upload
                bs,
                running_bs,
            )
            group.set_counts(
                {
                    f"{prefix}kv_indptr": running_bs + 1,
                    f"{prefix}kv_last_page_lens": running_bs,
                }
            )
            attn_metadata.kv_indptr = var[f"{prefix}kv_indptr"].gpu[: bs + 1]
            attn_metadata.kv_last_page_lens = var[f"{prefix}kv_last_page_lens"].gpu[:bs]
        attn_metadata.dtype_q = self.dtype_q
        return attn_metadata

    def _prefill_reads_token_csr(self, attn_metadata) -> bool:
        return hasattr(self.model_runner, "drafter") or (
            attn_metadata.has_cached and not self.kv_shuffled
        )

    def _attach_selection(
        self,
        attn_metadata: AttentionMetaData,
        first_token: int = 0,
        ubatch_idx: int = 0,
    ) -> None:
        """Hand ``attn_metadata`` its forward's share of the sparse selection
        buffers, from token ``first_token`` on: a TBO micro-batch's starts at
        its own first token, so the two never share one. Its compact indptr,
        a token's entry plus a closing one, also starts ``ubatch_idx`` later:
        the closing entry of one micro-batch is not the opening of the next."""
        if not self.is_sparse:
            return
        attn_metadata.sparse_kv_indices = self._sparse_kv_indices_gpu[
            first_token * self.index_topk_out :
        ]
        attn_metadata.dcp_sparse_kv_indptr = self._dcp_sparse_kv_indptr_gpu[
            first_token + ubatch_idx :
        ]
        attn_metadata.dcp_owned_counts = self._dcp_owned_counts_gpu[first_token:]

    def _sparse_prefill_work_plan(
        self,
        attn_metadata: AttentionMetaData,
        prefix: str,
        num_heads: int,
        last_page_lens: torch.Tensor,
    ) -> None:
        """The persistent sparse prefill's work plan over ``attn_metadata``'s
        query rows, into the ``prefix`` set's buffers; none for the
        non-persistent DCP sparse prefill, which reads no plan."""
        if self.dcp_sparse_prefill_nonps:
            return
        var = self.model_runner.forward_vars
        work = {
            name: var[f"{prefix}sparse_prefill_{name}"]
            for name in PERSISTENT_WORK_BUFFERS
        }
        get_mla_metadata_v1(
            attn_metadata.sparse_cu_seqlens_q,
            attn_metadata.sparse_kv_indptr,
            last_page_lens,
            num_heads,
            1,  # nhead_kv
            True,
            work["work_meta_data"],
            work["work_info_set"],
            work["work_indptr"],
            work["reduce_indptr"],
            work["reduce_final_map"],
            work["reduce_partial_map"],
            page_size=self.block_size,
            dtype_q=self.dtype_q,
            dtype_kv=self.dtype_kv,
            kv_granularity=max(self.block_size, 16),
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=1,
            max_split_per_batch=_MLA_SPLIT_BUDGET_AUTO,
        )
        for name, buffer in work.items():
            setattr(attn_metadata, f"sparse_prefill_{name}", buffer)

    def _finish_prefill_batch(self, batch, attn_metadata, prefix=""):
        super()._finish_prefill_batch(batch, attn_metadata, prefix)
        bs = batch.scheduled_bs
        var = self.model_runner.forward_vars
        staged = self._sparse_prefill_staged.pop(prefix)
        if staged is not None:
            if self.dcp_world_size > 1:
                self._build_dcp_indexer_prefill_meta(attn_metadata, bs, prefix)
            self._sparse_prefill_work_plan(
                attn_metadata,
                prefix,
                self.sparse_prefill_num_heads,
                attn_metadata.sparse_kv_last_page_lens,
            )
            if staged.pcp_split:
                self._apply_pcp_reindex(
                    attn_metadata, batch.scheduled_tokens, staged.sparse_counts, prefix
                )
            self._finish_prefill_index_rows(
                attn_metadata,
                bs,
                staged.index_rows,
                staged.total_kv,
                prefix,
                staged.pcp_split,
            )

        if self._prefill_reads_token_csr(attn_metadata):
            kv_indices_generate_triton(
                var[f"{prefix}block_tables"].gpu[:bs],
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
            attn_metadata.mla_chunk_meta = self._build_mla_chunk_meta_dcp(batch)
        elif (
            self.attn_prefill_chunk_size > 0
            and attn_metadata.has_cached
            and int(batch.cached_lens.sum()) > self.attn_prefill_chunk_size
        ):
            attn_metadata.mla_chunk_meta = (
                self._build_mla_chunk_meta_shuffle(batch, prefix)
                if self.kv_shuffled
                else self._build_mla_chunk_meta(batch)
            )

        self._attach_kv_shuffled_prefill_meta(attn_metadata, bs, prefix)

    def _attach_kv_shuffled_prefill_meta(
        self, attn_metadata, bs: int, prefix: str
    ) -> None:
        # Only the single-pass gather reads the full-context CSR; the chunked
        # path carries its own per-chunk CSR in mla_chunk_meta.
        if not (self.kv_shuffled and attn_metadata.has_cached):
            return
        if getattr(attn_metadata, "mla_chunk_meta", None) is not None:
            return
        # The shuffled cached-prefix gather (gather_kv_b_proj with
        # shuffled_kv_cache=True) reads block_size-token blocks, so it needs
        # block-granular CSR indices (logical block ids) instead of the
        # token-granular kv_indices used by the plain layout. Build them from
        # the full per-seq context (cached + just-written new tokens).
        block_size = self.model_runner.block_size
        var = self.model_runner.forward_vars
        # Counts from the host mirror of context_lens: no device sync, and the
        # indptr is one upload rather than a chain of tiny kernels.
        block_counts = cdiv(
            var[f"{prefix}context_lens"].np[:bs].astype(np.int64), block_size
        )
        indptr = np.zeros(bs + 1, dtype=np.int32)
        np.cumsum(block_counts, out=indptr[1:])
        indptr_gpu = upload_numpy(indptr, self.device)
        indices = torch.empty(int(indptr[-1]), dtype=torch.int32, device=self.device)
        kv_indices_generate_triton(
            var[f"{prefix}block_tables"].gpu[:bs],
            indices,
            indptr_gpu,
            1,
            int(block_counts.max(initial=0)),
        )
        attn_metadata.shuffle_kv_block_indptr = indptr_gpu
        attn_metadata.shuffle_kv_block_indices = indices

    def _build_mla_chunk_meta_shuffle(
        self, batch: PrefillBatch, prefix: str = ""
    ) -> MLAChunkContextMetadata | None:
        """Block-aligned variant of ``_build_mla_chunk_meta`` for the shuffled
        KV layout.

        The shuffled gather reads whole ``block_size``-token blocks, so chunks
        are split along the global *block* axis (``attn_prefill_chunk_size //
        block_size`` blocks per chunk) rather than the token axis. Each chunk
        carries block-granular CSR (``shuffle_kv_block_indptr/indices``) plus
        token-granular ``cu_seqlens_k`` (output positions / flash_attn lens).

        All chunks are computed at once on the host as ``[num_chunks, bs]``
        arrays, then sent in a single upload; the per-chunk tensors are views
        into it.
        """
        block_size = self.model_runner.block_size
        chunk_blocks = max(1, self.attn_prefill_chunk_size // block_size)

        bs = batch.scheduled_bs
        cached = batch.cached_lens.astype(np.int64)
        per_seq_blocks = cdiv(cached, block_size)
        blk_offsets = np.zeros(bs + 1, dtype=np.int64)
        np.cumsum(per_seq_blocks, out=blk_offsets[1:])
        total_blocks = int(blk_offsets[-1])
        if total_blocks == 0:
            return None
        num_chunks = cdiv(total_blocks, chunk_blocks)

        # Global logical block list in per-seq CSR order: the leading cached
        # blocks of each row of the host block table (row-major mask order).
        # Scan only the columns this batch reaches, not the configured max.
        valid_cols = int(per_seq_blocks.max(initial=0))
        block_tables = self.model_runner.forward_vars[f"{prefix}block_tables"].np[
            :bs, :valid_cols
        ]
        col = np.arange(valid_cols)
        global_blocks = block_tables[col[None, :] < per_seq_blocks[:, None]]

        # [num_chunks, bs]: intersect each seq's block range with each chunk's.
        gb_start = np.arange(num_chunks, dtype=np.int64)[:, None] * chunk_blocks
        gb_end = np.minimum(gb_start + chunk_blocks, total_blocks)
        seq_lo = np.clip(blk_offsets[None, :-1], gb_start, gb_end)
        seq_hi = np.clip(blk_offsets[None, 1:], gb_start, gb_end)
        local_lo = seq_lo - blk_offsets[:-1]
        local_hi = seq_hi - blk_offsets[:-1]
        # Full blocks * block_size, clamped to the cached length for the seq's
        # final (partial) block.
        chunk_tokens = np.maximum(
            np.minimum(local_hi * block_size, cached) - local_lo * block_size, 0
        )

        # One packed upload: [blk_indptr | cu_seqlens_k | global_blocks].
        packed = np.zeros(2 * num_chunks * (bs + 1) + total_blocks, dtype=np.int32)
        blk_indptr = packed[: num_chunks * (bs + 1)].reshape(num_chunks, bs + 1)
        cu_k = packed[num_chunks * (bs + 1) : 2 * num_chunks * (bs + 1)].reshape(
            num_chunks, bs + 1
        )
        np.cumsum(seq_hi - seq_lo, axis=1, out=blk_indptr[:, 1:])
        np.cumsum(chunk_tokens, axis=1, out=cu_k[:, 1:])
        packed[2 * num_chunks * (bs + 1) :] = global_blocks
        packed_gpu = upload_numpy(packed, self.device)
        blk_indptr_gpu = packed_gpu[: num_chunks * (bs + 1)].view(num_chunks, bs + 1)
        cu_k_gpu = packed_gpu[num_chunks * (bs + 1) : 2 * num_chunks * (bs + 1)].view(
            num_chunks, bs + 1
        )
        blocks_gpu = packed_gpu[2 * num_chunks * (bs + 1) :]

        blk_indptr_list = list(blk_indptr_gpu.unbind(0))
        blk_indices_list = list(blocks_gpu.split(chunk_blocks))
        return MLAChunkContextMetadata(
            kv_indptr=blk_indptr_list,  # unused by shuffle gather; kept for parity
            kv_indices=blk_indices_list,
            cu_seqlens_k=list(cu_k_gpu.unbind(0)),
            total_tokens=cu_k[:, -1].tolist(),
            max_seqlen_k=chunk_tokens.max(axis=1).tolist(),
            num_chunks=num_chunks,
            k_workspace=self.k_chunk_workspace,
            v_workspace=self.v_chunk_workspace,
            shuffle_kv_block_indptr=blk_indptr_list,
            shuffle_kv_block_indices=blk_indices_list,
        )

    def _build_mla_chunk_meta(
        self, batch: PrefillBatch
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

        bs = batch.scheduled_bs
        cached_lens = batch.cached_lens.astype(np.int64)
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
        self, batch: PrefillBatch
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

        bs = batch.scheduled_bs
        cached_lens = batch.cached_lens.astype(np.int64)
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
        prefix: str,
    ) -> None:
        """Reduce the per-query sparse-prefill metadata to this PCP rank's
        1/pcp round-robin queries, its work plan into the ``prefix`` set.

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
        ks_padded = pcp_pad_dense(attn_metadata.cu_seqlen_ks, n_pad)
        attn_metadata.cu_seqlen_ks = ks_padded[owned_q].contiguous()
        # The whole batch's map, cut to this rank's.
        bids = attn_metadata.batch_id_per_q_token[:s_real]
        bid_padded = pcp_pad_dense(bids, n_pad)
        attn_metadata.batch_id_per_q_token = bid_padded[owned_q].contiguous()

        # --- one query per row (incl dummies) -> sparse_cu_seqlens_q = arange.
        var = self.model_runner.forward_vars
        attn_metadata.sparse_cu_seqlens_q = var["sparse_cu_seqlens_q"].gpu[
            : n_owned + 1
        ]

        # --- sparse_kv_indptr: cumsum of min(sparse_counts, topk); dummy -> 0.
        sparse_counts_t = torch.as_tensor(sparse_counts, device=device)
        owned_counts = pcp_pad_dense(sparse_counts_t, n_pad)[owned_q].to(torch.int64)
        owned_counts = torch.clamp(owned_counts, max=self.index_topk)
        indptr_owned = torch.zeros(n_owned + 1, dtype=torch.int32, device=device)
        indptr_owned[1:] = torch.cumsum(owned_counts, 0).to(torch.int32)
        attn_metadata.sparse_kv_indptr = indptr_owned

        # --- rebuild the sparse-prefill work buffers for the owned queries,
        #     one page per owned query.
        self._sparse_prefill_work_plan(
            attn_metadata,
            prefix,
            self.padded_num_attention_heads,
            var["sparse_kv_last_page_lens"].gpu[:n_owned],
        )

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

    def _build_token_block_tables(
        self, attn_metadata, running_tokens: int, max_seqlen_q: int
    ) -> None:
        """Give an indexer that reads its keys a query row at a time one
        block-table row per query token (`indexes_by_token_table`): its
        sequence's (`batch_id_per_q_token`). A step with one query per sequence
        already has that table, so only MTP verify gathers.
        """
        if not self.indexes_by_token_table:
            return
        block_tables = attn_metadata.block_tables
        if max_seqlen_q == 1:
            attn_metadata.token_block_tables = block_tables
            return
        rows = self._token_block_tables_gpu[:running_tokens]
        batch_ids = self.model_runner.forward_vars["batch_id_per_q_token"].gpu
        # A padding token (`-1`) reads the first sequence's row; its row end
        # admits no key.
        torch.index_select(
            block_tables, 0, batch_ids[:running_tokens].clamp(min=0), out=rows
        )
        attn_metadata.token_block_tables = rows

    def _concat_prefill_scored(self) -> bool:
        """Whether a prefill scores FP8 index rows against its requests' keys
        gathered end to end (`ConcatKeys`): every sparse indexer's but FP4's
        and a pooled one's."""
        return self.is_sparse and not self._indexer_fp4 and self.index_kpool == 1

    def _concat_requests(self, attn_metadata, bs, prefix, pcp_split, rows):
        """Host cumsums of the ``prefix`` set's requests' index rows (under PCP
        the rows this rank owns) and committed keys (`concat_rows`)."""
        var = self.model_runner.forward_vars
        cu_q = var[f"{prefix}cu_seqlens_q"].np[: bs + 1]
        if pcp_split:
            cu_q = pcp_owned_rows_by_request(
                cu_q, get_pcp_rank(), get_pcp_world_size(), rows
            )
        committed = "cu_seqlens_k" if attn_metadata.has_cached else "cu_seqlens_q"
        return cu_q, var[prefix + committed].np[: bs + 1]

    def _prefill_concat_plane(self) -> LogitsWorkspace | None:
        """The plane FP8 prefill bands are scored into; None when aiter's
        scorer allocates its own."""
        return self.index_logits_workspace if FP8_MQA_LOGITS_TAKES_OUT else None

    def _alloc_prefill_concat_keys(self) -> None:
        """Allocate the FP8 prefill indexer's gathered-key workspace: one
        request of the model length plus a chunk, and under DCP the
        all-gathered shards of that many keys. A step whose requests together
        hold more grows it (`ConcatKeyWorkspace`).

        Sized by the configuration, not the step: allocated per step these are
        as long as the step's context, so every chunk of a long prefill asks
        the caching allocator for a slightly larger block than the last,
        which it never has, until HBM runs out."""
        config = self.model_runner.config
        world = self.dcp_world_size
        keys = config.max_model_len + self.max_num_batched_tokens
        gathered = 0
        if world > 1:
            # A request's shard is padded to its interleave super-block.
            shard = -(-keys // world) + self.max_bs * self.cp_kv_cache_interleave_size
            gathered = world * shard
        self.index_prefill_key_workspace = ConcatKeyWorkspace(
            keys, config.hf_config.index_head_dim, dtypes.fp8, self.device, gathered
        )
        logger.info(
            "FP8 prefill indexer key workspace: %.2f GiB (%d keys)",
            self.index_prefill_key_workspace.nbytes / (1 << 30),
            keys,
        )

    def _prefill_concat_plane_capacity(self) -> int:
        """fp32 elements the logits plane needs for FP8 prefill bands: a
        rank's share of a step's rows (`RowSplit`) across a request of the
        model length, within the logits budget. 0 when aiter's scorer cannot
        write into a plane: its bands are then allocated per call."""
        if not FP8_MQA_LOGITS_TAKES_OUT:
            logger.warning(
                "aiter fp8_mqa_logits takes no logits buffer: FP8 prefill index "
                "logits are allocated per band, a block that grows with the "
                "context through a long prefill"
            )
            return 0
        rows = -(-self.max_num_batched_tokens // self.dcp_world_size)
        width = concat_plane_width(self.model_runner.config.max_model_len)
        return logits_rows(rows, width) * width

    def _publishes_decode_index_rows(self) -> bool:
        """Whether a decode step publishes its index rows (`IndexRows`): every
        sparse indexer's but DCP's, whose decode scores in its fused exchange."""
        return self.is_sparse and self.dcp_world_size == 1

    def _index_decode_width(self) -> int:
        """Columns of the indexer's decode plane: every key a request up to the
        model length has -- a pooled indexer's (kpool), every pool."""
        max_model_len = self.model_runner.config.max_model_len
        if self.index_kpool > 1:
            return pool_plane_width(max_model_len, self.index_kpool)
        return max_model_len

    def _decode_reads_row_ends(self) -> bool:
        """Whether decode rows read a bound of their own (`index_row_ends`)
        rather than their request's context: FP4's, and a pooled indexer's."""
        return self._publishes_decode_index_rows() and (
            self._indexer_fp4 or self.index_kpool > 1
        )

    def _closed_pools(self, keys: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """`decode_row_ends` on the device: the pools ``keys`` keys close,
        capped at the last the plane holds, into ``out``."""
        return torch.floor_divide(keys, self.index_kpool, out=out).clamp_max_(
            self._index_decode_width()
        )

    def _decode_index_rows(
        self,
        meta,
        bs,
        next_n,
        tokens,
        ubatch: str = "",
        row_ends=None,
    ) -> IndexRows | None:
        """`bs` decoding requests' index rows, ``tokens`` of them, at most
        `next_n` a request, over the step's published buffers (`meta`'s), each
        row to its bound -- the published `index_row_ends` of `ubatch` (its
        buffer prefix, the step's by default), unless ``row_ends``
        says otherwise: FP8 a request's rows reading its keys together, FP4 a
        request's rows scored together -- a decode is recorded, so its
        launches keep `ubatch`'s fixed homes (`Fp4LaunchStorage`) -- a pooled
        indexer's rows each off its own table row (`token_block_tables`)."""
        if not self._publishes_decode_index_rows():
            return None
        width = self._index_decode_width()
        workspace = self.index_logits_workspace
        if row_ends is None and self._decode_reads_row_ends():
            row_ends = self.model_runner.forward_vars[f"{ubatch}index_row_ends"].gpu
        if self.index_kpool > 1:
            return fp8_rows(
                row_ends[:tokens], width, workspace, meta.token_block_tables[:tokens]
            )
        if not self._indexer_fp4:
            return fp8_rows(
                meta.context_lens[:bs],
                width,
                workspace,
                meta.block_tables[:bs],
                next_n=next_n,
            )
        return fp4_rows(
            row_ends[:tokens],
            width,
            workspace,
            Fp4MqaRaggedMetadata(
                meta.cu_seqlens_q[: bs + 1], next_n, meta.block_tables
            ),
            heads=self.index_n_heads,
            page_size=self.model_runner.block_size,
            storage=self._fp4_launches[ubatch],
        )

    def _mtp_index_rows(
        self, rows: IndexRows | None, ubatch: str = ""
    ) -> IndexRows | None:
        """A pooled indexer's decode `rows` as an MTP layer reads them: its
        positions run a token behind, so each row sees the pools one position
        earlier closed -- `ubatch`'s `index_mtp_row_ends`."""
        if rows is None or self.index_kpool == 1:
            return None
        ends = self.model_runner.forward_vars[f"{ubatch}index_mtp_row_ends"].gpu
        return rows._replace(row_ends=ends[: rows.row_ends.shape[0]])

    def _stage_prefill_index_rows(
        self,
        bs: int,
        running_bs: int,
        row_ends: np.ndarray,
        pcp_split: bool,
        has_cached: bool,
        group,
        prefix: str,
    ) -> StagedIndexRows:
        """A prefill's index-row bounds off host `row_ends` (each query row's
        visible keys local to its request), on the host, into the ``prefix``
        set: under PCP the rows this rank owns."""
        if self.index_kpool > 1:
            return self._stage_prefill_pool_rows(
                bs, running_bs, row_ends, has_cached, group, prefix
            )
        var = self.model_runner.forward_vars
        query_starts = None
        if pcp_split:
            world, rank = get_pcp_world_size(), get_pcp_rank()
            padded = np.zeros(pcp_pad_len(len(row_ends), world), dtype=np.int32)
            padded[: len(row_ends)] = row_ends
            row_ends = padded[rank::world]
            if self._indexer_fp4:
                starts = pcp_owned_rows_by_request(
                    var[f"{prefix}cu_seqlens_q"].np[: bs + 1],
                    rank,
                    world,
                    len(row_ends),
                )
                query_starts = len(starts)
                var[f"{prefix}index_query_start_loc"].np[:query_starts] = starts
                group.set_counts({f"{prefix}index_query_start_loc": query_starts})
        rows = len(row_ends)
        var[f"{prefix}index_row_ends"].np[:rows] = row_ends
        group.set_counts({f"{prefix}index_row_ends": rows})
        return StagedIndexRows(
            rows, max(int(row_ends.max(initial=0)), 1), query_starts=query_starts
        )

    def _finish_prefill_index_rows(
        self,
        attn_metadata: AttentionMetaData,
        bs: int,
        staged: StagedIndexRows,
        total_kv: int,
        prefix: str,
        pcp_split: bool,
    ) -> None:
        """A prefill's index rows (`IndexRows`) over the ``prefix`` set's
        uploaded bounds: FP8 against the requests' keys gathered end to end,
        FP4 a request's rows together off its pages. Under DCP this rank's
        share of them.

        Reads `cu_seqlen_ks` as `_apply_pcp_reindex` leaves it, so runs after.
        """
        if self.index_kpool > 1:
            self._finish_prefill_pool_rows(attn_metadata, staged, prefix)
            return
        var = self.model_runner.forward_vars
        cu_seqlens_q = (
            attn_metadata.cu_seqlens_q[: bs + 1]
            if staged.query_starts is None
            else var[f"{prefix}index_query_start_loc"].gpu[: staged.query_starts]
        )
        rows = staged.rows
        ends = var[f"{prefix}index_row_ends"].gpu[:rows]
        split = RowSplit.of(rows, self.dcp_rank, self.dcp_world_size)
        if not self._indexer_fp4:
            keys = ConcatKeys(
                block_tables=attn_metadata.block_tables,
                cu_committed=(
                    attn_metadata.cu_seqlens_k
                    if attn_metadata.has_cached
                    else attn_metadata.cu_seqlens_q
                ),
                total=total_kv,
                row_starts=attn_metadata.cu_seqlen_ks,
                workspace=self.index_prefill_key_workspace,
            )
            attn_metadata.index_rows = concat_rows(
                ends,
                keys,
                split,
                workspace=self._prefill_concat_plane(),
                requests=self._concat_requests(
                    attn_metadata, bs, prefix, pcp_split, rows
                ),
            )
            return
        # A prefill is eager, so its plane is as wide as its longest row.
        attn_metadata.index_rows = self._prefill_fp4_rows(
            ends,
            staged.width,
            attn_metadata,
            cu_seqlens_q,
            attn_metadata.max_seqlen_q,
            split,
        )

    def _prefill_fp4_rows(
        self,
        ends,
        width,
        attn_metadata,
        cu_seqlens_q,
        max_seqlen_q,
        split,
    ) -> IndexRows:
        """FP4 prefill rows to `ends`, a request's rows scored together over
        its block-table row -- DCP's indexer table under DCP."""
        tables = (
            attn_metadata.dcp_indexer_fp4_block_tables
            if self.dcp_world_size > 1
            else attn_metadata.block_tables
        )
        return fp4_rows(
            ends,
            width,
            self.index_logits_workspace,
            Fp4MqaRaggedMetadata(cu_seqlens_q, max_seqlen_q, tables),
            heads=self.index_n_heads,
            page_size=self.model_runner.block_size,
            split=split,
        )

    @staticmethod
    def _pool_row_slices(running_bs: int, rows: int):
        """`index_pool_rows`' four runs: pool prefix (a span a block-table row,
        padding included), row starts, ends, MTP ends."""
        return (
            slice(0, running_bs + 1),
            slice(running_bs + 1, running_bs + 1 + rows),
            slice(running_bs + 1 + rows, running_bs + 1 + 2 * rows),
            slice(running_bs + 1 + 2 * rows, running_bs + 1 + 3 * rows),
        )

    def _stage_prefill_pool_rows(
        self,
        bs: int,
        running_bs: int,
        row_ends: np.ndarray,
        has_cached: bool,
        group,
        prefix: str,
    ) -> StagedIndexRows:
        """A pooled prefill's row bounds, on the host, into the ``prefix``
        set: each row against its request's closed pools, gathered end to end,
        up to the pools its own position closed -- and an MTP layer's, a token
        behind.

        `cp_gather_indexer_k_quant_cache` gives every block-table row a thread
        that claims the tokens of its pool span: a padding request's span is
        empty, so none claims a scheduled request's."""
        var = self.model_runner.forward_vars
        committed = "cu_seqlens_k" if has_cached else "cu_seqlens_q"
        pools = np.diff(var[prefix + committed].np[: bs + 1]) // self.index_kpool
        counts = np.diff(var[f"{prefix}cu_seqlens_q"].np[: bs + 1])
        rows = len(row_ends)
        host = var[f"{prefix}index_pool_rows"].np
        _, starts, ends, mtp_ends = self._pool_row_slices(running_bs, rows)
        host[0] = 0
        host[1 : bs + 1] = np.cumsum(pools)
        total = int(host[bs])
        host[bs + 1 : running_bs + 1] = total
        if total:
            host[starts] = np.repeat(host[:bs], counts)
            host[ends] = row_ends // self.index_kpool
            host[mtp_ends] = (row_ends - 1) // self.index_kpool
            group.set_counts({f"{prefix}index_pool_rows": mtp_ends.stop})
        return StagedIndexRows(rows, 0, pools=total)

    def _finish_prefill_pool_rows(
        self,
        attn_metadata: AttentionMetaData,
        staged_rows: StagedIndexRows,
        prefix: str,
    ) -> None:
        """The pooled prefill's index rows over the uploaded bounds; None when
        no request has closed a pool yet: nothing to score."""
        total = staged_rows.pools
        if not total:
            attn_metadata.index_rows = attn_metadata.mtp_index_rows = None
            return
        cu, starts, ends, mtp_ends = self._pool_row_slices(
            attn_metadata.block_tables.shape[0], staged_rows.rows
        )
        staged = self.model_runner.forward_vars[f"{prefix}index_pool_rows"].gpu
        keys = ConcatKeys(
            block_tables=attn_metadata.block_tables,
            cu_committed=staged[cu],
            total=total,
            row_starts=staged[starts],
        )
        attn_metadata.index_rows = concat_rows(staged[ends], keys)
        attn_metadata.mtp_index_rows = attn_metadata.index_rows._replace(
            row_ends=staged[mtp_ends]
        )

    def stage_decode(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
        group,
    ):
        self._check_metadata_writable(
            "mla_csr", "mla_sparse", "seq_metadata", "positions"
        )
        scheduled_bs = batch.total_seqs_num_decode
        dropout_p = 0.0

        var = self.model_runner.forward_vars
        context_lens = np.asarray(batch.context_lens, dtype=np.int32)
        block_tables = batch.block_tables
        if max_seqlen_q > 1:
            # Already in current batch order (prepare_input_ids).
            context_lens = self.model_runner.tokenID_processor.verify_ctx_lens(
                context_lens
            )
            num_blocks = cdiv(context_lens, self.model_runner.block_size)
            block_tables = [bt[:n] for bt, n in zip(block_tables, num_blocks)]
        _, lengths, cu_seqlens_q = self.decode_spans(batch)
        positions = decode_positions(context_lens, lengths, cu_seqlens_q)

        # Before the slots, not after: `slot_mapping` reads this packed table.
        # DCP still walks the trimmed ragged rows -- its slot is a per-rank
        # filter, not an address this table can answer.
        self.prepare_block_tables(batch, running_bs)

        if self.dcp_world_size > 1:
            # DCP round-robin: each new token is written only on the rank
            # owning its global pos.
            slots = [
                self._dcp_round_robin_slot(block_tables[request], pos)
                for request, pos in zip(
                    np.repeat(np.arange(scheduled_bs), lengths), positions
                )
            ]
        else:
            slots = slot_mapping(
                positions,
                lengths,
                var["block_tables"].np,
                self.model_runner.block_size,
                scratch=self.token_axis_scratch,
            )

        scheduled_tokens = batch.total_tokens_num_decode
        var["slot_mapping"].np[:scheduled_tokens] = slots
        var["positions"].np[:scheduled_tokens] = positions
        self._pad_tokens(scheduled_tokens, running_tokens)
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
                        positions,
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
                        context_lens,
                        batch.is_first_decode_without_local_prefill,
                    )
                )
            ]
        else:
            num_blocks_per_seq = cdiv(context_lens, self.block_size)
        kv_indptr = np.cumsum(num_blocks_per_seq)
        sum_blocks = kv_indptr[-1] if kv_indptr.size else 0

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
            ("kv_indptr", running_bs + 1),
            ("slot_mapping", running_tokens),
            ("context_lens", running_bs),
            ("kv_last_page_lens", running_bs),
        ]
        if self._publishes_dcp_local_lens:
            vars_used.append(("dcp_local_context_lens", running_tokens))

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

        max_model_len = self.model_runner.config.max_model_len
        max_seqlen_k = context_lens.max(initial=0)

        ctx = {el: var[el].gpu[:num] for el, num in vars_used}
        counts = dict(vars_used, positions=running_tokens)
        if self._decode_reads_row_ends():
            # Read through `index_rows`, not a metadata field of its own.
            row_ends = var["index_row_ends"].np
            row_ends[:scheduled_tokens] = decode_row_ends(
                positions, self.index_kpool, max_model_len
            )
            row_ends[scheduled_tokens:running_tokens] = 0
            counts["index_row_ends"] = running_tokens
        if self.index_kpool > 1:
            # Read through `mtp_index_rows`, not a metadata field of its own.
            mtp_ends = var["index_mtp_row_ends"].np
            mtp_ends[:scheduled_tokens] = decode_row_ends(
                positions - 1, self.index_kpool, max_model_len
            )
            mtp_ends[scheduled_tokens:running_tokens] = 0
            counts["index_mtp_row_ends"] = running_tokens
        if self.dcp_world_size > 1:
            counts["g_kv_indptr"] = running_bs + 1
        group.set_counts(counts)
        ctx["block_tables"] = block_table_state(var["block_tables"]).stage(
            running_bs, group
        )
        ctx["kv_indices"] = var["kv_indices"].gpu
        # A view: `prepare_query_metadata` already staged it this step, and
        # nothing here writes the host copy.
        ctx["cu_seqlens_q"] = var["cu_seqlens_q"].gpu[: running_bs + 1]
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
            var["g_kv_indptr"].gpu[: running_bs + 1]
            if self.dcp_world_size > 1
            else None
        )

        is_sparse_mtp = self.is_sparse and max_seqlen_q > 1
        if is_sparse_mtp or self._indexer_fp4:
            attn_metadata.sparse_cu_seqlens_q = var["sparse_cu_seqlens_q"].gpu[
                : running_tokens + 1
            ]
        if is_sparse_mtp:
            attn_metadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:running_tokens]
        elif self.is_sparse:
            # Non-MTP sparse decode (single token per seq): the sparse KV is
            # packed at page_size=1, so last_page_len is 1 for every seq. Expose
            # the all-1s buffer so _forward_decode passes it to mla_decode_fwd
            # instead of the dense per-block kv_last_page_lens (which would make
            # the kernel over-read past the written sparse-index region).
            attn_metadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:running_bs]
        return attn_metadata, var["positions"].gpu[:running_tokens]

    def finish_decode(
        self,
        batch: ScheduledBatch,
        attn_metadata: AttentionMetaData,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
    ) -> None:
        super().finish_decode(
            batch, attn_metadata, running_bs, running_tokens, max_seqlen_q
        )
        self._attach_selection(attn_metadata)
        scheduled_bs = batch.total_seqs_num_decode
        scheduled_tokens = batch.total_tokens_num_decode
        kv_indices_generate_triton(
            attn_metadata.block_tables,
            attn_metadata.kv_indices,
            attn_metadata.kv_indptr,
            self.block_ratio,
            attn_metadata.max_seqlen_k,
        )

        is_sparse_mtp = self.is_sparse and max_seqlen_q > 1
        if is_sparse_mtp:
            ctx_mla_ps = self.set_mla_persistent_worker_buffers(
                running_bs, max_seqlen_q
            )
            # Plan the sparse per-token work over the REAL tokens only, as the
            # cprr branch below does for its requests. This plan is not the
            # target's alone: the MTP drafter's step 0 runs the target's token
            # stream through its own sparse MLA layer reusing this metadata,
            # and that stream is NOT padded (eagle_proposer: "Step 0 carries
            # the target's stream and pads nothing"), so its q and o hold
            # scheduled_tokens rows. The persistent kernel reads q and writes o
            # for every work item, zero-KV padding included, so a plan over
            # running_tokens reads q and writes o past their ends whenever the
            # batch was padded: silent corruption, or a GPU memory fault when
            # the next page is unmapped (gfx1250 FlyDSL PS1, LLM-Booster #287).
            # The target itself runs padded q/o, and the kernel is
            # work_indptr-driven, so its padded rows are simply left unplanned.
            ctx_mla_ps_sparse = self._set_mla_persistent_worker_buffers_sparse_mtp(
                scheduled_tokens
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
            # The plan lays each sequence's queries `max_seqlen_q` rows apart
            # (`uni_seqlen_qo`); planned off `cu_seqlens_q` instead, aiter's
            # kernel still misreads some shapes.
            _, lengths, _ = self.decode_spans(batch)
            if (
                ctx_mla_ps
                and not query_is_uniform(lengths, max_seqlen_q)
                and self._decode_runs_persistent()
            ):
                raise ragged_decode_refusal(
                    "the persistent MLA decode", lengths, max_seqlen_q
                )
            ctx_mla_ps_sparse = {}
        for name, value in (*ctx_mla_ps.items(), *ctx_mla_ps_sparse.items()):
            setattr(attn_metadata, name, value)
        self._build_token_block_tables(attn_metadata, running_tokens, max_seqlen_q)
        attn_metadata.index_rows = self._decode_index_rows(
            attn_metadata, scheduled_bs, max_seqlen_q, batch.total_tokens_num
        )
        attn_metadata.mtp_index_rows = self._mtp_index_rows(attn_metadata.index_rows)

    def _decode_runs_persistent(self) -> bool:
        """Whether any MLA layer decodes on the persistent kernel over the
        step's dense work plan (`MLAAttention.decode_is_persistent`)."""
        sfc = self.model_runner.config.compilation_config.static_forward_context
        return any(
            isinstance(impl := getattr(module, "impl", None), MLAAttention)
            and impl.decode_is_persistent()
            for module in sfc.values()
        )

    def _ubatch_decode_rows(self):
        return (*super()._ubatch_decode_rows(), "kv_last_page_lens")

    def _ubatch_decode_indptrs(self):
        return ("kv_indptr", *(("sparse_kv_indptr",) if self.is_sparse else ()))

    def _ubatch_decode_token_rows(self):
        var = self.model_runner.forward_vars
        return (
            *super()._ubatch_decode_token_rows(),
            *(name for name in _UBATCH_INDEX_TOKEN_ROWS if name in var),
        )

    def _capture_layout_counts(self, bs: int, max_seqlen_q: int) -> dict[str, int]:
        # One key a request is a whole page with a one-token last page: the
        # split-KV asm kernel takes `page_count - (tail_len != 0)` full pages,
        # which an empty request underflows into a loop that never exits.
        counts = super()._capture_layout_counts(bs, max_seqlen_q)
        var = self.model_runner.forward_vars
        tokens = bs * max_seqlen_q
        var["kv_last_page_lens"].np[:bs] = 1
        counts["kv_last_page_lens"] = bs
        if self.is_sparse:
            # One selected key a row: a request's (decode), a token's (verify).
            var["sparse_kv_indptr"].np[: tokens + 1] = np.arange(
                tokens + 1, dtype=np.int32
            )
            counts["sparse_kv_indptr"] = tokens + 1
        if self._publishes_dcp_local_lens:
            var["dcp_local_context_lens"].np[:tokens] = 1
            counts["dcp_local_context_lens"] = tokens
        if self.dcp_world_size > 1 and max_seqlen_q > 1:
            # The cprr kernel masks a verify on GLOBAL positions, which only
            # `g_kv_indptr` gives it: one local key a rank is a global length
            # of `dcp_world_size`, the round-robin being token-level.
            var["g_kv_indptr"].np[: bs + 1] = (
                np.arange(bs + 1, dtype=np.int32) * self.dcp_world_size
            )
            counts["g_kv_indptr"] = bs + 1
        # A row's index bound: its one key, or a pooled indexer's closed pools,
        # none.
        if "index_row_ends" in var:
            var["index_row_ends"].np[:tokens] = 0 if self.index_kpool > 1 else 1
            counts["index_row_ends"] = tokens
        if "index_mtp_row_ends" in var:
            var["index_mtp_row_ends"].np[:tokens] = 0
            counts["index_mtp_row_ends"] = tokens
        return counts

    def build_for_cudagraph_capture(
        self, bs: int, *, decode_ubatch_slices=None
    ) -> AttentionMetaData:
        self._check_metadata_writable("mla_csr")
        var = self.model_runner.forward_vars
        max_q_len = var["mtp_k"] + 1 if "mtp_k" in var else 1
        self.publish_capture_layout(bs, max_q_len)
        sparse_kv_indptr = var["sparse_kv_indptr"].gpu if self.is_sparse else None
        scheduled_tokens = bs * max_q_len
        is_sparse_mtp = self.is_sparse and max_q_len > 1
        cp_round_robin = self.dcp_world_size > 1 and max_q_len > 1
        dcp_local_context_lens = None
        if self._publishes_dcp_local_lens:
            dcp_local_context_lens = var["dcp_local_context_lens"].gpu[
                :scheduled_tokens
            ]
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
        attn_metadata = AttentionMetaData(
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
        attn_metadata.dtype_q = self.dtype_q
        # Attach the round-robin CP global kv_indptr for the captured graph so
        # replay (which overwrites the buffer with real values) matches. Only
        # consumed by _forward_decode when dcp>1 and max_q_len>1.
        attn_metadata.g_kv_indptr = (
            var["g_kv_indptr"].gpu[: bs + 1] if self.dcp_world_size > 1 else None
        )
        if ctx_mla_ps_sparse is not None:
            for k, v in ctx_mla_ps_sparse.items():
                setattr(attn_metadata, k, v)
        if is_sparse_mtp or self._indexer_fp4:
            attn_metadata.sparse_cu_seqlens_q = var["sparse_cu_seqlens_q"].gpu[
                : scheduled_tokens + 1
            ]
        if is_sparse_mtp:
            attn_metadata.sparse_kv_indptr = var["sparse_kv_indptr"].gpu[
                : scheduled_tokens + 1
            ]
            attn_metadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:scheduled_tokens]
            # Staged with the capture's query prefix (`ModelRunner`).
            attn_metadata.batch_id_per_q_token = var["batch_id_per_q_token"].gpu[
                :scheduled_tokens
            ]
        elif self.is_sparse:
            # Non-MTP sparse decode capture: all-1s per-token last-page lens,
            # matching prepare_decode so _forward_decode reads the sparse buffer.
            attn_metadata.sparse_kv_last_page_lens = var[
                "sparse_kv_last_page_lens"
            ].gpu[:bs]
        self._build_token_block_tables(attn_metadata, scheduled_tokens, max_q_len)
        self._attach_selection(attn_metadata)
        attn_metadata.index_rows = self._decode_index_rows(
            attn_metadata, bs, max_q_len, scheduled_tokens
        )
        attn_metadata.mtp_index_rows = self._mtp_index_rows(attn_metadata.index_rows)
        if decode_ubatch_slices:
            self.capture_ubatch_decode(
                attn_metadata, bs, decode_ubatch_slices, max_q_len
            )
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
        return attn_metadata, context

    def _ubatch_decode_metadata(
        self, ubatch_idx: int, ub: UBatchSlice, max_seqlen_q: int
    ) -> AttentionMetaData:
        attn = super()._ubatch_decode_metadata(ubatch_idx, ub, max_seqlen_q)
        var = self.model_runner.forward_vars
        p = f"ub{ubatch_idx}_"
        running_bs = ub.running_bs
        running_tokens = ub.running_tokens
        attn.kv_last_page_lens = var[f"{p}kv_last_page_lens"].gpu[:running_bs]
        if self._publishes_dcp_local_lens:
            attn.dcp_local_context_lens = var[f"{p}dcp_local_context_lens"].gpu[
                :running_tokens
            ]
        if self.is_sparse:
            attn.sparse_kv_indptr = var[f"{p}sparse_kv_indptr"].gpu[: running_bs + 1]
            attn.sparse_kv_last_page_lens = var["sparse_kv_last_page_lens"].gpu[
                :running_bs
            ]
        attn.dtype_q = self.dtype_q
        self._attach_selection(attn, ub.token_slice.start, ubatch_idx)
        if self.indexes_by_token_table:
            # Config refuses decode TBO with speculation on a sparse model, so a
            # ubatch runs one query per sequence.
            attn.token_block_tables = attn.block_tables
        if self._indexer_fp4:
            attn.sparse_cu_seqlens_q = var["sparse_cu_seqlens_q"].gpu[
                : running_tokens + 1
            ]
        return attn

    def finish_ubatch(
        self,
        ubatch_idx: int,
        ub: UBatchSlice,
        ub_attn: AttentionMetaData,
        max_seqlen_q: int,
    ) -> None:
        super().finish_ubatch(ubatch_idx, ub, ub_attn, max_seqlen_q)
        prefix = f"ub{ubatch_idx}_"
        self.set_mla_persistent_worker_buffers(
            ub.running_bs, max_seqlen_q, prefix=prefix
        )
        ub_attn.index_rows = self._decode_index_rows(
            ub_attn,
            ub.running_bs,
            max_seqlen_q,
            ub.running_tokens,
            ubatch=prefix,
        )
        ub_attn.mtp_index_rows = self._mtp_index_rows(ub_attn.index_rows, prefix)
