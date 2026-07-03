from typing import Optional
import logging

from dataclasses import dataclass

import torch

from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.dist.parallel_state import get_dp_group, get_tp_group
from aiter.jit.utils.chip_info import get_gfx
from atom.config import get_current_atom_config
from atom.model_ops.attention_mla import _MLA_MIN_HEADS
from atom.plugin.vllm.attention.layer_mla import (
    disabled_mla_persistent_metadata,
    mla_fold_kv_metadata_triton,
)
from atom.utils import CpuGpuBuffer
from atom.utils.block_convert import kv_indices_generate_triton
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
)

logger = logging.getLogger("atom")

_PARTITION_SIZE_ROCM = 256
_CP_TOKENS_PER_ITER_ROCM = 32 * 1024


def get_aiter_kv_cache_dtype(config) -> torch.dtype:
    kv_cache_dtype = config.cache_config.cache_dtype
    if kv_cache_dtype == "auto":
        kv_cache_dtype = "bf16"
    elif kv_cache_dtype == "bfloat16":
        kv_cache_dtype = "bf16"
    elif kv_cache_dtype == "float16":
        kv_cache_dtype = "fp16"
    return dtypes.d_dtypes[kv_cache_dtype]


@dataclass
class AiterMhaPhaseMetadata:
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor


@dataclass
class AiterChunkSlidingWindowMetadata:
    swa_seqlens: torch.Tensor
    swa_cu_seqlens: torch.Tensor
    swa_seq_starts: torch.Tensor
    swa_token_to_batch: torch.Tensor
    swa_max_seqlens: int
    swa_total_tokens: int
    swa_workspace: torch.Tensor


@dataclass
class AiterChunkContextMetadata:
    workspace: torch.Tensor
    cu_seq_lens_chunk: torch.Tensor
    chunk_starts: torch.Tensor
    token_to_batch: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]
    seq_lens: torch.Tensor
    num_chunks: int
    total_token_per_batch: list[int]
    swa_metadata: Optional[AiterChunkSlidingWindowMetadata] = None


@dataclass
class AiterChunkPrefillMetadata:
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    chunk_context_metadata: AiterChunkContextMetadata


@dataclass
class AiterMhaMetadataForVllm:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    num_actual_kv_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor

    # prefill and decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    num_extends: int
    num_extend_tokens: int
    dropout_p: float = 0.0

    decode_metadata: Optional[AiterMhaPhaseMetadata] = None
    prefill_metadata: Optional[AiterMhaPhaseMetadata] = None
    extend_metadata: Optional[AiterChunkPrefillMetadata] = None

    use_cascade: bool = False
    common_prefix_len: int = 0
    total_tokens: int = 0


@dataclass
class AiterMlaDecodeMetadataForVllm:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    dcp_tot_seq_lens: torch.Tensor | None
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: torch.Tensor | None = None
    # The page indices of the paged kv cache
    paged_kv_indices: torch.Tensor | None = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: torch.Tensor | None = None
    # The query indptr, shape : [num_decode + 1]
    qo_indptr: torch.Tensor | None = None
    # The dtype of MLA out tensor
    attn_out_dtype: torch.dtype = torch.bfloat16
    # The max query output length: int
    max_qo_len: int | None = None
    # Whether dense MLA persistent metadata was built for this decode batch.
    use_persistent_metadata: bool = False
    # The fold factor for handling mqa_ratio=64 in non-persistent mode
    fold_factor: int | None = None
    # Fold buffers for the MLA nhead-fold workaround. These are populated by
    # the metadata builder outside the CUDA graph capture region
    fold_kv_indptr: torch.Tensor | None = None
    fold_kv_indices: torch.Tensor | None = None
    fold_qo_indptr: torch.Tensor | None = None
    fold_kv_last_page_len: torch.Tensor | None = None


@dataclass
class AiterMlaPersistentMetadataForVllm:
    # All fields are None when persistent metadata is disabled
    # (see disabled_mla_persistent_metadata()), e.g. under DP.
    work_meta_data: torch.Tensor | None
    work_indptr: torch.Tensor | None
    work_info_set: torch.Tensor | None
    reduce_indptr: torch.Tensor | None
    reduce_final_map: torch.Tensor | None
    reduce_partial_map: torch.Tensor | None


@dataclass
class AiterMlaPrefillMetadataForVllm:
    """Prefill Specific Metadata"""

    @dataclass
    class AiterMlaChunkedContextMetadataForVllm:
        # New for MLA (compared to FlashAttention)
        # For handling chunked prefill
        cu_seq_lens: torch.Tensor
        starts: torch.Tensor
        seq_tot: list[int]
        max_seq_lens: list[int]
        seq_lens: torch.Tensor
        workspace: torch.Tensor
        token_to_seq: torch.Tensor
        chunk_total_token: list[int]
        prefill_tokens_with_context: int | None = None

        # for mla DCP
        padded_local_chunk_seq_lens: list[list[int]] | None = None
        local_context_lens_allranks: list[list[int]] | None = None
        padded_local_cu_seq_lens: torch.Tensor | None = None
        cu_seq_lens_lst: list[list[int]] | None = None
        chunk_size: int | None = None

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    max_query_len: int
    chunked_context: AiterMlaChunkedContextMetadataForVllm | None = None
    query_seq_lens: torch.Tensor | None = None
    workspace_buffer: torch.Tensor | None = None
    q_data_type: torch.dtype | None = None
    output_dtype: torch.dtype | None = None


@dataclass
class AiterMlaMetadataForVllm:
    """vLLM metadata for ATOM MLA attention.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # The dimension of the attention heads
    head_dim: int | None = None

    decode: AiterMlaDecodeMetadataForVllm | None = None
    prefill: AiterMlaPrefillMetadataForVllm | None = None
    persistent_metadata: AiterMlaPersistentMetadataForVllm | None = None


@dataclass
class AiterMlaSparseIndexerPrefillChunkMetadataForVllm:
    block_table: torch.Tensor
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int


@dataclass
class AiterMlaSparseIndexerPrefillMetadataForVllm:
    chunks: list[AiterMlaSparseIndexerPrefillChunkMetadataForVllm]


@dataclass
class AiterMlaSparseIndexerDecodeMetadataForVllm:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor
    use_large_context_topk: bool
    offsets: torch.Tensor | None  # Precomputed offsets for speculative decoding


@dataclass
class AiterMlaSparseIndexerMetadataForVllm:
    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor

    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    # The dimension of the attention heads
    head_dim: int

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: AiterMlaSparseIndexerDecodeMetadataForVllm | None = None
    prefill: AiterMlaSparseIndexerPrefillMetadataForVllm | None = None


# TODO (zyongye) optimize this, this is now vibe coded
def kv_spans_from_batches(
    start_seq_loc: torch.Tensor, seq_len_per_batch: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      start_seq_loc: 1D long tensor [B+1], cumulative counts of
                     selected tokens per batch.
            Example: [0, 2, 4, 7] ->
                     batch sizes (selected) [2, 2, 3], N=7 tokens total.
      seq_len_per_batch: 1D long tensor [B],
                         full sequence length (KV length) of each batch.
                         Example: [5, 9, 4].

    Returns:
      start_tensor: 1D long tensor [N], start offset in the
                    concatenated KV cache for each token's batch.
      end_location: 1D long tensor [N],
                    **exclusive** end = start + token's local position.
                    (So the attended KV slice is kv[start:end].)

    Assumes each batch contributes its full `seq_len_per_batch[i]`
    keys to the KV cache, andthe selected tokens within a batch
    are the **last** `counts[i]` positions of that sequence.
    """
    q = start_seq_loc.to(dtype=torch.long)
    L = seq_len_per_batch.to(dtype=torch.long)
    assert q.dim() == 1 and L.dim() == 1
    assert q.numel() == L.numel() + 1, "start_seq_loc must have length B+1"

    # Selected tokens per batch and totals
    counts = q[1:] - q[:-1]  # [B]
    N = int(q[-1].item())  # total selected tokens
    B = L.numel()

    if N == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    # KV start offsets per batch in the concatenated KV cache
    kv_starts_per_batch = torch.cumsum(L, dim=0) - L  # [B]

    # For each selected token, which batch does it belong to?
    batch_id = torch.repeat_interleave(torch.arange(B), counts)  # [N]

    # Map batch KV start to each token
    start_tensor = kv_starts_per_batch[batch_id]  # [N]

    # End-align local positions inside each batch:
    # local_pos = L[b] - counts[b] + (1..counts[b])  for each batch b
    L_expand = torch.repeat_interleave(L, counts)  # [N]
    m_expand = torch.repeat_interleave(counts, counts)  # [N]
    # position within the selected block: 1..counts[b]
    pos_within = (
        torch.arange(N, dtype=torch.long) - torch.repeat_interleave(q[:-1], counts) + 1
    )

    local_pos = L_expand - m_expand + pos_within  # [N], 1-based
    end_location = start_tensor + local_pos  # exclusive end

    return start_tensor.int().to(device), end_location.int().to(device)


def get_max_prefill_buffer_size(max_model_len: int):
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40


@dataclass
class AiterMlaSparseMetadataForVllm:
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    seq_lens: torch.Tensor

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor

    qo_indptr: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_indptr: torch.Tensor
    attn_out_dtype: torch.dtype

    block_size: int = 1
    topk_tokens: int = 2048

    work_meta_data: torch.Tensor | None = None
    work_indptr: torch.Tensor | None = None
    work_info_set: torch.Tensor | None = None
    reduce_indptr: torch.Tensor | None = None
    reduce_final_map: torch.Tensor | None = None
    reduce_partial_map: torch.Tensor | None = None


@dataclass
class MinimaxM3SparsePrefillMetadata:
    qo_indptr: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int


@dataclass
class MinimaxM3SparseDecodeMetadata:
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int = 1


@dataclass
class MinimaxM3SparseMetadata:
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    block_table: torch.Tensor
    max_query_len: int
    prefill: MinimaxM3SparsePrefillMetadata | None = None
    decode: MinimaxM3SparseDecodeMetadata | None = None


class MinimaxM3SparseAttentionMetadataBuilder(AttentionMetadataBuilder):
    # MiniMax-M3 sparse attention owns dynamic index/topk/cache state that must
    # be refreshed outside full cudagraph capture.
    _cudagraph_support = AttentionCGSupport.NEVER
    reorder_batch_threshold = 1

    def __init__(
        self,
        kv_cache_spec=None,
        layer_names=None,
        config=None,
        device=None,
        model_runner=None,
    ):
        del model_runner
        super().__init__(kv_cache_spec, layer_names, config, device)
        logger.info("init MinimaxM3SparseAttentionMetadataBuilder")
        from vllm.config import VllmConfig

        assert isinstance(config, VllmConfig)
        self.vllm_config = config
        self.model_config = config.model_config
        self.cache_config = config.cache_config
        self.scheduler_config = config.scheduler_config
        self.block_size = kv_cache_spec.block_size
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(
        self,
        common_prefix_len: int = 0,
        common_attn_metadata=None,
        fast_build: bool = False,
    ):
        del fast_build
        if common_prefix_len > 0:
            raise ValueError("ATOM does not support cascade attention yet")
        assert common_attn_metadata is not None

        from atom.model_ops.minimax_m3.sparse_attn import SPARSE_BLOCK_SIZE
        from vllm.v1.attention.backends.utils import (
            split_decodes_prefills_and_extends,
        )

        (
            num_decodes,
            num_extends,
            num_prefills,
            num_decode_tokens,
            _num_extend_tokens,
            _num_prefill_tokens,
        ) = split_decodes_prefills_and_extends(
            common_attn_metadata=common_attn_metadata,
            decode_threshold=getattr(self, "reorder_batch_threshold", 1) or 1,
        )
        num_tokens = common_attn_metadata.num_actual_tokens
        num_prefills_total = num_extends + num_prefills
        num_prefill_tokens = num_tokens - num_decode_tokens
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor

        prefill_metadata: MinimaxM3SparsePrefillMetadata | None = None
        if num_prefills_total > 0:
            # MiniMax-M3 sparse attention uses the prefill kernel for any mixed
            # decode+prefill batch, because it builds per-token causal sparse
            # block tables. Only pure decode batches use the decode kernel.
            # The vLLM scheduler orders request rows as decode, extend, prefill.
            # The prefill metadata below must therefore start after decode rows
            # and must shift query_start_loc back to this phase's local token
            # slice; otherwise sparse prefill reads decode requests as prefixes.
            prefill_start = num_decodes
            prefill_stop = prefill_start + num_prefills_total
            prefill_token_start = num_decode_tokens
            prefill_seq_lens = seq_lens[prefill_start:prefill_stop]
            prefill_query_start = common_attn_metadata.query_start_loc[
                prefill_start : prefill_stop + 1
            ].to(torch.int32)
            prefill_query_start = prefill_query_start - prefill_token_start
            context_lens = common_attn_metadata.compute_num_computed_tokens()[
                prefill_start:prefill_stop
            ]
            seq_lens_cpu = getattr(
                common_attn_metadata, "seq_lens_cpu_upper_bound", None
            )
            if seq_lens_cpu is None:
                seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
            if seq_lens_cpu is None:
                seq_lens_cpu = seq_lens.cpu()
            prefill_max_seq_len = int(
                seq_lens_cpu[prefill_start:prefill_stop].max().item()
            )
            query_lens_cpu = (
                common_attn_metadata.query_start_loc_cpu[1:]
                - common_attn_metadata.query_start_loc_cpu[:-1]
            )
            prefill_max_query_len = max(
                1, int(query_lens_cpu[prefill_start:prefill_stop].max().item())
            )
            if self.block_size != SPARSE_BLOCK_SIZE:
                raise ValueError(
                    f"MiniMax-M3 sparse block size must be {SPARSE_BLOCK_SIZE}."
                )
            qo_indptr = torch.arange(
                num_prefill_tokens + 1, dtype=torch.int32, device=seq_lens.device
            )
            prefill_metadata = MinimaxM3SparsePrefillMetadata(
                qo_indptr=qo_indptr,
                cu_seqlens_q=prefill_query_start,
                seq_lens=prefill_seq_lens,
                context_lens=context_lens,
                block_table=block_table[prefill_start:prefill_stop],
                max_query_len=prefill_max_query_len,
                max_seq_len=prefill_max_seq_len,
            )

        decode_metadata: MinimaxM3SparseDecodeMetadata | None = None
        if num_decodes > 0:
            query_lens_cpu = (
                common_attn_metadata.query_start_loc_cpu[1:]
                - common_attn_metadata.query_start_loc_cpu[:-1]
            )
            decode_max_query_len = max(
                1, int(query_lens_cpu[:num_decodes].max().item())
            )
            decode_metadata = MinimaxM3SparseDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
                max_query_len=decode_max_query_len,
            )

        return MinimaxM3SparseMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills_total,
            num_prefill_tokens=num_prefill_tokens,
            block_table=block_table,
            max_query_len=common_attn_metadata.max_query_len,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata=None):
        return self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )


# vLLM metadata builders
class AiterMhaMetadataBuilderForVllm(AttentionMetadataBuilder):
    """vLLM-only MHA metadata builder."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold = 1

    def __init__(
        self,
        kv_cache_spec=None,
        layer_names=None,
        config=None,
        device=None,
        model_runner=None,
    ):
        super().__init__(kv_cache_spec, layer_names, config, device)
        logger.info("init AiterMhaMetadataBuilderForVllm")
        from vllm.config import VllmConfig, get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

        assert isinstance(config, VllmConfig)

        self.vllm_config = config
        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.cache_config = config.cache_config

        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.head_dim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.aot_sliding_window: tuple[int, int] | None = None
        self.total_tokens: int = 0

        self.scheduler_config = config.scheduler_config
        self.block_ratio = 1

        sliding_window_sizes: set[tuple[int, int] | None] = set()
        layers = get_layers_from_vllm_config(config, AttentionLayerBase, layer_names)
        for layer in layers.values():
            from atom.plugin.vllm.attention.layer import AttentionForVllmMHA

            assert isinstance(layer, AttentionForVllmMHA)
            sliding_window = layer.sliding_window
            if sliding_window is None or sliding_window == -1:
                sliding_window_sizes.add(None)
            elif isinstance(sliding_window, tuple):
                sliding_window_sizes.add(sliding_window)
            else:
                sliding_window_sizes.add((sliding_window - 1, 0))

        while len(sliding_window_sizes) > 0:
            sliding_window_config = sliding_window_sizes.pop()
            if sliding_window_config is not None and sliding_window_config[0] != -1:
                assert (
                    self.aot_sliding_window is None
                ), "Aiter Backend only support one valid sliding window"
                self.aot_sliding_window = sliding_window_config

        self.extend_workspace = torch.empty(
            [2, _CP_TOKENS_PER_ITER_ROCM, self.num_heads_kv, self.head_dim],
            dtype=self.model_config.dtype,
            device=device,
        )
        workspace_bytes = (
            2
            * _CP_TOKENS_PER_ITER_ROCM
            * self.num_heads_kv
            * self.head_dim
            * torch.tensor([], dtype=self.model_config.dtype).element_size()
        )
        workspace_mib = workspace_bytes / (1024 * 1024)
        logger.warning(
            "ATOM allocates extend_workspace outside vLLM memory accounting: "
            "shape=%s dtype=%s size=%.2f MiB. "
            "This untracked GPU memory can increase OOM risk when "
            "gpu_mem_utilization is high.",
            tuple(self.extend_workspace.shape),
            self.model_config.dtype,
            workspace_mib,
        )

        max_num_batched_tokens = config.scheduler_config.max_num_batched_tokens
        i64_kwargs = {"dtype": torch.int64, "device": device}
        self.positions = CpuGpuBuffer(max_num_batched_tokens, **i64_kwargs)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(
        self,
        common_prefix_len: int = 0,
        common_attn_metadata=None,
        fast_build: bool = False,
    ):
        if common_prefix_len > 0:
            raise ValueError("ATOM does not support cascade attention yet")

        from vllm.v1.attention.backends.utils import split_decodes_prefills_and_extends

        # decode_threshold tracks reorder_batch_threshold so MTP/EAGLE
        # multi-token verification (query_len > 1) routes through decode.
        decode_threshold = getattr(self, "reorder_batch_threshold", 1) or 1
        split_ret = split_decodes_prefills_and_extends(
            common_attn_metadata=common_attn_metadata,
            decode_threshold=decode_threshold,
        )

        (
            num_decodes,
            num_extends,
            num_prefills,
            num_decode_tokens,
            num_extend_tokens,
            num_prefill_tokens,
        ) = split_ret

        prefill_only = num_decodes == 0 and num_extends == 0 and num_prefills > 0
        decode_only = num_decodes > 0 and num_extends == 0 and num_prefills == 0
        mixed = not (prefill_only or decode_only)

        # common_attn_metadata._seq_lens_cpu is equal to common_attn_metadata.seq_lens.cpu(),
        # but using seq_lens.cpu() can get the better performance in low concurrency.
        # seq_lens = common_attn_metadata._seq_lens_cpu
        seq_lens = common_attn_metadata.seq_lens.cpu()
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        prefill_max_query_len = decode_max_query_len = (
            common_attn_metadata.max_query_len
        )
        prefill_max_seq_len = decode_max_seq_len = common_attn_metadata.max_seq_len
        prefill_query_start_loc = decode_query_start_loc = (
            common_attn_metadata.query_start_loc
        )

        if mixed:
            prefill_start = num_decodes + num_extends
            if num_prefills > 0:
                prefill_max_query_len = query_lens_cpu[prefill_start:].max().item()
                prefill_max_seq_len = seq_lens[prefill_start:].max().item()
                prefill_query_start_loc = (
                    prefill_query_start_loc[prefill_start:]
                    - prefill_query_start_loc[prefill_start]
                )
            if num_decodes > 0:
                decode_max_query_len = query_lens_cpu[:num_decodes].max().item()
                decode_max_seq_len = seq_lens[:num_decodes].max().item()
                decode_query_start_loc = decode_query_start_loc[: num_decodes + 1]

        prefill_metadata = None
        decode_metadata = None
        extend_metadata = None

        if num_prefills > 0:
            prefill_metadata = AiterMhaPhaseMetadata(
                max_query_len=prefill_max_query_len,
                max_seq_len=prefill_max_seq_len,
                query_start_loc=prefill_query_start_loc,
            )

        if num_decodes > 0:
            decode_metadata = AiterMhaPhaseMetadata(
                max_query_len=decode_max_query_len,
                max_seq_len=decode_max_seq_len,
                query_start_loc=decode_query_start_loc,
            )

        if num_extends > 0:
            num_extends_slice = slice(num_decodes, num_decodes + num_extends)
            query_lens_extend = query_lens_cpu[num_extends_slice]
            seq_lens_extend = seq_lens[num_extends_slice]
            # In DBO, the second ubatch's continuation request keeps the full
            # seq_len but has its query_len reduced by split_attn_metadata, so
            # use seq_len - query_len to correctly count the KV that precedes
            # this ubatch's queries
            computed_kv_lens = seq_lens_extend - query_lens_extend

            swa_metadata = None
            if self.aot_sliding_window is not None:
                swa_seqlen_for_extend = torch.minimum(
                    seq_lens_extend,
                    query_lens_extend + self.aot_sliding_window[0] + 1,
                )
                cu_seq_lens = torch.zeros(
                    num_extends + 1,
                    dtype=torch.int32,
                    device=seq_lens_extend.device,
                )
                torch.cumsum(
                    swa_seqlen_for_extend,
                    dim=0,
                    dtype=cu_seq_lens.dtype,
                    out=cu_seq_lens[1:],
                )
                token_to_seq = torch.arange(
                    0,
                    num_extends,
                    dtype=torch.int32,
                    device=seq_lens_extend.device,
                )
                token_to_seq = torch.repeat_interleave(
                    token_to_seq, swa_seqlen_for_extend
                )
                fetched_shape = cu_seq_lens[-1].item()
                swa_workspace = torch.empty(
                    (2, fetched_shape, self.num_heads_kv, self.head_dim),
                    dtype=self.vllm_config.model_config.dtype,
                    device=self.device,
                )

                seq_starts = seq_lens_extend - swa_seqlen_for_extend
                max_seqlen_k = swa_seqlen_for_extend.max().item()
                total_tokens = cu_seq_lens[-1].item()

                swa_metadata = AiterChunkSlidingWindowMetadata(
                    swa_seqlens=swa_seqlen_for_extend.to(
                        self.device, non_blocking=True
                    ),
                    swa_cu_seqlens=cu_seq_lens.to(self.device, non_blocking=True),
                    swa_seq_starts=seq_starts.to(self.device, non_blocking=True),
                    swa_token_to_batch=token_to_seq.to(self.device, non_blocking=True),
                    swa_max_seqlens=max_seqlen_k,
                    swa_total_tokens=total_tokens,
                    swa_workspace=swa_workspace,
                )

            # allocate the equal amount of workspace for
            # each chunk prefill request
            max_context_chunk = _CP_TOKENS_PER_ITER_ROCM // num_extends
            from vllm.utils.math_utils import cdiv

            num_chunks = cdiv(computed_kv_lens.max().item(), max_context_chunk)

            chunk_starts = (
                torch.arange(num_chunks, dtype=torch.int32)
                .unsqueeze(1)
                .expand(-1, num_extends)
                * max_context_chunk
            )
            chunk_ends = torch.min(
                computed_kv_lens.unsqueeze(0), chunk_starts + max_context_chunk
            )
            chunk_seq_lens = (chunk_ends - chunk_starts).clamp(
                min=0
            )  # [num_chunks, num_extends]
            cu_seq_lens_cpu = torch.zeros(
                [num_chunks, num_extends + 1], dtype=torch.int32, pin_memory=True
            )
            torch.cumsum(
                chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:], dtype=torch.int32
            )
            max_cum_tokens = cu_seq_lens_cpu[:, -1].max().item()

            # Build token->batch mapping robustly, even with zero-length batches.
            token_to_batch_tensor = torch.zeros(
                (num_chunks, max_cum_tokens), dtype=torch.int32, pin_memory=True
            )
            batch_ids = torch.arange(num_extends, dtype=torch.int32)
            for chunk_idx in range(num_chunks):
                total_tokens = cu_seq_lens_cpu[chunk_idx, -1].item()
                if total_tokens == 0:
                    continue
                token_to_batch = torch.repeat_interleave(
                    batch_ids, chunk_seq_lens[chunk_idx].to(torch.int64)
                )
                token_to_batch_tensor[chunk_idx, :total_tokens] = token_to_batch

            chunk_context_metadata = AiterChunkContextMetadata(
                workspace=self.extend_workspace,
                cu_seq_lens_chunk=cu_seq_lens_cpu.to(self.device, non_blocking=True),
                chunk_starts=chunk_starts.to(self.device, non_blocking=True),
                seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                seq_lens=chunk_seq_lens,
                token_to_batch=token_to_batch_tensor.to(self.device, non_blocking=True),
                num_chunks=num_chunks,
                total_token_per_batch=cu_seq_lens_cpu[:, -1].tolist(),
                swa_metadata=swa_metadata,
            )

            query_start_loc_device = common_attn_metadata.query_start_loc[
                num_decodes : num_decodes + num_extends + 1
            ]
            seq_lens_device = common_attn_metadata.seq_lens[num_extends_slice]
            cu_seq_lens = torch.zeros(
                num_extends + 1, dtype=torch.int32, device=seq_lens_device.device
            )
            torch.cumsum(
                seq_lens_device, dim=0, dtype=cu_seq_lens.dtype, out=cu_seq_lens[1:]
            )
            extend_metadata = AiterChunkPrefillMetadata(
                max_query_len=query_lens_extend.max().item(),
                max_seq_len=seq_lens[num_extends_slice].max().item(),
                query_start_loc=query_start_loc_device - query_start_loc_device[0],
                chunk_context_metadata=chunk_context_metadata,
            )
        # num_actual_kv_tokens = torch.sum(seq_lens).item()
        num_actual_kv_tokens = 0

        use_cascade = False

        num_actual_tokens = common_attn_metadata.num_actual_tokens

        attn_metadata = AiterMhaMetadataForVllm(
            num_actual_tokens=num_actual_tokens,
            num_actual_kv_tokens=num_actual_kv_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_extends=num_extends,
            num_extend_tokens=num_extend_tokens,
            dropout_p=0.0,
            decode_metadata=decode_metadata,
            prefill_metadata=prefill_metadata,
            extend_metadata=extend_metadata,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            total_tokens=self.total_tokens,
        )

        return attn_metadata

    def build_for_drafting(
        self,
        common_attn_metadata,
        draft_index: int,
    ) -> AiterMhaMetadataForVllm:
        """
        Build attention metadata for draft model without CPU-GPU sync.

        During EAGLE/MTP drafting all requests are uniform decodes, so we can
        skip split_decodes_prefills_and_extends() and avoid all .cpu() /
        .item() calls that would otherwise break CUDA graph capture.
        """
        query_start_loc = common_attn_metadata.query_start_loc_cpu
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        is_prefill = query_lens > self.reorder_batch_threshold

        if torch.any(is_prefill):
            return self.build(
                common_prefix_len=0, common_attn_metadata=common_attn_metadata
            )

        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        decode_metadata = AiterMhaPhaseMetadata(
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            query_start_loc=common_attn_metadata.query_start_loc,
        )
        return AiterMhaMetadataForVllm(
            num_actual_tokens=num_tokens,
            num_actual_kv_tokens=0,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_reqs,
            num_decode_tokens=num_tokens,
            num_prefills=0,
            num_prefill_tokens=0,
            num_extends=0,
            num_extend_tokens=0,
            decode_metadata=decode_metadata,
            prefill_metadata=None,
            extend_metadata=None,
            use_cascade=False,
            common_prefix_len=0,
            total_tokens=self.total_tokens,
        )

    # this method will be called by vllm, so it follows the vllm's interface convention
    def build_for_cudagraph_capture(
        self,
        common_attn_metadata=None,
    ):
        self.total_tokens = (
            self.model_config.max_model_len
            * self.vllm_config.scheduler_config.max_num_partial_prefills
        )
        attn_metadata = self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )
        self.total_tokens = 0
        return attn_metadata


class AiterMlaMetadataBuilderForVllm(MLACommonMetadataBuilder):
    """vLLM-only dense MLA metadata builder."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    reorder_batch_threshold = 1
    query_len_support = QueryLenSupport.UNIFORM

    def __init__(
        self,
        kv_cache_spec=None,
        layer_names=None,
        config=None,
        device=None,
        model_runner=None,
    ):
        super().__init__(kv_cache_spec, layer_names, config, device)
        logger.info("init AiterMlaMetadataBuilderForVllm")
        from vllm.config import VllmConfig

        assert isinstance(config, VllmConfig)

        self.vllm_config = config
        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.cache_config = config.cache_config

        self.compilation_config = self.vllm_config.compilation_config
        self.decode_attn_out_dtype = self.vllm_config.model_config.dtype

        max_num_pages_per_req = self.vllm_config.model_config.max_model_len
        max_num_reqs = self.vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req

        hf_config = config.model_config.hf_config
        text_config = getattr(hf_config, "text_config", None)
        num_attention_heads = getattr(
            hf_config, "num_attention_heads", None
        ) or getattr(text_config, "num_attention_heads", None)
        assert (
            num_attention_heads is not None
        ), "num_attention_heads is not found in config"

        self.num_attention_heads = num_attention_heads // get_tp_group().world_size
        self.padded_num_attention_heads = max(self.num_attention_heads, _MLA_MIN_HEADS)
        self.block_size = kv_cache_spec.block_size
        self.max_bs = max_num_reqs
        self.dtype_kv = get_aiter_kv_cache_dtype(config)
        # MLA decode path in ATOM-vLLM quantizes Q to FP8 when the KV cache is FP8,
        # so aiter metadata must be sized/generated with the same dtype.
        self.dtype_q = dtypes.fp8 if self.dtype_kv == dtypes.fp8 else torch.bfloat16

        self.paged_kv_last_page_len = torch.ones(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.paged_kv_indptr = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.paged_kv_indices = torch.zeros(
            max_num_pages, dtype=torch.int32, device=device
        )
        self.qo_indptr = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)

        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            max_num_reqs,
            1,
            self.padded_num_attention_heads,
            self.dtype_q,
            self.dtype_kv,
            is_sparse=False,
            fast_mode=True,
        )

        self.mla_persistent_metadata = {
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
        }

        # Workaround for the missing MLA fp8/fp8 nhead=64 qseqlen=1
        # non-persistent kernel on gfx950. Leverage the pre-existing
        # 8-head non-persistent kernels, folding the q/o tensors to
        # 8 heads
        self._mla_fold_enabled = (
            self.padded_num_attention_heads in [64, 32]
            and self.dtype_kv == dtypes.fp8
            and get_gfx() == "gfx950"
        )
        self._mla_fold_factor = (
            self.padded_num_attention_heads // 8 if self._mla_fold_enabled else 1
        )
        # For 64-head fp8/fp8 qseqlen=1 MLA, use native persistent instead of fold
        self._mla_dp_native_persistent_enabled = (
            self._mla_fold_enabled
            and self.padded_num_attention_heads == 64
            and self.dtype_q == dtypes.fp8
            and self.dtype_kv == dtypes.fp8
        )

        # Allocate the fold buffers for the nhead-folding workaround outside CUDA
        # graph capture and refill them in `_build_decode`.
        if self._mla_fold_enabled and not self._mla_dp_native_persistent_enabled:
            fold_factor = self._mla_fold_factor
            max_fold_bs = max_num_reqs * fold_factor
            self.fold_kv_indptr = torch.zeros(
                max_fold_bs + 1, dtype=torch.int32, device=device
            )
            self.fold_kv_indices = torch.empty(
                max_num_pages * fold_factor, dtype=torch.int32, device=device
            )
            # qo_indptr and last_page_len are constant for qseqlen==1 decode.
            self.fold_qo_indptr = torch.arange(
                max_fold_bs + 1, dtype=torch.int32, device=device
            )
            self.fold_kv_last_page_len = torch.ones(
                max_fold_bs, dtype=torch.int32, device=device
            )

    # TODO: support mtp and sparse
    def _set_mla_persistent_worker_buffers(
        self, bs: int, cu_seqlens_q: torch.Tensor, max_q_len: int = 1
    ):
        split_params = {
            "kv_granularity": max(self.block_size, 16),
            "max_seqlen_qo": max_q_len,
            "uni_seqlen_qo": max_q_len,
            "fast_mode": 1,
            "max_split_per_batch": 16,
        }
        var = self.mla_persistent_metadata
        work_meta_data = var["work_meta_data"]
        work_info_set = var["work_info_set"]
        work_indptr = var["work_indptr"]
        reduce_indptr = var["reduce_indptr"]
        reduce_final_map = var["reduce_final_map"]
        reduce_partial_map = var["reduce_partial_map"]
        get_mla_metadata_v1(
            cu_seqlens_q,
            self.paged_kv_indptr[: bs + 1],  # TODO: support sparse
            self.paged_kv_last_page_len[:bs],
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

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ):
        # kernel block size is always 1, although the kv block size is not 1.
        device = self.device
        num_reqs = seq_lens_device.size(0)

        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]

        torch.cumsum(
            seq_lens_device,
            dim=0,
            dtype=torch.int32,
            out=self.paged_kv_indptr[1 : 1 + num_reqs],
        )
        paged_kv_indptr = self.paged_kv_indptr[: 1 + num_reqs]

        qo_len = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_qo_len = qo_len.max().item() if qo_len.numel() > 0 else 1

        kv_indices_generate_triton(
            block_table_tensor,
            self.paged_kv_indices,
            paged_kv_indptr,
            1,
            max_seq_len,
        )
        paged_kv_indices = self.paged_kv_indices

        # For pure decode, query_start_loc is [0,1,2,...,N]; skip the DtoD copy
        # and populate qo_indptr using an in-place arange when possible.
        if num_decode_tokens == num_reqs:
            if (
                not getattr(self, "_qo_indptr_arange_ready", False)
                or getattr(self, "_qo_indptr_arange_n", 0) != num_reqs
            ):
                torch.arange(
                    0,
                    num_reqs + 1,
                    dtype=torch.int32,
                    device=device,
                    out=self.qo_indptr[: num_reqs + 1],
                )
                if num_reqs + 1 < self.qo_indptr.shape[0]:
                    self.qo_indptr[num_reqs + 1 :] = num_reqs
                self._qo_indptr_arange_ready = True
                self._qo_indptr_arange_n = num_reqs
        else:
            self._qo_indptr_arange_ready = False
            self.qo_indptr[: 1 + num_reqs].copy_(
                query_start_loc_device, non_blocking=True
            )
            if 1 + num_reqs < self.qo_indptr.shape[0]:
                self.qo_indptr[1 + num_reqs :] = num_decode_tokens
        qo_indptr = self.qo_indptr[: 1 + num_reqs]

        # Disable persistent MLA in DP mode: pre-computed metadata buffers
        # are invalid when request counts vary across DP ranks each step.
        dp_enabled = get_dp_group().world_size > 1
        use_persistent_metadata = (not dp_enabled) or (
            self._mla_dp_native_persistent_enabled and max_qo_len == 1
        )
        if use_persistent_metadata:
            ctx_mla_ps = self._set_mla_persistent_worker_buffers(
                num_reqs,
                qo_indptr,
                max_qo_len,
            )
            self.mla_persistent_metadata.update(ctx_mla_ps)

        fold_factor = (
            self._mla_fold_factor
            if (
                self._mla_fold_enabled
                and dp_enabled
                and max_qo_len == 1
                and not use_persistent_metadata
            )
            else None
        )

        fold_kv_indptr = fold_kv_indices = None
        fold_qo_indptr = fold_kv_last_page_len = None
        if fold_factor is not None and fold_factor > 1:
            new_bs = num_reqs * fold_factor
            # Keep the view sized to this step's worst case so aiter's
            # non-persistent split heuristic sees avg_kv == max_seq_len.
            # During full CUDA graph capture max_seq_len is max_model_len,
            # which is the replay upper bound.
            fold_kv_indices_len = num_reqs * max_seq_len * fold_factor
            assert fold_kv_indices_len <= self.fold_kv_indices.numel(), (
                f"fold_kv_indices overflow: need {fold_kv_indices_len}, "
                f"have {self.fold_kv_indices.numel()}"
            )
            fold_kv_indptr = self.fold_kv_indptr[: new_bs + 1]
            fold_kv_indices = self.fold_kv_indices[:fold_kv_indices_len]
            fold_qo_indptr = self.fold_qo_indptr[: new_bs + 1]
            fold_kv_last_page_len = self.fold_kv_last_page_len[:new_bs]

            mla_fold_kv_metadata_triton(
                paged_kv_indptr,
                paged_kv_indices,
                fold_kv_indptr,
                fold_kv_indices,
                fold_factor=fold_factor,
                num_reqs=num_reqs,
            )

        attn_metadata = AiterMlaDecodeMetadataForVllm(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            qo_indptr=qo_indptr,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            max_qo_len=max_qo_len,
            attn_out_dtype=self.decode_attn_out_dtype,
            use_persistent_metadata=use_persistent_metadata,
            fold_factor=fold_factor,
            fold_kv_indptr=fold_kv_indptr,
            fold_kv_indices=fold_kv_indices,
            fold_qo_indptr=fold_qo_indptr,
            fold_kv_last_page_len=fold_kv_last_page_len,
        )

        return attn_metadata

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata=None,
    ):
        return self.build(0, common_attn_metadata)

    def build(
        self,
        common_prefix_len: int = 0,
        common_attn_metadata=None,
        fast_build: bool = False,
    ):

        from vllm.v1.attention.backends.utils import split_decodes_and_prefills
        from vllm.model_executor.layers.attention.mla_attention import (
            QueryLenSupport,
        )

        from vllm.utils.math_utils import cdiv, round_down
        from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.device
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=(self.query_len_support != QueryLenSupport.VARLEN),
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        prefill_metadata = None
        if num_prefills > 0:
            reqs_start = num_decodes  # prefill_start

            # In DBO, an ubatch can contain only part of a prefill request.
            # Derive context lengths from the sliced CPU query lengths and
            # seq_lens upper bound to match upstream vLLM's MLA builder,
            # instead of forcing a device->host sync through
            # compute_num_computed_tokens().
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            prefill_query_lens_cpu = (
                query_start_loc_cpu[reqs_start + 1 : num_reqs + 1]
                - query_start_loc_cpu[reqs_start:num_reqs]
            )
            context_lens_cpu = (
                seq_lens_cpu[reqs_start:num_reqs] - prefill_query_lens_cpu
            )
            max_context_len_cpu = context_lens_cpu.max().item()
            num_prefills_with_context_cpu = (context_lens_cpu > 0).sum().item()
            prefill_query_start_loc = (
                query_start_loc[reqs_start:] - query_start_loc[reqs_start]
            )
            prefill_query_start_loc_cpu = (
                query_start_loc_cpu[reqs_start:] - query_start_loc_cpu[reqs_start]
            )

            chunked_context_metadata = None
            if max_context_len_cpu > 0:
                # NOTE: it is recommend you read the `Chunked Prefill` section
                # in the comment at the top of the file before trying to
                # understand the following code

                # currently we allocate an equal amount of workspace for each
                # prefill in the batch, we could probably use a more advanced
                # algorithm here and allocate more workspace to prefills with
                # longer context lengths
                max_context_chunk = (
                    self.chunked_prefill_workspace_size // num_prefills_with_context_cpu
                )

                if self.aot_schedule:
                    # align max_context_chunk to page_size by rounding down,
                    # currently the `gather_and_maybe_dequant_cache` kernel
                    # cannot handle `context_chunk_starts` that are not aligned
                    # to page_size
                    max_context_chunk = round_down(max_context_chunk, self.page_size)

                assert max_context_chunk > 0
                num_chunks = cdiv(max_context_len_cpu, max_context_chunk)

                # if `max_context_chunk = 256`, `num_chunks = 3`, and
                #   `num_prefills_with_context = 4`, create a tensor that looks
                # like
                #  [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512]]
                # Note(simon): this is done in CPU because of downstream's
                # of `to_list`.
                chunk_starts = (
                    torch.arange(num_chunks, dtype=torch.int32)
                    .unsqueeze(1)
                    .expand(-1, num_prefills)
                    * max_context_chunk
                )
                chunk_ends = torch.min(
                    context_lens_cpu.unsqueeze(0), chunk_starts + max_context_chunk
                )
                chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)

                cu_seq_lens_cpu = torch.zeros(
                    num_chunks, num_prefills + 1, dtype=torch.int32, pin_memory=True
                )
                torch.cumsum(
                    chunk_seq_lens,
                    dim=1,
                    out=cu_seq_lens_cpu[:, 1:],
                    dtype=torch.int32,
                )
                chunk_total_token = cu_seq_lens_cpu[:, -1]

                max_token_num_over_chunk = chunk_total_token.max().item()
                token_to_seq_tensor_cpu = torch.zeros(
                    [num_chunks, max_token_num_over_chunk], dtype=torch.int32
                )
                range_idx = torch.arange(num_prefills, dtype=torch.int32)
                for i in range(num_chunks):
                    chunk_token_to_seq_tensor = torch.repeat_interleave(
                        range_idx, chunk_seq_lens[i]
                    )
                    chunk_len = chunk_token_to_seq_tensor.shape[0]
                    token_to_seq_tensor_cpu[i, :chunk_len] = chunk_token_to_seq_tensor

                if self.dcp_world_size > 1:
                    local_context_lens_allranks = get_dcp_local_seq_lens(
                        context_lens_cpu,
                        self.dcp_world_size,
                        None,
                        self.dcp_local_block_size,
                    )
                    # Note(qcs): The max local context lengths
                    # padded to `dcp_local_block_size`.
                    padded_local_context_lens_cpu: torch.Tensor = (
                        cdiv(
                            context_lens_cpu,
                            self.dcp_virtual_block_size,
                        )
                        * self.dcp_local_block_size
                    )
                    # Note(hc): The above max_context_chunk already enforces
                    # block_size alignment, DCP just need the block_size can
                    # be divisible by dcp_world_size, because DCP use
                    # cp_gather_cache which not require `cp_chunk_starts`
                    # aligned to page_size.
                    assert max_context_chunk % self.dcp_world_size == 0
                    padded_local_max_context_chunk_across_ranks = (
                        cdiv(
                            max_context_chunk,
                            self.dcp_virtual_block_size,
                        )
                        * self.dcp_local_block_size
                    )
                    local_chunk_starts = (
                        torch.arange(num_chunks, dtype=torch.int32)
                        .unsqueeze(1)
                        .expand(-1, num_prefills)
                        * padded_local_max_context_chunk_across_ranks
                    )
                    local_chunk_ends = torch.min(
                        padded_local_context_lens_cpu.unsqueeze(0),
                        local_chunk_starts
                        + padded_local_max_context_chunk_across_ranks,
                    )
                    padded_local_chunk_seq_lens = (
                        local_chunk_ends - local_chunk_starts
                    ).clamp(min=0)

                    padded_local_cu_chunk_seq_lens_cpu = torch.zeros(
                        num_chunks,
                        num_prefills + 1,
                        dtype=torch.int32,
                        pin_memory=True,
                    )
                    torch.cumsum(
                        padded_local_chunk_seq_lens,
                        dim=1,
                        out=padded_local_cu_chunk_seq_lens_cpu[:, 1:],
                        dtype=torch.int32,
                    )

                chunked_context_metadata_cls = (
                    AiterMlaPrefillMetadataForVllm.AiterMlaChunkedContextMetadataForVllm
                )
                prefill_tokens_with_context = None
                if num_prefills_with_context_cpu > 0:
                    prefill_tokens_with_context = prefill_query_start_loc_cpu[
                        num_prefills_with_context_cpu
                    ].item()
                if self.dcp_world_size > 1:
                    chunked_context_metadata = chunked_context_metadata_cls(
                        cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
                        starts=local_chunk_starts.to(device, non_blocking=True),
                        seq_tot=padded_local_chunk_seq_lens.sum(dim=1).tolist(),
                        max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                        seq_lens=chunk_seq_lens,
                        token_to_seq=token_to_seq_tensor_cpu.to(
                            device, non_blocking=True
                        ),
                        chunk_total_token=chunk_total_token.tolist(),
                        workspace=self.chunked_prefill_workspace,
                        padded_local_chunk_seq_lens=padded_local_chunk_seq_lens.tolist(),
                        local_context_lens_allranks=local_context_lens_allranks.tolist(),
                        padded_local_cu_seq_lens=padded_local_cu_chunk_seq_lens_cpu.to(
                            device, non_blocking=True
                        ),
                        cu_seq_lens_lst=cu_seq_lens_cpu.tolist(),
                        chunk_size=padded_local_max_context_chunk_across_ranks,
                        prefill_tokens_with_context=prefill_tokens_with_context,
                    )
                else:
                    chunked_context_metadata = chunked_context_metadata_cls(
                        cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
                        starts=chunk_starts.to(device, non_blocking=True),
                        seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                        max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                        seq_lens=chunk_seq_lens,
                        token_to_seq=token_to_seq_tensor_cpu.to(
                            device, non_blocking=True
                        ),
                        chunk_total_token=chunk_total_token,
                        workspace=self.chunked_prefill_workspace,
                        prefill_tokens_with_context=prefill_tokens_with_context,
                    )

                assert (
                    max(chunked_context_metadata.max_seq_lens)
                    <= self.chunked_prefill_workspace_size
                )

            prefill_metadata = AiterMlaPrefillMetadataForVllm(
                block_table=block_table_tensor[reqs_start:, ...],
                query_start_loc=prefill_query_start_loc,
                max_query_len=max_query_len,
                chunked_context=chunked_context_metadata,
            )

        decode_metadata = None
        if num_decodes > 0:
            dcp_tot_seq_lens_device = None
            if self.dcp_world_size > 1:
                dcp_tot_seq_lens_device = seq_lens[:num_decodes]
                seq_lens = dcp_local_seq_lens

                # After DCP distribution, the maximum number of tokens for any rank is
                # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
                # and I is cp_kv_cache_interleave_size.
                # This eliminates GPU->CPU sync while minimizing workspace
                # over-allocation.
                num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
                max_seq_len = (
                    (max_seq_len + num_partitions - 1) // num_partitions
                ) * self.cp_kv_cache_interleave_size

            decode_metadata = self._build_decode(
                block_table_tensor=block_table_tensor[:num_decodes, ...],
                seq_lens_device=seq_lens[:num_decodes],
                max_seq_len=max_seq_len,
                query_start_loc_cpu=query_start_loc_cpu[: num_decodes + 1],
                query_start_loc_device=query_start_loc[: num_decodes + 1],
                num_decode_tokens=num_decode_tokens,
                dcp_tot_seq_lens_device=dcp_tot_seq_lens_device,
            )

        attn_metadata = AiterMlaMetadataForVllm(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=query_start_loc,
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            # MLA metadata chunk prefill specific
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        # TODO: support mtp
        use_persistent_metadata = (
            decode_metadata is not None and decode_metadata.use_persistent_metadata
        )
        ctx_mla_ps = (
            self.mla_persistent_metadata
            if use_persistent_metadata
            else disabled_mla_persistent_metadata()
        )
        persistent_metadata = AiterMlaPersistentMetadataForVllm(**ctx_mla_ps)

        attn_metadata.persistent_metadata = persistent_metadata
        return attn_metadata


class AiterMlaSparseMetadataBuilder(AttentionMetadataBuilder):
    """vLLM-only metadata builder for sparse MLA main attention."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold = 1

    def __init__(
        self,
        kv_cache_spec=None,
        layer_names=None,
        config=None,
        device=None,
        model_runner=None,
    ):
        AttentionMetadataBuilder.__init__(
            self, kv_cache_spec, layer_names, config, device
        )
        from vllm.config import VllmConfig
        from vllm.model_executor.layers.attention.mla_attention import (
            get_mla_dims,
        )

        assert isinstance(config, VllmConfig)

        self.vllm_config = config
        self.model_config = config.model_config
        self.model_dtype = self.model_config.dtype
        self.kv_cache_spec = kv_cache_spec
        self.device = device
        max_num_batched_tokens = config.scheduler_config.max_num_batched_tokens
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        parallel_config = config.parallel_config
        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.padded_num_heads = max(self.num_heads, _MLA_MIN_HEADS)
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = config.model_config.hf_config.index_topk
        self.max_model_len_tensor = torch.tensor(
            [self.model_config.max_model_len], device=device, dtype=torch.int32
        )
        self.dummy_block_table = torch.empty(
            (1, 1), dtype=torch.int32, device=self.device
        )

        # zeros (not empty) so the shrink-tail fast path in build() can assume
        # entries past the current extent are already 0 without a full-buffer
        # fill_(0) every step.
        self.req_id_per_token_buffer = torch.zeros(
            (max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )
        self.qo_indptr = torch.arange(
            0, max_num_batched_tokens + 1, dtype=torch.int32, device=device
        )
        self.paged_kv_last_page_len = torch.ones(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )
        self.paged_kv_indptr = torch.zeros(
            [max_num_batched_tokens + 1], dtype=torch.int32, device=device
        )
        # The indexer writes topk indices to paged_kv_indices and sparse MLA reads
        # from this buffer. The indexer module is shared across ubatches in DBO
        # settings, so we bind a single shared buffer onto every indexer this builder
        # serves and let other ubatches for the same layer reuse it so that sparse
        # MLA doesn't read from unwritten per-builder buffers
        self.paged_kv_indices = self._bind_shared_sparse_kv_indices(
            layer_names,
            config,
            device,
            max_num_batched_tokens * self.topk_tokens,
        )

        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            max_num_batched_tokens,
            1,
            self.padded_num_heads,
            get_aiter_kv_cache_dtype(config),
            get_aiter_kv_cache_dtype(config),
            is_sparse=True,
            fast_mode=True,
        )
        self._mla_work_meta_data = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        )
        self._mla_work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        )
        self._mla_work_info_set = torch.empty(
            work_info_set_size, dtype=work_info_set_type, device=device
        )
        self._mla_reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        )
        self._mla_reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        )
        self._mla_reduce_partial_map = torch.empty(
            reduce_partial_map_size,
            dtype=reduce_partial_map_type,
            device=device,
        )

        # ----- Decode-orchestration caches (see build()) -----
        # Track the previously-written extents of the persistent buffers so we
        # only re-zero the shrink tail, and fingerprint the get_mla_metadata_v1
        # inputs so we can skip recomputing an identical work-split schedule.
        self._prev_req_extent = 0
        self._prev_indices_extent = 0
        self._prev_metadata_key = None

    def _bind_shared_sparse_kv_indices(self, layer_names, config, device, numel):
        # Resolve and bind a single shared paged_kv_indices buffer.
        # Reuse the buffer the other ubatch already bound if it exists, otherwise
        # allocate a new one
        default_sfc = (
            get_current_atom_config().compilation_config.static_forward_context
        )
        vllm_sfc = getattr(config.compilation_config, "static_forward_context", {})

        def _resolve_indexer(layer_name):
            attention_prefix = (
                layer_name[: -len(".attn")]
                if layer_name.endswith(".attn")
                else layer_name
            )
            indexer_cache = vllm_sfc.get(f"{attention_prefix}.indexer.k_cache")
            owner_atom_config = getattr(indexer_cache, "atom_config", None)
            sfc = (
                owner_atom_config.compilation_config.static_forward_context
                if owner_atom_config is not None
                else default_sfc
            )
            return (
                attention_prefix,
                sfc.get(f"{attention_prefix}.indexer"),
                sfc.get(attention_prefix),
                owner_atom_config,
            )

        # Reuse the buffer a sibling ubatch builder already bound onto the shared
        # indexer module (the indexer's initial torch.empty(0) has numel 0, so the
        # first builder allocates and later builders reuse). Reusing -- never
        # re-allocating -- keeps the tensor identity stable for torch.compile and
        # the device address stable for CUDA graphs.
        shared_buffer = None
        for layer_name in layer_names or []:
            _, indexer, _, _ = _resolve_indexer(layer_name)
            existing_buffer = getattr(indexer, "sparse_kv_indices_buffer", None)
            if existing_buffer is not None and existing_buffer.numel() >= numel:
                shared_buffer = existing_buffer
                break
        if shared_buffer is None:
            shared_buffer = torch.zeros([numel], dtype=torch.int32, device=device)

        for layer_name in layer_names or []:
            attention_prefix, indexer, sparse_attn, owner_atom_config = (
                _resolve_indexer(layer_name)
            )
            if indexer is not None:
                indexer.sparse_kv_indices_buffer = shared_buffer
            if sparse_attn is not None and hasattr(
                sparse_attn, "sparse_kv_indices_buffer"
            ):
                sparse_attn.sparse_kv_indices_buffer = shared_buffer
            if indexer is None or sparse_attn is None:
                logger.warning(
                    "Sparse MLA buffer binding incomplete for %s "
                    "(indexer=%s, sparse_attn=%s, owner_atom_config=%s)",
                    attention_prefix,
                    indexer is not None,
                    sparse_attn is not None,
                    owner_atom_config is not None,
                )
        return shared_buffer

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = common_attn_metadata.query_start_loc_cpu.to(torch.int32)
        seg_lengths = torch.diff(starts)
        req_id_per_token = torch.repeat_interleave(
            torch.arange(seg_lengths.shape[0], dtype=torch.int32), seg_lengths
        )
        # Shrink-tail-only zeroing instead of three full-buffer fill_(0) every
        # step (the buffers are persistent across decode steps and zeros-init):
        #   - req_id_per_token_buffer / paged_kv_indices: the kernel only reads
        #     the ranges defined by paged_kv_indptr / num_tokens, so entries
        #     past the new extent are never read; we only need to re-zero the
        #     tail left over from a previous, larger batch.
        #   - paged_kv_indptr: index 0 stays 0 (never written by the cumsum
        #     below, which starts at index 1) and indices >= 1 are fully
        #     rewritten by the cumsum + scalar broadcast, so its fill_(0) is
        #     redundant and dropped entirely.
        new_req_extent = int(req_id_per_token.shape[0])
        new_indices_extent = num_tokens * self.topk_tokens
        if self._prev_req_extent > new_req_extent:
            self.req_id_per_token_buffer[new_req_extent : self._prev_req_extent].fill_(
                0
            )
        if self._prev_indices_extent > new_indices_extent:
            self.paged_kv_indices[new_indices_extent : self._prev_indices_extent].fill_(
                0
            )
        self._prev_req_extent = new_req_extent
        self._prev_indices_extent = new_indices_extent
        self.req_id_per_token_buffer[:new_req_extent].copy_(
            req_id_per_token, non_blocking=True
        )

        query_lens = (
            common_attn_metadata.query_start_loc[1:]
            - common_attn_metadata.query_start_loc[:-1]
        )
        seq_lens = common_attn_metadata.seq_lens

        from atom.plugin.vllm.attention.layer_sparse_mla import (
            generate_sparse_seqlen_triton,
        )

        sparse_seqlen = generate_sparse_seqlen_triton(
            query_lens,
            seq_lens,
            common_attn_metadata.query_start_loc,
            self.topk_tokens,
            num_tokens,
            common_attn_metadata.max_query_len,
        )
        torch.cumsum(
            sparse_seqlen,
            dim=0,
            out=self.paged_kv_indptr[1 : num_tokens + 1],
        )
        self.paged_kv_indptr[num_tokens + 1 :].fill_(self.paged_kv_indptr[num_tokens])

        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]
        qo_indptr = self.qo_indptr[: num_tokens + 1]
        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_tokens]
        paged_kv_indices = self.paged_kv_indices[: num_tokens * self.topk_tokens]
        paged_kv_indptr = self.paged_kv_indptr[: num_tokens + 1]

        # ----- Compute persistent MLA metadata -----
        # The aiter sparse decode kernel uses qseqlen=1 (each query token is
        # treated as its own batch entry), so persistent metadata can always
        # be precomputed here. The kernel switches to the persistent
        # work-stealing path automatically when work_meta_data is non-None.
        #
        # Pure-decode skip-cache: only valid when every request contributes
        # exactly one query token. max_query_len == 1 alone is NOT enough --
        # it allows empty/padded request slots (q_len == 0), where
        # num_actual_tokens < num_reqs and the per-token sparse stream no longer
        # lines up with the per-request clamped seq_lens used as the key. We
        # therefore also require num_tokens == num_reqs; together they imply
        # exactly one token per request, so clamped_seq_lens[:num_reqs] IS the
        # per-token sparse stream and the work-split schedule is a deterministic
        # function of (num_tokens, padded_num_heads, per-request
        # min(seq_len, topk_tokens)) -- topk_tokens, page_size and the qo layout
        # are constant. Fingerprint those CPU-side and skip the launch when
        # unchanged; for long-context decode (seq_len >= topk_tokens) the key is
        # shape-determined, so the hit rate is ~100%.
        #
        # MTP/spec or padded batches always recompute and invalidate the key.
        # The fingerprint is only built on the cacheable path, so the slow path
        # adds no extra CPU work.
        num_reqs = common_attn_metadata.num_reqs
        decode_only = (
            int(common_attn_metadata.max_query_len) == 1 and num_tokens == num_reqs
        )
        metadata_key = None
        if decode_only:
            seq_lens_cpu = getattr(common_attn_metadata, "seq_lens_cpu", None)
            if seq_lens_cpu is None:
                seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
            if seq_lens_cpu is None:
                seq_lens_cpu = seq_lens.cpu()
            clamped_seq_lens = torch.clamp(
                seq_lens_cpu[:num_reqs], max=self.topk_tokens
            )
            metadata_key = (
                num_tokens,
                self.padded_num_heads,
                clamped_seq_lens.to(torch.int32).numpy().tobytes(),
            )
        if metadata_key is None or metadata_key != self._prev_metadata_key:
            get_mla_metadata_v1(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_last_page_len,
                self.padded_num_heads,
                1,
                True,
                self._mla_work_meta_data,
                self._mla_work_info_set,
                self._mla_work_indptr,
                self._mla_reduce_indptr,
                self._mla_reduce_final_map,
                self._mla_reduce_partial_map,
                page_size=1,
                kv_granularity=16,
                max_seqlen_qo=1,
                uni_seqlen_qo=1,
                fast_mode=True,
                dtype_q=get_aiter_kv_cache_dtype(self.vllm_config),
                dtype_kv=get_aiter_kv_cache_dtype(self.vllm_config),
            )
            # metadata_key is the exact fingerprint on the cacheable path and
            # None for MTP/spec/padded batches -- storing it as-is records a
            # hittable key on decode and invalidates the cache otherwise.
            self._prev_metadata_key = metadata_key

        attn_metadata = AiterMlaSparseMetadataForVllm(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            attn_out_dtype=self.model_dtype,
            topk_tokens=self.topk_tokens,
            qo_indptr=qo_indptr,
            paged_kv_last_page_len=paged_kv_last_page_len,
            paged_kv_indices=paged_kv_indices,
            paged_kv_indptr=paged_kv_indptr,
            work_meta_data=self._mla_work_meta_data,
            work_indptr=self._mla_work_indptr,
            work_info_set=self._mla_work_info_set,
            reduce_indptr=self._mla_reduce_indptr,
            reduce_final_map=self._mla_reduce_final_map,
            reduce_partial_map=self._mla_reduce_partial_map,
        )

        return attn_metadata


class AiterMlaSparseIndexerMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold = 1

    def __init__(
        self,
        kv_cache_spec=None,
        layer_names=None,
        config=None,
        device=None,
        model_runner=None,
    ):
        AttentionMetadataBuilder.__init__(
            self, kv_cache_spec, layer_names, config, device
        )
        from vllm.config import VllmConfig

        try:
            from vllm.utils.platform_utils import num_compute_units
        except ImportError:
            from vllm.utils.platform_utils import get_cu_count as num_compute_units
        from vllm.v1.worker.cp_utils import get_total_cp_world_size
        from vllm.utils.math_utils import cdiv
        from atom.models.utils import extract_layer_index

        assert isinstance(config, VllmConfig)

        self.vllm_config = config
        self.model_config = config.model_config
        self.kv_cache_spec = kv_cache_spec
        self.device = device
        max_num_batched_tokens = config.scheduler_config.max_num_batched_tokens

        self.max_prefill_buffer_size = get_max_prefill_buffer_size(
            self.model_config.max_model_len
        )
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        # Determine if this builder is for draft model layers (MTP).
        # Draft model layers have layer indices >= num_hidden_layers.
        # The draft model itself does not do speculative decoding, so
        # num_speculative_tokens should be 0 for its builders.
        is_draft_layer = False
        if layer_names:
            num_hidden_layers = config.model_config.hf_config.num_hidden_layers
            layer_indices = [
                extract_layer_index(layer_name) for layer_name in layer_names
            ]
            is_draft_layer = all(idx >= num_hidden_layers for idx in layer_indices)
        if is_draft_layer:
            self.num_speculative_tokens = 0
        else:
            self.num_speculative_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
                if self.vllm_config.speculative_config
                else 0
            )

        sm_count = num_compute_units(self.device.index)
        self.num_sms = sm_count

        self.decode_lens_buffer = torch.empty(
            (max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.arange_buffer = torch.arange(
            config.scheduler_config.max_num_seqs * (1 + self.num_speculative_tokens),
            dtype=torch.int32,
            device=self.device,
        )
        self.expanded_seq_lens_buffer = torch.zeros(
            (max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )
        self.expanded_block_table_buffer = torch.zeros(
            (
                max_num_batched_tokens,
                max_num_blocks_per_req,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )

    def _build_indexer_one_prefill_chunk(
        self, reqs_start, reqs_end, query_start_loc_cpu, seq_lens_cpu, block_table
    ):
        prefill_query_start_loc = (
            query_start_loc_cpu[reqs_start : reqs_end + 1]
            - query_start_loc_cpu[reqs_start]
        )
        cu_seqlen_ks, cu_seqlen_ke = kv_spans_from_batches(
            prefill_query_start_loc, seq_lens_cpu[reqs_start:reqs_end], self.device
        )
        token_start = query_start_loc_cpu[reqs_start].item()
        token_end = query_start_loc_cpu[reqs_end].item()
        total_seq_lens = seq_lens_cpu[reqs_start:reqs_end].sum()
        seq_idx = torch.arange(0, reqs_end - reqs_start, dtype=torch.int32)
        token_to_seq = torch.repeat_interleave(
            seq_idx, seq_lens_cpu[reqs_start:reqs_end]
        ).to(self.device)
        assert total_seq_lens <= self.max_prefill_buffer_size
        cu_seq_lens = (
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32),
                    seq_lens_cpu[reqs_start:reqs_end].cumsum(dim=0),
                ]
            )
            .to(torch.int32)
            .to(self.device)
        )
        return AiterMlaSparseIndexerPrefillChunkMetadataForVllm(
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
            cu_seq_lens=cu_seq_lens,
            token_to_seq=token_to_seq,
            total_seq_lens=total_seq_lens,
            block_table=block_table[reqs_start:reqs_end],
            token_start=token_start,
            token_end=token_end,
            num_reqs=reqs_end - reqs_start,
        )

    def _build_indexer(
        self,
        common_prefix_len: int,
        common_attn_metadata=None,
        fast_build: bool = False,
    ) -> AiterMlaSparseIndexerMetadataForVllm:
        from vllm.v1.attention.backends.utils import (
            split_decodes_and_prefills,
            split_prefill_chunks,
        )
        from vllm.platforms import current_platform
        from vllm.utils.deep_gemm import (
            get_paged_mqa_logits_metadata,
            is_deep_gemm_supported,
        )

        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        prefill_metadata = None
        if num_prefills > 0:
            chunk_seq_ids = split_prefill_chunks(
                common_attn_metadata.seq_lens_cpu[num_decodes:],
                self.max_prefill_buffer_size,
                request_offset=num_decodes,
            )
            chunks = [
                self._build_indexer_one_prefill_chunk(
                    reqs_start,
                    reqs_end,
                    query_start_loc_cpu,
                    common_attn_metadata.seq_lens_cpu,
                    common_attn_metadata.block_table_tensor,
                )
                for reqs_start, reqs_end in chunk_seq_ids
            ]
            prefill_metadata = AiterMlaSparseIndexerPrefillMetadataForVllm(
                chunks=chunks,
            )

        decode_metadata = None
        if num_decodes > 0:
            torch.diff(
                common_attn_metadata.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            decode_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            )

            seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

            # Padded CUDA graph requests have block_table entries of -1.
            # Clamp to 0 to prevent OOB access in the DeepGEMM kernel.
            # This is safe because padded requests have seq_lens=0, so the
            # kernel produces no meaningful output for those rows.
            block_table.clamp_(min=0)

            max_decode_len = int(decode_lens_cpu.max().item())
            if max_decode_len > 1:
                # Flatten multi-token decode requests into single-token
                # batch entries, expanding seq_lens and block tables so
                # the kernel always sees next_n=1.

                # Assume 4 requests with seq_lens [10, 7, 12, 0] (the final req is
                # padding) and decode_lens [3, 1, 4, 0] in the below example comments.
                # The context lengths are therefore
                # [10-3, 7-1, 12-4, 0-0] = [7, 6, 8, 0].

                # 3 + 1 + 4 + 0 = 8
                actual_expanded = int(decode_lens_cpu.sum().item())

                # [7, 6, 8, 0] -> [7, 7, 7, 6, 8, 8, 8, 8]
                expanded_base = torch.repeat_interleave(
                    seq_lens - decode_lens, decode_lens, output_size=actual_expanded
                )

                # [0, 3, 4, 8] -> [0, 0, 0, 3, 4, 4, 4, 4]
                expanded_starts = torch.repeat_interleave(
                    common_attn_metadata.query_start_loc[:num_decodes],
                    decode_lens,
                    output_size=actual_expanded,
                )

                # [0, 1, 2, 0, 0, 1, 2, 3]
                positions_within = (
                    self.arange_buffer[:actual_expanded] - expanded_starts
                )

                # [8, 9, 10, 7, 9, 10, 11, 12, ...] where ... is unused buffer space
                self.expanded_seq_lens_buffer[:actual_expanded] = (
                    expanded_base + positions_within + 1
                )
                self.expanded_seq_lens_buffer[actual_expanded:] = 0
                seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]

                # Give each of the flattened entries the same block table row as the
                # original request.
                self.expanded_block_table_buffer[:actual_expanded] = (
                    torch.repeat_interleave(
                        block_table, decode_lens, dim=0, output_size=actual_expanded
                    )
                )
                if actual_expanded < num_decode_tokens:
                    self.expanded_block_table_buffer[
                        actual_expanded:num_decode_tokens, 0
                    ] = 0
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]

                # All reqs now have decode_len=1
                self.decode_lens_buffer[:num_decode_tokens] = 1
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                offsets = None
                batch_size = num_decode_tokens
            else:
                next_n = 1 + self.num_speculative_tokens
                if next_n > 1:
                    offsets = torch.arange(
                        next_n, device=self.device, dtype=torch.int32
                    )
                else:
                    offsets = None
                batch_size = num_decodes

            # DeepGEMM is required for the paged MQA logits on CUDA devices
            if current_platform.is_cuda() and is_deep_gemm_supported():
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.kv_cache_spec.block_size,
                    self.num_sms,
                )

            # Decide which top-k kernel to use based on batch size and sequence length
            # Decision logic based on micro-benchmark results:
            # - large_context_topk wins for batch <= 128 and seq_len > 8K
            # - top_k_per_row_decode wins for batch > 128 or seq_len <= 8K
            _is_large_context = common_attn_metadata.max_seq_len > 8192
            use_large_context_topk = batch_size <= 128 and _is_large_context

            decode_metadata = AiterMlaSparseIndexerDecodeMetadataForVllm(
                block_table=block_table,
                seq_lens=seq_lens,
                decode_lens=decode_lens,
                requires_padding=False,
                schedule_metadata=self.scheduler_metadata_buffer,
                use_large_context_topk=use_large_context_topk,
                offsets=offsets,
            )

        indexer_metadata = AiterMlaSparseIndexerMetadataForVllm(
            seq_lens=common_attn_metadata.seq_lens,
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            head_dim=128,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        return indexer_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        indexer_metadata = self._build_indexer(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )
        return indexer_metadata
