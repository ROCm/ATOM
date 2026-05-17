from typing import Type

import logging
import torch
from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.dist.parallel_state import get_tp_group
from atom.model_ops.attention_mla import _MLA_MIN_HEADS
from atom.utils import CpuGpuBuffer
from atom.utils.block_convert import kv_indices_generate_triton
from atom.utils.forward_context import Context
from atom.plugin.vllm.attention.metadata import (
    _CP_TOKENS_PER_ITER_ROCM,
    AiterChunkContextMetadata,
    AiterChunkSlidingWindowMetadata,
    AiterChunkPrefillMetadata,
    AiterMlaDecodeMetadataForVllm,
    AiterMlaMetadataForVllm,
    AiterMlaPersistentMetadataForVllm,
    AiterMlaPrefillMetadataForVllm,
    AiterMhaMetadataForVllm,
    AiterMhaPhaseMetadata,
)
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
)

logger = logging.getLogger("atom")


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

        num_computed_tokens_cpu = common_attn_metadata._num_computed_tokens_cpu

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
            computed_kv_lens = num_computed_tokens_cpu[num_extends_slice]

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

        context_batch_size = 0
        has_prefill = bool(num_prefills > 0 or num_extends > 0)
        if has_prefill:
            context_batch_size = num_prefills + num_extends
        else:
            context_batch_size = num_decodes
        context_graph_bs = context_batch_size

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        context = Context(
            positions=None,
            is_prefill=has_prefill,
            batch_size=context_batch_size,
            graph_bs=context_graph_bs,
        )

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
            context=context,
        )

        return attn_metadata

    def build_for_drafting(
        self,
        common_attn_metadata,
        draft_index: int,
    ) -> AttentionMetaData:
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
        decode_metadata = AiterFlashAttentionDecodeMetadata(
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            query_start_loc=common_attn_metadata.query_start_loc,
        )
        attn_metadata_for_plugin_mode = AiterFlashAttentionMetadataForPluginMode(
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

        attn_metadata = AttentionMetaData(
            max_seqlen_q=common_attn_metadata.max_query_len,
            block_tables=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            plugin_metadata=attn_metadata_for_plugin_mode,
        )
        return attn_metadata

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
            torch.bfloat16,
            dtypes.d_dtypes[config.cache_config.cache_dtype],
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

        max_qo_len = (
            (query_start_loc_cpu[-1] - query_start_loc_cpu[-2]).item()
            if query_start_loc_cpu.numel() > 1
            else 1
        )

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

        ctx_mla_ps = self._set_mla_persistent_worker_buffers(num_reqs, qo_indptr, 1)
        self.mla_persistent_metadata.update(ctx_mla_ps)

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
            num_computed_tokens_cpu = (
                common_attn_metadata.compute_num_computed_tokens().cpu()
            )

            reqs_start = num_decodes  # prefill_start

            context_lens_cpu = num_computed_tokens_cpu[reqs_start:num_reqs]
            max_context_len_cpu = context_lens_cpu.max().item()
            num_prefills_with_context_cpu = (context_lens_cpu > 0).sum().item()
            prefill_query_start_loc = (
                query_start_loc[reqs_start:] - query_start_loc[reqs_start]
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
                    )

                if self._use_cudnn_prefill:
                    chunked_context_metadata.seq_lens = chunk_seq_lens

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
        persistent_metadata = AiterMlaPersistentMetadataForVllm(
            **self.mla_persistent_metadata
        )

        attn_metadata.persistent_metadata = persistent_metadata
        return attn_metadata


class AiterMhaBackendForVllm:
    """vLLM-facing MHA backend surface for ATOM attention layers."""

    accept_output_buffer: bool = False
    supported_dtypes: list = [torch.float16, torch.bfloat16]
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @classmethod
    def get_kv_cache_block_dim(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int:
        sentinel = 1234567
        shape = cls.get_kv_cache_shape(
            sentinel,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str=cache_dtype_str,
        )
        return shape.index(sentinel)

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        if cls.supports_block_size(default_block_size):
            return default_block_size
        return 16

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return False

    @staticmethod
    def get_required_kv_cache_layout():
        return None

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> Type["AiterMhaMetadataBuilderForVllm"]:
        return AiterMhaMetadataBuilderForVllm

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmMHA

        return AttentionForVllmMHA

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class AiterMlaBackendForVllm:
    """vLLM-facing dense MLA backend surface for ATOM attention layers."""

    accept_output_buffer: bool = True
    supported_dtypes: list = [torch.float16, torch.bfloat16]
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [1]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 1

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @staticmethod
    def get_required_kv_cache_layout():
        return None

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return (1, 0, 2, 3) if include_num_layers_dimension else (0, 1, 2)

    @staticmethod
    def get_builder_cls() -> Type["AiterMlaMetadataBuilderForVllm"]:
        return AiterMlaMetadataBuilderForVllm

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmMLA

        return AttentionForVllmMLA

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class AiterSparseMlaBackendForVllm(AiterMlaBackendForVllm):
    """vLLM-facing sparse MLA backend surface for ATOM attention layers."""

    @staticmethod
    def get_builder_cls() -> Type["AiterMLASparseMetadataBuilder"]:
        from atom.plugin.vllm.attention_backend.mla_sparse import (
            AiterMLASparseMetadataBuilder,
        )

        return AiterMLASparseMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmMLA

        return AttentionForVllmMLA

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class AiterSparseMlaIndexerBackendForVllm(AiterMlaBackendForVllm):
    """vLLM-facing sparse MLA indexer backend surface."""

    @staticmethod
    def get_builder_cls() -> Type["AiterMLASparseIndexerMetadataBuilder"]:
        from atom.plugin.vllm.attention_backend.mla_sparse import (
            AiterMLASparseIndexerMetadataBuilder,
        )

        return AiterMLASparseIndexerMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmMLA

        return AttentionForVllmMLA

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)
