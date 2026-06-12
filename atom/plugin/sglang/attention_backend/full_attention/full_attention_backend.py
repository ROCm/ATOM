from __future__ import annotations

# SGLang full-attention backend replacing sglang's built-in AiterAttnBackend.
# Shared by ALL full-attention models (DeepSeek, Qwen3, etc.) — handles KV
# cache writes, page-table fixup, pa_persistent_fwd decode path, and MLA
# prefill kernels. Sits at the lowest layer of the attention stack:
# sglang's RadixAttention delegates the actual kernel dispatch here.
#
# TODO: rewrite this file once sglang's attention flow is unified into ATOM's
# attention layer — KV cache management and attention kernel dispatch will then
# be handled by ATOM's native backend, making sglang-specific overrides
# unnecessary.

from typing import TYPE_CHECKING, Optional

import torch

import sglang.srt.layers.attention.aiter_backend as _sglang_aiter
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.layers.attention.utils import (
    create_flashinfer_kv_indices_triton,
    launch_reshape_and_cache_flash,
    pad_sequence_with_mask,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import get_bool_env_var

from atom.plugin.sglang.attention_backend.full_attention.kv_cache import (
    set_kv_buffer_with_layout_shuffle as _set_kv_buffer_with_layout_shuffle,
)
from atom.plugin.sglang.attention_backend.full_attention.metadata import ForwardMetadata
from atom.plugin.sglang.attention_backend.full_attention.pa_metadata import (
    allocate_pa_metadata_buffers as _allocate_pa_metadata_buffers,
    build_pa_metadata_for_decode as _build_pa_metadata_for_decode,
    build_pa_metadata_for_prefill as _build_pa_metadata_for_prefill,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

try:
    from aiter import (
        QuantType,
        flash_attn_varlen_func,
        dtypes,
        get_pa_metadata_info_v1,
        get_hip_quant,
        mha_batch_prefill_func,
        pa_fwd_asm,
        pa_persistent_fwd,
        paged_attention_ragged,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import 'aiter', which provides AMD-specific attention kernels "
        "required by full_attention_backend. Please ensure 'aiter' is installed and "
        f"available on your AMD system. Original import error: {e}"
    ) from e

# MLA prefill kernels - imported separately to avoid breaking the main aiter imports
mla_prefill_ps_asm_fwd = None
mla_reduce_v1 = None
mla_prefill_fwd = None
mla_decode_fwd = None
fused_gemm_afp4wfp4_preshuffle_split_cat = None
try:
    from aiter import mla_prefill_ps_asm_fwd
except ImportError:
    pass
try:
    from aiter import mla_reduce_v1
except ImportError:
    pass
try:
    from aiter.mla import mla_prefill_fwd
    from aiter.mla import mla_decode_fwd
except ImportError:
    pass
try:
    from aiter.ops.triton.fused_gemm_afp4wfp4_split_cat import (
        fused_gemm_afp4wfp4_preshuffle_split_cat,
    )
except ImportError:
    pass


class ATOMAttnBackendForSgl(AiterAttnBackend):
    """ATOM's custom attention backend for sglang plugin mode.

    Extends sglang's AiterAttnBackend with ATOM-specific optimisations:
    page-table management, pa_persistent_fwd decode path, and MLA
    prefill kernels (fp8, decompress, absorbed).  Registered to sglang
    via atom.plugin.register._register_custom_attention_to_sglang().
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        topk: int = 1,
    ):
        super().__init__(model_runner, skip_prefill, kv_indptr_buf, topk)
        mapping = getattr(
            model_runner.token_to_kv_pool, "full_attention_layer_id_mapping", None
        )

        if isinstance(mapping, dict) and mapping:
            first_full_attn_id = next(iter(mapping.keys()))
        else:
            first_full_attn_id = 0

        # Pre-initialized qo_indptr for pa_persistent_fwd decode mode: [0, 1, 2, ..., max_bs]
        # In decode mode, each sequence has 1 token, so this is always [0, 1, 2, ..., batch_size]
        max_bs = model_runner.req_to_token_pool.size
        self.pa_decode_qo_indptr = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=model_runner.device
        )
        self.seq_lens = torch.zeros(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.page_table = torch.zeros(
            (max_bs, self.max_context_len // self.page_size),
            dtype=torch.int32,
            device=model_runner.device,
        )
        # Pre-compute strided indices for page_table construction (used in both CUDA Graph and non-CUDA Graph modes)
        self.strided_indices = torch.arange(
            0, self.max_context_len, self.page_size, device=model_runner.device
        )

        if not self.use_mla:
            # Pre-allocate buffers for pa_persistent_fwd (used in both CUDA graph and non-CUDA graph modes)
            max_num_blocks_per_seq = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size
            max_total_blocks = max_bs * max_num_blocks_per_seq
            self.pa_kv_indices = torch.zeros(
                max_total_blocks, dtype=torch.int32, device=self.device
            )
            # Pre-allocate pa_kv_indptr buffer (similar to self.kv_indptr, but dedicated for pa_persistent_fwd)
            self.pa_kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )
            # Pre-initialized batch indices [0, 1, 2, ..., max_bs-1] for Triton kernel
            self.pa_batch_indices = torch.arange(
                0, max_bs, dtype=torch.int32, device=self.device
            )

        # Pre-allocated descale tensors for FP8 attention (q, k, v all use scale=1.0)

        self.forward_metadata: ForwardMetadata = None

        self.pa_metadata_buffers = None

        k_buffer, _ = model_runner.token_to_kv_pool.get_kv_buffer(first_full_attn_id)
        num_slots, num_kv_heads, _ = k_buffer.shape
        block_size = self.page_size
        num_blocks = num_slots // block_size
        max_total_tokens = num_blocks * block_size
        self.k_qscale = torch.ones(
            num_kv_heads, max_total_tokens, dtype=torch.float32, device=self.device
        )
        self.v_qscale = torch.ones(
            num_kv_heads, max_total_tokens, dtype=torch.float32, device=self.device
        )
        self.decode_using_pa_ps = self.page_size == 1024

    def _cuda_graph_mla_max_seqlen_qo(self) -> int:
        """Largest q length used by MLA CUDA graph speculative paths."""
        max_seqlen_qo = 1
        if self.num_draft_tokens is not None:
            max_seqlen_qo = max(max_seqlen_qo, self.num_draft_tokens)
        if self.speculative_num_steps is not None:
            max_seqlen_qo = max(max_seqlen_qo, self.speculative_num_steps + 1)
        return max_seqlen_qo

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""
        if forward_batch.forward_mode.is_decode_or_idle():
            self._init_forward_metadata_decode(forward_batch)
        elif self.use_mla and forward_batch.forward_mode.is_draft_extend_v2():
            self._init_draft_extend_v2_mla(forward_batch.batch_size, forward_batch)
        elif self.use_mla and forward_batch.forward_mode.is_draft_extend():
            self._init_draft_extend_mla(forward_batch.batch_size, forward_batch)
        elif self.use_mla and forward_batch.forward_mode.is_target_verify():
            self._init_target_verify_mla(forward_batch.batch_size, forward_batch)
        else:
            self._init_forward_metadata_extend(forward_batch)
        self._fixup_page_table(forward_batch)

    def _init_forward_metadata_decode(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        spec_info = forward_batch.spec_info

        if spec_info is None:
            kv_indptr = self.kv_indptr
            kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                forward_batch.seq_lens_sum, dtype=torch.int32, device=self.device
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
            bs = kv_indptr.shape[0] - 1

        if self.use_mla:
            self._init_decode_mla(bs, kv_indptr, kv_indices)
        else:
            self._init_decode_mha(bs, kv_indptr, kv_indices, forward_batch)

    def _init_decode_mla(self, bs, kv_indptr, kv_indices):
        qo_indptr = self.qo_indptr_[: bs + 1]
        qo_indptr[1 : bs + 1] = torch.cumsum(self.kv_last_page_len[:bs], dim=0)
        kv_last_page_len = self.kv_last_page_len[:bs]
        max_q_len = 1

        work_metadata = None
        work_indptr = None
        work_info_set = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None
        num_kv_splits = None

        if _sglang_aiter._use_mla_ps_kernel:
            (
                work_metadata,
                work_indptr,
                work_info_set,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
            ) = self.make_mla_decode_meta_data_buffer(max_q_len, bs)
            num_kv_splits = self.max_split_per_batch
            self.make_mla_meta_data(
                qo_indptr,
                kv_indptr,
                kv_last_page_len,
                work_metadata,
                work_info_set,
                work_indptr,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                max_q_len,
                fast_mode=_sglang_aiter.fast_mode,
                max_split_per_batch=num_kv_splits,
                intra_batch_mode=_sglang_aiter.intra_batch_mode,
            )

        self.forward_metadata = ForwardMetadata(
            kv_indptr,
            kv_indices,
            qo_indptr,
            kv_last_page_len,
            max_q_len,
            None,
            None,
            None,
            work_metadata=work_metadata,
            work_info_set=work_info_set,
            work_indptr=work_indptr,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            num_kv_splits=num_kv_splits,
        )

    def _init_decode_mha(self, bs, kv_indptr, kv_indices, forward_batch):
        seq_lens_i32 = forward_batch.seq_lens[:bs]
        if seq_lens_i32.dtype != torch.int32:
            seq_lens_i32 = seq_lens_i32.to(torch.int32)

        if self.decode_using_pa_ps:
            seq_lens_cpu = forward_batch.seq_lens_cpu
            if seq_lens_cpu is None:
                seq_lens_cpu = forward_batch.seq_lens.cpu()

            page_table, seq_lens = self._update_decode_page_table(
                bs,
                forward_batch.req_pool_indices,
                seq_lens_i32,
                seq_lens_cpu=seq_lens_cpu,
            )
            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                None,
                None,
                1,
                None,
                page_table,
                seq_lens,
            )
            _build_pa_metadata_for_decode(self, bs, tp_q_head_num=self.num_head)
        else:
            page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, :
            ]
            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                None,
                None,
                1,
                None,
                page_table,
                seq_lens_i32,
            )

    def _init_forward_metadata_extend(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size

        if self.use_mla:
            self._init_extend_mla(bs, forward_batch)
        else:
            self._init_extend_mha(bs, forward_batch)

    def _init_draft_extend_mla(self, bs, forward_batch):
        """Init MLA metadata for speculative draft_extend."""
        spec_info = forward_batch.spec_info
        if spec_info is None:
            raise RuntimeError("MLA draft_extend requires speculative metadata")

        kv_indices, kv_indptr, qo_indptr, _ = spec_info.generate_attn_arg_prefill(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.seq_lens_sum,
            self.req_to_token,
        )

        extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        if extend_seq_lens_cpu is not None:
            max_q_len = (
                int(extend_seq_lens_cpu.max().item())
                if isinstance(extend_seq_lens_cpu, torch.Tensor)
                else max(extend_seq_lens_cpu)
            )
        elif forward_batch.extend_seq_lens is not None:
            max_q_len = int(forward_batch.extend_seq_lens.max().item())
        elif getattr(spec_info, "accept_length", None) is not None:
            max_q_len = int(spec_info.accept_length.max().item())
        else:
            raise RuntimeError("MLA draft_extend is missing extend sequence lengths")

        seq_lens_cpu = forward_batch.seq_lens_cpu
        max_kv_len = (
            (
                int(seq_lens_cpu.max().item())
                if isinstance(seq_lens_cpu, torch.Tensor)
                else max(seq_lens_cpu)
            )
            if seq_lens_cpu is not None
            else int(forward_batch.seq_lens.max().item())
        )

        work_metadata = None
        work_indptr = None
        work_info_set = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None
        num_kv_splits = None

        if _sglang_aiter._use_mla_ps_kernel:
            (
                work_metadata,
                work_indptr,
                work_info_set,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
            ) = self.make_mla_decode_meta_data_buffer(max_q_len, bs)
            num_kv_splits = self.max_split_per_batch
            self.make_mla_meta_data(
                qo_indptr,
                kv_indptr,
                self.kv_last_page_len[:bs],
                work_metadata,
                work_info_set,
                work_indptr,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                max_q_len,
                fast_mode=_sglang_aiter.fast_mode,
                max_split_per_batch=num_kv_splits,
                intra_batch_mode=_sglang_aiter.intra_batch_mode,
            )

        self.forward_metadata = ForwardMetadata(
            kv_indptr,
            kv_indices,
            qo_indptr,
            self.kv_last_page_len[:bs],
            max_q_len,
            max_kv_len,
            None,
            None,
            work_metadata=work_metadata,
            work_info_set=work_info_set,
            work_indptr=work_indptr,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            num_kv_splits=num_kv_splits,
            run_graph=False,
        )

    def _init_draft_extend_v2_mla(self, bs, forward_batch):
        """Init MLA metadata for speculative DRAFT_EXTEND_V2.

        SpecV2 draft-extend uses a fixed number of draft tokens per request,
        unlike V1 where each request can have a different accepted length.
        """
        spec_info = forward_batch.spec_info
        if spec_info is None:
            raise RuntimeError("MLA draft_extend_v2 requires speculative metadata")

        self._ensure_spec_v2_topk_supported()
        num_draft_tokens = self._resolve_v2_num_draft_tokens(
            forward_batch.extend_seq_lens,
            forward_batch.extend_seq_lens_cpu,
        )
        device = forward_batch.seq_lens.device
        qo_indptr = self._set_uniform_qo_indptr(bs, num_draft_tokens, device)

        kv_indptr = self.kv_indptr[: bs + 1]
        kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
        kv_indices = torch.empty(
            forward_batch.seq_lens_sum, dtype=torch.int32, device=device
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token.stride(0),
        )

        max_kv_len = (
            int(forward_batch.seq_lens_cpu.max().item())
            if forward_batch.seq_lens_cpu is not None
            else int(forward_batch.seq_lens.max().item())
        )

        work_metadata = None
        work_indptr = None
        work_info_set = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None
        num_kv_splits = None

        if _sglang_aiter._use_mla_ps_kernel:
            (
                work_metadata,
                work_indptr,
                work_info_set,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
            ) = self.make_mla_decode_meta_data_buffer(num_draft_tokens, bs)
            num_kv_splits = self.max_split_per_batch
            self.make_mla_meta_data(
                qo_indptr,
                kv_indptr,
                self.kv_last_page_len[:bs],
                work_metadata,
                work_info_set,
                work_indptr,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                num_draft_tokens,
                fast_mode=_sglang_aiter.fast_mode,
                max_split_per_batch=num_kv_splits,
                intra_batch_mode=_sglang_aiter.intra_batch_mode,
            )

        self.forward_metadata = ForwardMetadata(
            kv_indptr,
            kv_indices,
            qo_indptr,
            self.kv_last_page_len[:bs],
            num_draft_tokens,
            max_kv_len,
            None,
            None,
            work_metadata=work_metadata,
            work_info_set=work_info_set,
            work_indptr=work_indptr,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            num_kv_splits=num_kv_splits,
            run_graph=False,
        )

    def _init_target_verify_mla(self, bs, forward_batch):
        """Init MLA metadata for speculative target_verify."""
        spec_info = forward_batch.spec_info
        if spec_info is None:
            raise RuntimeError("MLA target_verify requires speculative metadata")

        draft_num = spec_info.draft_token_num
        kv_lens = forward_batch.seq_lens + draft_num
        kv_lens_sum = forward_batch.seq_lens_sum + draft_num * bs
        device = forward_batch.seq_lens.device

        qo_indptr = torch.arange(
            0,
            (1 + bs) * draft_num,
            step=draft_num,
            dtype=torch.int32,
            device=device,
        )
        kv_indptr = self.kv_indptr
        kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]
        kv_indices = torch.empty(
            kv_lens_sum,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            forward_batch.req_pool_indices,
            kv_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token.stride(0),
        )

        work_metadata = None
        work_indptr = None
        work_info_set = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None
        num_kv_splits = None

        if _sglang_aiter._use_mla_ps_kernel:
            (
                work_metadata,
                work_indptr,
                work_info_set,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
            ) = self.make_mla_decode_meta_data_buffer(draft_num, bs)
            num_kv_splits = self.max_split_per_batch
            self.make_mla_meta_data(
                qo_indptr,
                kv_indptr,
                self.kv_last_page_len[:bs],
                work_metadata,
                work_info_set,
                work_indptr,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                draft_num,
                fast_mode=_sglang_aiter.fast_mode,
                max_split_per_batch=num_kv_splits,
                intra_batch_mode=_sglang_aiter.intra_batch_mode,
            )

        self.forward_metadata = ForwardMetadata(
            kv_indptr,
            kv_indices,
            qo_indptr,
            self.kv_last_page_len[:bs],
            draft_num,
            None,
            None,
            None,
            work_metadata=work_metadata,
            work_info_set=work_info_set,
            work_indptr=work_indptr,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            num_kv_splits=num_kv_splits,
            run_graph=False,
        )

    def _init_extend_mla(self, bs, forward_batch):
        self.mla_indices_updater_prefill.update(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.seq_lens_sum,
            forward_batch.extend_seq_lens,
            forward_batch.extend_seq_lens.max().item(),
            forward_batch.seq_lens.max().item(),
            spec_info=None,
        )

        max_q_len = self.mla_indices_updater_prefill.max_q_len
        qo_indptr = self.mla_indices_updater_prefill.qo_indptr
        kv_indptr = self.mla_indices_updater_prefill.kv_indptr

        work_metadata = None
        work_indptr = None
        work_info_set = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None
        fp8_prefill_kv_indices = None

        from sglang.srt.utils import is_gfx95_supported

        _use_fp8_prefill_attn = (
            get_bool_env_var("SGLANG_AITER_FP8_PREFILL_ATTN", "True")
            and is_gfx95_supported()
        )
        if _use_fp8_prefill_attn:
            tile_q = 256
            qlen_granularity = tile_q // (self.num_head // self.num_kv_head)
            (
                work_metadata,
                work_indptr,
                work_info_set,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
            ) = self.make_mla_prefill_ps_meta_data_buffer(
                bs, max_q_len, qlen_granularity
            )
            self.make_mla_prefill_ps_meta_data(
                qo_indptr,
                kv_indptr,
                forward_batch.seq_lens,
                work_metadata,
                work_indptr,
                work_info_set,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                is_causal=True,
            )
            total_s = int(forward_batch.seq_lens_sum)
            fp8_prefill_kv_indices = torch.arange(
                total_s, device=self.device, dtype=torch.int32
            )

        self.forward_metadata = ForwardMetadata(
            kv_indptr,
            self.mla_indices_updater_prefill.kv_indices,
            qo_indptr,
            self.kv_last_page_len[:bs],
            max_q_len,
            self.mla_indices_updater_prefill.max_kv_len,
            None,
            None,
            work_metadata=work_metadata,
            work_info_set=work_info_set,
            work_indptr=work_indptr,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            fp8_prefill_kv_indices=fp8_prefill_kv_indices,
        )

    def _init_extend_mha(self, bs, forward_batch):
        self.indices_updater_prefill.update(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.seq_lens_sum,
            forward_batch.extend_prefix_lens,
            encoder_lens=forward_batch.encoder_lens,
            spec_info=None,
        )
        self.forward_metadata = ForwardMetadata(
            self.indices_updater_prefill.kv_indptr,
            self.indices_updater_prefill.kv_indices,
            self.qo_indptr[: bs + 1],
            None,
            self._max_len(
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_seq_lens,
            ),
            self._max_len(forward_batch.seq_lens_cpu, forward_batch.seq_lens),
            None,
            forward_batch.seq_lens,
        )

    def _fixup_page_table(self, forward_batch: ForwardBatch):
        """Post-process page_table for non-MLA extend mode."""
        if (
            forward_batch.forward_mode.is_extend()
            and not self.use_mla
            and self.forward_metadata.page_table is not None
        ):
            if self.page_size > 1:
                seq_lens_cpu = forward_batch.seq_lens_cpu
                if seq_lens_cpu is None:
                    seq_lens_cpu = forward_batch.seq_lens.cpu()
                max_seq_pages = (
                    seq_lens_cpu.max().item() + self.page_size - 1
                ) // self.page_size + 1
                self.forward_metadata.page_table = (
                    self.forward_metadata.page_table[
                        :, self.strided_indices[:max_seq_pages]
                    ]
                    // self.page_size
                )
            if self.decode_using_pa_ps:
                _build_pa_metadata_for_prefill(self, forward_batch.batch_size)
        if (
            not self.decode_using_pa_ps
            and self.page_size > 1
            and self.forward_metadata.page_table is not None
        ):
            self.forward_metadata.page_table = (
                self.forward_metadata.page_table[:, self.strided_indices]
                // self.page_size
            )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_kv_last_page_len = torch.ones(
            max_bs, dtype=torch.int, device=self.device
        )
        assert self.cuda_graph_kv_last_page_len.is_cuda, (
            "ATOMAttnBackendForSgl.init_cuda_graph_state created "
            f"non-CUDA cuda_graph_kv_last_page_len on {self.cuda_graph_kv_last_page_len.device}, "
            f"backend={type(self)}"
        )
        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_bs * self.max_context_len),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        # Always use preshuffle layout for pa_fwd_asm
        self.page_table = torch.zeros(
            (max_bs, self.max_context_len // self.page_size),
            dtype=torch.int32,
            device=self.device,
        )
        self.seq_lens = torch.zeros((max_bs,), dtype=torch.int32, device=self.device)
        self.strided_indices = torch.arange(
            0, self.max_context_len, self.page_size, device=self.device
        )

        if self.use_mla and _sglang_aiter._use_mla_ps_kernel:
            max_seqlen_qo = self._cuda_graph_mla_max_seqlen_qo()
            (
                self.work_metadata,
                self.work_indptr,
                self.work_info_set,
                self.reduce_indptr,
                self.reduce_final_map,
                self.reduce_partial_map,
            ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, max_bs)
        elif self.use_mla:
            self.work_metadata = None
            self.work_indptr = None
            self.work_info_set = None
            self.reduce_indptr = None
            self.reduce_final_map = None
            self.reduce_partial_map = None

        if self.decode_using_pa_ps and not self.use_mla:
            buffer_specs = get_pa_metadata_info_v1(max_bs, self.num_kv_head)
            _allocate_pa_metadata_buffers(self, buffer_specs)

    def _init_mla_cuda_graph_metadata(self, bs, req_pool_indices, seq_lens):
        """Shared MLA decode metadata setup for CUDA graph capture/replay."""
        kv_indptr = self.kv_indptr
        kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]
        kv_indices = self.cuda_graph_kv_indices
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token.stride(0),
        )

        qo_indptr = self.qo_indptr_[: bs + 1]
        qo_indptr[1 : bs + 1] = torch.cumsum(
            self.cuda_graph_kv_last_page_len[:bs], dim=0
        )
        kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
        max_q_len = 1

        work_metadata = None
        work_indptr = None
        work_info_set = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None
        num_kv_splits = None

        if _sglang_aiter._use_mla_ps_kernel:
            num_kv_splits = self.max_split_per_batch

            self.make_mla_meta_data(
                qo_indptr,
                kv_indptr,
                kv_last_page_len,
                self.work_metadata,
                self.work_info_set,
                self.work_indptr,
                self.reduce_indptr,
                self.reduce_final_map,
                self.reduce_partial_map,
                max_q_len,
                fast_mode=_sglang_aiter.fast_mode,
                max_split_per_batch=num_kv_splits,
                intra_batch_mode=_sglang_aiter.intra_batch_mode,
            )

            work_metadata = self.work_metadata
            work_info_set = self.work_info_set
            work_indptr = self.work_indptr
            reduce_indptr = self.reduce_indptr
            reduce_final_map = self.reduce_final_map
            reduce_partial_map = self.reduce_partial_map

        self.forward_metadata = ForwardMetadata(
            kv_indptr,
            kv_indices,
            qo_indptr,
            kv_last_page_len,
            max_q_len,
            None,
            None,
            None,
            work_metadata=work_metadata,
            work_info_set=work_info_set,
            work_indptr=work_indptr,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            num_kv_splits=num_kv_splits,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        num_kv_splits = None
        work_metadata = None
        work_info_set = None
        work_indptr = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        if forward_mode.is_decode_or_idle():
            if self.use_mla:
                self._init_mla_cuda_graph_metadata(bs, req_pool_indices, seq_lens)
            else:
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                page_table, seq_lens_persistent = self._update_decode_page_table(
                    bs,
                    req_pool_indices,
                    seq_lens,
                    static_columns=True,
                )
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    None,
                    None,
                    1,
                    None,
                    page_table,
                    seq_lens_persistent,
                )
                if self.decode_using_pa_ps:
                    _build_pa_metadata_for_decode(self, bs, tp_q_head_num=self.num_head)
        elif forward_mode.is_target_verify():
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_lens = seq_lens + self.num_draft_tokens if self.use_mla else seq_lens
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = self.num_draft_tokens

            if self.use_mla:
                if _sglang_aiter._use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch
                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=_sglang_aiter.fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=_sglang_aiter.intra_batch_mode,
                    )
                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr
                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    kv_indptr[-1].item(),
                    None,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
                assert (
                    self.forward_metadata.kv_last_page_len is None
                    or self.forward_metadata.kv_last_page_len.is_cuda
                ), (
                    "capture_cuda_graph TARGET_VERIFY produced non-CUDA kv_last_page_len: "
                    f"{self.forward_metadata.kv_last_page_len.device}, "
                    f"backend={type(self)}, metadata_backend={type(self.forward_metadata)}"
                )
            else:
                custom_mask = self.cuda_graph_custom_mask
                assert spec_info is not None and spec_info.custom_mask is not None
                custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
                seq_mask_len = max_q_len * (seq_lens + max_q_len)
                mask_indptr = self.mask_indptr
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
                mask_indptr = mask_indptr[: bs + 1]
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    kv_indptr[-1].item(),
                    None,
                    None,
                    custom_mask=custom_mask,
                    mask_indptr=mask_indptr,
                    max_extend_len=max_q_len,
                )
                assert (
                    self.forward_metadata.kv_last_page_len is None
                    or self.forward_metadata.kv_last_page_len.is_cuda
                ), (
                    "capture_cuda_graph TARGET_VERIFY(non-MLA) produced non-CUDA kv_last_page_len: "
                    f"{self.forward_metadata.kv_last_page_len.device}, "
                    f"backend={type(self)}"
                )
        elif forward_mode.is_draft_extend_v2():
            self._ensure_spec_v2_topk_supported()
            num_tokens_per_bs = self._resolve_v2_num_draft_tokens()
            qo_indptr = self._set_uniform_qo_indptr(bs, num_tokens_per_bs, self.device)
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            if self.use_mla:
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = num_tokens_per_bs
                if _sglang_aiter._use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch
                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=_sglang_aiter.fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=_sglang_aiter.intra_batch_mode,
                    )
                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr
                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    kv_indptr[-1].item(),
                    None,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
                assert (
                    self.forward_metadata.kv_last_page_len is None
                    or self.forward_metadata.kv_last_page_len.is_cuda
                ), (
                    "capture_cuda_graph DRAFT_EXTEND_V2 produced non-CUDA kv_last_page_len: "
                    f"{self.forward_metadata.kv_last_page_len.device}, "
                    f"backend={type(self)}"
                )
            else:
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    num_tokens_per_bs,
                    None,
                    None,
                    None,
                    custom_mask=None,
                    mask_indptr=None,
                    max_extend_len=num_tokens_per_bs,
                )
        elif forward_mode.is_draft_extend():
            num_tokens_per_bs = self.speculative_num_steps + 1
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                step=num_tokens_per_bs,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            if self.use_mla:
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = num_tokens_per_bs
                if _sglang_aiter._use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch
                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=_sglang_aiter.fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=_sglang_aiter.intra_batch_mode,
                    )
                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr
                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    kv_indptr[-1].item(),
                    None,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
                assert (
                    self.forward_metadata.kv_last_page_len is None
                    or self.forward_metadata.kv_last_page_len.is_cuda
                ), (
                    "capture_cuda_graph DRAFT_EXTEND produced non-CUDA kv_last_page_len: "
                    f"{self.forward_metadata.kv_last_page_len.device}, "
                    f"backend={type(self)}"
                )
            else:
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    num_tokens_per_bs,
                    None,
                    None,
                    None,
                    custom_mask=None,
                    mask_indptr=None,
                    max_extend_len=num_tokens_per_bs,
                )
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        num_kv_splits = None
        work_metadata = None
        work_info_set = None
        work_indptr = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        if forward_mode.is_decode_or_idle():
            if self.use_mla:
                self._init_mla_cuda_graph_metadata(bs, req_pool_indices, seq_lens)
            else:
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                page_table, seq_lens_persistent = self._update_decode_page_table(
                    bs,
                    req_pool_indices,
                    seq_lens,
                    seq_lens_cpu=seq_lens_cpu,
                    static_columns=True,
                )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    None,
                    None,
                    1,
                    None,
                    page_table,
                    seq_lens_persistent[:bs],
                )
                if self.decode_using_pa_ps:
                    _build_pa_metadata_for_decode(self, bs, tp_q_head_num=self.num_head)
        elif forward_mode.is_target_verify():
            bs = len(req_pool_indices)
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_lens = seq_lens + self.num_draft_tokens if self.use_mla else seq_lens
            target_verify_kv_len_sum = seq_lens_sum + self.num_draft_tokens * bs
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = self.num_draft_tokens

            if self.use_mla:
                if _sglang_aiter._use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch
                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=_sglang_aiter.fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=_sglang_aiter.intra_batch_mode,
                    )
                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr
                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    target_verify_kv_len_sum,
                    None,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
                assert (
                    self.forward_metadata.kv_last_page_len is None
                    or self.forward_metadata.kv_last_page_len.is_cuda
                ), (
                    "replay_cuda_graph TARGET_VERIFY produced non-CUDA kv_last_page_len: "
                    f"{self.forward_metadata.kv_last_page_len.device}, "
                    f"backend={type(self)}, metadata_backend={type(self.forward_metadata)}"
                )
            else:
                custom_mask = self.cuda_graph_custom_mask
                assert spec_info is not None and spec_info.custom_mask is not None
                custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
                seq_mask_len = max_q_len * (seq_lens + max_q_len)
                mask_indptr = self.mask_indptr
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
                mask_indptr = mask_indptr[: bs + 1]
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    target_verify_kv_len_sum,
                    None,
                    None,
                    custom_mask=custom_mask,
                    mask_indptr=mask_indptr,
                    max_extend_len=max_q_len,
                )
                assert (
                    self.forward_metadata.kv_last_page_len is None
                    or self.forward_metadata.kv_last_page_len.is_cuda
                ), (
                    "replay_cuda_graph TARGET_VERIFY(non-MLA) produced non-CUDA kv_last_page_len: "
                    f"{self.forward_metadata.kv_last_page_len.device}, "
                    f"backend={type(self)}"
                )
        elif forward_mode.is_draft_extend_v2():
            self._ensure_spec_v2_topk_supported()
            num_tokens_per_bs = self._resolve_v2_num_draft_tokens()
            qo_indptr = self._set_uniform_qo_indptr(bs, num_tokens_per_bs, self.device)
            seq_lens = seq_lens[:bs]
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            if self.use_mla:
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = num_tokens_per_bs
                if _sglang_aiter._use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch
                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=_sglang_aiter.fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=_sglang_aiter.intra_batch_mode,
                    )
                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr
                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    seq_lens_sum,
                    None,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
                assert (
                    self.forward_metadata.kv_last_page_len is None
                    or self.forward_metadata.kv_last_page_len.is_cuda
                ), (
                    "replay_cuda_graph DRAFT_EXTEND_V2 produced non-CUDA kv_last_page_len: "
                    f"{self.forward_metadata.kv_last_page_len.device}, "
                    f"backend={type(self)}"
                )
            else:
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    num_tokens_per_bs,
                    None,
                    None,
                    None,
                    custom_mask=None,
                    mask_indptr=None,
                    max_extend_len=num_tokens_per_bs,
                )
        elif forward_mode.is_draft_extend():
            num_tokens_per_bs = self.speculative_num_steps + 1
            seq_lens = seq_lens[:bs]
            accept_lens = spec_info.accept_length[:bs]
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(accept_lens, dim=0)
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            if self.use_mla:
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = num_tokens_per_bs
                if _sglang_aiter._use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch
                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=_sglang_aiter.fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=_sglang_aiter.intra_batch_mode,
                    )
                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr
                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    seq_lens_sum,
                    None,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
                assert (
                    self.forward_metadata.kv_last_page_len is None
                    or self.forward_metadata.kv_last_page_len.is_cuda
                ), (
                    "replay_cuda_graph DRAFT_EXTEND produced non-CUDA kv_last_page_len: "
                    f"{self.forward_metadata.kv_last_page_len.device}, "
                    f"backend={type(self)}"
                )
            else:
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    num_tokens_per_bs,
                    None,
                    None,
                    None,
                    custom_mask=None,
                    mask_indptr=None,
                    max_extend_len=num_tokens_per_bs,
                )
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    @staticmethod
    def _max_len(cpu_values, device_values) -> int:
        values = cpu_values if cpu_values is not None else device_values
        if isinstance(values, torch.Tensor):
            return int(values.max().item())
        return int(max(values))

    def _update_decode_page_table(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        static_columns: bool = False,
    ):
        page_table_persistent = self.page_table
        seq_lens_persistent = self.seq_lens
        seq_lens_persistent.fill_(0)
        page_table_persistent.fill_(0)
        seq_lens_persistent[:bs].copy_(seq_lens, non_blocking=True)

        if seq_lens_cpu is None:
            max_seq_len = int(seq_lens.max().item())
        elif isinstance(seq_lens_cpu, torch.Tensor):
            max_seq_len = int(seq_lens_cpu.max().item())
        else:
            max_seq_len = int(max(seq_lens_cpu))

        max_seq_pages = (max_seq_len + self.page_size - 1) // self.page_size + 1
        max_seq_pages = min(max_seq_pages, page_table_persistent.shape[1])
        page_table = self.req_to_token[
            req_pool_indices[:, None],
            self.strided_indices[:max_seq_pages][None, :],
        ]
        page_table_persistent[:bs, :max_seq_pages].copy_(
            page_table // self.page_size, non_blocking=True
        )

        if static_columns:
            return page_table_persistent[:bs, :], seq_lens_persistent[:bs]
        return page_table_persistent[:bs, :max_seq_pages], seq_lens_persistent[:bs]

    def _should_use_native_dense_mha(self, layer) -> bool:
        sliding_window_size = getattr(layer, "sliding_window_size", None)
        return (
            not self.use_mla
            and not layer.is_cross_attention
            and layer.head_dim == 256
            and layer.qk_head_dim == 256
            and layer.v_head_dim == 256
            and (sliding_window_size is None or sliding_window_size <= -1)
        )

    def _kv_descales(self, layer):
        if self.kv_cache_dtype != dtypes.fp8:
            return None, None
        k_descale = layer.k_scale if layer.k_scale is not None else self.k_scale
        v_descale = layer.v_scale if layer.v_scale is not None else self.v_scale
        return k_descale, v_descale

    def _ensure_fp8_kv_scales(self, layer, k_buffer, block_size):
        if self.kv_cache_dtype != dtypes.fp8:
            return layer.k_scale, layer.v_scale

        num_slots, num_kv_heads, _ = k_buffer.shape
        num_blocks = num_slots // block_size
        expected_shape = (num_blocks, num_kv_heads, block_size)

        def needs_alloc(scale):
            return (
                scale is None
                or scale.numel() <= 1
                or tuple(scale.shape) != expected_shape
                or scale.device != k_buffer.device
            )

        if needs_alloc(layer.k_scale):
            layer.k_scale = torch.nn.Parameter(
                torch.ones(expected_shape, dtype=torch.float32, device=k_buffer.device),
                requires_grad=False,
            )
        if needs_alloc(layer.v_scale):
            layer.v_scale = torch.nn.Parameter(
                torch.ones(expected_shape, dtype=torch.float32, device=k_buffer.device),
                requires_grad=False,
            )

        return layer.k_scale, layer.v_scale

    def _get_aiter_paged_ragged_kv_cache_dtype(self) -> str:
        if self.kv_cache_dtype != dtypes.fp8:
            return "auto"
        return "fp8_e4m3"

    def _set_kv_buffer_native_dense(self, layer, cache_loc, k, v, forward_batch):
        k_descale, v_descale = self._kv_descales(layer)
        if self.kv_cache_dtype == dtypes.fp8:
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            launch_reshape_and_cache_flash(
                k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                k_cache.view(
                    -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                ),
                v_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.v_head_dim),
                cache_loc,
                k_scale=k_descale,
                v_scale=v_descale,
            )
            return

        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, cache_loc, k, v, k_descale, v_descale
        )

    def set_kv_buffer_with_layout_shuffle(
        self,
        cache_loc,
        k,
        v,
        k_buffer,
        v_buffer,
        k_scale,
        v_scale,
        block_size,
    ):
        _set_kv_buffer_with_layout_shuffle(
            cache_loc,
            k,
            v,
            k_buffer,
            v_buffer,
            k_scale,
            v_scale,
            block_size,
        )

    def _forward_sparse_mla(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        topk_indices: torch.Tensor,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        from atom.plugin.sglang.attention_backend.sparse_mla_indexer import (
            forward_sparse_mla_for_sglang,
        )

        return forward_sparse_mla_for_sglang(
            q,
            k,
            v,
            layer,
            forward_batch,
            topk_indices,
            save_kv_cache=save_kv_cache,
            input_dtype=self.input_dtype,
        )

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        topk_indices = kwargs.get("topk_indices")
        if self.use_mla and topk_indices is not None:
            return self._forward_sparse_mla(
                q, k, v, layer, forward_batch, topk_indices, save_kv_cache
            )

        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        use_native_dense_mha = self._should_use_native_dense_mha(layer)

        if k is not None:
            assert v is not None
            if save_kv_cache:
                if use_native_dense_mha:
                    self._set_kv_buffer_native_dense(
                        layer, cache_loc, k, v, forward_batch
                    )
                elif self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
                else:
                    k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
                    k_scale, v_scale = self._ensure_fp8_kv_scales(
                        layer, k_buffer, self.page_size
                    )
                    self.set_kv_buffer_with_layout_shuffle(
                        cache_loc,
                        k,
                        v,
                        k_buffer,
                        v_buffer,
                        k_scale,
                        v_scale,
                        self.page_size,
                    )

        if self.use_mla:
            return self._forward_extend_mla(q, k, v, layer, forward_batch)
        if use_native_dense_mha:
            return self._forward_extend_native_dense_mha(q, layer, forward_batch)
        else:
            return self._forward_extend_mha(q, k, v, layer, forward_batch)

    def _forward_extend_native_dense_mha(self, q, layer, forward_batch):
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        q_descale, k_descale, v_descale = None, None, None

        if self.kv_cache_dtype == dtypes.fp8:
            q = q.to(dtypes.fp8)
            q_descale = layer.k_scale if layer.k_scale is not None else self.k_scale
            k_descale = layer.k_scale if layer.k_scale is not None else self.k_scale
            v_descale = layer.v_scale if layer.v_scale is not None else self.v_scale

        bs0 = forward_batch.batch_size + 1
        o = mha_batch_prefill_func(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            k_cache,
            v_cache,
            self.forward_metadata.qo_indptr[:bs0],
            self.forward_metadata.kv_indptr[:bs0],
            self.forward_metadata.kv_indices,
            self.forward_metadata.max_q_len,
            self.forward_metadata.max_kv_len,
            causal=True,
            logits_soft_cap=layer.logit_cap,
            alibi_slopes=None,
            return_lse=False,
            return_attn_probs=False,
            window_size=(-1, -1),
            sink_ptr=None,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _forward_extend_mha(self, q, k, v, layer, forward_batch):
        """Non-MLA extend path: standard MHA with flash_attn_varlen_func."""
        seqlens_in_batch = forward_batch.seq_lens
        cu_seqlens_q = torch.nn.functional.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )
        if q.dtype != k.dtype and k.dtype == dtypes.fp8:
            q = q.to(dtypes.fp8)
        o = flash_attn_varlen_func(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            k.contiguous().view(-1, layer.tp_k_head_num, layer.head_dim),
            v.contiguous().view(-1, layer.tp_v_head_num, layer.head_dim),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=self.forward_metadata.max_q_len,
            max_seqlen_k=self.forward_metadata.max_kv_len,
            min_seqlen_q=0,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            window_size=(-1, -1, 0),
            sink_ptr=None,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _forward_extend_mla(self, q, k, v, layer, forward_batch):
        """MLA extend path: ported from sglang aiter_backend forward_extend MLA logic."""
        max_q_len = self.forward_metadata.max_q_len
        max_kv_len = self.forward_metadata.max_kv_len
        kv_indptr = self.forward_metadata.kv_indptr
        kv_indices = self.forward_metadata.kv_indices
        qo_indptr = self.forward_metadata.qo_indptr

        K_Buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        assert len(q.shape) == 3

        if forward_batch.forward_mode.is_target_verify():
            return self._forward_extend_mla_speculative(
                q,
                layer,
                K_Buffer,
                qo_indptr,
                forward_batch,
            )
        if forward_batch.forward_mode.is_draft_extend(include_v2=True) and (
            k is None or v is None or layer.qk_head_dim == K_Buffer.shape[-1]
        ):
            return self._forward_extend_mla_speculative(
                q,
                layer,
                K_Buffer,
                qo_indptr,
                forward_batch,
            )
        if (
            not forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            raise ValueError(
                f"Invalid forward mode for MLA extend: {forward_batch.forward_mode=}"
            )
        if k is None or v is None:
            raise RuntimeError("MLA normal extend requires explicit k/v tensors")

        V_Buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        kv_lora_rank = V_Buffer.shape[-1]
        qk_rope_head_dim = K_Buffer.shape[-1] - kv_lora_rank
        qk_nope_head_dim = k.shape[-1] - qk_rope_head_dim

        assert len(k.shape) == 3
        assert len(v.shape) == 3
        return self._forward_extend_mla_normal(
            q,
            k,
            v,
            layer,
            forward_batch,
            K_Buffer,
            V_Buffer,
            kv_lora_rank,
            qk_rope_head_dim,
            qk_nope_head_dim,
            max_q_len,
            max_kv_len,
            kv_indptr,
            kv_indices,
            qo_indptr,
        )

    def _forward_extend_mla_normal(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        K_Buffer,
        V_Buffer,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_nope_head_dim,
        max_q_len,
        max_kv_len,
        kv_indptr,
        kv_indices,
        qo_indptr,
    ):
        """MLA prefill/extend using explicit q/k/v instead of absorbed decode."""
        extend_prefix_lens_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        extend_no_prefix = (
            False if extend_prefix_lens_cpu is None else not any(extend_prefix_lens_cpu)
        )

        if kv_indices.shape[0] == 0 or extend_no_prefix:
            return self._extend_mla_no_prefix(
                q,
                k,
                v,
                layer,
                kv_lora_rank,
                qk_rope_head_dim,
                max_q_len,
                qo_indptr,
            )
        elif layer.qk_head_dim != (kv_lora_rank + qk_rope_head_dim):
            # non-absorbed MLA: qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            return self._extend_mla_decompress_prefix(
                q,
                layer,
                forward_batch,
                K_Buffer,
                kv_lora_rank,
                qk_rope_head_dim,
                qk_nope_head_dim,
                max_q_len,
                max_kv_len,
                kv_indptr,
                kv_indices,
                qo_indptr,
            )
        else:
            # absorbed MLA: qk_head_dim = kv_lora_rank + qk_rope_head_dim
            return self._extend_mla_absorbed_prefix(
                q,
                layer,
                K_Buffer,
                kv_indptr,
                kv_indices,
                qo_indptr,
            )

    def _try_fused_mxfp4_kv_b_proj_fp8(
        self,
        layer,
        kvc_for_gemm,
        k_pe,
        qk_nope_head_dim,
    ):
        """Return FP8 k/v from MXFP4 kv_b_proj when the fused split-cat path fits."""
        if fused_gemm_afp4wfp4_preshuffle_split_cat is None:
            return None
        weight = getattr(layer.kv_b_proj, "weight", None)
        weight_scale = getattr(layer.kv_b_proj, "weight_scale", None)
        if weight is None or weight_scale is None:
            return None
        if weight.dtype not in (torch.uint8, getattr(dtypes, "fp4x2", None)):
            return None
        if weight.dim() != 2 or weight.shape[0] % 16 != 0:
            return None
        if weight_scale.shape[0] % 32 != 0:
            return None

        m = kvc_for_gemm.shape[0]
        q_input, x_scale = get_hip_quant(QuantType.per_1x32)(
            kvc_for_gemm,
            quant_dtype=dtypes.fp4x2,
            shuffle=(m >= 32),
        )
        if m >= 32:
            x_scale = x_scale.view(torch.uint8).view(x_scale.shape[0] // 32, -1)
        else:
            x_scale = x_scale[:m, ...].view(torch.uint8)

        return fused_gemm_afp4wfp4_preshuffle_split_cat(
            q_input.view(torch.uint8),
            weight.view(torch.uint8).view(weight.shape[0] // 16, -1),
            k_pe.expand((-1, layer.tp_k_head_num, -1)),
            x_scale,
            weight_scale.view(torch.uint8).view(weight_scale.shape[0] // 32, -1),
            qk_nope_head_dim,
            layer.v_head_dim,
            dtypes.fp8,
        )

    def _extend_mla_no_prefix(
        self,
        q,
        k,
        v,
        layer,
        kv_lora_rank,
        qk_rope_head_dim,
        max_q_len,
        qo_indptr,
    ):
        """No-prefix prefill: FP8 kernel, mla_prefill_fwd, or flash_attn fallback."""
        if self.forward_metadata.fp8_prefill_kv_indices is not None:
            return self._extend_mla_fp8_prefill(q, k, v, layer, max_q_len, qo_indptr)

        if (
            layer.qk_head_dim == (kv_lora_rank + qk_rope_head_dim)
            and mla_prefill_fwd is not None
        ):
            # Absorbed MLA: head_dim (576) exceeds CK limit (256),
            # use mla_prefill_fwd which natively supports large MLA head dims.
            if layer.qk_head_dim != layer.v_head_dim:
                output = q.new_empty(
                    (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                )
            else:
                output = torch.empty_like(q)
            total_s = q.shape[0]
            temp_kv_indices = torch.arange(total_s, device=q.device, dtype=torch.int32)
            mla_prefill_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k.view(-1, 1, 1, layer.qk_head_dim),
                output.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                qo_indptr,
                qo_indptr,
                temp_kv_indices,
                self.forward_metadata.kv_last_page_len,
                max_q_len,
                layer.scaling,
                layer.logit_cap,
            )
            return output

        return flash_attn_varlen_func(
            q,
            k,
            v,
            qo_indptr,
            qo_indptr,
            max_q_len,
            max_q_len,
            softmax_scale=layer.scaling,
            causal=True,
        )

    def _extend_mla_fp8_prefill(self, q, k, v, layer, max_q_len, qo_indptr=None):
        """FP8 prefill path using mla_prefill_ps_asm_fwd + mla_reduce_v1."""
        total_s = q.shape[0]
        nhead = layer.tp_q_head_num
        v_head_dim = layer.v_head_dim
        md = self.forward_metadata
        if qo_indptr is None:
            qo_indptr = md.qo_indptr

        if q.dtype != dtypes.fp8:
            q = q.to(dtypes.fp8)
        if k.dtype != dtypes.fp8:
            k = k.to(dtypes.fp8)
        if v.dtype != dtypes.fp8:
            v = v.to(dtypes.fp8)
        one_scale = torch.ones((), dtype=torch.float32, device=q.device)

        tile_q = 256
        logits = torch.empty(
            (md.reduce_partial_map.size(0) * tile_q, nhead, v_head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        attn_lse = torch.empty(
            (md.reduce_partial_map.size(0) * tile_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        final_lse = torch.empty((total_s, nhead), dtype=torch.float32, device=q.device)
        output = q.new_empty((total_s, nhead, v_head_dim), dtype=self.input_dtype)

        mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            qo_indptr,
            md.kv_indptr,
            md.fp8_prefill_kv_indices,
            md.work_indptr,
            md.work_info_set,
            max_q_len,
            layer.scaling,
            True,
            logits,
            attn_lse,
            output,
            one_scale,
            one_scale,
            one_scale,
        )
        mla_reduce_v1(
            logits,
            attn_lse,
            md.reduce_indptr,
            md.reduce_final_map,
            md.reduce_partial_map,
            tile_q,
            output,
            final_lse,
        )
        return output

    def _extend_mla_decompress_prefix(
        self,
        q,
        layer,
        forward_batch,
        K_Buffer,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_nope_head_dim,
        max_q_len,
        max_kv_len,
        kv_indptr,
        kv_indices,
        qo_indptr,
    ):
        """Has prefix, absorbed weights differ: decompress via kv_b_proj + flash_attn."""
        K_Buffer = torch.index_select(K_Buffer, 0, kv_indices)
        kvc, k_pe = torch.split(K_Buffer, [kv_lora_rank, qk_rope_head_dim], dim=-1)

        if self.kv_cache_dtype == dtypes.fp8:
            dtype = q.dtype
            kvc = kvc.to(dtype)
            k_pe = k_pe.to(dtype)

        # The staged MHA-form MLA cache write keeps a singleton KV-head axis
        # ([tokens, 1, kv_lora_rank]). Flatten it before kv_b_proj GEMM.
        if kvc.ndim == 3:
            assert (
                kvc.shape[1] == 1
            ), f"Unexpected prefix latent shape for kv_b_proj: {tuple(kvc.shape)}"
        kvc_for_gemm = kvc.reshape(-1, kv_lora_rank).contiguous()

        if self.forward_metadata.fp8_prefill_kv_indices is not None:
            fused_kv = self._try_fused_mxfp4_kv_b_proj_fp8(
                layer,
                kvc_for_gemm,
                k_pe,
                qk_nope_head_dim,
            )
            if fused_kv is not None:
                k_prefix, v_prefix = fused_kv
                return self._extend_mla_fp8_prefill(
                    q, k_prefix, v_prefix, layer, max_q_len
                )

        kvprefix = layer.kv_b_proj(kvc_for_gemm)
        if isinstance(kvprefix, tuple):
            kvprefix = kvprefix[0]
        kvprefix = kvprefix.view(
            -1, layer.tp_k_head_num, qk_nope_head_dim + layer.v_head_dim
        )
        k_prefix, v_prefix = torch.split(
            kvprefix, [qk_nope_head_dim, layer.v_head_dim], dim=-1
        )
        k_prefix = torch.cat(
            [
                k_prefix,
                torch.broadcast_to(
                    k_pe,
                    (k_pe.shape[0], layer.tp_k_head_num, k_pe.shape[2]),
                ),
            ],
            dim=-1,
        )

        if self.forward_metadata.fp8_prefill_kv_indices is not None:
            return self._extend_mla_fp8_prefill(q, k_prefix, v_prefix, layer, max_q_len)

        extend_prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)
        if extend_prefix_lens is not None:
            assert extend_prefix_lens.shape == forward_batch.extend_seq_lens.shape

        return flash_attn_varlen_func(
            q,
            k_prefix,
            v_prefix,
            qo_indptr,
            kv_indptr,
            max_q_len,
            max_kv_len,
            softmax_scale=layer.scaling,
            causal=True,
        )

    def _extend_mla_absorbed_prefix(
        self,
        q,
        layer,
        K_Buffer,
        kv_indptr,
        kv_indices,
        qo_indptr,
    ):
        """Has prefix, qk_head_dim == kv_lora_rank + qk_rope_head_dim: mla_prefill_fwd."""
        k_selected = torch.index_select(K_Buffer, 0, kv_indices)
        if k_selected.dtype != q.dtype:
            k_selected = k_selected.to(q.dtype)
        compact_kv_indices = torch.arange(
            k_selected.shape[0], device=q.device, dtype=torch.int32
        )

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        mla_prefill_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k_selected.view(-1, 1, 1, layer.qk_head_dim),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            qo_indptr,
            kv_indptr,
            compact_kv_indices,
            self.forward_metadata.kv_last_page_len,
            self.forward_metadata.max_q_len,
            layer.scaling,
            layer.logit_cap,
        )
        return o

    def _call_mla_decode_fwd(self, q, k_buffer, o, layer):
        """Common mla_decode_fwd invocation shared across decode/extend paths."""
        md = self.forward_metadata
        head_repeat_factor = getattr(self, "head_repeat_factor", 1)
        if head_repeat_factor > 1:
            q_in = q.repeat_interleave(head_repeat_factor, dim=1)
            o_padded = q.new_empty(
                (q.shape[0], self.num_head_padded, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            mla_decode_fwd(
                q_in,
                k_buffer.view(-1, 1, 1, layer.qk_head_dim),
                o_padded,
                md.qo_indptr,
                md.kv_indptr,
                md.kv_indices,
                md.kv_last_page_len,
                md.max_q_len,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                work_meta_data=md.work_metadata,
                work_indptr=md.work_indptr,
                work_info_set=md.work_info_set,
                reduce_indptr=md.reduce_indptr,
                reduce_final_map=md.reduce_final_map,
                reduce_partial_map=md.reduce_partial_map,
                q_scale=layer.k_scale,
                kv_scale=layer.k_scale,
                intra_batch_mode=_sglang_aiter.intra_batch_mode,
                num_kv_splits=md.num_kv_splits,
            )
            o.copy_(o_padded[:, ::head_repeat_factor, :])
            return

        mla_decode_fwd(
            q,
            k_buffer.view(-1, 1, 1, layer.qk_head_dim),
            o,
            md.qo_indptr,
            md.kv_indptr,
            md.kv_indices,
            md.kv_last_page_len,
            md.max_q_len,
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
            work_meta_data=md.work_metadata,
            work_indptr=md.work_indptr,
            work_info_set=md.work_info_set,
            reduce_indptr=md.reduce_indptr,
            reduce_final_map=md.reduce_final_map,
            reduce_partial_map=md.reduce_partial_map,
            q_scale=layer.k_scale,
            kv_scale=layer.k_scale,
            intra_batch_mode=_sglang_aiter.intra_batch_mode,
            num_kv_splits=md.num_kv_splits,
        )

    def _forward_extend_mla_speculative(
        self, q, layer, K_Buffer, qo_indptr, forward_batch
    ):
        """MLA speculative path (target_verify / draft_extend)."""
        md = self.forward_metadata

        if forward_batch.forward_mode.is_target_verify():
            o = q.new_empty(
                (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            self._call_mla_decode_fwd(q, K_Buffer, o, layer)
            return o

        if forward_batch.forward_mode.is_draft_extend(include_v2=True):
            if md.run_graph is not True:
                bs, q_pad, _ = pad_sequence_with_mask(
                    q.view(q.shape[0], -1),
                    qo_indptr[:-1],
                    forward_batch.extend_seq_lens,
                    md.max_q_len,
                )
                o = q.new_empty(
                    (bs * md.max_q_len, layer.tp_q_head_num, layer.v_head_dim),
                    dtype=self.input_dtype,
                )
                self._call_mla_decode_fwd(
                    q_pad.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    K_Buffer,
                    o,
                    layer,
                )
                total_valid_q = int(qo_indptr[-1].item())
                return o[:total_valid_q]

            o = q.new_empty(
                (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            self._call_mla_decode_fwd(q, K_Buffer, o, layer)
            return o

        raise ValueError(
            f"Invalid forward mode for MLA speculative path: {forward_batch.forward_mode=}"
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        topk_indices = kwargs.get("topk_indices")
        if self.use_mla and topk_indices is not None:
            return self._forward_sparse_mla(
                q, k, v, layer, forward_batch, topk_indices, save_kv_cache
            )

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        batch_size = q.shape[0]
        head_dim_out = (
            layer.v_head_dim
            if layer.qk_head_dim != layer.v_head_dim
            else layer.head_dim
        )

        if self.use_mla:
            o = q.new_empty(
                (batch_size, layer.tp_q_head_num * head_dim_out),
                dtype=self.input_dtype,
            )
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            self._call_mla_decode_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k_buffer,
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                layer,
            )
            return o

        # Non-MLA decode paths
        use_native_dense_mha = self._should_use_native_dense_mha(layer)
        if use_native_dense_mha:
            if save_kv_cache:
                self._set_kv_buffer_native_dense(
                    layer, forward_batch.out_cache_loc, k, v, forward_batch
                )
            return self._forward_decode_native_dense_mha(q, layer, forward_batch)

        o = q.new_empty((batch_size, layer.tp_q_head_num, head_dim_out))

        if save_kv_cache:
            k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            k_scale, v_scale = self._ensure_fp8_kv_scales(
                layer, k_buffer, self.page_size
            )
            self.set_kv_buffer_with_layout_shuffle(
                forward_batch.out_cache_loc,
                k,
                v,
                k_buffer,
                v_buffer,
                k_scale,
                v_scale,
                self.page_size,
            )

        k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        block_size = self.page_size
        num_slots, num_kv_heads, head_size = k_buffer.shape
        num_blocks = num_slots // block_size
        k_buffer = k_buffer[: num_blocks * block_size].view(
            num_blocks, block_size, num_kv_heads, head_size
        )
        v_buffer = v_buffer[: num_blocks * block_size].view(
            num_blocks, block_size, num_kv_heads, head_size
        )
        x = 16 // k_buffer.element_size()
        new_key_cache = k_buffer.view(
            num_blocks, num_kv_heads, head_size // x, block_size, x
        )
        new_value_cache = v_buffer.view(
            num_blocks, num_kv_heads, block_size // x, head_size, x
        )

        if self.decode_using_pa_ps:
            total_tokens = num_blocks * block_size
            q_3d = q.view(batch_size, layer.tp_q_head_num, layer.head_dim)
            pa_persistent_fwd(
                Q=q_3d,
                K=new_key_cache,
                V=new_value_cache,
                output=o,
                max_qlen=self.forward_metadata.pa_metadata_max_qlen,
                qo_indptr=self.forward_metadata.pa_metadata_qo_indptr,
                kv_indptr=self.forward_metadata.pa_metadata_pages_kv_indptr,
                kv_indices=self.forward_metadata.pa_metadata_kv_indices,
                context_lens=self.forward_metadata.pa_metadata_context_lens,
                work_indptr=self.pa_metadata_buffers["work_indptr"],
                work_info=self.pa_metadata_buffers["work_info"],
                reduce_indptr=self.pa_metadata_buffers["reduce_indptr"],
                reduce_final_map=self.pa_metadata_buffers["reduce_final_map"],
                reduce_partial_map=self.pa_metadata_buffers["reduce_partial_map"],
                K_QScale=self.k_qscale[:, :total_tokens],
                V_QScale=self.v_qscale[:, :total_tokens],
                softmax_scale=layer.scaling,
                mask=1,
            )
        else:
            q_3d = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            if self.forward_metadata.page_table.dtype != torch.int32:
                raise TypeError(
                    "pa_fwd_asm block_tables must be torch.int32, got "
                    f"{self.forward_metadata.page_table.dtype}"
                )
            if self.forward_metadata.kv_lens.dtype != torch.int32:
                raise TypeError(
                    "pa_fwd_asm context_lens must be torch.int32, got "
                    f"{self.forward_metadata.kv_lens.dtype}"
                )
            k_qscale = (
                layer.k_scale if self.kv_cache_dtype == dtypes.fp8 else self.k_scale
            )
            v_qscale = (
                layer.v_scale if self.kv_cache_dtype == dtypes.fp8 else self.v_scale
            )
            pa_fwd_asm(
                Q=q_3d,
                K=new_key_cache,
                V=new_value_cache,
                block_tables=self.forward_metadata.page_table,
                context_lens=self.forward_metadata.kv_lens,
                block_tables_stride0=self.forward_metadata.page_table.stride(0),
                K_QScale=k_qscale,
                V_QScale=v_qscale,
                out_=o,
            )

        return o.view(-1, layer.tp_q_head_num * head_dim_out)

    def _forward_decode_native_dense_mha(self, q, layer, forward_batch):
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        aiter_kv_str = self._get_aiter_paged_ragged_kv_cache_dtype()

        o = torch.empty_like(q, dtype=self.input_dtype)
        paged_attention_ragged(
            o.view(-1, layer.tp_q_head_num, layer.head_dim),
            self.workspace_buffer,
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            k_cache.view(-1, 1, layer.tp_k_head_num, layer.head_dim),
            v_cache.view(-1, 1, layer.tp_v_head_num, layer.v_head_dim),
            layer.scaling,
            self.forward_metadata.kv_indptr,
            self.forward_metadata.kv_indices,
            self.kv_last_page_len,
            1,
            self.max_num_partitions,
            None,
            aiter_kv_str,
            "NHD",
            layer.logit_cap,
            self.k_scale,
            self.v_scale,
            None,
            getattr(_sglang_aiter, "_AITER_PARTITION_SIZE_ROCM", 256),
        )
        return o
