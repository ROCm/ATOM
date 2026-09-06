# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging

import aiter
import numpy as np
import torch
import triton
import triton.language as tl
from aiter.dist.parallel_state import get_tp_group

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mha import PagedAttentionImpl, use_pa_decode_bf16_asm
from atom.utils import CpuGpuBuffer, envs, pack_rows, upload_numpy
from atom.utils.block_convert import (
    block_table_convert_triton,
    kv_indices_generate_triton,
)
from atom.utils.forward_context import AttentionMetaData, Context, get_forward_context
from atom.utils.tbo import TokenSplitPrefillState

from .backends import AttentionBackend, CommonAttentionBuilder
from .mha_kv_pool import MhaKvPool
from .pool_layout.sub_pool_spec import SubPoolSpec, page_pool
from .token_layout.decode import decode_positions
from .token_layout.slots import slot_mapping

logger = logging.getLogger("atom")


def cdiv(a, b):
    return (a + b - 1) // b


def _is_indexed_sparse_attention(module) -> bool:
    """True only for the MiniMax-M3 sparse ``Attention`` layer (the one that owns
    the sparse impl), so binding reads ``module.impl``.

    ``model.modules()`` walks both the outer ``MiniMaxM3SparseAttention`` wrapper
    AND its child ``Attention`` layer. Only the child carries ``.impl`` (a
    ``SparseMHAPagedAttentionImpl``) and the KV-cache slot; the wrapper must be
    skipped (return None from build_kv_cache_tensor). So key off the impl flag,
    NOT the wrapper's own ``is_indexed_sparse_attention`` class attribute."""
    impl = getattr(module, "impl", None)
    return bool(getattr(impl, "is_indexed_sparse_attention", False))


def _resolve_index_cache_dtype(config) -> torch.dtype:
    from aiter import dtypes

    index_cache_dtype = getattr(config, "index_cache_dtype", None)
    if index_cache_dtype is None:
        index_cache_dtype = getattr(config, "kv_cache_dtype", "bf16")
    if index_cache_dtype in ("bf16", "fp8"):
        return dtypes.d_dtypes[index_cache_dtype]
    raise ValueError(
        "index_cache_dtype must be 'bf16' or 'fp8', " f"got {index_cache_dtype!r}"
    )


@triton.jit
def _mtp_prepare_decode_metadata_kernel(
    context_lens_ptr,
    block_tables_ptr,
    slot_mapping_ptr,
    positions_in_ptr,
    positions_out_ptr,
    last_token_indices_ptr,
    bs,
    skip_update: tl.constexpr,
    update_context_lens: tl.constexpr,
    update_positions: tl.constexpr,
    select_positions: tl.constexpr,
    block_size: tl.constexpr,
    block_table_stride: tl.constexpr,
    position_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if not skip_update:
        seq = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = seq < bs

        ctx = tl.load(context_lens_ptr + seq, mask=mask, other=1).to(tl.int64)
        if update_context_lens:
            ctx += 1
            tl.store(context_lens_ptr + seq, ctx, mask=mask)

        if update_positions:
            pos_idx = seq
            if select_positions:
                pos_idx = tl.load(last_token_indices_ptr + seq, mask=mask, other=0)
            pos = tl.load(positions_in_ptr + pos_idx, mask=mask, other=0)
            tl.store(positions_out_ptr + seq * position_stride, pos + 1, mask=mask)

        last_pos = tl.maximum(ctx - 1, 0)
        block_col = last_pos // block_size
        within_block = last_pos - block_col * block_size

        phys_block = tl.load(
            block_tables_ptr + seq * block_table_stride + block_col,
            mask=mask,
            other=0,
        ).to(tl.int64)
        tl.store(
            slot_mapping_ptr + seq, phys_block * block_size + within_block, mask=mask
        )


class AiterBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ATOM_ATTENTION"

    @staticmethod
    def get_builder_cls() -> type["AiterAttentionMetadataBuilder"]:
        return AiterAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return PagedAttentionImpl

    @staticmethod
    def make_kv_pool(hf_config, *, world_size: int, block_size: int, kv_dtype):
        return MhaKvPool.from_hf_config(
            hf_config,
            world_size=world_size,
            block_size=block_size,
            kv_dtype=kv_dtype,
        )


class AiterAttentionMetadataBuilder(CommonAttentionBuilder):
    BLOCK_TABLE_EXTENDER: list[list[int]] = [[]]
    # EagleProposer fuses the per-draft-step position bump into
    # prepare_mtp_decode's kernel when this is set (block-paged MHA draft).
    fuse_mtp_decode_position_update = True

    def __init__(
        self,
        kv_cache_spec=None,
        layer_names=None,
        config=None,
        device=None,
        model_runner=None,
    ):
        hf_config = model_runner.config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        sparse_cfg = getattr(text_config, "sparse_attention_config", None)
        from atom.config import _is_minimax_m3_config

        self._has_sparse_attention = bool(sparse_cfg) and _is_minimax_m3_config(
            hf_config
        )
        # Which indexer layer the next sparse module binds. Reset by whatever
        # backs the pool, so a rollout re-allocation hands out the same slices
        # in the same order.
        self._next_index_layer = 0
        if self._has_sparse_attention and (
            sparse_block_size := sparse_cfg.get("sparse_block_size")
        ):
            # MiniMax-M3 sparse kernels operate on sparse_attention_config's
            # block size. The scheduler/KV manager block size may be larger as
            # long as it is divisible by this logical attention block size.
            self.block_size = sparse_block_size
        else:
            self.block_size = (
                model_runner.block_size
                if model_runner.block_size in (256, 1024)
                else 16
            )
        if envs.ATOM_USE_UNIFIED_ATTN:
            # SHUFFLE (pre-shuffled) KV cache: use the logical block size directly
            # as the physical block size so block_ratio == 1 and
            # unified_attention's block_table needs no logical->physical
            # conversion. Pass --block-size equal to the performant physical
            # page: fp8 packs x=16 - 128; bf16 packs x=8 - 64 (both keep a
            # 128-byte physical page, i.e. block_size // x == 8).
            expected = 128 if model_runner.kv_cache_dtype in ("fp8",) else 64
            if model_runner.block_size != expected:
                logger.warning(
                    "ATOM_USE_UNIFIED_ATTN=1 expects --block-size %s for %s KV "
                    "cache (so block_ratio == 1), got --block-size %s. Continuing "
                    "with the requested block size.",
                    expected,
                    model_runner.kv_cache_dtype,
                    model_runner.block_size,
                )
            self.block_size = model_runner.block_size

        assert (
            model_runner.block_size % self.block_size == 0
        ), f"model_runner.block_size must be divisible by block_size but got {model_runner.block_size=}, block_size={self.block_size}, please set --block-size (model_runner.block_size) to be divisible by {self.block_size}"
        super().__init__(model_runner)
        # Set by `allocate_kv_cache_tensors` / `adopt_imported_kv_pool`, both of
        # which run long after construction. MiMo-V2 leaves it None and gives
        # each module a pool of its own instead.
        self.kv_pool: MhaKvPool | None = None
        # MiMo-V2's region and how much of it the bind walk has handed out.
        self._page_buf: torch.Tensor | None = None
        self._page_cursor = 0
        config = model_runner.config
        hf_config = config.hf_config
        from atom.utils import envs as _envs

        self._tbo_token_split = bool(
            config.enable_tbo and _envs.ATOM_TBO_PREFILL_TOKEN_SPLIT
        )
        # Snapshot of the current prefill batch, set by
        # `_stash_tbo_token_split_prefill_state`; None when not token-splitting.
        self._tbo_prefill_state = None
        # `self.num_attention_heads` set by CommonAttentionBuilder.__init__.
        # For speculative decode (MTP), max_qlen = num_speculative_tokens + 1
        if (
            config.speculative_config is not None
            and config.speculative_config.num_speculative_tokens is not None
        ):
            max_qlen = config.speculative_config.num_speculative_tokens + 1
        else:
            max_qlen = 1

        num_head_k = max(1, hf_config.num_key_value_heads // get_tp_group().world_size)
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = aiter.get_pa_metadata_info_v1(
            self.max_bs,
            num_head_k,
        )

        i32_kwargs = {"dtype": torch.int32, "device": self.device}
        if self._has_sparse_attention and self.block_ratio > 1:
            self.model_runner.forward_vars["sparse_attention_block_tables"] = (
                CpuGpuBuffer(
                    self.max_bs,
                    self.max_num_blocks_per_seq,
                    **i32_kwargs,
                )
            )
        self._pa_decode_bf16_asm_enabled = (
            use_pa_decode_bf16_asm() and model_runner.block_size == 256
        )

        pa_persistent_metadata = {
            "max_qlen": max_qlen,
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
        }
        self.model_runner.forward_vars.update(pa_persistent_metadata)
        # Per-ubatch buffers for CUDAGraph TBO
        if model_runner.config.enable_tbo:
            self._allocate_ubatch_buffers(
                max_qlen,
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
        ub_max_bs = self.max_bs

        for ub_idx in range(self._NUM_TBO_UBATCHES):
            p = f"ub{ub_idx}_"
            var[f"{p}kv_indptr"] = CpuGpuBuffer(ub_max_bs + 1, **i32_kwargs)
            var[f"{p}kv_indices"] = CpuGpuBuffer(
                self.max_bs * self.max_num_blocks_per_seq,
                **i32_kwargs,
            )
            var[f"{p}context_lens"] = CpuGpuBuffer(ub_max_bs, **i32_kwargs)
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

            # PA work buffers per ubatch (GPU only)
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

    def set_aiter_persistent_worker_buffers(self, bs: int):
        config = self.model_runner.config
        hf_config = config.hf_config
        num_query_heads = self.num_attention_heads
        num_kv_heads = max(
            1, hf_config.num_key_value_heads // get_tp_group().world_size
        )
        block_size = self.block_size

        var = self.model_runner.forward_vars
        max_qlen = var["max_qlen"]

        qo_indptr = var["cu_seqlens_q"].gpu[: bs + 1]
        kv_indptr = var["kv_indptr"].gpu[: bs + 1]
        seq_lens_kv = var["context_lens"].gpu[:bs]

        work_meta_data = var["work_meta_data"]
        work_indptr = var["work_indptr"]
        work_info_set = var["work_info_set"]
        reduce_indptr = var["reduce_indptr"]
        reduce_final_map = var["reduce_final_map"]
        reduce_partial_map = var["reduce_partial_map"]

        aiter.get_pa_metadata_v1(
            qo_indptr,
            kv_indptr,
            seq_lens_kv,
            num_query_heads // num_kv_heads,
            num_kv_heads,
            True,
            work_meta_data,
            work_indptr,
            work_info_set,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            kv_granularity=max(block_size, 16),
            block_size=block_size,
            max_seqlen_qo=int(max_qlen),
            uni_seqlen_qo=max_qlen,
            fast_mode=True,
            max_split_per_batch=-1,
        )

        return {
            "work_meta_data": work_meta_data,
            "work_indptr": work_indptr,
            "work_info_set": work_info_set,
            "reduce_indptr": reduce_indptr,
            "reduce_final_map": reduce_final_map,
            "reduce_partial_map": reduce_partial_map,
        }

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        """One paged KV pool, priced by the declaration it is built from."""
        return [page_pool(self._paged_entry_bytes())]

    def paged_pool_bytes(self, blocks: int) -> int:
        """The same declaration, times the blocks the budget bought."""
        return self._paged_entry_bytes() * blocks

    def _paged_entry_bytes(self) -> int:
        """What one scheduler block of this model's MHA pool costs:

        - Standard models: `[2, num_hidden_layers, blocks, block_size,
          num_kv_heads, head_dim]` for kv_cache + matching kv_scale (fp32).
        - MiMo-V2-Flash: per-layer-type accounting (full vs SWA layers
          have different num_kv_heads).
        - MiniMax-M3: plus the indexer key cache its sparse layers own,
          which is a field of the same pool.

        One function because the byte budget, the allocation's size and the
        per-module walk that fills it are three readings of one number.
        """
        runner = self.model_runner
        hf_config = runner.config.hf_config
        num_kv_heads = runner._get_num_kv_heads()
        total_num_layers = runner._get_total_num_layers()

        if runner.is_mimo_v2():
            # Mixed full + SWA layers, possibly different num_kv_heads.
            pattern = hf_config.hybrid_layer_pattern
            num_swa_layers = sum(
                1 for i in range(hf_config.num_hidden_layers) if pattern[i] == 1
            )
            num_full_layers = hf_config.num_hidden_layers - num_swa_layers
            num_draft_layers = total_num_layers - hf_config.num_hidden_layers
            num_swa_layers += num_draft_layers

            _swa_raw = getattr(hf_config, "swa_num_key_value_heads", 0)
            swa_kv_heads = (
                _swa_raw // runner.world_size
                if _swa_raw >= runner.world_size
                else (1 if _swa_raw else 0)
            )
            # Two pools' worth of block: the layer kinds differ in
            # `num_kv_heads` and nothing else. Allocation walks the modules
            # instead and takes a one-layer pool each, which comes to the same
            # bytes as long as a layer's own fields are whole multiples of the
            # entry alignment -- and when they are not, the last module's
            # arena is handed a region that is short and says so.
            block_bytes = self._declare_kv_pool(
                num_kv_heads, layers=num_full_layers
            ).entry_bytes
            return (
                block_bytes
                + self._declare_kv_pool(swa_kv_heads, layers=num_swa_layers).entry_bytes
            )

        # Standard MHA path: the same declaration the allocation is built from,
        # indexer cache included.
        return self._declare_kv_pool(num_kv_heads, layers=total_num_layers).entry_bytes

    def _kv_pool_layers(self) -> int:
        """Layers this backend's paged pool holds rows for.

        A hook because a hybrid holds rows for only some of its layers, and
        sizing, allocation and the P/D adopt path have to agree on the count
        or the arena refuses the buffer.
        """
        return self.model_runner._get_total_num_layers()

    def _index_cache_decl(self) -> dict:
        """Indexer-cache constructor arguments, empty when there is none.

        Keyed off `_has_sparse_attention` -- the same answer the block size and
        the metadata buffers use -- not a second reading of the config. MiMo-V2
        declares two pools and sums them, so a second predicate would have to
        be kept from double-charging; this one is false for it by construction.
        """
        if not self._has_sparse_attention:
            return {}
        runner = self.model_runner
        hf_config = runner.config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        sparse_cfg = text_config.sparse_attention_config
        return {
            "index_layers": sum(
                1 for enabled in sparse_cfg.get("sparse_attention_freq", []) if enabled
            ),
            "index_dim": sparse_cfg["sparse_index_dim"],
            "index_dtype": _resolve_index_cache_dtype(runner.config),
        }

    def _declare_kv_pool(
        self, num_kv_heads: int, layers: int | None = None
    ) -> MhaKvPool:
        """This model's MHA layers as a pool, declared but not allocated.

        Sizing asks before a block count exists, so both steps come from here
        and the pool that is charged for is the pool that gets built.
        """
        from aiter import dtypes

        runner = self.model_runner
        return MhaKvPool(
            layers=self._kv_pool_layers() if layers is None else layers,
            # The scheduler's block, which is the entry class `page_pool`
            # charges. How many of this backend's own pages make one up is
            # `block_ratio`, and it stays inside the backend.
            block_size=runner.block_size,
            num_kv_heads=num_kv_heads,
            head_dim=runner.config.hf_config.head_dim,
            kv_dtype=dtypes.d_dtypes[runner.config.kv_cache_dtype],
            **self._index_cache_decl(),
        )

    def adopt_imported_kv_pool(self, blocks: int, buf) -> None:
        runner = self.model_runner
        if runner.is_mimo_v2():
            # The pools are still one region, but which module gets which slice
            # is decided by the bind walk, and this side's `_page_cursor` would
            # have to be rewound to the same place first. Untried, so refused.
            raise NotImplementedError(
                "MiMo-V2 carves its region per module during the bind walk, "
                "which the P/D decode side has not been made to replay."
            )
        self.kv_pool = self._declare_kv_pool(runner._get_num_kv_heads())
        self._next_index_layer = 0
        self.kv_pool.allocate(blocks, runner.device, buf=buf)
        self.num_blocks = blocks * self.block_ratio

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int, *, blocks: int, buf
    ) -> dict:
        """Allocate this model's MHA pool inside the runner's paged region.

        Nothing shaped goes back to the runner: the cache, the scales and the
        indexer keys were three named attributes and are three regions of the
        one buffer it holds. So the only reference that can outlive a rollout
        sleep is the pool's, and `MhaKvPool.release` is what drops it.
        """
        self.num_blocks = blocks * self.block_ratio
        runner = self.model_runner
        hf_config = runner.config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        self._next_index_layer = 0

        if runner.is_mimo_v2():
            # Deferred to `build_kv_cache_tensor`: a module's pool is sized by
            # its own num_kv_heads, so the region is handed out one module at a
            # time, in bind order.
            self._page_buf = buf
            self._page_cursor = 0
            return {"_kv_layer_cache_store": []}

        self.kv_pool = self._declare_kv_pool(num_kv_heads)
        self.kv_pool.allocate(blocks, runner.device, buf=buf)
        tensors: dict = {}
        # Not under the indexer cache: this is a per-run memo dict, keyed off
        # `use_index_cache` alone. Its one reader is in the sparse branch of
        # `build_kv_cache_tensor`, so gating it on the pool as well would only
        # claim a dependency that is not there.
        if getattr(text_config, "use_index_cache", False) or getattr(
            hf_config, "use_index_cache", False
        ):
            tensors["_sparse_attention_topk_cache_state"] = {}
        return tensors

    def build_kv_cache_tensor(self, layer_id: int, module):
        """Bind one MHA (non-MLA) attention module to its KV slice.

        Handles both standard hybrid models (Qwen3-Next pattern: full-attn
        layers interleaved with linear-attn) and MiMo-V2-Flash (per-layer
        allocation with potentially different num_kv_heads per module).

        Returns the KVCacheTensor to register, or None if the module is not
        an MHA attention this builder owns. Side effects: sets module
        `k_cache`, `v_cache`, `k_scale`, `v_scale`, `max_model_len`.
        """

        from atom.config import KVCacheTensor

        if _is_indexed_sparse_attention(module):
            # MiniMax-M3 sparse attention. The KV cache uses the SAME allocation
            # and binding as standard MHA — we only additionally bind the separate
            # indexer-key cache here, then fall through to the standard branch for
            # all K/V + scale binding (it sets module.k_cache/v_cache/k_scale/
            # v_scale and returns the KVCacheTensor). The standard binding is
            # page-128 SHUFFLE; SparseMHAPagedAttentionImpl.rope_cache re-views it
            # to page-16 SHUFFLE (zero-copy) at attention time. index_cache is a
            # genuinely separate field (not derivable from the K/V ones), so
            # each sparse layer takes its own slice here, in bind order. The
            # pool shapes that slice in scheduler blocks and the impl addresses
            # it in this backend's pages; the reshape is where that is said.
            runner = self.model_runner
            index_view = self.kv_pool.index_view(self._next_index_layer)
            self._next_index_layer += 1
            module.impl.index_cache = index_view.view(
                -1, self.block_size, self.kv_pool.index_dim
            )
            module.impl.max_model_len = runner.config.max_model_len
            module.impl.index_topk_cache_state = getattr(
                runner, "_sparse_attention_topk_cache_state", None
            )
            # NOTE: no return — fall through to the standard MHA binding below.

        if not (
            hasattr(module, "base_attention")
            and hasattr(module, "use_mla")
            and not module.use_mla
        ):
            return None

        runner = self.model_runner
        config = runner.config

        # attn_idx: hybrid models (Qwen3-Next) skip linear-attention layers
        # in the kv_cache slot ordering; non-hybrid models use layer_id 1:1.
        if runner.is_qwen_next():
            mtp_start = runner.mtp_start_layer_idx
            if layer_id < mtp_start:
                attn_idx = layer_id // runner.full_attention_interval
            else:
                attn_idx = runner.num_full_attn + (layer_id - mtp_start)
        else:
            attn_idx = layer_id

        fp8 = config.kv_cache_dtype == "fp8"
        if runner.is_mimo_v2():
            # A one-layer pool per module: each has its own num_kv_heads, so
            # its own per-block cost. Same declaration, kept alive by the views
            # it hands out.
            pool = self._declare_kv_pool(module.num_kv_heads, layers=1)
            # A pool entry is a scheduler block, so back out of this backend's
            # page. `allocate_kv_cache_tensors` records the count before it
            # defers to here, which is why it does so ahead of that return.
            blocks = self.num_blocks // self.block_ratio
            # The next slice of this builder's region, in bind order. Sizing
            # priced the same layers in two aggregate pools rather than one per
            # module; the two agree, and a region too short to finish the walk
            # is refused by the arena that runs off the end of it.
            take = pool.pool_bytes(blocks)
            start = self._page_cursor
            self._page_cursor += take
            pool.allocate(
                blocks, runner.device, buf=self._page_buf[start : start + take]
            )
            k_cache, v_cache = pool.kv_views(0)
            if fp8:
                module.k_scale, module.v_scale = pool.scale_views(0)
            runner._kv_layer_cache_store.append(
                (k_cache, v_cache, module.k_scale, module.v_scale)
            )
        else:
            k_cache, v_cache = self.kv_pool.kv_views(attn_idx)
            if fp8:
                module.k_scale, module.v_scale = self.kv_pool.scale_views(attn_idx)

        module.max_model_len = config.max_model_len
        module.k_cache = k_cache
        module.v_cache = v_cache
        return KVCacheTensor(
            layer_num=layer_id,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=module.k_scale,
            v_scale=module.v_scale,
        )

    def _get_sparse_attention_block_tables(
        self,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        bs: int,
    ) -> torch.Tensor:
        """Return MiniMax-M3 sparse-kernel block tables.

        `block_tables` is produced by the scheduler at `model_runner.block_size`
        granularity. MiniMax-M3 sparse attention indexes blocks at
        `sparse_block_size` granularity, so when the scheduler block is larger
        we expand each scheduler page id into its logical sparse pages.
        """
        if self.block_ratio == 1:
            return block_tables
        sparse_block_tables = self.model_runner.forward_vars[
            "sparse_attention_block_tables"
        ].gpu[:bs]
        return block_table_convert_triton(
            block_tables,
            sparse_block_tables,
            seq_lens,
            self.block_ratio,
        )

    def get_kv_transfer_tensors(self):
        from atom.kv_transfer.disaggregation.types import (
            KVTransferRegion,
            KVTransferTensors,
        )

        runner = self.model_runner
        has_unified_kv = self.kv_pool is not None
        # for MiMoV2 model per layer kv binding
        has_per_layer_kv = (
            hasattr(runner, "_kv_layer_cache_store") and runner._kv_layer_cache_store
        )
        if not has_unified_kv and not has_per_layer_kv:
            return None

        block_regions: list[KVTransferRegion] = []

        def _add_region(tensor):
            bpb = tensor.stride(0) * tensor.element_size()
            block_regions.append(
                KVTransferRegion(
                    base_addr=tensor.data_ptr(),
                    total_bytes=tensor.numel() * tensor.element_size(),
                    unit_bytes=bpb,
                )
            )

        if hasattr(runner, "_kv_layer_cache_store") and runner._kv_layer_cache_store:
            for k_cache, v_cache, k_scale, v_scale in runner._kv_layer_cache_store:
                _add_region(k_cache)
                _add_region(v_cache)
                if k_scale is not None:
                    _add_region(k_scale)
                if v_scale is not None:
                    _add_region(v_scale)
        else:
            # Every field of the pool, indexer cache included: a region per
            # (field, layer), in declared order.
            for tensor in self.kv_pool.region_tensors():
                _add_region(tensor)

        return KVTransferTensors(
            block_regions=block_regions,
            slot_regions=[],
        )

    def prepare_mtp_decode(
        self,
        bs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        positions: torch.Tensor,
        only_update: bool = False,
        num_reject_tokens: torch.Tensor = None,
        *,
        update_context_lens: bool = False,
        positions_out: torch.Tensor | None = None,
        last_token_indices: torch.Tensor | None = None,
    ):
        """Per-draft-step metadata for a block-paged MHA Eagle3 draft.

        Called by EagleProposer.propose at mid-step iters. The draft's decode
        kernels (``paged_attention_{asm,triton}``) read ``block_tables`` +
        ``context_lens``. Eagle can pre-bump ``context_lens`` before this call,
        or ask this fused kernel to update it in place. The block_size==1024
        persistent path is the only one consuming ``kv_indptr``/``kv_indices``;
        MiniMax-M3 runs at ``--block-size 128`` so the kernel never reads them -
        no rebuild.

        The one value we must (re)compute is the write slot for the new draft
        token in the draft's own block-paged KV cache:

            slot = block_tables[seq, (ctx-1)//B] * B + (ctx-1) % B,   B = block_size

        Returned under ``slot_mapping`` so EagleProposer skips its token-granular
        (MLA physical block_size==1) flat-kv slot derivation, which would yield
        a bare block id for ``B > 1``.

        ``only_update`` / ``num_reject_tokens`` are MLA/V4-specific knobs and are
        unused here: there are no persistent worker buffers to roll over for
        ``block_size != 1024``.
        """
        running_bs = int(positions.shape[-1])  # rows; see the base contract
        var = self.model_runner.forward_vars
        slot_mapping = var["slot_mapping"].gpu[:running_bs]
        block_tables = var["block_tables"].gpu
        context_lens = var["context_lens"].gpu
        update_positions = positions_out is not None
        select_positions = update_positions and last_token_indices is not None
        if positions_out is None:
            positions_out = positions
        if last_token_indices is None:
            last_token_indices = slot_mapping
        # Dummy runs skip the draft attention, so keep this launch as a no-op:
        # their synthetic context_lens can point past block_tables.
        _mtp_prepare_decode_metadata_kernel[(max(1, triton.cdiv(running_bs, 128)),)](
            context_lens,
            block_tables,
            slot_mapping,
            positions,
            positions_out,
            last_token_indices,
            running_bs,
            running_bs == 0 or get_forward_context().context.is_dummy_run,
            update_context_lens,
            update_positions,
            select_positions,
            self.model_runner.block_size,
            block_tables.stride(0),
            positions_out.stride(0) if update_positions else 1,
            BLOCK=128,
        )
        workinfos = {"slot_mapping": slot_mapping}
        if self._has_sparse_attention:
            # `attention_mha` picks this up off the shared metadata with a plain
            # `getattr`, and `prepare_decode` cut it at `scheduled_bs` -- so a
            # mid-step would read seq_lens for fewer rows than it runs.
            from atom.model_ops.minimax_m3.sparse_attn import (
                make_sparse_decode_metadata,
            )

            workinfos["sparse_attention_metadata"] = make_sparse_decode_metadata(
                seq_lens=context_lens[:running_bs],
                block_table=self._get_sparse_attention_block_tables(
                    block_tables[:running_bs], context_lens[:running_bs], running_bs
                ),
                slot_mapping=slot_mapping,  # this step's, not the verify fwd's
                max_seq_len=max_seqlen_k,
                # A mid-step is one row per sequence. Not the `max_seqlen_q`
                # argument: on the `only_update` path that carries the verify
                # forward's width.
                max_query_len=1,
            )
        return workinfos

    def prepare_prefill(self, batch: ScheduledBatch, running_bs: int):
        attn_metadata, positions = CommonAttentionBuilder.prepare_prefill(
            self, batch, running_bs
        )
        if self._has_sparse_attention and not attn_metadata.has_cached:
            bs = batch.total_seqs_num_prefill
            attn_metadata.block_tables = self.model_runner.forward_vars[
                "block_tables"
            ].copy_to_gpu(bs)
        # `prefill_attention_triton` reads the paged KV cache, so it needs a
        # block_table even with no cached tokens. The base builder marshals one
        # every step but only uploads it when `has_cached`.
        if (
            attn_metadata.block_tables is None
            and envs.ATOM_USE_UNIFIED_ATTN
            and batch.block_tables
        ):
            bs = batch.total_seqs_num_prefill
            attn_metadata.block_tables = self.model_runner.forward_vars[
                "block_tables"
            ].copy_to_gpu(bs)
        if self._has_sparse_attention:
            from atom.model_ops.minimax_m3.sparse_attn import (
                make_sparse_prefill_metadata,
            )

            bs = batch.total_seqs_num_prefill
            sparse_block_tables = self._get_sparse_attention_block_tables(
                attn_metadata.block_tables[:bs],
                attn_metadata.context_lens[:bs],
                bs,
            )
            # Sliced like the block tables above -- `num_prefills` counts
            # requests, and these two are padded past it.
            attn_metadata.sparse_attention_metadata = make_sparse_prefill_metadata(
                cu_seqlens_q=attn_metadata.cu_seqlens_q[: bs + 1],
                seq_lens=attn_metadata.context_lens[:bs],
                block_table=sparse_block_tables,
                slot_mapping=attn_metadata.slot_mapping,
                max_query_len=attn_metadata.max_seqlen_q,
                max_seq_len=attn_metadata.max_seqlen_k,
                num_prefills=bs,
                num_prefill_tokens=batch.total_tokens_num_prefill,
            )
        if self._tbo_token_split:
            self._stash_tbo_token_split_prefill_state(batch)
        # TBO: publish CPU length copies for zero-sync ubatch splits.
        # Attached last, once all metadata is finalized. No-op unless TBO is on.
        self._attach_tbo_prefill_cpu_lens(attn_metadata, batch.total_seqs_num_prefill)
        return attn_metadata, positions

    # ================================================================
    # TBO PREFILL TOKEN-SPLIT (ATOM_TBO_PREFILL_TOKEN_SPLIT) — MHA path
    # ================================================================

    def _stash_tbo_token_split_prefill_state(self, batch: ScheduledBatch):
        """Snapshot full-batch block_tables / cu_tokens / num_cached so a later
        micro-batch can rebuild the straddled request's cached prefix."""
        self._tbo_prefill_state = None
        if not batch.block_tables:
            return
        bs = batch.total_seqs_num_prefill
        # Per-request prefix-cache hit length (tokens already in the paged cache
        # from previous steps). A straddled/ubatch request's FULL visible K is
        # num_cached + (its tokens in this ubatch), so the gather must include it.
        self._tbo_prefill_state = TokenSplitPrefillState(
            # Handed over as-is: the reader wants `len(row)` and a slice
            # assignment, which the rows serve directly. Referencing is safe
            # because BlockManager only appends between steps, and stash and
            # read are both inside this one forward.
            block_tables=batch.block_tables[:bs],
            cu_tokens=np.asarray(
                self.model_runner.forward_vars["cu_seqlens_q"].np[: bs + 1],
                dtype=np.int64,
            ).copy(),
            num_cached=np.asarray(batch.num_cached_tokens[:bs], dtype=np.int64),
        )

    def build_ubatch_prefill_metadata(
        self,
        attn_metadata: AttentionMetaData,
        ub_slice,
        running_bs: int,
        ubatch_idx: int = 0,
    ) -> AttentionMetaData:
        del ubatch_idx
        from atom.utils.tbo.ubatch_splitting import (
            derive_prefill_lens_from_positions,
            split_attn_metadata,
        )

        ub_attn = split_attn_metadata(attn_metadata, ub_slice, running_bs)
        self._attach_tbo_token_split_straddle_prefix(ub_attn, ub_slice)
        if self._has_sparse_attention:
            from atom.model_ops.minimax_m3.sparse_attn import (
                make_sparse_prefill_metadata,
            )

            ub_num_reqs = ub_slice.request_slice.stop - ub_slice.request_slice.start
            ub_num_tokens = ub_slice.token_slice.stop - ub_slice.token_slice.start
            ub_cu_seqlens_q = ub_attn.cu_seqlens_q[: ub_num_reqs + 1]
            var = self.model_runner.forward_vars
            _, ub_seq_lens_np = derive_prefill_lens_from_positions(
                var["positions"].np[
                    ub_slice.token_slice.start : ub_slice.token_slice.stop
                ],
                var["cu_seqlens_q"].np,
                ub_slice,
            )
            ub_seq_lens = torch.from_numpy(ub_seq_lens_np).to(
                self.device, non_blocking=True
            )
            ub_max_seq_len = int(ub_seq_lens_np.max()) if ub_num_reqs > 0 else 0
            sparse_block_tables = self._get_sparse_attention_block_tables(
                ub_attn.block_tables[:ub_num_reqs],
                ub_seq_lens,
                ub_num_reqs,
            )
            sparse_md = make_sparse_prefill_metadata(
                cu_seqlens_q=ub_cu_seqlens_q,
                seq_lens=ub_seq_lens,
                block_table=sparse_block_tables,
                slot_mapping=ub_attn.slot_mapping,
                max_query_len=ub_attn.max_seqlen_q,
                max_seq_len=ub_max_seq_len,
                num_prefills=ub_num_reqs,
                num_prefill_tokens=ub_num_tokens,
            )
            ub_attn.sparse_attention_metadata = sparse_md
        return ub_attn

    def _attach_tbo_token_split_straddle_prefix(self, ub_attn, ub_slice):
        """Re-attach the straddled request's already-cached first half as a
        paged cached prefix (block_tables + context_lens) so ubatch 1's dense
        attention sees the tokens ubatch 0 wrote. No-op when not straddling."""
        from atom.utils.tbo import compute_straddle_split_info

        state = self._tbo_prefill_state
        if state is None:
            return
        block_tables_host = state.block_tables
        cu_tokens = state.cu_tokens

        info = compute_straddle_split_info(cu_tokens, ub_slice)
        if not info.is_straddling:
            return

        rs = ub_slice.request_slice
        first_req = info.first_req
        ub_num_reqs = info.ub_num_reqs
        prefix_len = info.prefix_len
        device = self.device

        new_lens = (
            cu_tokens[rs.start + 1 : rs.stop + 1] - cu_tokens[rs.start : rs.stop]
        ).astype(np.int64)
        new_lens[0] -= prefix_len  # first request is the straddled one
        # Visible K per request = prior-step prefix cache (num_cached) + the
        # straddle prefix written by ubatch 0 (only req0) + this ubatch's new
        # tokens. Missing the num_cached term made the gather read from the
        # cached-prefix blocks instead of the freshly-written new blocks.
        if state.num_cached is not None:
            cached = state.num_cached[rs.start : rs.stop].astype(np.int64)
        else:
            cached = np.zeros(ub_num_reqs, dtype=np.int64)
        cached_prefix_lens = cached.copy()
        cached_prefix_lens[0] += prefix_len
        ctx_lens = cached_prefix_lens + new_lens
        total_kv = int(ctx_lens.sum())

        # Indexed, not sliced: a short slice would leave `bt` rows unwritten.
        rows = [block_tables_host[first_req + i] for i in range(ub_num_reqs)]
        max_blocks = max((len(row) for row in rows), default=1)
        bt = np.empty((ub_num_reqs, max_blocks), dtype=np.int32)  # pack_rows clears
        pack_rows(bt, rows)

        cu_k = np.zeros(ub_num_reqs + 1, dtype=np.int32)
        np.cumsum(ctx_lens.astype(np.int32), out=cu_k[1:])

        ub_attn.has_cached = True
        ub_attn.total_kv = total_kv
        ub_attn.context_lens = upload_numpy(ctx_lens.astype(np.int32), device)
        # `bt` is [requests x blocks], so a long enough context puts it past the
        # pageable limit and the copy would start synchronizing.
        ub_attn.block_tables = upload_numpy(bt, device)
        ub_attn.cu_seqlens_k = upload_numpy(cu_k, device)
        ub_attn.seq_starts = torch.zeros(ub_num_reqs, dtype=torch.int32, device=device)
        ub_attn.num_cached_tokens = upload_numpy(
            cached_prefix_lens.astype(np.int32), device
        )
        ub_attn.max_seqlen_k = int(ctx_lens.max())

    def prepare_decode(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
    ):
        scheduled_bs = batch.total_seqs_num_decode
        self.total_blocks = 0
        dropout_p = 0.0
        min_seqlen_q = 0

        context_lens = np.asarray(batch.context_lens, dtype=np.int32)
        block_tables = batch.block_tables

        if max_seqlen_q > 1:
            num_rejected = self.model_runner.tokenID_processor.num_rejected
            if num_rejected is not None:
                context_lens -= num_rejected
        positions = decode_positions(context_lens, max_seqlen_q)
        max_seqlen_k = np.max(context_lens)

        # Before the slots, not after: `slot_mapping` reads this packed table.
        self.prepare_block_tables(batch)

        var = self.model_runner.forward_vars
        scheduled_tokens = batch.total_tokens_num_decode
        var["slot_mapping"].np[:running_tokens] = -1
        if max_seqlen_q > 1:
            slots = slot_mapping(
                positions,
                np.full(scheduled_bs, max_seqlen_q, dtype=np.int32),
                var["block_tables"].np,
                self.model_runner.block_size,
                scratch=self.token_axis_scratch,
            )
        else:
            # One token per sequence reduces the gather to its last row, and
            # `last_block_num_tokens` names the offset without a position.
            slots = [
                block_table[-1] * self.model_runner.block_size + last_block_num - 1
                for block_table, last_block_num in zip(
                    block_tables, batch.last_block_num_tokens
                )
            ]
        if not batch.is_dummy_run:
            var["slot_mapping"].np[:scheduled_tokens] = slots[:scheduled_tokens]

        var["positions"].np[:scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens
        var["context_lens"].np[scheduled_bs:running_bs] = 0

        # Prepare kv_indptr and kv_indices for persistent attention
        num_blocks_per_seq = cdiv(context_lens, self.block_size)
        kv_indptr = np.cumsum(num_blocks_per_seq)
        sum_blocks = kv_indptr[-1] if len(kv_indptr) > 0 else 0

        var["kv_indptr"].np[0] = 0
        var["kv_indptr"].np[1 : scheduled_bs + 1] = kv_indptr
        var["kv_indptr"].np[scheduled_bs + 1 : running_bs + 1] = sum_blocks

        vars_used = [
            ("slot_mapping", running_tokens),
            ("context_lens", running_bs),
            ("block_tables", running_bs),
            ("kv_indptr", running_bs + 1),
        ]

        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        # A view: `publish_cu_seqlens_q` already uploaded it this step, and
        # nothing here writes the host copy.
        ctx["cu_seqlens_q"] = var["cu_seqlens_q"].gpu[: running_bs + 1]
        if self.block_size in (256, 1024):
            ctx_pa_ps = self.set_aiter_persistent_worker_buffers(running_bs)
            ctx.update(ctx_pa_ps)

        ctx["kv_indices"] = var["kv_indices"].gpu
        max_seqlen_k = context_lens.max()
        kv_indices_generate_triton(
            ctx["block_tables"],
            ctx["kv_indices"],
            ctx["kv_indptr"],
            self.block_ratio,
            max_seqlen_k,
        )
        attn_metadata = AttentionMetaData(
            dropout_p=dropout_p,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            **ctx,
        )
        if self._has_sparse_attention:
            from atom.model_ops.minimax_m3.sparse_attn import (
                make_sparse_decode_metadata,
            )

            # Plain decode (q==1) and eagle3 spec-verify (q==num_spec+1) both run
            # the DECODE sparse path; the decode kernels handle q>1 per-token
            # causal internally via max_query_len.
            sparse_block_tables = self._get_sparse_attention_block_tables(
                attn_metadata.block_tables[:scheduled_bs],
                attn_metadata.context_lens[:scheduled_bs],
                scheduled_bs,
            )
            attn_metadata.sparse_attention_metadata = make_sparse_decode_metadata(
                seq_lens=attn_metadata.context_lens[:scheduled_bs],
                block_table=sparse_block_tables,
                slot_mapping=attn_metadata.slot_mapping,
                max_seq_len=int(max_seqlen_k),
                max_query_len=int(max_seqlen_q),
            )
        mrope_positions = self._build_mrope_decode_positions(
            batch, context_lens, max_seqlen_q
        )
        if mrope_positions is not None:
            positions = mrope_positions
        else:
            positions = var["positions"].copy_to_gpu(scheduled_tokens)
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
        """Compute per-ubatch forward_vars for CUDAGraph TBO.

        Splits the full-batch data into per-ubatch CpuGpuBuffers.
        The split point is bs // 2 to match CUDAGraph's baked-in token slices.
        """
        var = self.model_runner.forward_vars
        N = self._NUM_TBO_UBATCHES
        half = bs // N

        ub_ranges = [(0, half), (half, bs)]
        running_bs_list = [half, bs - half]

        for ub_idx, ((req_start, req_end), running_bs) in enumerate(
            zip(ub_ranges, running_bs_list)
        ):
            p = f"ub{ub_idx}_"
            ub_real_reqs = max(0, min(scheduled_bs, req_end) - req_start)

            var[f"{p}context_lens"].np[:ub_real_reqs] = var["context_lens"].np[
                req_start : req_start + ub_real_reqs
            ]
            var[f"{p}context_lens"].np[ub_real_reqs:running_bs] = 0

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
                (f"{p}slot_mapping", ub_running_tokens),
                (f"{p}block_tables", running_bs),
                (f"{p}kv_indptr", running_bs + 1),
                (f"{p}cu_seqlens_q", running_bs + 1),
            ]
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

            # Set PA persistent worker buffers for this ubatch
            if self.block_size in (256, 1024):
                self._set_ubatch_pa_buffers(running_bs, max_seqlen_q, ub_idx)

    def _set_ubatch_pa_buffers(self, running_bs, max_q_len, ubatch_idx):
        """Compute PA work buffers for a per-ubatch forward_vars set."""
        config = self.model_runner.config
        hf_config = config.hf_config
        num_query_heads = self.num_attention_heads
        num_kv_heads = max(
            1, hf_config.num_key_value_heads // get_tp_group().world_size
        )
        p = f"ub{ubatch_idx}_"
        var = self.model_runner.forward_vars

        aiter.get_pa_metadata_v1(
            var[f"{p}cu_seqlens_q"].gpu[: running_bs + 1],
            var[f"{p}kv_indptr"].gpu[: running_bs + 1],
            var[f"{p}context_lens"].gpu[:running_bs],
            num_query_heads // num_kv_heads,
            num_kv_heads,
            True,
            var[f"{p}work_meta_data"],
            var[f"{p}work_indptr"],
            var[f"{p}work_info_set"],
            var[f"{p}reduce_indptr"],
            var[f"{p}reduce_final_map"],
            var[f"{p}reduce_partial_map"],
            kv_granularity=max(self.block_size, 16),
            block_size=self.block_size,
            max_seqlen_qo=max_q_len,
            uni_seqlen_qo=max_q_len,
            fast_mode=True,
            max_split_per_batch=-1,
        )

    def build_ubatch_metadata(
        self,
        ubatch_idx: int,
        running_bs: int,
    ) -> AttentionMetaData:
        """Create per-ubatch AttentionMetaData from pre-allocated forward_vars."""
        var = self.model_runner.forward_vars
        p = f"ub{ubatch_idx}_"
        max_q_len = var["max_qlen"]

        # Compute PA work buffers for this ubatch
        if self.block_size in (256, 1024):
            self._set_ubatch_pa_buffers(running_bs, max_q_len, ubatch_idx)

        attn = AttentionMetaData(
            slot_mapping=var[f"{p}slot_mapping"].gpu[: running_bs * max_q_len],
            context_lens=var[f"{p}context_lens"].gpu[:running_bs],
            block_tables=var[f"{p}block_tables"].gpu[:running_bs],
            max_seqlen_q=max_q_len,
            max_seqlen_k=self.model_runner.config.max_model_len,
            cu_seqlens_q=var[f"{p}cu_seqlens_q"].gpu[: running_bs + 1],
            kv_indptr=var[f"{p}kv_indptr"].gpu[: running_bs + 1],
            kv_indices=var[f"{p}kv_indices"].gpu,
            work_meta_data=var[f"{p}work_meta_data"],
            work_info_set=var[f"{p}work_info_set"],
            work_indptr=var[f"{p}work_indptr"],
            reduce_indptr=var[f"{p}reduce_indptr"],
            reduce_final_map=var[f"{p}reduce_final_map"],
            reduce_partial_map=var[f"{p}reduce_partial_map"],
        )
        if self._has_sparse_attention:
            if max_q_len > 1:
                raise NotImplementedError(
                    "MiniMax M3 sparse TBO decode with max_q_len > 1 is not "
                    "implemented."
                )
            from atom.model_ops.minimax_m3.sparse_attn import (
                make_sparse_decode_metadata,
            )

            sparse_block_tables = self._get_sparse_attention_block_tables(
                attn.block_tables,
                attn.context_lens,
                running_bs,
            )
            attn.sparse_attention_metadata = make_sparse_decode_metadata(
                seq_lens=attn.context_lens,
                block_table=sparse_block_tables,
                slot_mapping=attn.slot_mapping,
                max_seq_len=attn.max_seqlen_k,
            )
        return attn

    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        var = self.model_runner.forward_vars
        max_seqlen_k = self.model_runner.config.max_model_len
        max_q_len = int(var["max_qlen"])

        if self.block_size in (256, 1024):
            ctx_pa_ps = self.set_aiter_persistent_worker_buffers(bs)
        else:
            ctx_pa_ps = {}
        scheduled_tokens = bs * max_q_len
        attn_metadata = AttentionMetaData(
            slot_mapping=var["slot_mapping"].gpu[:scheduled_tokens],
            context_lens=var["context_lens"].gpu[:bs],
            block_tables=var["block_tables"].gpu[:bs],
            max_seqlen_q=max_q_len,
            cu_seqlens_q=var["cu_seqlens_q"].gpu[: bs + 1],
            kv_indptr=var["kv_indptr"].gpu[: bs + 1],
            kv_indices=var["kv_indices"].gpu,
            max_seqlen_k=max_seqlen_k,
            **ctx_pa_ps,
        )
        if self._has_sparse_attention:
            from atom.model_ops.minimax_m3.sparse_attn import (
                make_sparse_decode_metadata,
            )

            seq_lens = attn_metadata.context_lens
            # Both plain decode (q==1) and eagle3 spec-verify (q==num_spec+1) use
            # the DECODE sparse path; the decode kernels handle q>1 per-token causal
            # internally (max_query_len), so no separate prefill graph is needed.
            sparse_block_tables = self._get_sparse_attention_block_tables(
                attn_metadata.block_tables,
                seq_lens,
                bs,
            )
            attn_metadata.sparse_attention_metadata = make_sparse_decode_metadata(
                seq_lens=seq_lens,
                block_table=sparse_block_tables,
                slot_mapping=attn_metadata.slot_mapping,
                max_seq_len=attn_metadata.max_seqlen_k,
                max_query_len=max_q_len,
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
