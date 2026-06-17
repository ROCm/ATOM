# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Type

import aiter
import numpy as np
import torch
from aiter.dist.parallel_state import get_tp_group
from atom.model_engine.scheduler import ScheduledBatch
from atom.utils import CpuGpuBuffer
from atom.utils.block_convert import kv_indices_generate_triton
from atom.model_ops.attention_mha import PagedAttentionImpl
from atom.utils.forward_context import AttentionMetaData, Context
from atom.utils.tbo import TokenSplitPrefillState
from atom.utils import envs

from .backends import AttentionBackend, CommonAttentionBuilder


def cdiv(a, b):
    return (a + b - 1) // b


class AiterBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ATOM_ATTENTION"

    @staticmethod
    def get_builder_cls() -> Type["AiterAttentionMetadataBuilder"]:
        return AiterAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return PagedAttentionImpl


class AiterAttentionMetadataBuilder(CommonAttentionBuilder):
    BLOCK_TABLE_EXTENDER: list[list[int]] = [[]]

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
        self._is_minimax_m3_sparse = bool(
            getattr(text_config, "sparse_attention_config", None)
        )
        if self._is_minimax_m3_sparse:
            from atom.model_ops.minimax_m3.sparse_attn import SPARSE_BLOCK_SIZE

            if model_runner.block_size != SPARSE_BLOCK_SIZE:
                raise ValueError(
                    "MiniMax-M3 native sparse attention requires "
                    f"--block-size {SPARSE_BLOCK_SIZE}, got {model_runner.block_size}."
                )
            self.block_size = SPARSE_BLOCK_SIZE
        else:
            self.block_size = 1024 if model_runner.block_size == 1024 else 16
        if envs.ATOM_USE_UNIFIED_ATTN and not self._is_minimax_m3_sparse:
            # SHUFFLE (pre-shuffled) KV cache: use the logical block size directly
            # as the physical block size so block_ratio == 1 and
            # unified_attention's block_table needs no logical->physical
            # conversion. Pass --block-size equal to the performant physical
            # page: fp8 packs x=16 - 128; bf16 packs x=8 - 64 (both keep a
            # 128-byte physical page, i.e. block_size // x == 8).
            expected = 128 if model_runner.kv_cache_dtype in ("fp8",) else 64
            assert model_runner.block_size == expected, (
                f"ATOM_USE_UNIFIED_ATTN=1 expects --block-size {expected} "
                f"for {model_runner.kv_cache_dtype} KV cache (so block_ratio == 1), "
                f"got --block-size {model_runner.block_size}"
            )
            self.block_size = model_runner.block_size
        assert (
            model_runner.block_size % self.block_size == 0
        ), f"model_runner.block_size must be divisible by block_size but got {model_runner.block_size=}, block_size={self.block_size}, please set --block-size (model_runner.block_size) to be divisible by {self.block_size}"
        super().__init__(model_runner)
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

    def compute_block_bytes(self) -> int:
        """Standard split-K/V MHA per-block bytes.

        - Standard models: `[2, num_hidden_layers, blocks, block_size,
          num_kv_heads, head_dim]` for kv_cache + matching kv_scale (fp32).
        - MiMo-V2-Flash: per-layer-type accounting (full vs SWA layers
          have different num_kv_heads).
        """
        from aiter import dtypes

        runner = self.model_runner
        config = runner.config
        hf_config = config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        num_kv_heads = runner._get_num_kv_heads()
        total_num_layers = runner._get_total_num_layers()
        kv_dtype_size = dtypes.d_dtypes[config.kv_cache_dtype].itemsize

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
            block_bytes = (
                2
                * num_full_layers
                * runner.block_size
                * num_kv_heads
                * hf_config.head_dim
                * kv_dtype_size
            )
            block_bytes += (
                2
                * num_swa_layers
                * runner.block_size
                * swa_kv_heads
                * hf_config.head_dim
                * kv_dtype_size
            )
            block_bytes += (
                2
                * num_full_layers
                * num_kv_heads
                * runner.physical_block_size
                * 4  # float32 kv_scale
            )
            block_bytes += (
                2
                * num_swa_layers
                * swa_kv_heads
                * runner.physical_block_size
                * 4  # float32 kv_scale
            )
            return block_bytes

        # Standard MHA path.
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * runner.block_size
            * num_kv_heads
            * hf_config.head_dim
            * kv_dtype_size
        )
        block_bytes += (
            2
            * hf_config.num_hidden_layers
            * num_kv_heads
            * runner.physical_block_size
            * 4  # float32 kv_scale
        )
        sparse_cfg = getattr(text_config, "sparse_attention_config", None)
        if sparse_cfg:
            sparse_layers = sum(
                1 for enabled in sparse_cfg.get("sparse_attention_freq", []) if enabled
            )
            index_dim = sparse_cfg["sparse_index_dim"]
            block_bytes += (
                sparse_layers
                * runner.physical_block_size
                * index_dim
                * torch.empty((), dtype=config.torch_dtype).element_size()
            )
        return block_bytes

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict:
        """Allocate the standard split-K/V paged KV cache.

        - MiMo-V2-Flash defers per-module allocation to build_kv_cache_tensor
          (each module has its own num_kv_heads), returning sentinels here.
        - All other models use a single `[2, num_hidden_layers, ...]` tensor
          shared across layers; per-layer slicing happens in build_kv_cache_tensor.
        """
        from aiter import dtypes

        runner = self.model_runner
        config = runner.config
        hf_config = config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)

        if runner.is_mimo_v2():
            # Per-layer allocation deferred (each module gets its own
            # correctly-sized tensor matching its num_kv_heads).
            return {
                "kv_cache": None,
                "kv_scale": None,
                "_kv_layer_cache_store": [],
            }

        tensors = {
            "kv_cache": torch.zeros(
                2,
                hf_config.num_hidden_layers,
                runner.num_physical_kvcache_blocks,
                runner.physical_block_size,
                num_kv_heads,
                hf_config.head_dim,
                dtype=dtypes.d_dtypes[config.kv_cache_dtype],
                device="cuda",
            ),
            "kv_scale": torch.zeros(
                2,
                hf_config.num_hidden_layers,
                runner.num_physical_kvcache_blocks,
                num_kv_heads,
                runner.physical_block_size,
                dtype=dtypes.fp32,
                device="cuda",
            ),
        }
        sparse_cfg = getattr(text_config, "sparse_attention_config", None)
        if sparse_cfg:
            sparse_layers = sum(
                1 for enabled in sparse_cfg.get("sparse_attention_freq", []) if enabled
            )
            tensors["minimax_m3_index_cache"] = torch.zeros(
                sparse_layers,
                runner.num_physical_kvcache_blocks,
                runner.physical_block_size,
                sparse_cfg["sparse_index_dim"],
                dtype=config.torch_dtype,
                device="cuda",
            )
            tensors["_minimax_m3_sparse_cache_next"] = 0
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
        from aiter import dtypes

        if getattr(module, "is_minimax_m3_sparse_attention", False):
            runner = self.model_runner
            config = runner.config
            sparse_idx = runner._minimax_m3_sparse_cache_next
            runner._minimax_m3_sparse_cache_next += 1
            module.kv_cache = runner.kv_cache[:, layer_id].permute(1, 0, 2, 3, 4)
            module.index_cache = runner.minimax_m3_index_cache[sparse_idx]
            key_cache, value_cache = module.kv_cache.unbind(1)
            module.max_model_len = config.max_model_len
            return KVCacheTensor(
                layer_num=layer_id,
                k_cache=key_cache,
                v_cache=value_cache,
                k_scale=None,
                v_scale=None,
            )

        if not (
            hasattr(module, "base_attention")
            and hasattr(module, "use_mla")
            and not module.use_mla
        ):
            return None

        runner = self.model_runner
        config = runner.config
        hf_config = config.hf_config

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

        if runner.is_mimo_v2():
            # Per-layer allocation: each module gets its own correctly-sized
            # tensor matching its num_kv_heads.
            kv_dtype = dtypes.d_dtypes[config.kv_cache_dtype]
            x = 16 // kv_dtype.itemsize
            module_kv_heads = module.num_kv_heads
            k_cache = torch.zeros(
                runner.num_physical_kvcache_blocks,
                module_kv_heads,
                hf_config.head_dim // x,
                runner.physical_block_size,
                x,
                dtype=kv_dtype,
                device="cuda",
            )
            v_cache = torch.zeros(
                runner.num_physical_kvcache_blocks,
                module_kv_heads,
                runner.physical_block_size // x,
                hf_config.head_dim,
                x,
                dtype=kv_dtype,
                device="cuda",
            )
            if config.kv_cache_dtype == "fp8":
                module.k_scale = torch.zeros(
                    runner.num_physical_kvcache_blocks,
                    module_kv_heads,
                    runner.physical_block_size,
                    dtype=dtypes.fp32,
                    device="cuda",
                )
                module.v_scale = torch.zeros(
                    runner.num_physical_kvcache_blocks,
                    module_kv_heads,
                    runner.physical_block_size,
                    dtype=dtypes.fp32,
                    device="cuda",
                )
            runner._kv_layer_cache_store.append(
                (k_cache, v_cache, module.k_scale, module.v_scale)
            )
        else:
            x = 16 // runner.kv_cache.element_size()
            k_cache = runner.kv_cache[0, attn_idx].view(
                runner.num_physical_kvcache_blocks,
                runner.num_kv_heads,
                hf_config.head_dim // x,
                runner.physical_block_size,
                x,
            )
            # V cache uses the same 5D SHUFFLE layout as the MiMo-V2 per-module
            # allocator above: [num_blocks, num_kv_heads, block_size//x, head_dim, x].
            v_cache = runner.kv_cache[1, attn_idx].view(
                runner.num_physical_kvcache_blocks,
                runner.num_kv_heads,
                runner.physical_block_size // x,
                hf_config.head_dim,
                x,
            )
            if config.kv_cache_dtype == "fp8":
                module.k_scale = runner.kv_scale[0, attn_idx]
                module.v_scale = runner.kv_scale[1, attn_idx]

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

    def get_kv_transfer_tensors(self):
        from atom.kv_transfer.disaggregation.types import (
            KVTransferRegion,
            KVTransferTensors,
        )

        runner = self.model_runner
        has_unified_kv = hasattr(runner, "kv_cache") and runner.kv_cache is not None
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
            num_layers = runner.kv_cache.shape[1]
            for layer_id in range(num_layers):
                _add_region(runner.kv_cache[0, layer_id])  # K
                _add_region(runner.kv_cache[1, layer_id])  # V
            if hasattr(runner, "kv_scale") and runner.kv_scale is not None:
                for layer_id in range(num_layers):
                    _add_region(runner.kv_scale[0, layer_id])
                    _add_region(runner.kv_scale[1, layer_id])

        return KVTransferTensors(
            block_regions=block_regions,
            slot_regions=[],
            num_blocks=runner.num_physical_kvcache_blocks,
        )

    def prepare_prefill(self, batch: ScheduledBatch):
        attn_metadata, positions = CommonAttentionBuilder.prepare_prefill(self, batch)
        if self._is_minimax_m3_sparse and not attn_metadata.has_cached:
            bs = batch.total_seqs_num_prefill
            self.prepare_block_tables(batch)
            attn_metadata.block_tables = self.model_runner.forward_vars[
                "block_tables"
            ].copy_to_gpu(bs)
        if self._is_minimax_m3_sparse:
            from atom.model_ops.minimax_m3.metadata import (
                MiniMaxM3SparseMetadata,
                MiniMaxM3SparsePrefillMetadata,
            )
            from atom.model_ops.minimax_m3.sparse_attn import SPARSE_BLOCK_SIZE

            bs = batch.total_seqs_num_prefill
            cu_q = attn_metadata.cu_seqlens_q
            seq_lens = attn_metadata.context_lens
            query_lens = cu_q[1 : bs + 1] - cu_q[:bs]
            prefix_lens = seq_lens - query_lens
            seq_lens_cpu = np.asarray(batch.context_lens[:bs], dtype=np.int64)
            total_kv_blocks = int(
                ((seq_lens_cpu + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE).sum()
            )
            # Per-query-token request id + absolute position, built ONCE here on
            # the CPU (numpy, no device sync) and reused by every sparse layer's
            # ASM prefill block-table builder. Layer-invariant: depends only on
            # cu_seqlens_q + prefix lengths. Uses the host cu_seqlens_q mirror to
            # avoid a D2H copy (a GPU repeat_interleave would force a sync).
            cu_np = np.asarray(
                self.model_runner.forward_vars["cu_seqlens_q"].np[: bs + 1],
                dtype=np.int64,
            )
            q_lens_np = cu_np[1:] - cu_np[:bs]
            total_q = int(cu_np[bs])
            prefix_np = (seq_lens_cpu - q_lens_np).astype(np.int64)
            req_id_np = np.repeat(np.arange(bs, dtype=np.int32), q_lens_np)
            abs_pos_np = (
                prefix_np[req_id_np]
                + (np.arange(total_q, dtype=np.int64) - cu_np[req_id_np])
            ).astype(np.int32)
            dev = attn_metadata.block_tables.device
            query_req_id = torch.from_numpy(req_id_np).to(dev, non_blocking=True)
            query_abs_pos = torch.from_numpy(abs_pos_np).to(dev, non_blocking=True)
            per_token_qo_indptr = torch.arange(
                total_q + 1, dtype=torch.int32, device=dev
            )
            prefill = MiniMaxM3SparsePrefillMetadata(
                cu_seqlens_q=cu_q,
                cu_seqlens_k=attn_metadata.cu_seqlens_k,
                seq_lens=seq_lens,
                context_lens=prefix_lens,
                block_table=attn_metadata.block_tables,
                max_query_len=attn_metadata.max_seqlen_q,
                max_seq_len=attn_metadata.max_seqlen_k,
                total_kv_blocks=total_kv_blocks,
                query_req_id=query_req_id,
                query_abs_pos=query_abs_pos,
                per_token_qo_indptr=per_token_qo_indptr,
            )
            attn_metadata.minimax_m3_sparse_metadata = MiniMaxM3SparseMetadata(
                seq_lens=seq_lens,
                max_seq_len=attn_metadata.max_seqlen_k,
                slot_mapping=attn_metadata.slot_mapping,
                num_actual_tokens=batch.total_tokens_num_prefill,
                num_decodes=0,
                num_decode_tokens=0,
                num_prefills=bs,
                num_prefill_tokens=batch.total_tokens_num_prefill,
                prefill=prefill,
                decode=None,
            )
        if self._tbo_token_split:
            self._stash_tbo_token_split_prefill_state(batch)
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
            block_tables=[
                np.asarray(bt, dtype=np.int32) for bt in batch.block_tables[:bs]
            ],
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
        padded_bs: int,
        ubatch_idx: int = 0,
    ) -> AttentionMetaData:
        del ubatch_idx
        from atom.utils.tbo.ubatch_splitting import split_attn_metadata

        ub_attn = split_attn_metadata(attn_metadata, ub_slice, padded_bs)
        self._attach_tbo_token_split_straddle_prefix(ub_attn, ub_slice)
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
        ctx_lens = cached + new_lens
        ctx_lens[0] += prefix_len
        total_kv = int(ctx_lens.sum())

        max_blocks = max(
            (len(block_tables_host[first_req + i]) for i in range(ub_num_reqs)),
            default=1,
        )
        bt = np.zeros((ub_num_reqs, max_blocks), dtype=np.int32)
        for i in range(ub_num_reqs):
            row = block_tables_host[first_req + i]
            bt[i, : len(row)] = row

        cu_k = np.zeros(ub_num_reqs + 1, dtype=np.int32)
        np.cumsum(ctx_lens.astype(np.int32), out=cu_k[1:])

        ub_attn.has_cached = True
        ub_attn.total_kv = total_kv
        ub_attn.context_lens = torch.from_numpy(ctx_lens.astype(np.int32)).to(
            device, non_blocking=True
        )
        ub_attn.block_tables = torch.from_numpy(bt).to(device, non_blocking=True)
        ub_attn.cu_seqlens_k = torch.from_numpy(cu_k).to(device, non_blocking=True)
        ub_attn.seq_starts = torch.zeros(ub_num_reqs, dtype=torch.int32, device=device)
        ub_attn.num_cached_tokens = torch.from_numpy(ctx_lens.astype(np.int32)).to(
            device, non_blocking=True
        )
        ub_attn.max_seqlen_k = int(ctx_lens.max())

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        scheduled_bs = batch.total_seqs_num_decode
        self.total_blocks = 0
        dropout_p = 0.0
        max_seqlen_q = batch.num_spec_step + 1
        min_seqlen_q = 0

        context_lens = np.asarray(batch.context_lens, dtype=np.int32)
        block_tables = batch.block_tables

        if max_seqlen_q > 1:
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
        max_seqlen_k = np.max(context_lens)

        self.prepare_block_tables(batch)

        var = self.model_runner.forward_vars
        sum_scheduled_tokens = batch.total_tokens_num_decode
        var["slot_mapping"].np[: bs * max_seqlen_q] = -1
        if not batch.is_dummy_run:
            var["slot_mapping"].np[:sum_scheduled_tokens] = slot_mapping[
                :sum_scheduled_tokens
            ]

        var["positions"].np[:sum_scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens
        var["context_lens"].np[scheduled_bs:bs] = 0

        # Prepare kv_indptr and kv_indices for persistent attention
        num_blocks_per_seq = cdiv(context_lens, self.block_size)
        kv_indptr = np.cumsum(num_blocks_per_seq)
        sum_blocks = kv_indptr[-1] if len(kv_indptr) > 0 else 0

        var["kv_indptr"].np[0] = 0
        var["kv_indptr"].np[1 : scheduled_bs + 1] = kv_indptr
        var["kv_indptr"].np[scheduled_bs + 1 : bs + 1] = sum_blocks

        vars_used = [
            ("slot_mapping", bs * max_seqlen_q),
            ("context_lens", bs),
            ("cu_seqlens_q", bs + 1),
            ("block_tables", bs),
            ("kv_indptr", bs + 1),
        ]

        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        if self.block_size == 1024:
            ctx_pa_ps = self.set_aiter_persistent_worker_buffers(bs)
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
        if self._is_minimax_m3_sparse:
            from atom.model_ops.minimax_m3.metadata import (
                MiniMaxM3SparseDecodeMetadata,
                MiniMaxM3SparseMetadata,
                MiniMaxM3SparsePrefillMetadata,
            )
            from atom.model_ops.minimax_m3.sparse_attn import SPARSE_BLOCK_SIZE

            if max_seqlen_q <= 1:
                decode = MiniMaxM3SparseDecodeMetadata(
                    seq_lens=attn_metadata.context_lens[:scheduled_bs],
                    block_table=attn_metadata.block_tables[:scheduled_bs],
                )
                attn_metadata.minimax_m3_sparse_metadata = MiniMaxM3SparseMetadata(
                    seq_lens=attn_metadata.context_lens[:scheduled_bs],
                    max_seq_len=int(max_seqlen_k),
                    slot_mapping=attn_metadata.slot_mapping,
                    num_actual_tokens=sum_scheduled_tokens,
                    num_decodes=scheduled_bs,
                    num_decode_tokens=sum_scheduled_tokens,
                    num_prefills=0,
                    num_prefill_tokens=0,
                    prefill=None,
                    decode=decode,
                )
            else:
                cu_q = attn_metadata.cu_seqlens_q
                seq_lens = attn_metadata.context_lens[:scheduled_bs]
                query_lens = cu_q[1 : scheduled_bs + 1] - cu_q[:scheduled_bs]
                prefix_lens = seq_lens - query_lens
                seq_lens_cpu = np.asarray(context_lens[:scheduled_bs], dtype=np.int64)
                total_kv_blocks = int(
                    ((seq_lens_cpu + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE).sum()
                )
                prefill = MiniMaxM3SparsePrefillMetadata(
                    cu_seqlens_q=cu_q[: scheduled_bs + 1],
                    cu_seqlens_k=cu_q[: scheduled_bs + 1],
                    seq_lens=seq_lens,
                    context_lens=prefix_lens,
                    block_table=attn_metadata.block_tables[:scheduled_bs],
                    max_query_len=max_seqlen_q,
                    max_seq_len=int(max_seqlen_k),
                    total_kv_blocks=total_kv_blocks,
                )
                attn_metadata.minimax_m3_sparse_metadata = MiniMaxM3SparseMetadata(
                    seq_lens=seq_lens,
                    max_seq_len=int(max_seqlen_k),
                    slot_mapping=attn_metadata.slot_mapping,
                    num_actual_tokens=sum_scheduled_tokens,
                    num_decodes=0,
                    num_decode_tokens=0,
                    num_prefills=scheduled_bs,
                    num_prefill_tokens=sum_scheduled_tokens,
                    prefill=prefill,
                    decode=None,
                )
        mrope_positions = self._build_mrope_decode_positions(
            batch, context_lens, max_seqlen_q
        )
        if mrope_positions is not None:
            positions = mrope_positions
        else:
            positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
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
        """Compute per-ubatch forward_vars for CUDAGraph TBO.

        Splits the full-batch data into per-ubatch CpuGpuBuffers.
        The split point is bs // 2 to match CUDAGraph's baked-in token slices.
        """
        var = self.model_runner.forward_vars
        N = self._NUM_TBO_UBATCHES
        half = bs // N

        ub_ranges = [(0, half), (half, bs)]
        padded_bs_list = [half, bs - half]

        for ub_idx, ((req_start, req_end), padded_bs) in enumerate(
            zip(ub_ranges, padded_bs_list)
        ):
            p = f"ub{ub_idx}_"
            ub_real_reqs = max(0, min(scheduled_bs, req_end) - req_start)

            var[f"{p}context_lens"].np[:ub_real_reqs] = var["context_lens"].np[
                req_start : req_start + ub_real_reqs
            ]
            var[f"{p}context_lens"].np[ub_real_reqs:padded_bs] = 0

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
                (f"{p}slot_mapping", padded_tok_count),
                (f"{p}block_tables", padded_bs),
                (f"{p}kv_indptr", padded_bs + 1),
                (f"{p}cu_seqlens_q", padded_bs + 1),
            ]
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

            # Set PA persistent worker buffers for this ubatch
            if self.block_size == 1024:
                self._set_ubatch_pa_buffers(padded_bs, max_seqlen_q, ub_idx)

    def _set_ubatch_pa_buffers(self, padded_bs, max_q_len, ubatch_idx):
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
            var[f"{p}cu_seqlens_q"].gpu[: padded_bs + 1],
            var[f"{p}kv_indptr"].gpu[: padded_bs + 1],
            var[f"{p}context_lens"].gpu[:padded_bs],
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
        padded_bs: int,
    ) -> AttentionMetaData:
        """Create per-ubatch AttentionMetaData from pre-allocated forward_vars."""
        var = self.model_runner.forward_vars
        p = f"ub{ubatch_idx}_"
        max_q_len = var["max_qlen"]

        # Compute PA work buffers for this ubatch
        if self.block_size == 1024:
            self._set_ubatch_pa_buffers(padded_bs, max_q_len, ubatch_idx)

        attn = AttentionMetaData(
            slot_mapping=var[f"{p}slot_mapping"].gpu[: padded_bs * max_q_len],
            context_lens=var[f"{p}context_lens"].gpu[:padded_bs],
            block_tables=var[f"{p}block_tables"].gpu[:padded_bs],
            max_seqlen_q=max_q_len,
            cu_seqlens_q=var[f"{p}cu_seqlens_q"].gpu[: padded_bs + 1],
            kv_indptr=var[f"{p}kv_indptr"].gpu[: padded_bs + 1],
            kv_indices=var[f"{p}kv_indices"].gpu,
            work_meta_data=var[f"{p}work_meta_data"],
            work_info_set=var[f"{p}work_info_set"],
            work_indptr=var[f"{p}work_indptr"],
            reduce_indptr=var[f"{p}reduce_indptr"],
            reduce_final_map=var[f"{p}reduce_final_map"],
            reduce_partial_map=var[f"{p}reduce_partial_map"],
        )
        return attn

    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        var = self.model_runner.forward_vars
        if self.block_size == 1024:
            ctx_pa_ps = self.set_aiter_persistent_worker_buffers(bs)
        else:
            ctx_pa_ps = {}
        max_q_len = var["max_qlen"]
        total_tokens = bs * max_q_len
        attn_metadata = AttentionMetaData(
            slot_mapping=var["slot_mapping"].gpu[:total_tokens],
            context_lens=var["context_lens"].gpu[:bs],
            block_tables=var["block_tables"].gpu[:bs],
            max_seqlen_q=max_q_len,
            cu_seqlens_q=var["cu_seqlens_q"].gpu[: bs + 1],
            kv_indptr=var["kv_indptr"].gpu[: bs + 1],
            kv_indices=var["kv_indices"].gpu,
            max_seqlen_k=self.model_runner.config.max_model_len,
            **ctx_pa_ps,
        )
        if self._is_minimax_m3_sparse:
            from atom.model_ops.minimax_m3.metadata import (
                MiniMaxM3SparseDecodeMetadata,
                MiniMaxM3SparseMetadata,
                MiniMaxM3SparsePrefillMetadata,
            )
            from atom.model_ops.minimax_m3.sparse_attn import SPARSE_BLOCK_SIZE

            seq_lens = attn_metadata.context_lens
            if max_q_len <= 1:
                decode = MiniMaxM3SparseDecodeMetadata(
                    seq_lens=seq_lens,
                    block_table=attn_metadata.block_tables,
                )
                attn_metadata.minimax_m3_sparse_metadata = MiniMaxM3SparseMetadata(
                    seq_lens=seq_lens,
                    max_seq_len=attn_metadata.max_seqlen_k,
                    slot_mapping=attn_metadata.slot_mapping,
                    num_actual_tokens=total_tokens,
                    num_decodes=bs,
                    num_decode_tokens=total_tokens,
                    num_prefills=0,
                    num_prefill_tokens=0,
                    prefill=None,
                    decode=decode,
                )
            else:
                query_lens = (
                    attn_metadata.cu_seqlens_q[1 : bs + 1]
                    - attn_metadata.cu_seqlens_q[:bs]
                )
                prefix_lens = seq_lens - query_lens
                total_kv_blocks = (
                    torch.div(
                        seq_lens + SPARSE_BLOCK_SIZE - 1,
                        SPARSE_BLOCK_SIZE,
                        rounding_mode="floor",
                    )
                    .sum()
                    .item()
                )
                prefill = MiniMaxM3SparsePrefillMetadata(
                    cu_seqlens_q=attn_metadata.cu_seqlens_q,
                    cu_seqlens_k=attn_metadata.cu_seqlens_q,
                    seq_lens=seq_lens,
                    context_lens=prefix_lens,
                    block_table=attn_metadata.block_tables,
                    max_query_len=max_q_len,
                    max_seq_len=attn_metadata.max_seqlen_k,
                    total_kv_blocks=int(total_kv_blocks),
                )
                attn_metadata.minimax_m3_sparse_metadata = MiniMaxM3SparseMetadata(
                    seq_lens=seq_lens,
                    max_seq_len=attn_metadata.max_seqlen_k,
                    slot_mapping=attn_metadata.slot_mapping,
                    num_actual_tokens=total_tokens,
                    num_decodes=0,
                    num_decode_tokens=0,
                    num_prefills=bs,
                    num_prefill_tokens=total_tokens,
                    prefill=prefill,
                    decode=None,
                )

        positions = var["positions"].copy_to_gpu(total_tokens)
        context = Context(
            positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs
        )
        return attn_metadata, context
