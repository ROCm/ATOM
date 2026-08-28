# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import torch
from aiter import dtypes

from atom.config import _MQA_LOGITS_PRESHUFFLE_ROWS
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mla import MLAAttention
from atom.model_ops.glm5_next.kpool import pooled_path_enabled
from atom.utils import envs

from .aiter_mla import AiterMLAMetadataBuilder, mla_kv_entry_dim
from .backends import AttentionBackend
from .gdn_attn import GDNStateMixin
from .sub_pool_spec import SubPoolSpec, page_pool, state_pool
from .triton_mla import TritonMLAMetadataBuilder


class KimiMLAGDNBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "KIMI_MLA_GDN"

    @staticmethod
    def get_builder_cls() -> type["_KimiMLAGDNCommon"]:
        if envs.ATOM_USE_TRITON_MLA:
            return KimiTritonMLAGDNMetadataBuilder
        return KimiAiterMLAGDNMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["MLAAttention"]:
        return MLAAttention


class _KimiMLAGDNCommon(GDNStateMixin):
    def __init__(self, model_runner):
        super().__init__(model_runner=model_runner)
        self.mla_idx_by_layer = {
            layer: index
            for index, layer in enumerate(model_runner.full_attention_layers)
        }
        self.kda_idx_by_layer = {
            layer: index
            for index, layer in enumerate(model_runner.kda_attention_layers)
        }

    def _num_cache_rows(self) -> int:
        """Rows in the MLA pool: the target's full-attention layers plus any
        draft layers that share this pool.

        Derived from `_get_total_num_layers()` rather than from the
        `num_draft_layers` argument ModelRunner passes to
        `allocate_kv_cache_tensors`, so the row count the pool is SIZED for
        (`sub_pool_specs`) and the row count it is ALLOCATED with can never
        disagree: a draft that owns a sibling pool is excluded from both at
        once. Mirrors `AiterMLAMetadataBuilder`, which reads the same
        method in both places.
        """
        runner = self.model_runner
        hf = runner.config.hf_config
        num_draft = runner._get_total_num_layers() - hf.num_hidden_layers
        return runner.num_full_attn + num_draft

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        """MLA paged KV for the full-attention layers, plus the KDA/GDN
        per-request state pool (`GDNStateMixin.state_spec`)."""
        runner = self.model_runner
        config = runner.config
        hf = config.hf_config
        entry = mla_kv_entry_dim(hf)
        kv_dtype_size = dtypes.d_dtypes[config.kv_cache_dtype].itemsize
        block_bytes = self._num_cache_rows() * runner.block_size * entry * kv_dtype_size
        if runner.is_deepseek_v32:
            # Sparse-indexer key cache rides the same paged pool (GLM-5.3-Flash).
            index_cache_layer_ids, _ = self._index_cache_layout()
            block_bytes += len(index_cache_layer_ids) * self._index_cache_block_bytes()
        return [page_pool(block_bytes), self.state_spec()]

    def _aligned_index_dim(self) -> int:
        """Indexer entry width, padded to 16B so inductor sees aligned rows."""
        hf = self.model_runner.config.hf_config
        return ((hf.index_head_dim + 4 + 15) // 16) * 16

    # ---- kpool tail buffer -------------------------------------------------
    #
    # GLM-5.3-Flash's indexer caches one POOLED key per `index_kpool` tokens, so
    # the in-progress pool's raw K and gate score have to outlive the step that
    # produced them. They ride the per-request state slots KDA already owns
    # rather than a second paged cache: the buffer is `index_kpool - 1` useful
    # rows of 2 x head_dim bf16 per request per indexer layer -- well under a MB
    # for the whole engine -- and it inherits the state pool's lifetime, fork
    # and relocation semantics for free.

    def _kpool_size(self) -> int:
        """``index_kpool``, or 1 when this model does not pool indexer keys."""
        hf = self.model_runner.config.hf_config
        return int(getattr(hf, "index_kpool", 1) or 1)

    def _index_rows_per_block(self) -> int:
        """Index-cache rows one scheduler block owns.

        With the pooled path on, one cached key covers ``index_kpool`` tokens,
        so a block of ``block_size`` tokens needs ``block_size // index_kpool``
        rows rather than one per token. `Config` picks the block size so this
        stays a multiple of the preshuffled row count that
        `deepgemm_fp8_paged_mqa_logits` requires.

        Sizing, allocation, binding and the transfer-region byte count all read
        this one method, so they cannot disagree about how large the cache is.
        """
        runner = self.model_runner
        kpool = self._kpool_size()
        if not pooled_path_enabled(kpool):
            return runner.block_size
        assert runner.block_size % kpool == 0, (
            f"kv_cache_block_size={runner.block_size} is not divisible by "
            f"index_kpool={kpool}; Config sets the block size for exactly this"
        )
        rows = runner.block_size // kpool
        assert rows % _MQA_LOGITS_PRESHUFFLE_ROWS == 0, (
            f"{rows} pooled rows per block is not a multiple of "
            f"{_MQA_LOGITS_PRESHUFFLE_ROWS}, so deepgemm_fp8_paged_mqa_logits "
            "cannot stay in the preshuffled layout -- the only one it computes "
            "correctly. Raise kv_cache_block_size."
        )
        return rows

    def _index_cache_block_bytes(self, index_cache_layer=None) -> int:
        """Bytes one scheduler block occupies in one layer of the index cache.

        This cache is indexed by scheduler block already, and with pooling it
        holds fewer rows than tokens, so the base's `block_ratio` factor does
        not apply. Called with no tensor by the sizing path, which needs the
        number before anything is allocated.
        """
        return (
            self._index_rows_per_block()
            * self._aligned_index_dim()
            * dtypes.fp8.itemsize
        )

    def _kpool_tail_bytes(self) -> int:
        """Per-request tail bytes across every indexer-owning layer."""
        kpool = self._kpool_size()
        if kpool <= 1 or not self.model_runner.is_deepseek_v32:
            return 0
        hf = self.model_runner.config.hf_config
        index_cache_layer_ids, _ = self._index_cache_layout()
        per_layer = 2 * kpool * hf.index_head_dim * torch.bfloat16.itemsize
        return len(index_cache_layer_ids) * per_layer

    def state_spec(self) -> SubPoolSpec:
        """KDA recurrent state, plus GLM-5.3's kpool tail in the same entry.

        Widening the existing entry rather than declaring a second class keeps
        one slot id per request: the tail must be addressed by exactly the
        index KDA's state is, or a request would read another's partial pool.
        """
        base = super().state_spec()
        extra = self._kpool_tail_bytes()
        if not extra:
            return base
        return state_pool(
            base.name,
            base.entry_bytes + extra,
            entries_per_req=base.entries_per_req,
            extra_entries=base.extra_entries,
        )

    def allocate_per_req_cache(self, entries: dict[str, int]) -> dict:
        out = super().allocate_per_req_cache(entries)
        if not self._kpool_tail_bytes():
            return out
        hf = self.model_runner.config.hf_config
        index_cache_layer_ids, _ = self._index_cache_layout()
        out["kpool_tail_cache"] = torch.zeros(
            (
                len(index_cache_layer_ids),
                entries.get(STATE_SLOT_CLASS, 0),
                2,  # 0 = K, 1 = gate score
                self._kpool_size(),
                hf.index_head_dim,
            ),
            dtype=torch.bfloat16,
            device="cuda",
        )
        return out

    def relocate_state_slots(self, pairs) -> None:
        """Move the tail with the KDA state it shares a slot group with.

        Missing this would leave a relocated request reading the partial pool
        of whichever request previously held its new slot -- a corruption that
        only shows up once the pool boundary moves under load.
        """
        super().relocate_state_slots(pairs)
        tail = getattr(self.model_runner, "kpool_tail_cache", None)
        if tail is None or not pairs:
            return
        span = 1 + self.num_spec
        dsts, srcs = [], []
        for src_group, dst_group in pairs:
            src, dst = src_group * span, dst_group * span
            dsts.append(tail[:, dst : dst + span])
            srcs.append(tail[:, src : src + span])
        torch._foreach_copy_(dsts, srcs)

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict:
        del num_kv_heads, num_draft_layers
        runner = self.model_runner
        config = runner.config
        hf = config.hf_config
        num_layers = self._num_cache_rows()
        entry = mla_kv_entry_dim(hf)
        out: dict = {
            "kv_cache": torch.zeros(
                num_layers,
                runner.num_physical_kvcache_blocks,
                runner.physical_block_size,
                entry,
                dtype=dtypes.d_dtypes[config.kv_cache_dtype],
                device="cuda",
            )
        }
        if runner.is_deepseek_v32:
            # Sparse indexer key cache, one compact row per indexer-owning
            # layer. Mirrors AiterMLAMetadataBuilder.allocate_kv_cache_tensors,
            # which this mixin shadows.
            aligned = self._aligned_index_dim()
            index_cache_layer_ids, _ = self._index_cache_layout()
            out["aligned_index_dim"] = aligned
            out["index_cache_layer_ids"] = index_cache_layer_ids
            out["index_cache_layer_map"] = {
                global_layer_id: compact_layer_id
                for compact_layer_id, global_layer_id in enumerate(
                    index_cache_layer_ids
                )
            }
            # Indexed by SCHEDULER block, not by physical row: with the
            # pooled path on there are fewer index rows than tokens, so the
            # `num_physical_kvcache_blocks x physical_block_size` shape the
            # token-granular cache uses no longer describes this one. Blocks x
            # rows-per-block does, at any compression, and it makes the bytes
            # one scheduler block owns a plain `stride(0)`.
            num_sched_blocks = (
                runner.num_physical_kvcache_blocks
                * runner.physical_block_size
                // runner.block_size
            )
            out["index_cache"] = torch.zeros(
                len(index_cache_layer_ids),
                num_sched_blocks,
                self._index_rows_per_block(),
                aligned,
                dtype=dtypes.fp8,
                device="cuda",
            )
        return out

    def build_kv_cache_tensor(self, layer_id: int, module):
        from atom.config import KVCacheTensor

        runner = self.model_runner
        if hasattr(module, "base_linear_attention"):
            row = self.kda_idx_by_layer[layer_id]
            return KVCacheTensor(
                layer_num=layer_id,
                k_cache=runner.mamba_k_cache[row],
                v_cache=runner.mamba_v_cache[row],
                k_scale=None,
                v_scale=None,
            )

        if hasattr(module, "base_attention") and getattr(module, "use_mla", False):
            hf = runner.config.hf_config
            row = self.mla_idx_by_layer.get(layer_id)
            if row is None:
                assert layer_id >= hf.num_hidden_layers, (
                    f"MLA model layer {layer_id} is neither a K3 full-attention "
                    "layer nor a draft layer"
                )
                row = runner.num_full_attn + (layer_id - hf.num_hidden_layers)
            allocated_rows = runner.kv_cache.shape[0]
            assert row < allocated_rows, (
                f"MLA cache row {row} for model layer {layer_id} "
                f"exceeds {allocated_rows} allocated rows"
            )
            entry = mla_kv_entry_dim(hf)
            kv_cache = runner.kv_cache[row].view(-1, 1, entry)
            module.max_model_len = runner.config.max_model_len
            if runner.is_deepseek_v32 and getattr(module, "indexer", None) is not None:
                if layer_id not in runner.index_cache_layer_map:
                    raise RuntimeError(
                        "Sparse MLA indexer layer is missing from the compact "
                        f"index cache layout: layer_num={layer_id}"
                    )
                index_cache = runner.index_cache[runner.index_cache_layer_map[layer_id]]
                # Flat row view: `indexer_k_quant_and_cache` addresses a
                # slot as a single row id, and the pooled writer computes that
                # id from the block table itself.
                module.indexer.k_cache.kv_cache[0] = index_cache.view(
                    index_cache.shape[0] * index_cache.shape[1],
                    1,
                    runner.aligned_index_dim,
                )
                # kpool: this layer's slice of the per-request tail buffer,
                # bound here for the same reason the index cache is -- the
                # indexer has no other route to a runner-owned tensor.
                tail = getattr(runner, "kpool_tail_cache", None)
                if tail is not None:
                    module.indexer.kpool_tail_cache = tail[
                        runner.index_cache_layer_map[layer_id]
                    ]
            module.kv_cache = kv_cache
            return KVCacheTensor(
                layer_num=layer_id,
                k_cache=kv_cache,
                v_cache=None,
                k_scale=None,
                v_scale=None,
            )

        return None

    def prepare_prefill(self, batch: ScheduledBatch):
        attn_metadata, positions = super().prepare_prefill(batch)
        if batch.block_tables == []:
            attn_metadata.gdn_metadata = None
            return attn_metadata, positions
        attn_metadata.gdn_metadata = self.prepare_gdn_metadata(
            batch,
            attn_metadata,
            is_prefill=True,
            prepare_block_tables=False,
        )
        return attn_metadata, positions

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        attn_metadata, positions = super().prepare_decode(batch, bs)
        self._attach_gdn_decode_metadata(
            batch,
            attn_metadata,
            prepare_block_tables=False,
        )
        return attn_metadata, positions

    def build_for_cudagraph_capture(self, bs: int):
        if self.block_size == 1:
            var = self.model_runner.forward_vars
            var["kv_indptr"].np[: bs + 1] = np.arange(bs + 1, dtype=np.int32)
            var["kv_indptr"].copy_to_gpu(bs + 1)
            var["kv_indices"].gpu[:bs].zero_()
            var["kv_last_page_lens"].gpu[:bs].fill_(1)

        attn_metadata, context = super().build_for_cudagraph_capture(bs)
        attn_metadata.gdn_metadata = self._build_gdn_capture_metadata(bs)
        return attn_metadata, context


class KimiAiterMLAGDNMetadataBuilder(_KimiMLAGDNCommon, AiterMLAMetadataBuilder):
    pass


class KimiTritonMLAGDNMetadataBuilder(_KimiMLAGDNCommon, TritonMLAMetadataBuilder):
    pass
