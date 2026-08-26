# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import torch
from aiter import dtypes

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.state_runtime import StateTransfer
from atom.model_ops.attention_mla import MLAAttention
from atom.utils import envs

from .aiter_mla import AiterMLAMetadataBuilder
from .backends import AttentionBackend
from .gdn_attn import GDNStateMixin
from .sub_pool_spec import SubPoolSpec, page_pool
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

    def state_transfer(self) -> StateTransfer:
        """`GDNStateMixin`'s fork, minus its midstep claim.

        KDA does not run the chunk kernel this file's GDN sibling does: it
        calls `fla.ops.kda.chunk_kda`, which returns only the final state and
        never exposes the per-chunk states an interior checkpoint is sliced
        out of. `_checkpoint_targets` is also unreachable here, because
        `prepare_prefill` below overrides the one that calls it.

        Inheriting the mixin's answer would therefore be actively wrong rather
        than merely optimistic: `BlockManager` would stop cutting prefill
        chunks onto checkpoint positions and `checkpointers_at` would stop
        keeping them, so KDA would keep *zero* checkpoints — silently, since
        nothing on that path can tell a checkpoint that was skipped from one
        that was never wanted. Overridden here rather than fixed by making the
        mixin's flag conditional, so that the class which cannot do the thing
        is the class that says so.

        Two reasons, not one, and the second outlives the first. Porting the
        branch's `chunk_kda_paged` would expose per-chunk states here and make
        this override look removable — but KDA is also the single model whose
        state pool is *wider* than those states: `_state_dtypes` gives
        kimi_linear an fp32 v side while the chunked states are bf16. The GDN
        sibling's checkpoints are exact only because those two dtypes match
        (see `GDNStateMixin.state_transfer`), so here the same copy would
        silently hand cached requests a bf16-rounded state where uncached ones
        get fp32. Whoever ports the kernel has to answer that too.
        """
        return StateTransfer.fork(1)

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        """MLA paged KV for the full-attention layers, plus the KDA/GDN
        per-request state pool (`GDNStateMixin.state_spec`)."""
        runner = self.model_runner
        config = runner.config
        hf = config.hf_config
        entry = hf.kv_lora_rank + hf.qk_rope_head_dim
        kv_dtype_size = dtypes.d_dtypes[config.kv_cache_dtype].itemsize
        block_bytes = self._num_cache_rows() * runner.block_size * entry * kv_dtype_size
        return [page_pool(block_bytes), self.state_spec()]

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict:
        del num_kv_heads, num_draft_layers
        runner = self.model_runner
        config = runner.config
        hf = config.hf_config
        num_layers = self._num_cache_rows()
        entry = hf.kv_lora_rank + hf.qk_rope_head_dim
        return {
            "kv_cache": torch.zeros(
                num_layers,
                runner.num_physical_kvcache_blocks,
                runner.physical_block_size,
                entry,
                dtype=dtypes.d_dtypes[config.kv_cache_dtype],
                device="cuda",
            )
        }

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
                replay_buf_k=(runner.replayssm_buf_k[row] if self.replayssm else None),
                replay_buf_u=(runner.replayssm_buf_u[row] if self.replayssm else None),
                replay_buf_g=(runner.replayssm_buf_g[row] if self.replayssm else None),
                # KDA recurrent state: slot-addressed, not paged. Registered
                # because the forward reads it from `kv_cache_data`, but
                # excluded from every block-addressed transfer.
                per_request_state=True,
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
            entry = hf.kv_lora_rank + hf.qk_rope_head_dim
            kv_cache = runner.kv_cache[row].view(-1, 1, entry)
            module.max_model_len = runner.config.max_model_len
            module.kv_cache = kv_cache
            return KVCacheTensor(
                layer_num=layer_id,
                k_cache=kv_cache,
                v_cache=None,
                k_scale=None,
                v_scale=None,
            )

        return None

    def get_kv_transfer_tensors(self):
        """The MLA base's block regions, plus the KDA state pool it omits.

        This backend's cache is in two halves and the base class describes one:
        the full-attention layers sit in the paged pool it walks, the KDA
        layers in the slot-indexed recurrent-state pool `build_kv_cache_tensor`
        tags `per_request_state=True`. A request's KDA state is not derivable
        from its MLA blocks, so shipping only the base's answer would leave
        decode running most of the model on a zeroed state -- fluently, and
        with nothing downstream able to tell. The connector transfers whatever
        regions it is handed and reports success either way, which is why the
        two halves have to be joined here rather than checked later.

        Joined in this class rather than in `GDNStateMixin` because only a
        hybrid has a base answer to extend; the pure-GDN backends own no block
        regions. The slot half itself comes from the mixin, next to the other
        two methods that name the same bytes.
        """
        from atom.kv_transfer.disaggregation.factory import resolve_pd_backend

        tensors = super().get_kv_transfer_tensors()
        if tensors is None:
            return None

        # A non-empty `kv_transfer_config` alone does not mean disaggregation:
        # the aggregated LMCache offload tier configures one too, and it reads
        # the state pool through `state_backend` instead of through regions.
        connector = resolve_pd_backend(self.model_runner.config.kv_transfer_config)
        if connector is None:
            return tensors

        config = self.model_runner.config
        model_type = config.hf_config.model_type

        if connector != "mooncake":
            raise NotImplementedError(
                f"{model_type} disaggregated serving requires the mooncake "
                f"connector, but kv_connector={connector!r} is configured. Only "
                "mooncake transfers the slot-indexed regions the KDA recurrent "
                f"state lives in; every other connector would move the "
                f"{len(self.mla_idx_by_layer)} full-attention layers and "
                f"silently drop the state of the other "
                f"{len(self.kda_idx_by_layer)}. Set "
                'kv_transfer_config["kv_connector"] = "mooncake".'
            )

        if getattr(config, "pipeline_parallel_size", 1) > 1:
            # `_consumer_region_map` shifts a stage's regions onto the peer's
            # list group-major over `num_hidden_layers`. The slot regions below
            # are group-major over the KDA layers only, a shorter axis, so the
            # shift would land them on the wrong peer regions -- writing one
            # layer's state over another's rather than failing.
            raise NotImplementedError(
                f"{model_type} disaggregated serving does not support pipeline "
                f"parallelism (pipeline_parallel_size="
                f"{config.pipeline_parallel_size}): the KDA state regions are "
                "keyed by KDA layer index, which `_consumer_region_map` cannot "
                "align across stages. Run the P and D instances with PP=1."
            )

        tensors.slot_regions, tensors.num_slots = self.state_slot_regions()
        return tensors

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
