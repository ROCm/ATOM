# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from aiter import dtypes

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.state_pool import StateTransfer
from atom.model_ops.attention_mla import MLAAttention
from atom.utils import envs

from .aiter_mla import AiterMLAMetadataBuilder
from .backends import AttentionBackend
from .gdn_attn import GDNStateMixin
from .sub_pool_spec import SubPoolSpec, page_pool
from .triton_mla import TritonMLAMetadataBuilder

if TYPE_CHECKING:
    from atom.model_engine.superblock_geometry import SuperblockGeometry

logger = logging.getLogger("atom")


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
    #: K3 carves its KDA state from the same superblocks the MLA blocks come
    #: from, so the slot pool may grow to one slot per superblock rather than
    #: stopping at the count `plan_pools` reserved. `_unified_geometry`
    #: decides per run whether that path is taken; this states that it can be.
    uses_unified_pool = True

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

    def _unified_geometry(self) -> "SuperblockGeometry | None":
        """The unified pool's arithmetic, or None to keep two separate tensors.

        Built only when the runner derived a superblock size from the sizing
        plan (`model_runner.py`, `ceil(slot_bytes / block_bytes)`). Returning
        None leaves `allocate_kv_cache_tensors` and `allocate_per_req_cache`
        exactly as they were, which is what every non-K3 model gets.
        """
        runner = self.model_runner
        if not int(getattr(runner.config, "blocks_per_superblock", 0) or 0):
            return None
        if getattr(self, "_geometry", None) is not None:
            return self._geometry

        from atom.model_engine.superblock_geometry import (
            SuperblockGeometry,
            plan_state_fields,
        )

        config = runner.config
        hf = config.hf_config
        entry = hf.kv_lora_rank + hf.qk_rope_head_dim
        kv_size = dtypes.d_dtypes[config.kv_cache_dtype].itemsize
        block_bytes = self._num_cache_rows() * runner.block_size * entry * kv_size

        shape_k, shape_v = self._state_shape_for_runner()
        dt_k, dt_v = self._state_dtypes()
        # v first: it is the wider dtype (fp32 for kimi_linear against a bf16
        # k side), so anchoring it at the field start keeps its stride whole
        # without padding the narrower one.
        per_layer = [(shape_v, dt_v.itemsize), (shape_k, dt_k.itemsize)]
        fields = plan_state_fields(per_layer * runner.num_gdn_attn_state)

        # The pool spans BOTH pools' bytes, not just the paged one.
        # `plan_pools` reserves the STATE floor first and gives PAGE the
        # remainder, so `num_physical_kvcache_blocks` already excludes the
        # state; sizing from it alone would leave every state view carved out
        # of bytes a live block is using. Sum them back and re-divide.
        plan = runner.pool_plan
        state_class = self.state_spec().name
        # `block_bytes` above is one LOGICAL block -- `runner.block_size`
        # tokens across every MLA layer, the unit `sub_pool_specs` priced and
        # `plan_pools` counted. `num_physical_kvcache_blocks` counts something
        # else: K3's `block_ratio` is 128, so a physical block is one token and
        # there are 128x as many. Multiplying the two mixes the units and
        # oversizes the pool by exactly that ratio -- 4234 GiB instead of 33.
        #
        # Take the byte total from the plan, which states it in bytes and so
        # cannot be read at the wrong granularity.
        kv_bytes = int(plan.reserved_bytes.get(plan.paged_class or "", 0))
        state_bytes = int(plan.reserved_bytes.get(state_class, 0))
        num_supers = (kv_bytes + state_bytes) // (
            int(config.blocks_per_superblock) * block_bytes
        )
        logger.info(
            "[Unified pool] sizing: paged %.2f GiB + state %.2f GiB, "
            "bps=%d -> %d supers",
            kv_bytes / (1 << 30),
            state_bytes / (1 << 30),
            int(config.blocks_per_superblock),
            num_supers,
        )

        self._geometry = SuperblockGeometry(
            block_bytes=block_bytes,
            state_fields=fields,
            num_supers=num_supers,
        )
        self._state_dt = (dt_k, dt_v)
        return self._geometry

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict:
        del num_kv_heads, num_draft_layers
        runner = self.model_runner
        config = runner.config
        hf = config.hf_config
        num_layers = self._num_cache_rows()
        entry = hf.kv_lora_rank + hf.qk_rope_head_dim

        geo = self._unified_geometry()
        if geo is not None:
            # One flat pool. Blocks and slots are two readings of the same
            # bytes, so neither is sized at the other's expense and the split
            # between them is a host-side count rather than a tensor shape.
            # Logged BEFORE the allocation, not after: an oversized pool dies
            # inside `torch.zeros` with a byte count and no way to see which
            # term produced it, and the line that would have said so never
            # runs.
            logger.info(
                "[Unified pool] %d superblocks = %.2f GiB, %s",
                geo.num_supers,
                geo.total_bytes / (1 << 30),
                geo.describe(),
            )
            plan = runner.pool_plan
            budget = int(plan.reserved_bytes.get(plan.paged_class or "", 0)) + int(
                plan.reserved_bytes.get(self.state_spec().name, 0)
            )
            assert geo.total_bytes <= budget, (
                f"unified pool wants {geo.total_bytes / (1 << 30):.1f} GiB but "
                f"the plan reserved {budget / (1 << 30):.1f} GiB "
                f"(paged + state); num_supers={geo.num_supers}, "
                f"super_bytes={geo.super_bytes}"
            )
            self._pool = torch.zeros(
                geo.num_supers, geo.super_bytes, dtype=torch.uint8, device="cuda"
            )
            kv_dtype = dtypes.d_dtypes[config.kv_cache_dtype]
            # The MLA view keeps the shape every reader downstream expects --
            # `(rows, physical_blocks, physical_block_size, entry)`, which
            # `runner.kv_cache[row].view(-1, 1, entry)` below then flattens.
            #
            # `geo` counts LOGICAL blocks: `runner.block_size` tokens across
            # every MLA layer, the unit the pool was priced and planned in.
            # The tensor is indexed in PHYSICAL blocks, and K3's `block_ratio`
            # is 128 -- a physical block is one token. Converting here rather
            # than teaching the geometry about both keeps one unit per object.
            phys_blocks = geo.num_blocks * self.block_ratio
            kv = self._pool.view(-1)[: geo.num_blocks * geo.block_bytes].view(kv_dtype)
            return {
                "kv_cache": kv.view(
                    phys_blocks, num_layers, runner.physical_block_size, entry
                ).permute(1, 0, 2, 3)
            }

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

    def allocate_per_req_cache(self, entries: dict) -> dict:
        """Slot-major state views into the unified pool, or GDN's own tensors.

        With a unified pool there is no second allocation: each KDA layer gets
        an `as_strided` view whose slot stride is a whole superblock. Its shape
        is identical to `mamba_v_cache[layer]`'s, so `state[indices]` keeps
        working and no kernel signature changes -- Phase 0.1 measured every
        read and in-place write bit-exact through such a view.

        The stride is what makes it non-contiguous, and one kernel cares:
        aiter's KDA decode recomputes the slot stride from the shape and would
        read a neighbouring slot. `assert_reads_tensor_stride` states that.
        """
        geo = self._unified_geometry()
        if geo is None:
            return super().allocate_per_req_cache(entries)

        from atom.model_engine.superblock_geometry import assert_reads_tensor_stride
        from atom.model_ops.fla_ops.fused_sigmoid_gating import (
            fused_sigmoid_gating_delta_rule_update,
        )

        assert_reads_tensor_stride(fused_sigmoid_gating_delta_rule_update)

        dt_k, dt_v = self._state_dt
        n = self.model_runner.num_gdn_attn_state
        flat = self._pool.view(-1)

        # `plan_state_fields` was given (v, k) per layer, so layer L owns
        # fields 2L (v) and 2L+1 (k).
        layer_step = geo.uniform_layer_stride(first=0, per_layer=2)
        if layer_step < 0:
            raise ValueError(
                "KDA layers are not identically laid out inside a superblock, "
                "so no single layer stride addresses them all"
            )

        def _view(field_index: int, dtype: torch.dtype) -> torch.Tensor:
            """One `(num_layers, num_slots, *shape)` alias over the pool.

            Layer-major on the outside so `runner.mamba_k_cache[row]` still
            names a layer, slot-major inside so that row indexes by slot --
            the same shape today's dense tensor has. Built as a single
            `as_strided` rather than stacking per-layer views because
            `torch.stack` would copy, and these must alias the pool.
            """
            offset, slot_stride, shape = geo.state_view_params(field_index)
            if layer_step % dtype.itemsize:
                raise ValueError(
                    f"layer stride {layer_step} B is not a whole number of "
                    f"{dtype.itemsize}-byte elements"
                )
            inner, acc = [], 1
            for dim in reversed(shape):
                inner.append(acc)
                acc *= dim
            return torch.as_strided(
                flat.view(dtype),
                size=(n, geo.num_supers) + shape,
                stride=(layer_step // dtype.itemsize, slot_stride)
                + tuple(reversed(inner)),
                storage_offset=offset,
            )

        return {
            "mamba_k_cache": _view(1, dt_k),
            "mamba_v_cache": _view(0, dt_v),
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
