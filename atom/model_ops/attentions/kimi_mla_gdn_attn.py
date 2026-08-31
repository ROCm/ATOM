# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Sequence

import numpy as np
import torch
from aiter import dtypes
from aiter.dist.parallel_state import get_tp_group

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

    def _uses_paged_checkpoints(self) -> bool:
        """Whether this run keeps checkpoints as PAGE images rather than slots.

        Off under pipeline parallelism and RapidServe, which `get_num_blocks`
        *raises* on when the transfer copies. Answering yes there would turn
        "K3 under PP keeps no state cache" into "K3 under PP does not start".

        One predicate for both `state_transfer` and `state_spec`, because the
        two disagreeing would size a pool for one mechanism and run the other.
        """
        config = self.model_runner.config
        return not (
            config.pipeline_parallel_size > 1
            or getattr(config, "enable_rapidserve", False)
        )

    def state_transfer(self) -> StateTransfer:
        """A PAGE-image copy, not `GDNStateMixin`'s fork.

        A KDA slot is 53.6 MiB, so a checkpoint held as a slot competes with
        live requests for the pool that admits them. Held as PAGE units it is
        127 ordinary KV blocks — 0.112% of the paged pool — drawn from the same
        free list as everything else, and evicted by the same LRU.

        Not midstep-readable, for want of plumbing rather than of state: KDA
        goes through aiter's `chunk_kimi_delta_attn`, whose returned tuple
        carries only the final state. The per-chunk `h` an interior checkpoint
        would be sliced out of *is* computed — by the same
        `chunk_gated_delta_rule_fwd_h` the GDN path uses — and then dropped
        before the return. Exposing it is what this backend would need to
        answer `True`. `PagedStateCheckpointCoordinator` says `False` for its
        own reasons, so the two agree meanwhile.

        Dtype-safe by construction, which a checkpoint cut from `h` would not
        be here — `_state_dtypes` gives kimi_linear an fp32 v side. An image is
        copied slot to slot with no kernel output in between, so that fp32 side
        round-trips exactly. Both dtypes are named in the layout id, so a build
        that changed either cannot read another's images.
        """
        if not self._uses_paged_checkpoints():
            return StateTransfer.fork(1)
        shape_k, shape_v = self._state_shape_for_runner()
        dt_k, dt_v = self._state_dtypes()
        # Everything a reader needs to reassemble the image at the same byte
        # offsets. `order` is the one thing the shapes cannot say, and getting
        # it wrong puts every layer but the first in the wrong place; `spec`
        # because the conv state is `(conv_kernel - 1 + num_spec, ...)`, so two
        # otherwise-identical builds disagree on the image's size; `carry`
        # is the narrowing rule, and dropping the conv tail later would be a
        # `v2` rather than a silent reinterpretation of a v1 image.
        layout_id = (
            "kda-paged-state-v1"
            f":layers={self.model_runner.num_gdn_attn_state}"
            f":conv={tuple(shape_k)},{dt_k}"
            f":ssm={tuple(shape_v)},{dt_v}"
            ":order=conv-all-layers,ssm-all-layers"
            f":tp={get_tp_group().world_size}"
            f":spec={self.num_spec}"
            ":carry=all"
        )
        return StateTransfer.copy(layout_id)

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

    def _page_unit_regions(self) -> tuple[np.ndarray, np.ndarray]:
        """Base address and per-unit stride of every region a PAGE id owns.

        The destination side of a checkpoint copy. `GDNStateMixin` knows where
        a state slot's bytes are; this knows where a KV block's are, because
        this class owns the MLA pool.

        `kv_cache` is `(rows, physical_blocks, physical_block_size, entry)`,
        so a block owns one contiguous region per row and the rows are a fixed
        stride apart. Affine in the block id, and a property of the pool rather
        than of any block, so it is worked out once.

        The units are the trap. `unit_ids` carries **logical** block ids -- what
        `BlockPool` hands out and what `sub_pool_specs` priced -- while the
        tensor is shaped in **physical** blocks, and K3's `block_ratio` is 128.
        So a region is `runner.block_size` tokens wide, not
        `physical_block_size`, and the two differ by exactly that ratio. The
        assertion below is what makes a mix-up a startup error rather than 127
        blocks of scrambled state: it is the one relation that cannot hold if
        the granularity is wrong.
        """
        runner = self.model_runner
        cache = runner.kv_cache
        owner = cache.data_ptr()
        cached = getattr(self, "_page_unit_region_cache", None)
        if cached is not None and cached[0] == owner:
            return cached[1]

        if not cache.is_contiguous():
            raise RuntimeError("the MLA pool must be contiguous to be copied")
        item = cache.element_size()
        entry = cache.shape[3]
        rows = cache.shape[0]
        # One logical block's bytes inside one row.
        region = runner.block_size * entry * item
        row_stride = cache.stride(0) * item

        runtime = getattr(runner, "state_runtime", None)
        spec = None if runtime is None else runtime.checkpoint_spec
        page_unit_bytes = spec.page_unit_bytes if spec is not None else rows * region
        if rows * region != page_unit_bytes:
            raise RuntimeError(
                f"a PAGE unit is {page_unit_bytes} B but this pool gives a "
                f"logical block {rows} rows x {region} B = {rows * region} B; "
                "the two disagree about block granularity"
            )
        base = np.array(
            [owner + row * row_stride for row in range(rows)], dtype=np.int64
        )
        regions = (base, np.full(rows, region, dtype=np.int64))
        self._page_unit_region_cache = (owner, regions)
        return regions

    def _page_unit_bases(self, unit_ids: Sequence[Sequence[int]]) -> np.ndarray:
        """Start address of every destination segment, one row per image.

        `unit_ids` is `(images, units_per_checkpoint)`. A unit's regions are
        each at `base + id * stride`, so one image's worth is an outer product
        and a batch's is the same product with an image axis in front. Unit
        major, region minor -- the order `_checkpoint_copy_plan` built the
        destination stream in.
        """
        base, stride = self._page_unit_regions()
        ids = np.asarray(unit_ids, dtype=np.int64)
        return (base + ids[..., None] * stride).reshape(len(ids), -1)

    def _page_unit_stream_sizes(self, units: int) -> np.ndarray:
        """Bytes in each destination segment of an image of `units` units."""
        return np.tile(self._page_unit_regions()[1], units)

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

    def prepare_prefill(self, batch: ScheduledBatch, running_bs: int):
        attn_metadata, positions = super().prepare_prefill(batch, running_bs)
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

    def prepare_decode(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
    ):
        attn_metadata, positions = super().prepare_decode(
            batch, running_bs, running_tokens, max_seqlen_q
        )
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
