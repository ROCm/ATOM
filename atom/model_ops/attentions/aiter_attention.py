# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import ClassVar, NamedTuple

import aiter
import numpy as np
import torch
import triton
import triton.language as tl

from atom.config import _is_minimax_m3_config
from atom.distributed.ulysses_sp import local_num_kv_heads
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mha import PagedAttentionImpl, use_pa_decode_bf16_asm
from atom.model_ops.minimax_m3.sparse_attn import (
    make_sparse_decode_metadata,
    make_sparse_prefill_metadata,
)
from atom.utils import CpuGpuBuffer, envs
from atom.utils.block_convert import (
    block_table_convert_triton,
    kv_indices_generate_triton,
)
from atom.utils.block_tables import block_table_state
from atom.utils.forward_context import AttentionMetaData, Context, get_forward_context
from atom.utils.tbo import UBatchSlice

from .backends import (
    PERSISTENT_WORK_BUFFERS,
    AttentionBackend,
    CommonAttentionBuilder,
)
from .mha_kv_pool import MhaKvPool
from .pool_layout.entry_arena import EntryField, carve
from .pool_layout.pool_rows import KvGeometry
from .pool_layout.sub_pool_spec import SubPoolSpec, page_pool
from .token_layout.batch_ids import build_cu_seqlens
from .token_layout.decode import (
    decode_positions,
    query_is_uniform,
    ragged_decode_refusal,
)
from .token_layout.slots import slot_mapping

logger = logging.getLogger("atom")


def cdiv(a, b):
    return (a + b - 1) // b


class _IndexCacheSpec(NamedTuple):
    """MiniMax-M3's indexer key cache: one `dim`-wide key per token, on `rows`
    of this stage's layers. The pool is told its bytes and nothing else."""

    rows: int
    dim: int
    dtype: torch.dtype

    def __str__(self) -> str:
        """How a row space names itself in a log line."""
        return f"kv[{self.num_kv_heads}x{self.head_dim}]"


# The indexer keys' row space: a second numbering over the same modules, not a
# pool of its own.
INDEX_ROWS = "index"


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
        f"index_cache_dtype must be 'bf16' or 'fp8', got {index_cache_dtype!r}"
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
    scheduled_bs,
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

        # Rows past scheduled_bs write no KV, so read no block table either.
        scheduled = seq < scheduled_bs
        phys_block = tl.load(
            block_tables_ptr + seq * block_table_stride + block_col,
            mask=mask & scheduled,
            other=0,
        ).to(tl.int64)
        slot = tl.where(scheduled, phys_block * block_size + within_block, -1)
        tl.store(slot_mapping_ptr + seq, slot, mask=mask)


class AiterBackend(AttentionBackend):
    # An MHA draft keeps its own K and V; there is no target tensor its rows
    # could be.
    DRAFT_OWNS_KV_POOL: ClassVar[bool] = True

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
    def attn_block_size(hf_config, scheduler_block_size: int) -> int:
        """Tokens in the block this backend's kernels index.

        Two different things, tied only by the scheduler's being a whole number
        of these: the scheduler's is what prefix caching and the block manager
        work in, one for the whole process, while this one is a kernel detail
        each backend picks for itself. A target's, only: a draft takes the
        target builder's answer instead, for the reason in
        `DraftKvBuilder.kv_pool`.
        """
        if envs.ATOM_USE_UNIFIED_ATTN:
            # SHUFFLE cache read straight through, so the two coincide and
            # `unified_attention`'s block table needs no conversion.
            return scheduler_block_size
        # MiniMax-M3's sparse kernels index the block its config names.
        text_config = getattr(hf_config, "text_config", hf_config)
        sparse_cfg = getattr(text_config, "sparse_attention_config", None)
        sparse_block_size = sparse_cfg.get("sparse_block_size") if sparse_cfg else None
        if sparse_block_size and _is_minimax_m3_config(hf_config):
            return sparse_block_size
        return scheduler_block_size if scheduler_block_size in (256, 1024) else 16

    @staticmethod
    def make_kv_pool(
        hf_config,
        *,
        world_size: int,
        target_block_size: int,
        layers: int,
        kv_dtype,
    ):
        """An MHA pool, which is the one this backend's kernels read."""
        return MhaKvPool.from_hf_config(
            hf_config,
            world_size=world_size,
            block_size=target_block_size,
            layers=layers,
            kv_dtype=kv_dtype,
        )


class AiterAttentionMetadataBuilder(CommonAttentionBuilder):
    # Whose kernels this builder feeds. Read for `attn_block_size` below, so a
    # subclass that names another backend gets that backend's rule -- naming
    # this one outright would size the pool at one block and index it at
    # another the day the two stop agreeing.
    BACKEND: ClassVar[type[AiterBackend]] = AiterBackend
    BLOCK_TABLE_EXTENDER: ClassVar[list[list[int]]] = [[]]
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
        self._has_sparse_attention = bool(sparse_cfg) and _is_minimax_m3_config(
            hf_config
        )
        # The backend's own, from the one rule `make_kv_pool` also reads, so a
        # draft resolving here gets it from its own config the same way.
        # `CommonAttentionBuilder` holds it to dividing the scheduler's.
        self.block_size = type(self).BACKEND.attn_block_size(
            hf_config, model_runner.block_size
        )
        if envs.ATOM_USE_UNIFIED_ATTN:
            # The two coincide there, and only one scheduler block reads well:
            # fp8 packs x=16 -> 128, bf16 x=8 -> 64, both a 128 B physical page.
            expected = 128 if model_runner.kv_cache_dtype in ("fp8",) else 64
            if model_runner.block_size != expected:
                logger.warning(
                    "ATOM_USE_UNIFIED_ATTN=1 expects --block-size %s for %s KV "
                    "cache, got --block-size %s. Continuing with the requested "
                    "block size.",
                    expected,
                    model_runner.kv_cache_dtype,
                    model_runner.block_size,
                )
        super().__init__(model_runner)
        # Keyed by `(num_kv_heads, head_dim)` and filled by
        # `allocate_kv_cache_tensors`, which runs long after construction --
        # on the P/D decode side too, over the imported pool.
        self.kv_pools: dict[tuple[int, int], MhaKvPool] = {}
        config = model_runner.config
        hf_config = config.hf_config
        # `self.num_attention_heads` set by CommonAttentionBuilder.__init__.
        # For speculative decode (MTP), max_qlen = num_speculative_tokens + 1
        if (
            config.speculative_config is not None
            and config.speculative_config.num_speculative_tokens is not None
        ):
            max_qlen = config.speculative_config.num_speculative_tokens + 1
        else:
            max_qlen = 1

        num_head_k = local_num_kv_heads(hf_config.num_key_value_heads)
        # Kept flydsl work plan here and refreshed once per prepare_*.
        self._flydsl_kv_heads = num_head_k
        self._flydsl_plans: dict[tuple, object] = {}
        self._flydsl_plan_unplanned = False
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
        # MiniMax-M3's index selector reads one row per (index head, query
        # token); `num_head_k` is that head count, the same TP-local division
        # the sparse layers construct with.
        self._num_idx_heads = num_head_k if self._has_sparse_attention else 0
        if self._has_sparse_attention:
            # Decode only, and persistent because `make_sparse_decode_metadata`
            # refills it every step outside the graph while the captured
            # selector reads the pointer baked in at capture. Prefill is not
            # captured and allocates its own.
            self.model_runner.forward_vars[
                "sparse_attention_n_valid_column_per_row"
            ] = torch.empty(
                num_head_k * self.max_bs * max_qlen,
                **i32_kwargs,
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
            "kv_indptr": CpuGpuBuffer(
                self.max_bs + 1, publication_group="mha_csr", **i32_kwargs
            ),
            "kv_indices": CpuGpuBuffer(
                self.max_bs * self.max_num_blocks_per_seq,
                **i32_kwargs,
            ),
        }
        self.model_runner.forward_vars.update(pa_persistent_metadata)
        if model_runner.config.enable_tbo:
            self._allocate_ubatch_buffers(max_qlen)

    def h2d_group_members(self) -> dict[str, tuple[str, ...]]:
        # Ready together before the CSR kernel, sharing prefill destinations
        # without changing their addresses or padded publication counts.
        members = super().h2d_group_members()
        members["decode_metadata"] = members.get("decode_metadata", ()) + (
            "slot_mapping",
            "context_lens",
            "block_tables",
            "kv_indptr",
            "positions",
        )
        return members

    def _ubatch_buffer_shapes(self) -> dict[str, tuple[int, ...]]:
        var = self.model_runner.forward_vars
        shapes = super()._ubatch_buffer_shapes()
        if self._has_sparse_attention:
            # Both micro-batches' sparse metadata are alive at once.
            shapes["sparse_attention_n_valid_column_per_row"] = (
                self._num_idx_heads * self._ubatch_share(self.max_bs * var["max_qlen"]),
            )
            if self.block_ratio > 1:
                shapes["sparse_attention_block_tables"] = (
                    self.max_bs,
                    self.max_num_blocks_per_seq,
                )
        return shapes

    def set_aiter_persistent_worker_buffers(
        self, bs: int, max_qlen: int, prefix: str = ""
    ) -> dict[str, torch.Tensor]:
        """Plan the persistent PA kernel's work over the ``prefix`` buffers
        (``ub{i}_`` for a micro-batch); return the plan's buffers by name."""
        num_kv_heads = local_num_kv_heads(
            self.model_runner.config.hf_config.num_key_value_heads
        )
        var = self.model_runner.forward_vars
        work = {name: var[f"{prefix}{name}"] for name in PERSISTENT_WORK_BUFFERS}
        aiter.get_pa_metadata_v1(
            var[f"{prefix}cu_seqlens_q"].gpu[: bs + 1],
            var[f"{prefix}kv_indptr"].gpu[: bs + 1],
            var[f"{prefix}context_lens"].gpu[:bs],
            self.num_attention_heads // num_kv_heads,
            num_kv_heads,
            True,
            *work.values(),
            kv_granularity=max(self.block_size, 16),
            block_size=self.block_size,
            max_seqlen_qo=int(max_qlen),
            uni_seqlen_qo=max_qlen,
            fast_mode=True,
            max_split_per_batch=-1,
        )
        return work

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        """One paged KV pool, priced by the declaration it is built from."""
        return [page_pool(self._paged_entry_bytes())]

    def paged_pool_bytes(self, blocks: int) -> int:
        """The same declaration, times the blocks the budget bought."""
        return self._paged_entry_bytes() * blocks

    def _paged_entry_bytes(self) -> int:
        """What one scheduler block costs, over every pool this model needs."""
        return sum(pool.entry_bytes for pool in self._declare_kv_pools().values())

    def _module_kinds(self, module) -> tuple:
        """A KV row keyed by the module's own geometry, and an indexer row.

        Two row spaces over one module on MiniMax-M3: the indexer keys are a
        field of their own, so a sparse layer numbers separately there. The
        geometry is the key rather than a name because a hybrid can carry two
        head counts in one block-id space, and each needs its own pool.
        """
        kinds = []
        if _is_indexed_sparse_attention(module):
            kinds.append(INDEX_ROWS)
        if (
            hasattr(module, "base_attention")
            and hasattr(module, "use_mla")
            and not module.use_mla
        ):
            kinds.append(KvGeometry(module.num_kv_heads, module.head_dim))
        return tuple(kinds)

    def _declare_kv_pools(self) -> dict[KvGeometry, MhaKvPool]:
        """One pool per KV-head geometry, keyed by it, declared not allocated.

        A module says what it caches, so the pools are the rows the walk
        assigned per geometry: one pool for almost every model, and two for
        MiMo-V2-Flash, whose sliding-window layers carry a second head count --
        a second geometry sharing one block-id space, which is what sizing
        summed for it all along. Nothing here asks which model it is.
        """
        from aiter import dtypes

        runner = self.model_runner
        # `INDEX_ROWS` is a row space, not a pool: the indexer keys ride the
        # first geometry's pool, since declaring them per pool would charge for
        # the cache once per geometry. Insertion order is the walk's, so
        # "first" is the same pool every run.
        # One expression, handed to both: the field is priced per entry and the
        # pool is built per entry, and the two are a byte apart if they each
        # multiply it out.
        entry_tokens = self.block_size * self.block_ratio
        extra_fields = self._index_cache_fields(entry_tokens)
        pools = {}
        for geometry, count in self.row_counts().items():
            if not isinstance(geometry, KvGeometry):
                continue
            pools[geometry] = MhaKvPool(
                layers=count,
                # This backend's block, which is the only one its kernels
                # index: the block tables they read were expanded by
                # `block_ratio`, so their ids run over `blocks * ratio`. The
                # scheduler's block is `block_ratio` of ours and reaches the
                # pool only as the count `page_pool` charges per.
                block_size=self.block_size,
                blocks_per_entry=self.block_ratio,
                num_kv_heads=geometry.num_kv_heads,
                head_dim=geometry.head_dim,
                kv_dtype=dtypes.d_dtypes[runner.config.kv_cache_dtype],
                extra_fields=extra_fields,
            )
            extra_fields = ()
        return pools

    def _index_cache_spec(self) -> _IndexCacheSpec | None:
        """MiniMax-M3's indexer key cache, or None where there is none.

        Keyed off `_has_sparse_attention` -- the same answer the block size and
        the metadata buffers use -- not a second reading of the config. MiMo-V2
        declares two pools and sums them, so a second predicate would have to
        be kept from double-charging; this one is false for it by construction.
        """
        if not self._has_sparse_attention:
            return None
        runner = self.model_runner
        hf_config = runner.config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        sparse_cfg = text_config.sparse_attention_config
        return _IndexCacheSpec(
            # The walk, like every other count here. The config's
            # `sparse_attention_freq` says how many layers the *model* has an
            # indexer on; the rows are how many this stage holds, and on PP > 1
            # those differ -- the pool would be charged for the model's and
            # addressed by the stage's.
            rows=self.row_counts().get(INDEX_ROWS, 0),
            dim=sparse_cfg["sparse_index_dim"],
            dtype=_resolve_index_cache_dtype(runner.config),
        )

    def _index_cache_fields(self, entry_tokens: int) -> tuple[EntryField, ...]:
        """The indexer cache as a field of the KV entry, priced in bytes.

        Declared here and not in `MhaKvPool` because what it holds is this
        backend's subject; the pool is told how many bytes ride along, and
        gives them back through `field_view` in the shape `build_kv_cache_
        tensor` asks for.
        """
        spec = self._index_cache_spec()
        if spec is None:
            return ()
        return (
            EntryField(
                "index",
                spec.rows,
                (entry_tokens * spec.dim * spec.dtype.itemsize,),
                torch.uint8,
            ),
        )

    def release_kv_pools(self) -> None:
        for pool in self.kv_pools.values():
            pool.release()

    def _allocate_kv_pools(self, blocks: int, buf) -> None:
        """Back every declared pool, side by side in this builder's region.

        Same `carve` the runner placed that region with, so a model with two
        geometries is laid out by the rule one with a single geometry already
        was.
        """
        self.kv_pools = self._declare_kv_pools()
        pools = list(self.kv_pools.values())
        regions = carve(buf, [pool.pool_bytes(blocks) for pool in pools])
        for pool, region in zip(pools, regions):
            pool.allocate(blocks, self.model_runner.device, buf=region)
        self.num_blocks = blocks * self.block_ratio

    def allocate_kv_cache_tensors(self, *, blocks: int, buf) -> dict:
        """Allocate this model's MHA pools inside the runner's paged region.

        Nothing shaped goes back to the runner: the cache, the scales and the
        indexer keys were three named attributes and are three regions of the
        one buffer it holds. So the only reference that can outlive a rollout
        sleep is a pool's, and `MhaKvPool.release` is what drops it.
        """
        runner = self.model_runner
        hf_config = runner.config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        self._allocate_kv_pools(blocks, buf)
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

    def build_kv_cache_tensor(self, module):
        """Bind one MHA (non-MLA) attention module to its KV slice.

        Its row is the next one of its geometry -- a hybrid's linear layers and
        a draft's stack are simply not modules this builder owns, so nothing
        here has to know where they sit.

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
            # each sparse layer has its own row in that second space. The pool
            # was built at this backend's block, so its slice already is.
            runner = self.model_runner
            pool = next(p for p in self.kv_pools.values() if p.extra_fields)
            spec = self._index_cache_spec()
            module.impl.index_cache = pool.field_view(
                "index",
                self.pool_rows[INDEX_ROWS][module],
                spec.dtype,
                (-1, self.block_size, spec.dim),
            )
            module.impl.max_model_len = runner.config.max_model_len
            module.impl.index_topk_cache_state = getattr(
                runner, "_sparse_attention_topk_cache_state", None
            )
            # NOTE: no return — fall through to the standard MHA binding below.

        geometry = next(
            (k for k in self._module_kinds(module) if isinstance(k, KvGeometry)), None
        )
        if geometry is None:
            return None

        config = self.model_runner.config
        # The module's own geometry picks its pool, and the walk that sized
        # that pool says which of its rows this module got. Both are what the
        # module declared.
        pool = self.kv_pools[geometry]
        row = self.pool_rows[geometry][module]

        k_cache, v_cache = pool.kv_views(row)
        if config.kv_cache_dtype == "fp8":
            module.k_scale, module.v_scale = pool.scale_views(row)

        module.max_model_len = config.max_model_len
        module.k_cache = k_cache
        module.v_cache = v_cache
        return KVCacheTensor(
            layer_num=module.layer_num,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=module.k_scale,
            v_scale=module.v_scale,
        )

    def _n_valid_column_per_row_buffer(self, prefix: str = "") -> torch.Tensor:
        """The decode buffer `make_sparse_decode_metadata` refills each step.

        `prefix` selects the ubatch's, on the same `ub<i>_` convention as every
        other decode buffer here. Prefill passes no buffer at all: it is not
        captured, so a per-forward allocation is safe there and the two prefill
        ubatches then cannot collide.
        """
        return self.model_runner.forward_vars[
            f"{prefix}sparse_attention_n_valid_column_per_row"
        ]

    def _get_sparse_attention_block_tables(
        self,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        bs: int,
        prefix: str = "",
    ) -> torch.Tensor:
        """Return MiniMax-M3 sparse-kernel block tables, into the ``prefix``
        set's buffer.

        `block_tables` is produced by the scheduler at `model_runner.block_size`
        granularity. MiniMax-M3 sparse attention indexes blocks at
        `sparse_block_size` granularity, so when the scheduler block is larger
        we expand each scheduler page id into its logical sparse pages.
        """
        if self.block_ratio == 1:
            return block_tables
        sparse_block_tables = self.model_runner.forward_vars[
            f"{prefix}sparse_attention_block_tables"
        ].gpu[:bs]
        return block_table_convert_triton(
            block_tables,
            sparse_block_tables,
            seq_lens,
            self.block_ratio,
        )

    def get_kv_transfer_tensors(self):
        from atom.kv_transfer.disaggregation.page_region import page_region
        from atom.kv_transfer.disaggregation.types import KVTransferTensors

        if not self.kv_pools:
            return None
        return KVTransferTensors(
            # Every field of every pool, indexer cache included: a unit per
            # (pool, field, layer), in declared order. The geometry is in the
            # role because a hybrid declares two pools whose per-layer regions
            # are otherwise named alike.
            pages=[
                page_region(tensor, semantic_role=f"mha.{geometry}.{role}")
                for geometry, pool in self.kv_pools.items()
                for role, tensor in pool.region_tensors()
            ],
            # GQA/MQA KV heads are sharded or only partially replicated across
            # TP. In particular MiniMax-M3 must keep one stored shard per rank;
            # whole-object TP collapse is unsafe for this PAGE layout.
            tp_replication_factor=1,
        )

    def refresh_flydsl_plan(self, context_lens, *, create=False):
        """Build or refresh aiter's FlyDSL work plan for this forward.

        Depends only on context_lens, which every layer of one forward shares,
        so it is built here rather than per pa_decode call. Decode replays a
        captured graph, so the plan must be visible to the op at capture time
        (see build_for_cudagraph_capture) and its tensors must never be
        reallocated afterwards -- hence one entry per batch, kept forever, and
        refreshed in place before each replay. The refresh is a GPU kernel with
        no readback. None when FlyDSL is off, or when the batch is out of range.
        """
        # The plan only feeds FlyDSL; building one with FlyDSL off is a
        # refresh kernel per step that nothing reads.
        if not envs.ATOM_PA_FLYDSL:
            return None
        # From base_attention, not duplicated: the op checks the same bound.
        from aiter.ops.flydsl.pa_decode import plan_pa_decode

        from atom.model_ops.base_attention import (
            _FLYDSL_PLAN_MAX_BATCH,
            flydsl_plan_shape,
        )

        n = int(context_lens.shape[0])
        if not 1 <= n <= _FLYDSL_PLAN_MAX_BATCH:
            return None
        # plan_pa_decode rejects anything else. This builder serves several
        # models; one of them handing int64 lengths would raise on the first step.
        if (
            context_lens.dtype is not torch.int32
            or not context_lens.is_cuda
            or context_lens.ndim != 1
            or not context_lens.is_contiguous()
        ):
            return None

        # `max_partitions` is never the static split count: that clamps every
        # request alike and removes the planner's mechanism -- this tree did
        # that once. It takes plan_pa_decode's own default, except at head_dim
        # 256, where `flydsl_plan_shape` caps it at 64 with a 256-workgroup
        # budget to keep aiter's reducer off its slow serial path; remove that
        # once aiter #5949 is in the image. The shape is fixed per builder (its
        # KV pools are), so the key needs no part of it; the op's runtime plans
        # take the same shape from the call's own head size.
        key = (n, self._flydsl_kv_heads, context_lens.device.index)
        plan = self._flydsl_plans.get(key)
        if plan is None and not create:
            # This builder mints plans only during cudagraph capture, where the
            # batch is a ladder rung and the cost lands at startup. aiter's
            # planner takes batch as a tl.constexpr, so a new value is a kernel
            # specialization -- 65-72 ms cold -- and a plan that is never freed
            # because some captured graph may have baked its pointers in. A
            # runtime batch with no capture-owned plan gets a runtime plan in
            # run_pa_decode.
            # Only once a capture has happened: before the first one every
            # call lands here (profile run, eager warmup), and logging then
            # burns the one shot on a step that says nothing.
            if self._flydsl_plans and not self._flydsl_plan_unplanned:
                self._flydsl_plan_unplanned = True
                logger.info(
                    "flydsl: batch %d has no capture-owned plan; an explicit "
                    "runtime plan will be used; "
                    "captured batches are %s",
                    n,
                    sorted({k[0] for k in self._flydsl_plans}),
                )
            return None
        if plan is None:
            # One head size across this builder's pools, or aiter's defaults.
            head_dims = {geometry.head_dim for geometry in self.kv_pools}
            plan = plan_pa_decode(
                context_lens,
                self._flydsl_kv_heads,
                query_length=1,
                **flydsl_plan_shape(head_dims.pop() if len(head_dims) == 1 else None),
            )
            self._flydsl_plans[key] = plan
            logger.info(
                "flydsl work plan: num_seqs=%d kv_heads=%d max_partitions=%d "
                "capacity=%d",
                n,
                self._flydsl_kv_heads,
                int(plan.max_partitions),
                int(plan.capacity),
            )
        else:
            plan_pa_decode(
                context_lens,
                self._flydsl_kv_heads,
                query_length=1,
                plan=plan,
            )
        return plan

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
        # A warmup skips the draft attention, so keep this launch as a no-op:
        # its fabricated context_lens can point past block_tables.
        skip_update = get_forward_context().context.is_dummy_run
        _mtp_prepare_decode_metadata_kernel[(max(1, triton.cdiv(running_bs, 128)),)](
            context_lens,
            block_tables,
            slot_mapping,
            positions,
            positions_out,
            last_token_indices,
            running_bs,
            bs,
            skip_update,
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
            # `getattr`; the one `prepare_decode` built is the verify forward's,
            # with its query width and slot mapping, not this step's.
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
                num_idx_heads=self._num_idx_heads,
                n_valid_column_per_row_out=self._n_valid_column_per_row_buffer(),
            )
        # Same short-circuit as the metadata launch above. A dummy step would
        # refresh the live work_info from synthetic lengths, but every real
        # decode refreshes before it replays, so this is wasted work rather
        # than a wrong answer. Free here because skip_update is already
        # computed; the GDN and Qwen4 overrides would have to reach for the
        # forward context to save the same kernel.
        workinfos["flydsl_work_plan"] = (
            None if skip_update else self.refresh_flydsl_plan(context_lens[:running_bs])
        )
        return workinfos

    def _prefill_block_table_rows(self, batch, running_bs, has_cached):
        # Sparse attention and `prefill_attention_triton` read the paged KV
        # cache, so they need the block tables even with no cached tokens.
        if self._has_sparse_attention or (
            envs.ATOM_USE_UNIFIED_ATTN and batch.block_tables
        ):
            return running_bs
        return super()._prefill_block_table_rows(batch, running_bs, has_cached)

    def _finish_prefill_batch(self, batch, attn_metadata, prefix=""):
        super()._finish_prefill_batch(batch, attn_metadata, prefix)
        if self._has_sparse_attention:
            bs = batch.scheduled_bs
            sparse_block_tables = self._get_sparse_attention_block_tables(
                attn_metadata.block_tables[:bs],
                attn_metadata.context_lens[:bs],
                bs,
                prefix,
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
                num_prefill_tokens=batch.scheduled_tokens,
                num_idx_heads=self._num_idx_heads,
            )

    def _refuse_ragged_on_rectangular_layers(
        self, lengths: np.ndarray, max_seqlen_q: int
    ) -> None:
        """Refuse a ragged decode step if any MHA layer's decode kernel lays
        each sequence out ``max_seqlen_q`` rows apart
        (`PagedAttentionImpl.decode_reads_ragged_queries`): run, it would read
        the wrong rows."""
        sfc = self.model_runner.config.compilation_config.static_forward_context
        rectangular = [
            name
            for name, module in sfc.items()
            if isinstance(impl := getattr(module, "impl", None), PagedAttentionImpl)
            and not impl.decode_reads_ragged_queries(max_seqlen_q)
        ]
        if rectangular:
            raise ragged_decode_refusal(
                f"{rectangular[0]} (and {len(rectangular) - 1} more layers)",
                lengths,
                max_seqlen_q,
            )

    def stage_decode(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
        group,
    ):
        self._check_metadata_writable("mha_csr", "seq_metadata", "positions", "mrope")
        scheduled_bs = batch.total_seqs_num_decode
        self.total_blocks = 0
        dropout_p = 0.0
        min_seqlen_q = 0

        context_lens = np.asarray(batch.context_lens, dtype=np.int32)

        if max_seqlen_q > 1:
            context_lens = self.model_runner.tokenID_processor.verify_ctx_lens(
                context_lens
            )
        _, lengths, cu_seqlens_q = self.decode_spans(batch)
        if not query_is_uniform(lengths, max_seqlen_q):
            self._refuse_ragged_on_rectangular_layers(lengths, max_seqlen_q)
        positions = decode_positions(context_lens, lengths, cu_seqlens_q)
        max_seqlen_k = context_lens.max(initial=0)

        # Before the slots, not after: `slot_mapping` reads this packed table.
        self.prepare_block_tables(batch, running_bs)

        var = self.model_runner.forward_vars
        scheduled_tokens = batch.total_tokens_num_decode
        var["slot_mapping"].np[:scheduled_tokens] = slot_mapping(
            positions,
            lengths,
            var["block_tables"].np,
            self.model_runner.block_size,
            scratch=self.token_axis_scratch,
        )
        var["positions"].np[:scheduled_tokens] = positions
        self._pad_tokens(scheduled_tokens, running_tokens)
        var["context_lens"].np[:scheduled_bs] = context_lens
        var["context_lens"].np[scheduled_bs:running_bs] = 0

        # Prepare kv_indptr and kv_indices for persistent attention
        build_cu_seqlens(
            cdiv(context_lens, self.block_size), running_bs, out=var["kv_indptr"].np
        )

        vars_used = [
            ("slot_mapping", running_tokens),
            ("context_lens", running_bs),
            ("kv_indptr", running_bs + 1),
        ]

        counts = dict(vars_used)
        if not self.model_runner.use_mrope:
            counts["positions"] = running_tokens
        group.set_counts(counts)
        ctx = {el: var[el].gpu[:num] for el, num in vars_used}
        ctx["block_tables"] = block_table_state(var["block_tables"]).stage(
            running_bs, group
        )
        # A view: `prepare_query_metadata` already staged it this step, and
        # nothing here writes the host copy.
        ctx["cu_seqlens_q"] = var["cu_seqlens_q"].gpu[: running_bs + 1]
        ctx["kv_indices"] = var["kv_indices"].gpu
        attn_metadata = AttentionMetaData(
            dropout_p=dropout_p,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            **ctx,
        )
        mrope_positions = self._build_mrope_decode_positions(
            batch, context_lens, group, running_tokens=running_tokens
        )
        if mrope_positions is not None:
            positions = mrope_positions
        else:
            positions = var["positions"].gpu[:running_tokens]
        return attn_metadata, positions

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
        max_seqlen_k = attn_metadata.max_seqlen_k
        if self.block_size in (256, 1024):
            for name, value in self.set_aiter_persistent_worker_buffers(
                running_bs, self.model_runner.forward_vars["max_qlen"]
            ).items():
                setattr(attn_metadata, name, value)
        kv_indices_generate_triton(
            attn_metadata.block_tables,
            attn_metadata.kv_indices,
            attn_metadata.kv_indptr,
            self.block_ratio,
            max_seqlen_k,
        )
        attn_metadata.flydsl_work_plan = self.refresh_flydsl_plan(
            attn_metadata.context_lens
        )
        if self._has_sparse_attention:
            # Plain decode (q==1) and eagle3 spec-verify (q==num_spec+1) both run
            # the DECODE sparse path; the decode kernels handle q>1 per-token
            # causal internally via max_query_len.
            sparse_block_tables = self._get_sparse_attention_block_tables(
                attn_metadata.block_tables[:running_bs],
                attn_metadata.context_lens[:running_bs],
                running_bs,
            )
            attn_metadata.sparse_attention_metadata = make_sparse_decode_metadata(
                seq_lens=attn_metadata.context_lens[:running_bs],
                block_table=sparse_block_tables,
                slot_mapping=attn_metadata.slot_mapping,
                max_seq_len=int(max_seqlen_k),
                max_query_len=int(max_seqlen_q),
                num_idx_heads=self._num_idx_heads,
                n_valid_column_per_row_out=self._n_valid_column_per_row_buffer(),
            )

    def finish_ubatch(
        self,
        ubatch_idx: int,
        ub: UBatchSlice,
        ub_attn: AttentionMetaData,
        max_seqlen_q: int,
    ) -> None:
        super().finish_ubatch(ubatch_idx, ub, ub_attn, max_seqlen_q)
        running_bs = ub.running_bs
        if self.block_size in (256, 1024):
            self.set_aiter_persistent_worker_buffers(
                running_bs, max_seqlen_q, f"ub{ubatch_idx}_"
            )
        if self._has_sparse_attention:
            if max_seqlen_q > 1:
                raise NotImplementedError(
                    "MiniMax M3 sparse TBO decode with max_q_len > 1 is not "
                    "implemented."
                )
            sparse_block_tables = self._get_sparse_attention_block_tables(
                ub_attn.block_tables,
                ub_attn.context_lens,
                running_bs,
                f"ub{ubatch_idx}_",
            )
            ub_attn.sparse_attention_metadata = make_sparse_decode_metadata(
                seq_lens=ub_attn.context_lens,
                block_table=sparse_block_tables,
                slot_mapping=ub_attn.slot_mapping,
                max_seq_len=ub_attn.max_seqlen_k,
                num_idx_heads=self._num_idx_heads,
                n_valid_column_per_row_out=self._n_valid_column_per_row_buffer(
                    f"ub{ubatch_idx}_"
                ),
            )

    def build_for_cudagraph_capture(
        self, bs: int, *, decode_ubatch_slices=None
    ) -> AttentionMetaData:
        var = self.model_runner.forward_vars
        max_seqlen_k = self.model_runner.config.max_model_len
        max_q_len = int(var["max_qlen"])
        self.publish_capture_layout(bs, max_q_len)

        if self.block_size in (256, 1024):
            ctx_pa_ps = self.set_aiter_persistent_worker_buffers(bs, max_q_len)
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
            seq_lens = attn_metadata.context_lens
            # Both plain decode (q==1) and eagle3 spec-verify (q==num_spec+1) use
            # the DECODE sparse path; the decode kernels handle q>1 per-token causal
            # internally (max_query_len), so no separate prefill graph is needed.
            sparse_block_tables = self._get_sparse_attention_block_tables(
                attn_metadata.block_tables,
                seq_lens,
                bs,
            )
            # the buffer every replay's query metadata publishes into
            # (`prepare_query_metadata`); the capture's own staged by `ModelRunner`
            attn_metadata.batch_id_per_q_token = var["batch_id_per_q_token"].gpu[
                :scheduled_tokens
            ]
            attn_metadata.sparse_attention_metadata = make_sparse_decode_metadata(
                seq_lens=seq_lens,
                block_table=sparse_block_tables,
                slot_mapping=attn_metadata.slot_mapping,
                max_seq_len=attn_metadata.max_seqlen_k,
                max_query_len=max_q_len,
                num_idx_heads=self._num_idx_heads,
                n_valid_column_per_row_out=self._n_valid_column_per_row_buffer(),
            )

        positions = var["positions"].copy_to_gpu(scheduled_tokens)
        # Decode replays a captured graph, so the op must see a plan HERE
        attn_metadata.flydsl_work_plan = self.refresh_flydsl_plan(
            attn_metadata.context_lens, create=True
        )
        if decode_ubatch_slices:
            self.capture_ubatch_decode(
                attn_metadata, bs, decode_ubatch_slices, max_q_len
            )
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
