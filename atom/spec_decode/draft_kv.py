import logging

from atom.config import KVCacheTensor
from atom.model_ops.attentions.pool_layout.sub_pool_spec import SubPoolSpec, page_pool

logger = logging.getLogger("atom")


def draft_kv_builder(model_runner, draft_hf):
    """The KV builder a draft needs of its own, or None if it needs none.

    A draft is not an attention flavor; it is a model that has one. So the
    flavor comes from the draft's own config through the same selector the
    target uses, and that backend says whether a draft of its flavor wants a
    pool (`make_kv_pool`). None means it shares the target's, and the runner
    never sees a draft builder at all.

    Which is why speculative decoding imports no attention anywhere: adding a
    flavor is one `make_kv_pool` on that backend.
    """
    from aiter import dtypes

    from atom.utils.selector import get_attn_backend

    backend = get_attn_backend(
        model_runner.block_size,
        use_mla=bool(getattr(draft_hf, "kv_lora_rank", None)),
    )
    pool = backend.make_kv_pool(
        draft_hf,
        world_size=model_runner.world_size,
        block_size=model_runner.block_size,
        kv_dtype=dtypes.d_dtypes[model_runner.config.kv_cache_dtype],
    )
    return None if pool is None else DraftKvBuilder(model_runner, pool)


class DraftKvBuilder:
    """A draft model's own KV pool, riding the target model's block ids.

    Implements the subset of `AttentionMetadataBuilder` hooks ModelRunner
    consults for sizing and per-module binding, so a draft that cannot share
    the target's pool fits the builder protocol without leaking into the
    target's builder. It does NOT drive prepare_decode/prepare_prefill; it
    piggybacks on the target builder's metadata flow during propose.

    It knows nothing about attention: the caller that resolved the draft's
    flavor hands the pool in. What is left is the part that is genuinely about
    being a draft -- its blocks are the target's blocks, re-paged at its own
    block size.
    """

    def __init__(self, model_runner, kv_pool):
        self.model_runner = model_runner
        self.kv_pool = kv_pool
        self.block_size = kv_pool.block_size
        self._next_layer_id = 0  # consumed by build_kv_cache_tensor
        # Same name and unit as a real builder's, since this one also answers
        # the runner's allocate hook. No `block_ratio` factor: the assertion
        # below holds the draft's page to the scheduler's.
        self.num_blocks = 0  # set in allocate_kv_cache_tensors

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        """`page_pool` puts the draft in the target's entry class, so the two
        contributions sum into one per-block cost instead of a second pool."""
        return [page_pool(self.kv_pool.entry_bytes)]

    def paged_pool_bytes(self, blocks: int) -> int:
        """The draft's share of the runner's one paged allocation.

        The same `entry_bytes` `sub_pool_specs` adds to the target's, so the
        draft's region is exactly what the block budget already charged for
        it -- which is what "riding the target's block ids" costs.
        """
        return self.kv_pool.pool_bytes(blocks)

    def allocate_kv_cache_tensors(
        self, num_kv_heads, num_draft_layers, *, blocks: int, buf
    ) -> dict:
        """Back the draft's pool from its region. Nothing for the runner to
        setattr: the pool is this builder's, and its hooks below are the only
        readers.

        One entry per scheduler block, the count the target was built at --
        that is what riding the target's block ids means. The assertion is the
        other half: a pool paging at anything else would be charged per
        scheduler block and built per its own page.
        """
        runner = self.model_runner
        assert self.block_size == runner.block_size, (
            f"a draft pool has to page at the scheduler block to share its "
            f"ids: pool {self.block_size} vs scheduler {runner.block_size}"
        )
        self.num_blocks = blocks
        self.kv_pool.allocate(blocks, runner.device, buf=buf)
        logger.info(
            f"Allocated draft KV pool: {blocks} blocks, "
            f"{self.kv_pool.pool_bytes(blocks)} B of the paged allocation"
        )
        return {}

    def adopt_imported_kv_pool(self, blocks: int, buf) -> None:
        """Same declaration over the region of an imported pool.

        The draft rides the target's blocks, so its bytes travel inside the
        same handle; what makes them findable is that both sides carve with
        the same `paged_pool_bytes` walk.
        """
        self.num_blocks = blocks
        self.kv_pool.allocate(blocks, self.model_runner.device, buf=buf)

    def build_kv_cache_tensor(self, layer_id: int, module):
        """Bind one of the draft's attention modules to its own pool.

        Returns None for anything this pool does not hold, so ModelRunner falls
        through to the target builder.
        """
        if not (hasattr(module, "base_attention") and hasattr(module, "use_mla")):
            return None
        if module.use_mla:
            return None
        runner = self.model_runner
        idx = self._next_layer_id
        self._next_layer_id += 1
        k_cache, v_cache = self.kv_pool.kv_views(idx)
        module.max_model_len = runner.config.max_model_len
        if runner.config.kv_cache_dtype == "fp8":
            module.k_scale, module.v_scale = self.kv_pool.scale_views(idx)
        module.k_cache = k_cache
        module.v_cache = v_cache
        return KVCacheTensor(
            layer_num=layer_id,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=getattr(module, "k_scale", None),
            v_scale=getattr(module, "v_scale", None),
        )

    def get_kv_transfer_tensors(self) -> list:
        from atom.kv_transfer.disaggregation.types import KVTransferRegion

        return [
            KVTransferRegion(
                base_addr=t.data_ptr(),
                total_bytes=t.numel() * t.element_size(),
                unit_bytes=t.stride(0) * t.element_size(),
            )
            for t in self.kv_pool.region_tensors()
        ]
