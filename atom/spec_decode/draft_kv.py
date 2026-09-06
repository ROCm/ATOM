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
        self.num_blocks = 0  # set in allocate_kv_cache_tensors

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        """`page_pool` puts the draft in the target's entry class, so the two
        contributions sum into one per-block cost instead of a second pool."""
        return [page_pool(self.kv_pool.entry_bytes)]

    def allocate_kv_cache_tensors(self, num_kv_heads, num_draft_layers) -> dict:
        """Back the draft's pool. Nothing for the runner to setattr: the pool
        is this builder's, and its hooks below are the only readers."""
        runner = self.model_runner
        # Same total token capacity as the target pool, paged at the draft's
        # own block size.
        self.num_blocks = (
            runner.config.num_kvcache_blocks * runner.block_size // self.block_size
        )
        self.kv_pool.allocate(self.num_blocks, runner.device)
        logger.info(
            f"Allocated draft KV pool: {self.num_blocks} blocks, "
            f"{self.num_blocks * self.kv_pool.entry_bytes} B"
        )
        return {}

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
