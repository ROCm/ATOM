import logging
from functools import partial

from atom.config import KVCacheTensor
from atom.model_ops.attentions.pool_layout.pool_rows import PoolRowsMixin
from atom.model_ops.attentions.pool_layout.sub_pool_spec import SubPoolSpec, page_pool

logger = logging.getLogger("atom")

# This pool's one row space. Named, not a geometry: `make_kv_pool` already
# settled the draft's geometry, and there is only ever the one.
DRAFT_KV_ROWS = "draft_kv"


def draft_kv_builder(model_runner, draft_hf):
    """The KV builder a draft needs of its own, or None if it needs none.

    A draft is not an attention flavor; it is a model that has one. So the
    flavor comes from the draft's own config through the same selector the
    target uses, and that backend says whether a draft of its flavor caches
    into a pool of its own (`DRAFT_OWNS_KV_POOL`). Where it does not, the
    runner never sees a draft builder at all.

    Which is why speculative decoding imports no attention anywhere: adding a
    flavor is a flag and a `make_kv_pool` on that backend.
    """
    from aiter import dtypes

    from atom.utils.selector import attn_family, get_attn_backend

    backend = get_attn_backend(attn_family(draft_hf))
    if not backend.DRAFT_OWNS_KV_POOL:
        return None
    make_pool = partial(
        backend.make_kv_pool,
        draft_hf,
        world_size=model_runner.world_size,
        kv_dtype=dtypes.d_dtypes[model_runner.config.kv_cache_dtype],
    )
    # A way to build it, not the pool: how many rows it has is the walk's to
    # say, and this runs from the proposer's `__init__` -- `build_drafter`
    # still running, `runner.drafter` not yet set, nothing to walk.
    return DraftKvBuilder(model_runner, make_pool)


class DraftKvBuilder(PoolRowsMixin):
    """A draft model's own KV pool, riding the target model's block ids.

    Implements the subset of `AttentionMetadataBuilder` hooks ModelRunner
    consults for sizing and per-module binding, so a draft that cannot share
    the target's pool fits the builder protocol without leaking into the
    target's builder. It does NOT drive prepare_decode/prepare_prefill; it
    piggybacks on the target builder's metadata flow during propose.

    It knows nothing about attention: the caller that resolved the draft's
    flavor hands over a way to build the pool. What is left is the part that is
    genuinely about being a draft -- its blocks are the target's blocks,
    re-paged at its own block size.
    """

    def __init__(self, model_runner, make_pool):
        self.model_runner = model_runner
        self._make_pool = make_pool
        self._kv_pool = None
        self.num_blocks = 0  # set in allocate_kv_cache_tensors

    @property
    def kv_pool(self):
        """The pool, built at the row count the walk found.

        On demand because the draft model does not exist at `__init__`; every
        reader is a sizing or binding step, which all run after it does. The
        count is the walk's and not `num_hidden_layers`, or the pool is sized
        off one number and addressed by another.

        Indexed and not `.get(..., 0)`: this builder exists because the flavor
        owns a pool, so no rows is a contradiction, and a pool of none prices
        at zero and binds nothing -- a draft running on no KV at all. The
        `KeyError` is the one `pool_rows` promises.

        On demand is also what makes the block reachable, and it is the target
        builder's: `propose` hands the draft the target's block tables, and the
        draft's kernels take the block off the cache they were bound to, so a
        block of its own would address one page with an id that counted
        another. There is nowhere in a draft to convert between the two -- it
        has no metadata of its own. The builder is constructed after
        `build_drafter`, which is why this is read here and not bound earlier.
        """
        if self._kv_pool is None:
            self._kv_pool = self._make_pool(
                layers=self.row_counts()[DRAFT_KV_ROWS],
                target_block_size=self.model_runner.attn_metadata_builder.block_size,
            )
        return self._kv_pool

    def invalidate_pool_rows(self) -> None:
        """Drop the pool with the walk that sized it, so the two cannot part."""
        super().invalidate_pool_rows()
        self._kv_pool = None

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

    def allocate_kv_cache_tensors(self, *, blocks: int, buf) -> dict:
        """Back the draft's pool from its region. Nothing for the runner to
        setattr: the pool is this builder's, and its hooks below are the only
        readers.

        One entry per scheduler block, the count the target was built at --
        that is what riding the target's block ids means.
        """
        runner = self.model_runner
        pool = self.kv_pool
        # `blocks` counts SCHEDULER blocks -- it is the budget `paged_pool_bytes`
        # charged per -- and one entry answers one of them, so the block the
        # pool took has to be that one too. It took the target builder's, which
        # makes this the check that the target's and the scheduler's agree.
        # Raised and not asserted because `python -O` drops what it does not
        # run, and a short pool reads the next request's page rather than fault.
        if pool.block_size != runner.block_size:
            raise ValueError(
                f"the draft's blocks are {pool.block_size} tokens and the "
                f"scheduler's {runner.block_size}; a draft is allocated one "
                "entry per scheduler block and indexed by the target's block "
                "tables, so all three are one number"
            )
        self.num_blocks = blocks
        pool.allocate(blocks, runner.device, buf=buf)
        logger.info(
            "Allocated draft KV pool: %d blocks, %d B of the paged allocation",
            blocks,
            pool.pool_bytes(blocks),
        )
        return {}

    def release_kv_pools(self) -> None:
        # The field, not the property: releasing one nobody built would build it.
        if self._kv_pool is not None:
            self._kv_pool.release()

    def _pooled_models(self) -> list:
        """The draft's, and only the draft's -- this pool exists precisely
        because that model could not share the target's."""
        return [self.model_runner.drafter.model]

    def _module_kinds(self, module) -> tuple:
        """One row space, the draft's own KV rows."""
        is_paged = (
            hasattr(module, "base_attention")
            and hasattr(module, "use_mla")
            and not module.use_mla
        )
        return (DRAFT_KV_ROWS,) if is_paged else ()

    def build_kv_cache_tensor(self, module):
        """Bind one of the draft's attention modules to its own pool.

        Returns None for anything this pool does not hold, so ModelRunner falls
        through to the target builder.
        """
        if DRAFT_KV_ROWS not in self._module_kinds(module):
            return None
        runner = self.model_runner
        idx = self.pool_rows[DRAFT_KV_ROWS][module]
        k_cache, v_cache = self.kv_pool.kv_views(idx)
        module.max_model_len = runner.config.max_model_len
        if runner.config.kv_cache_dtype == "fp8":
            module.k_scale, module.v_scale = self.kv_pool.scale_views(idx)
        module.k_cache = k_cache
        module.v_cache = v_cache
        return KVCacheTensor(
            layer_num=module.layer_num,
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
                # The draft's rows sit in the same block ids as the target's,
                # so its regions need a name that says which stack they are.
                semantic_role=f"draft.{role}",
            )
            for role, t in self.kv_pool.region_tensors()
        ]
