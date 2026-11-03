import logging
from functools import partial

import torch

from atom.config import K3_DFLASH_ARCH, KVCacheTensor
from atom.model_ops.attentions.pool_layout.pool_rows import PoolRowsMixin
from atom.model_ops.attentions.pool_layout.sub_pool_spec import SubPoolSpec, page_pool

logger = logging.getLogger("atom")

# This pool's one row space. Named, not a geometry: `make_kv_pool` already
# settled the draft's geometry, and there is only ever the one.
DRAFT_KV_ROWS = "draft_kv"


#: Drafts whose pool `draft_kv_builder` builds outside the target's DCP group,
#: entry-granular in FLASH layout, with one per-tensor scale. Keyed on the
#: architecture `SpeculativeConfig.hf_config_override` renames the SpecForge
#: DFlash checkpoint to.
_DFLASH_POOL_ARCHS = frozenset({K3_DFLASH_ARCH})


def uses_dflash_pool(draft_hf) -> bool:
    """Whether this draft owns the DFlash pool rather than the generic one.

    Asked of the config rather than the model because this runs from the
    proposer's `__init__`, before the draft model exists.

    Keyed on the architecture, NOT on "a non-MLA block drafter": MiniMax-M3.1's
    DSpark draft is one of those too, and it binds a 5-D SHUFFLE pool with
    per-token fp8 scales that its per-row decode reads. Handing it this pool's
    FLASH layout and 1.0 scales would be wrong bytes with no error. An MLA draft
    owns no pool at all -- it binds into the target's latent rows.
    """
    arch = (getattr(draft_hf, "architectures", None) or [""])[0]
    return arch in _DFLASH_POOL_ARCHS


def draft_ctx_slots(out, positions, cu_ends, block_tables, tok_ramp, vblock):
    """The DFlash draft's own slot for each target token, into ``out``.

        slot = block_tables[req, pos // vblock] * vblock + pos % vblock

    ``cu_ends`` is ``cu_seqlens_q[1 : rows + 1]``, the requests' exclusive
    token ends; ``tok_ramp`` is ``arange(n)`` in their dtype. All on the
    device, no host sync.
    """
    rows = cu_ends.shape[0]
    # `right=True` on the EXCLUSIVE ends: token t belongs to the first request
    # whose end is strictly past it. The clamp is for the one token that can
    # equal the last end; a gather out of bounds is a fault rather than a wrong
    # answer, so it is cheap insurance.
    req = torch.searchsorted(cu_ends, tok_ramp, right=True).clamp_(max=rows - 1)
    page = torch.div(positions, vblock, rounding_mode="floor")
    # A column the table does not have is DROPPED, never clamped into another
    # request's page (the `_block_page_idx` rule).
    in_table = page < block_tables.shape[1]
    page = page.clamp_(max=block_tables.shape[1] - 1)
    out.copy_(
        torch.where(
            in_table,
            block_tables[req, page].to(torch.int64) * vblock
            + torch.remainder(positions, vblock),
            -1,
        )
    )
    return out


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

    dflash_pool = uses_dflash_pool(draft_hf)
    # The DFlash draft runs outside the target's DCP group, so one target block
    # id spans `dcp` of its entries. Its scales are per-tensor because
    # `reshape_and_cache_flash` and `unified_attention` each take one scale.
    dflash_kwargs = (
        {
            "blocks_per_entry": model_runner.config.decode_context_parallel_size,
            "per_tensor_scales": True,
        }
        if dflash_pool
        else {}
    )
    make_pool = partial(
        backend.make_kv_pool,
        draft_hf,
        world_size=model_runner.world_size,
        kv_dtype=dtypes.d_dtypes[model_runner.config.kv_cache_dtype],
        **dflash_kwargs,
    )
    return DraftKvBuilder(model_runner, make_pool, dflash_pool=dflash_pool)


class DraftKvBuilder(PoolRowsMixin):
    """A draft model's own KV pool, riding the target model's block ids.

    Implements the subset of `AttentionMetadataBuilder` hooks ModelRunner
    consults for sizing and per-module binding, so a draft that cannot share
    the target's pool fits the builder protocol without leaking into the
    target's builder. It does NOT drive prepare_decode/prepare_prefill; it
    piggybacks on the target builder's metadata flow during propose.

    The caller that resolved the draft's flavor hands over a way to build the
    pool; `dflash_pool` picks the block, the views and the scale binding for
    the DFlash pool. What is left is the part that is genuinely about being a
    draft -- its blocks are the target's blocks.
    """

    def __init__(self, model_runner, make_pool, dflash_pool: bool = False):
        self.model_runner = model_runner
        self._make_pool = make_pool
        # The DFlash pool binds FLASH views (`MhaKvPool.fa_views`) and one
        # shared per-tensor scale; any other pool binds SHUFFLE views and its
        # own scale planes. See `uses_dflash_pool`.
        self._dflash_pool = dflash_pool
        # The DFlash pool's one per-tensor scale, shared by every layer: a
        # one-element fp32 tensor because both `reshape_and_cache_flash` and
        # `unified_attention` take it as a device pointer. Built at the first
        # bind, never in a forward.
        self._unit_scale = None
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
            layers = self.row_counts()[DRAFT_KV_ROWS]
            runner = self.model_runner
            # The DFlash pool sits beside an MLA target, whose builder pages at
            # ATOM_MLA_PAGE_SIZE (a kernel page, 1 by default), not at a block
            # id; it takes the SCHEDULER's block, the unit the target's block
            # tables count in.
            block_size = (
                runner.block_size
                if self._dflash_pool
                else runner.attn_metadata_builder.block_size
            )
            self._kv_pool = self._make_pool(
                layers=layers,
                target_block_size=block_size,
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
            "Allocated draft KV pool: %d blocks, %d B per block entry, "
            "%d B of the paged allocation",
            blocks,
            pool.entry_bytes,
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
        # The layout the DRAFT's reader needs, which is not always the target's.
        # The DFlash read takes the 4-D FLASH page at the entry's width, which
        # is `block_size * dcp` tokens; SHUFFLE would need a power-of-two page
        # and its x-pack splits head_dim across two axes. Same bytes either
        # way.
        k_cache, v_cache = (
            self.kv_pool.fa_views(idx)
            if self._dflash_pool
            else self.kv_pool.kv_views(idx)
        )
        module.max_model_len = runner.config.max_model_len
        if self._dflash_pool:
            # One per-tensor scale of 1.0: the pool declares no scale planes
            # (`per_tensor_scales`), and the DFlash checkpoint ships no KV
            # scales (all tensors BF16, no `quantization_config`), so 1.0 is
            # the value both ends must agree on, as SGLang defaults it.
            if self._unit_scale is None:
                self._unit_scale = torch.ones(
                    1, dtype=torch.float32, device=k_cache.device
                )
            module.k_scale = module.v_scale = self._unit_scale
        elif runner.config.kv_cache_dtype == "fp8":
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

    def get_kv_transfer_tensors(self):
        from atom.kv_transfer.disaggregation.page_region import page_region
        from atom.kv_transfer.disaggregation.types import KVTransferTensors

        return KVTransferTensors(
            # The draft's rows sit in the same block ids as the target's, so
            # its regions need a name that says which stack they are.
            pages=[
                page_region(t, semantic_role=f"draft.{role}")
                for role, t in self.kv_pool.region_tensors()
            ],
            # A paged MHA/GQA draft is a per-rank shard. Appending it to an
            # otherwise replicated target must disable whole-object collapse.
            tp_replication_factor=1,
        )
