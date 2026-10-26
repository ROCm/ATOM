import logging
from functools import partial

from atom.config import KVCacheTensor
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
_DFLASH_POOL_ARCHS = frozenset({"K3DFlashModel"})


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


def uses_sglang_block_attn() -> bool:
    """Whether the GQA block read goes through SGLang's two-stage verify kernel.

    `verify_shared_kv_fwd` (vendored under `model_ops/attentions/sglang_port/`)
    splits the block's one attention into a PREFIX pass over the paged pool and
    an in-register L_EXT x L_EXT pass over the block's own freshly computed K/V,
    merged by LSE inside the kernel. That is the exact arrangement SGLang's
    `TritonAttnBackend.forward_extend` validated for this draft checkpoint, and
    it replaces the aiter `mha_v3.flash_attn_with_kvcache(causal=False)` read
    that attends prefix and block together out of the pool.

    NO DCP ARGUMENT, because the draft has no DCP. `verify_shared_kv_fwd`
    returns only `o` and never an LSE, so it cannot serve a rank holding a
    disjoint token shard -- and SGLang never asks it to: a draft runner there
    does not join the dcp group at all. A Qwen3.5 draft shards its KV at plain
    TP (`models/qwen3_5.py:1111-1113`, "Drafts are TP-sharded and do not
    replicate KV under DCP", `dcp_size = 1 if is_nextn else attn_dcp_size`) and
    DFlash's own attention never reads `attn_dcp_size` at all -- it shards q,
    k and v by `tp_size` flat (`models/dflash.py:146-171`). ATOM now mirrors
    that: the TARGET stays at DCP8, the draft runs at DCP1 over a pool whose
    entry is `block_size * dcp_size` tokens wide, and every rank holds the
    whole prefix for its own TP head slice.

    `ATOM_K3_DRAFT_SGLANG_ATTN=0` forces the aiter `mha_v3` read back on, so
    the two can be A/B'd against each other on one build. It is the SAME pool
    either way -- `fa_views` is entry-granular, which is the page the aiter
    read's `page_table` names.
    """
    from atom.utils import envs

    return envs.ATOM_K3_DRAFT_SGLANG_ATTN


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

    non_causal = uses_dflash_pool(draft_hf)
    kv_cache_dtype = model_runner.config.kv_cache_dtype

    if not non_causal:
        # Every other pool-owning draft (today: Eagle3-llama) keeps the geometry
        # and the fp8 scale planes it already had. Everything below this point
        # is specific to a non-causal block draft that sits OUTSIDE the target's
        # DCP group, and none of it is a no-op for a draft that does not.
        make_pool = partial(
            backend.make_kv_pool,
            draft_hf,
            world_size=model_runner.world_size,
            kv_dtype=dtypes.d_dtypes[kv_cache_dtype],
        )
        return DraftKvBuilder(model_runner, make_pool)

    # FULL TP, and a DCP group this pool is not in. SGLang's draft runner never
    # joins the dcp group -- `dcp_size = 1 if is_nextn else attn_dcp_size`
    # (`models/qwen3_5.py:1112`), and DFlash's attention shards q/k/v by
    # `tp_size` flat (`models/dflash.py:146-171`) -- so the draft's KV heads are
    # sharded normally and every rank holds the WHOLE token axis for its slice.
    # The model builds its QKV at the same `tp_full`; the two must be one number
    # or the pool and the projections describe different geometries over the same
    # bytes, with nothing to catch it (`reshape_and_cache_flash` takes the head
    # count from the key tensor and the stride from the cache, and stripes).
    #
    # Read off `model_runner.config` and NOT `get_dcp_world_size()`: this runs
    # from the proposer's `__init__`, where the global ATOM config may not
    # exist yet (`atom/distributed/dcp_utils.py`).
    tp_full = model_runner.config.tensor_parallel_size
    dcp_size = model_runner.config.decode_context_parallel_size
    assert (
        tp_full % dcp_size == 0
    ), f"draft KV pool needs TP ({tp_full}) divisible by DCP ({dcp_size})"
    # ONE TARGET BLOCK ID, `dcp_size` BLOCKS. The draft rides the target's block
    # ids (that is what `allocate_kv_cache_tensors` enforces), and under DCP a
    # target block id addresses `block_size` LOCAL tokens, i.e.
    # `block_size * dcp_size` GLOBAL positions -- the virtual block
    # `dcp_prefill_slot_mapping` is written in (`dcp_ops.py:1059`). A DCP1 draft
    # stores every one of those positions, so its entry has to be that wide or
    # the id space it is handed overruns the id space it owns.
    #
    # The per-rank byte count does not move: `dcp_size` times the tokens, and
    # `tp_full` rather than `tp_full // dcp_size` gives `dcp_size` times fewer
    # KV heads. `test_dflash_pool_footprint_is_dcp_invariant` pins it.
    if dcp_size > 1:
        logger.info(
            "Draft KV pool built at full TP %d with %d blocks per entry "
            "(tp=%d / target dcp=%d): the DRAFT runs at DCP1 -- KV heads are "
            "TP-sharded and every rank holds the whole token axis",
            tp_full,
            dcp_size,
            tp_full,
            dcp_size,
        )
    make_pool = partial(
        backend.make_kv_pool,
        draft_hf,
        world_size=tp_full,
        kv_dtype=dtypes.d_dtypes[kv_cache_dtype],
        blocks_per_entry=dcp_size,
        # PER-TENSOR, which is what makes the fp8 pool possible at all. The
        # draft's read is SGLang's `verify_shared_kv_fwd`, whose `k_scale` /
        # `v_scale` are python floats folded into `qk *= sm_scale * k_scale`
        # once per program; its write is `aiter.reshape_and_cache_flash`, whose
        # HIP kernel dereferences `*k_scale` once per launch. Neither can read
        # a per-(kv head, token) plane, and SGLang stores none -- it refuses
        # outright ("Only support per-tensor scaling factor for fp8 KV cache",
        # `srt/layers/quantization/kv_cache.py:76`). So the planes are not
        # declared rather than declared and ignored: `scale_views` then raises
        # the `KeyError` it promises if a binder ever asks for them.
        per_tensor_scales=True,
    )
    # A way to build it, not the pool: how many rows it has is the walk's to
    # say, and this runs from the proposer's `__init__` -- `build_drafter`
    # still running, `runner.drafter` not yet set, nothing to walk.
    return DraftKvBuilder(model_runner, make_pool, non_causal=non_causal)


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

    def __init__(self, model_runner, make_pool, non_causal: bool = False):
        self.model_runner = model_runner
        self._make_pool = make_pool
        # Decides the LAYOUT this pool is bound in, not its size. See
        # `uses_dflash_pool` and `MhaKvPool.fa_views`.
        self._non_causal = non_causal
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
            # The SCHEDULER's block, not the target builder's. The two agree for
            # a paged target, but an MLA target's builder pages at
            # ATOM_MLA_PAGE_SIZE -- 1 by default -- which is a kernel page, not
            # a block id. Sizing the draft at 1 contradicts the invariant
            # `allocate_kv_cache_tensors` enforces below (one entry per
            # scheduler block, indexed by the target's block tables, so all
            # three are one number), and an MHA draft on an MLA target cannot
            # allocate at all: MhaKvPool's SHUFFLE packing needs 16 to divide
            # the block size.
            self._kv_pool = self._make_pool(
                layers=self.row_counts()[DRAFT_KV_ROWS],
                target_block_size=self.model_runner.block_size,
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
        # The layout the DRAFT's reader needs, which is not always the target's.
        # A non-causal block drafter reads through mha_v3, and that kernel hard
        # unpacks four axes -- SHUFFLE's x-pack splits head_dim across two of
        # them, so it cannot be offered at all. Same bytes either way.
        k_cache, v_cache = (
            self.kv_pool.fa_views(idx)
            if self._non_causal
            else self.kv_pool.kv_views(idx)
        )
        module.max_model_len = runner.config.max_model_len
        if self._non_causal:
            # PYTHON FLOATS, not views, and never `scale_views`: a non-causal
            # draft's pool declares no scale planes (`per_tensor_scales`), so
            # asking for them is the `KeyError` that method promises. 1.0 is
            # SGLang's own default for an fp8 KV cache with nothing in the
            # checkpoint to load -- "If no scales were loaded ... use the
            # default value of 1.0" (`srt/layers/quantization/kv_cache.py`) --
            # and it is what both ends of this pool consume:
            # `reshape_and_cache_flash` divides by it on the write,
            # `verify_shared_kv_fwd` multiplies by it on the read. Set for bf16
            # too, so the two ends never have to ask the dtype what kind of
            # scale they hold.
            #
            # Gated, because a causal draft (Eagle3-llama) DOES own per-token
            # scale planes and must keep reading them.
            module.k_scale = module.v_scale = 1.0
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
