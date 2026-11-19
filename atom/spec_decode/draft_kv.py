import logging
import math
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import numpy as np
import torch

from atom.config import KVCacheTensor
from atom.model_engine.draft_window import DRAFT_WINDOW_CLASS, DraftWindowGeometry
from atom.model_ops.attentions.pool_layout.pool_rows import PoolRowsMixin
from atom.model_ops.attentions.pool_layout.sub_pool_spec import (
    SubPoolSpec,
    page_pool,
    state_pool,
)
from atom.utils import CpuGpuBuffer, envs

logger = logging.getLogger("atom")

# This pool's one row space. Named, not a geometry: `make_kv_pool` already
# settled the draft's geometry, and there is only ever the one.
DRAFT_KV_ROWS = "draft_kv"

# Blocks a forward's window table holds past `blocks_per_req`, for how far the
# scheduler runs ahead of the finalized tokens; a longer run is refused, not cut.
_WINDOW_TABLE_SLACK = 2


# The methods whose proposer calls `draft_kv_builder`; the others (MTP, ngram) run
# the target's own layers or have no draft model.
DRAFT_POOL_METHODS = ("eagle3", "dspark")


def draft_owns_kv_pool(config) -> bool:
    """Whether `config`'s speculative draft caches into a KV pool of its own.

    What `draft_kv_builder` will answer for the same draft, from the config alone so
    `Config` and the offload checks can ask before any runner exists: EAGLE3 / DSpark
    on a family in `DRAFT_POOL_FAMILIES` (not by resolving its backend, which would
    import aiter into the API process), native runner only.
    """
    from atom.plugin import is_plugin_mode

    # First: under a plugin the offload adapter hands over vLLM's own speculative
    # config unchanged, which has no `draft_model_hf_config` to read.
    if is_plugin_mode():
        return False
    spec = config.speculative_config
    if (
        spec is None
        or spec.method not in DRAFT_POOL_METHODS
        or spec.draft_model_hf_config is None
    ):
        return False
    from atom.utils.selector import DRAFT_POOL_FAMILIES, attn_family

    return attn_family(spec.draft_model_hf_config) in DRAFT_POOL_FAMILIES


@dataclass(frozen=True)
class DraftKvPoolSpec:
    """The pool a draft declares of its own, as the draft class's
    `kv_pool_spec`; None there is the generic SHUFFLE pool in the target's DCP
    group.

    Declaring one says the draft sits outside the DCP group, so every rank
    stores every position a target block id covers. An entry is then
    `block_size * dcp` tokens, which only the FLASH order reads, and the block
    is the scheduler's, the unit the target's block tables count in. The
    proposer's addressing and the pool's geometry both key on the declaration,
    so they cannot disagree.

    `kv_scale` is the one per-tensor scale K and V are stored at, which
    `DraftKvBuilder` places at bind. Only 1.0 is accepted: the pool has no
    scale planes and nothing loads a scale into it.
    """

    kv_scale: float = 1.0

    def pool_kwargs(self, dcp: int) -> dict:
        """What `make_kv_pool` takes beyond the geometry."""
        return {"blocks_per_entry": dcp, "flash_layout": True}

    def block_size(self, config) -> int:
        """The block the pool is built at, the scheduler's. The target's
        builder may page at a kernel page instead (MLA's ATOM_MLA_PAGE_SIZE),
        which a block id does not count in."""
        return config.kv_cache_block_size

    def entry_tokens(self, config) -> int:
        """The tokens one entry spans: the pool is built at it, and the draft's
        reader is built for it."""
        dcp = config.decode_context_parallel_size
        return self.block_size(config) * self.pool_kwargs(dcp)["blocks_per_entry"]

    def needs_own_ctx_slots(self, config) -> bool:
        """Whether the target's `slot_mapping` can't address this pool, so
        the target's builder stages the draft's context slots of its own.
        At DCP 1 an entry is one target block and no token is filtered, so
        the target's slots are the draft's, element for element."""
        return self.entry_tokens(config) != self.block_size(config)


class DraftRuns(NamedTuple):
    """Each request's run of held blocks in a windowed draft pool, which a block
    pass reads in place of the target's block tables."""

    tables: torch.Tensor  # [running_bs, W] the run's block ids, from `first` on
    first: torch.Tensor  # [running_bs] the block each run starts at
    extent: np.ndarray  # [running_bs] host: the block each run reaches


def draft_kv_dtype(config) -> torch.dtype:
    """The dtype a draft's own pool stores K and V in, the configured KV
    cache dtype: the pool is built in it, and the draft's reader is built for
    it."""
    from aiter import dtypes

    return dtypes.d_dtypes[config.kv_cache_dtype]


def draft_kv_builder(model_runner, draft_hf, model, *, spec_tokens: int = 0):
    """The KV builder a draft needs of its own, or None if it needs none.

    A draft is not an attention flavor; it is a model that has one. So the
    flavor comes from the draft's own config through the same selector the
    target uses, and that backend says whether a draft of its flavor caches
    into a pool of its own (`selector.DRAFT_POOL_FAMILIES`). Where it does not, the
    runner never sees a draft builder at all.

    Which is why speculative decoding imports no attention anywhere: adding a
    flavor is a table entry and a `make_kv_pool` on that backend.

    `model` is the built draft; its `kv_pool_spec` (None, or undeclared as on
    Eagle3, is the generic pool) adds to the pool's kwargs. Its `kv_read_window`
    sizes a windowed pool under `ATOM_DSPARK_DRAFT_KV_WINDOW`.
    """
    spec = getattr(model, "kv_pool_spec", None)
    if spec is not None and spec.kv_scale != 1.0:
        raise ValueError(
            f"{type(model).__name__} declares a KV scale of {spec.kv_scale}; a "
            "draft's own pool is stored at a unit scale only, since it has no "
            "scale planes and nothing loads a scale into it"
        )

    from aiter import dtypes

    from atom.utils.selector import DRAFT_POOL_FAMILIES, attn_family, get_attn_backend

    family = attn_family(draft_hf)
    if family not in DRAFT_POOL_FAMILIES:
        if spec is not None:
            raise ValueError(
                f"{type(model).__name__} declares a KV pool of its own, but its "
                f"attention family {family} cannot own a draft pool"
            )
        return None
    backend = get_attn_backend(family)

    if spec is None:
        spec_kwargs, block_size, kv_scale = {}, None, None
    else:
        dcp = model_runner.config.decode_context_parallel_size
        spec_kwargs = spec.pool_kwargs(dcp)
        block_size = spec.block_size(model_runner.config)
        kv_scale = spec.kv_scale
    make_pool = partial(
        backend.make_kv_pool,
        draft_hf,
        world_size=model_runner.world_size,
        # settled once, by `Config`
        kv_dtype=dtypes.d_dtypes[model_runner.config.speculative_config.draft_kv_dtype],
        **spec_kwargs,
    )
    # A way to build it, not the pool: how many rows it has is the walk's to
    # say, and this runs from the proposer's `__init__` -- `build_drafter`
    # still running, `runner.drafter` not yet set, nothing to walk.
    return DraftKvBuilder(
        model_runner,
        make_pool,
        block_size=block_size,
        kv_scale=kv_scale,
        window=draft_window_geometry(
            model_runner.config, getattr(model, "kv_read_window", None), spec_tokens
        ),
    )


def draft_window_geometry(
    config, read_window, spec_tokens: int
) -> DraftWindowGeometry | None:
    """The window a draft's pool is sized to, or None to ride the target's ids."""
    if read_window is None or not envs.ATOM_DSPARK_DRAFT_KV_WINDOW:
        return None
    window, read_page = read_window
    if not window:
        return None
    refuse_unsupported_draft_window(config)
    return DraftWindowGeometry(
        window=int(window),
        read_page=int(read_page),
        block_size=config.kv_cache_block_size,
        spec_tokens=int(spec_tokens),
    )


def refuse_unsupported_draft_window(config) -> None:
    """Refuse, at startup, the deployments a windowed draft pool does not run in:
    each moves or reads draft rows by the target's block ids, or splits them."""
    # The factory's own walk and names: an omitted `kv_connector` is MoRI-IO and
    # names fold case. None is a malformed `multi`, which its build refuses.
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    why = None
    if config.pipeline_parallel_size > 1:
        why = "pipeline parallelism (the head schedules a pool its last stage sizes)"
    elif config.prefill_context_parallel_size > 1:
        why = "prefill context parallelism (a rank holds a stripe of the context rows)"
    elif config.enable_rapidserve:
        why = "RapidServe (the decode side imports a pool laid out by block id)"
    for name, leaf in KVConnectorFactory.leaf_connectors(config) or ():
        if why is not None:
            break
        if name == "moriio":
            why = "the MoRI-IO connector (it transfers draft rows by target block)"
        elif leaf.kv_transfer_config.get("enable_chunked_transfer"):
            why = "chunked P/D transfer (it sends draft rows chunk by chunk)"
    if why is not None:
        raise ValueError(
            f"ATOM_DSPARK_DRAFT_KV_WINDOW=1 does not support {why}; unset it to "
            "keep the draft's KV in the target's block ids"
        )


def draft_context_slots(
    positions: np.ndarray,
    seqlens: np.ndarray,
    first: np.ndarray,
    extent: np.ndarray,
    tables: np.ndarray,
    block_size: int,
    *,
    rows_out: np.ndarray,
    slots_out: np.ndarray,
) -> int:
    """The context rows a windowed draft writes, and where: row `j` of request
    `r` is kept when its block is in `[first[r], extent[r])`. Returns the count.
    """
    req = np.repeat(np.arange(len(seqlens)), seqlens)
    col = positions // block_size - first[req]
    rows = np.flatnonzero((col >= 0) & (col < (extent - first)[req]))
    ids = tables[req[rows], col[rows]].astype(np.int64)
    # A held run has no hole in it; were one there, its rows go nowhere.
    rows, ids = rows[ids >= 0], ids[ids >= 0]
    n = len(rows)
    rows_out[:n] = rows
    slots_out[:n] = ids * block_size + positions[rows] % block_size
    return n


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

    `block_size` is the block the pool is built at; None takes the target
    builder's, which does not exist yet when this is constructed.

    `kv_scale` is the per-tensor scale the draft's `kv_pool_spec` declares;
    None, for a draft that declares no spec, binds the pool's own fp8 scale
    planes instead.

    With a `window`, a Pool.STATE pool of its own ids (`DraftWindowPool`'s)
    instead, whose runs each forward stages (`prepare_step`).
    """

    # Class-level so a builder made without `__init__` rides the target's ids.
    window: DraftWindowGeometry | None = None

    def __init__(
        self,
        model_runner,
        make_pool,
        block_size: int | None = None,
        kv_scale: float | None = None,
        window: DraftWindowGeometry | None = None,
    ):
        self.model_runner = model_runner
        self.window = window
        self._make_pool = make_pool
        self._block_size = block_size
        self._kv_scale = kv_scale
        # The declared scale, shared by every layer: a one-element fp32 tensor
        # because both `reshape_and_cache_flash` and `unified_attention` take
        # it as a device pointer. Built at the first bind, never in a forward.
        self._kv_scale_tensor = None
        self._kv_pool = None
        self.num_blocks = 0  # set in allocate_kv_cache_tensors

    @property
    def kv_pool(self):
        """The pool, built at the row count the walk found.

        On demand because the runner cannot reach the draft model
        (`runner.drafter`) at `__init__`; every reader is a sizing or binding
        step, which all run after it can. The count is the walk's and not
        `num_hidden_layers`, or the pool is sized off one number and addressed
        by another.

        Indexed and not `.get(..., 0)`: this builder exists because the flavor
        owns a pool, so no rows is a contradiction, and a pool of none prices
        at zero and binds nothing -- a draft running on no KV at all. The
        `KeyError` is the one `pool_rows` promises.

        On demand is also what makes the block reachable, and it is the target
        builder's: `_propose` hands the draft the target's block tables, and the
        draft's kernels take the block off the cache they were bound to, so a
        block of its own would address one page with an id that counted
        another. There is nowhere in a draft to convert between the two -- it
        has no metadata of its own. The builder is constructed before the
        target builder, which is why this is read here and not bound earlier.
        """
        if self._kv_pool is None:
            layers = self.row_counts()[DRAFT_KV_ROWS]
            block_size = self._block_size
            if block_size is None:
                block_size = self.model_runner.attn_metadata_builder.block_size
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
        contributions sum into one per-block cost instead of a second pool.

        A windowed pool is a STATE class instead: a window per request in
        flight, plus finished requests' windows kept for prefix hits.
        """
        if self.window is None:
            return [page_pool(self.kv_pool.entry_bytes)]
        per_req = self.window.blocks_per_req
        # Rounded up, so a fraction of a window per slot still keeps one.
        cached = math.ceil(
            envs.ATOM_DSPARK_DRAFT_KV_WINDOW_CACHE
            * self.model_runner.config.max_num_seqs
        )
        return [
            state_pool(
                DRAFT_WINDOW_CLASS,
                self.kv_pool.entry_bytes,
                entries_per_req=per_req,
                extra_entries=cached * per_req,
            )
        ]

    def paged_pool_bytes(self, blocks: int) -> int:
        """The draft's share of the runner's one paged allocation.

        The same `entry_bytes` `sub_pool_specs` adds to the target's, so the
        draft's region is exactly what the block budget already charged for
        it -- which is what "riding the target's block ids" costs. A windowed
        pool takes none of it: it is sized at its own count.
        """
        if self.window is not None:
            return 0
        return self.kv_pool.pool_bytes(blocks)

    def allocate_kv_cache_tensors(self, *, blocks: int, buf) -> dict:
        """Back the draft's pool from its region. Nothing for the runner to
        setattr: the pool is this builder's, and its hooks below are the only
        readers.

        One entry per scheduler block, the count the target was built at --
        that is what riding the target's block ids means. A windowed pool is
        built at its own class's count, in memory of its own.
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
        if self.window is None:
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
        self.num_blocks = runner.pool_plan.entries[DRAFT_WINDOW_CLASS]
        pool.allocate(self.num_blocks, runner.device)
        gib = 1 << 30
        logger.info(
            "Allocated windowed draft KV pool: %d blocks (%d per request for a "
            "%d-token window, %d cached past the running requests), %.2f GiB; "
            "one draft block per target block would be %d blocks, %.2f GiB",
            self.num_blocks,
            self.window.blocks_per_req,
            self.window.window,
            self.num_blocks - runner.config.max_num_seqs * self.window.blocks_per_req,
            pool.pool_bytes(self.num_blocks) / gib,
            blocks,
            pool.pool_bytes(blocks) / gib,
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
        pool = self.kv_pool
        idx = self.pool_rows[DRAFT_KV_ROWS][module]
        if pool.flash_layout:
            k_cache, v_cache = pool.fa_views(idx)
        else:
            k_cache, v_cache = pool.kv_views(idx)
        # A layer's reader may be built for one entry width and dtype; a view
        # of any other would be read at the wrong stride, with no error.
        check_cache = getattr(module, "check_cache", None)
        if check_cache is not None:
            check_cache(k_cache, v_cache)
        if self._kv_scale is not None:
            if self._kv_scale_tensor is None:
                self._kv_scale_tensor = torch.full(
                    (1,), self._kv_scale, dtype=torch.float32, device=k_cache.device
                )
            module.k_scale = module.v_scale = self._kv_scale_tensor
        elif pool.scale_fields:  # the layout's, as the target binder asks
            module.k_scale, module.v_scale = pool.scale_views(idx)
        module.max_model_len = runner.config.max_model_len
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
        """The draft's regions and how it keeps them: riding the target's ids,
        more PAGE units of its blocks; windowed, `draft_pages` on ids of its own."""
        from atom.kv_transfer.disaggregation.types import (
            DRAFT_LAYOUT_FULL,
            DRAFT_LAYOUT_WINDOW,
            KVTransferTensors,
        )

        pool, rows = self.kv_pool, self.pool_rows[DRAFT_KV_ROWS]
        modules = sorted(rows, key=rows.get)
        # Described by its pool like the target's, under a namespace of its own.
        pages = pool.transfer_pages(
            "draft",
            "draft",
            [m.layer_num for m in modules],
            # Each binding declares the page it reads its cache in.
            [
                pool.block_size if m.page_tokens is None else m.page_tokens
                for m in modules
            ],
        )
        # A paged MHA/GQA draft is a per-rank shard. Appending it to an
        # otherwise replicated target must disable whole-object collapse.
        if self.window is None:
            return KVTransferTensors(
                pages=pages, tp_replication_factor=1, draft_layout=DRAFT_LAYOUT_FULL
            )
        for page in pages:
            region = page.region
            if region.total_bytes != region.unit_bytes * self.num_blocks:
                raise ValueError(
                    f"draft region {region.semantic_role} holds "
                    f"{region.total_bytes} B, not {self.num_blocks} blocks of "
                    f"{region.unit_bytes} B"
                )
        return KVTransferTensors(
            tp_replication_factor=1,
            draft_layout=DRAFT_LAYOUT_WINDOW,
            draft_pages=pages,
        )

    # ------------------------- windowed step staging ------------------------- #
    def step_buffers(self) -> dict[str, CpuGpuBuffer]:
        """The per-forward staging a windowed pool reads, for `forward_vars`."""
        if self.window is None:
            return {}
        runner = self.model_runner
        max_bs = runner.config.max_num_seqs
        tokens = runner.config.max_num_batched_tokens
        pinned = torch.device(runner.device).type != "cpu"
        i32 = {"dtype": torch.int32, "device": runner.device, "pin_memory": pinned}
        i64 = {"dtype": torch.int64, "device": runner.device, "pin_memory": pinned}
        return {
            # Each request's held run of draft block ids, from its first block.
            "draft_window_tables": CpuGpuBuffer(
                max_bs, self.window.blocks_per_req + _WINDOW_TABLE_SLACK, **i32
            ),
            "draft_window_first": CpuGpuBuffer(max_bs, **i32),
            # The block each run reaches, read on the host only.
            "draft_window_extent": CpuGpuBuffer(max_bs, **i32),
            "draft_context_rows": CpuGpuBuffer(tokens, **i64),
            "draft_context_slots": CpuGpuBuffer(tokens, **i64),
        }

    def prepare_step(
        self,
        batch,
        attn_metadata,
        *,
        positions: np.ndarray,
        seqlens: np.ndarray,
        running_bs: int,
        decode: bool,
    ) -> None:
        """Stage this forward's runs, and on its own `attn_metadata` the context
        rows inside them; a pad row reads one page of block 0 and writes nothing."""
        if self.window is None:
            return
        var = self.model_runner.forward_vars
        tables = var["draft_window_tables"]
        first = var["draft_window_first"]
        extent = var["draft_window_extent"]
        tables.np[:running_bs] = 0
        first.np[:running_bs] = 0
        extent.np[:running_bs] = 1
        attn_metadata.draft_context = (
            var["draft_context_rows"].gpu[:0],
            var["draft_context_slots"].gpu[:0],
            False,
        )
        windows = getattr(batch, "draft_windows", None)
        bs = len(seqlens)
        if windows is not None and bs:
            starts, lengths, ids = windows
            starts, lengths = starts[:bs], lengths[:bs]
            width = tables.np.shape[1]
            if lengths.max(initial=0) > width:
                row = int(lengths.argmax())
                raise RuntimeError(
                    f"request row {row} holds {int(lengths[row])} draft window "
                    f"blocks, more than the {width} a forward's table has; the "
                    "scheduler ran further ahead of the finalized tokens than "
                    "`DraftWindowGeometry.scheduled_slack` allows for"
                )
            # One scatter for the whole batch: row `req[j]`, column `col[j]`.
            ends = np.cumsum(lengths)
            req = np.repeat(np.arange(bs), lengths)
            col = np.arange(int(ends[-1])) - np.repeat(ends - lengths, lengths)
            tables.np[req, col] = ids[: len(req)]
            first.np[:bs] = starts
            extent.np[:bs] = starts + lengths
            if decode:
                self._check_decode_reads(positions, seqlens, first.np[:bs])
            n = draft_context_slots(
                positions,
                seqlens,
                first.np[:bs],
                extent.np[:bs],
                tables.np[:bs],
                self.window.block_size,
                rows_out=var["draft_context_rows"].np,
                slots_out=var["draft_context_slots"].np,
            )
            if n:
                attn_metadata.draft_context = (
                    var["draft_context_rows"].copy_to_gpu(n),
                    var["draft_context_slots"].copy_to_gpu(n),
                    n == len(positions),
                )
        tables.copy_to_gpu(running_bs)
        first.copy_to_gpu(running_bs)

    def _check_decode_reads(
        self, positions: np.ndarray, seqlens: np.ndarray, first: np.ndarray
    ) -> None:
        """Refuse a decode whose block pass could read below a request's run,
        where the window table holds another request's rows."""
        geo = self.window
        starts = positions[np.cumsum(seqlens) - seqlens]
        read = np.maximum(0, starts + 2 - geo.window)
        need = (read - read % geo.read_page) // geo.block_size
        bad = np.flatnonzero(first > need)
        if len(bad):
            row = int(bad[0])
            raise RuntimeError(
                f"request row {row} holds draft blocks from {int(first[row])} "
                f"on, but its block pass anchored at or past position "
                f"{int(starts[row])} reads block {int(need[row])}"
            )

    def kept_context(self, attn_metadata, num_rows: int):
        """`(rows, slots)` of the context rows a forward of `num_rows` writes into a
        windowed pool, `rows` None when it is all of them; None when the pool rides."""
        if self.window is None:
            return None
        staged = getattr(attn_metadata, "draft_context", None)
        if staged is None:
            raise RuntimeError(
                "this forward staged no draft window (`DraftKvBuilder.prepare_step`)"
            )
        rows, slots, all_scheduled = staged
        return (None if all_scheduled and len(slots) == num_rows else rows), slots

    def block_runs(self, running_bs: int) -> DraftRuns | None:
        """The runs a block pass at `running_bs` rows reads, or None when the
        pool rides the target's ids."""
        if self.window is None:
            return None
        var = self.model_runner.forward_vars
        return DraftRuns(
            var["draft_window_tables"].gpu[:running_bs],
            var["draft_window_first"].gpu[:running_bs],
            var["draft_window_extent"].np[:running_bs],
        )
