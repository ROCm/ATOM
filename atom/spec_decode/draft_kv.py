import logging
from functools import partial

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

# Blocks a forward's window table holds past `blocks_per_req`: what a sequence
# holds is bounded by its finalized length, and the bound leans on how far the
# scheduler runs ahead of it. A run that outgrows the table is refused, not cut.
_WINDOW_TABLE_SLACK = 2


def draft_kv_builder(model_runner, draft_hf, *, read_window=None, spec_tokens=0):
    """The KV builder a draft needs of its own, or None if it needs none.

    A draft is not an attention flavor; it is a model that has one. So the
    flavor comes from the draft's own config through the same selector the
    target uses, and that backend says whether a draft of its flavor caches
    into a pool of its own (`DRAFT_OWNS_KV_POOL`). Where it does not, the
    runner never sees a draft builder at all.

    Which is why speculative decoding imports no attention anywhere: adding a
    flavor is a flag and a `make_kv_pool` on that backend.

    `read_window` is `(window, read_page)` for a draft whose block pass reads a
    sliding window (`DSparkDraftModel.kv_read_window`); under
    `ATOM_DSPARK_DRAFT_KV_WINDOW` its pool holds that window rather than riding
    the target's block ids. `spec_tokens` is the verify width.
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
    return DraftKvBuilder(
        model_runner,
        make_pool,
        window=draft_window_geometry(model_runner.config, read_window, spec_tokens),
    )


def draft_window_geometry(
    config, read_window, spec_tokens
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


def _kv_connectors(kv_config) -> list[dict]:
    if not isinstance(kv_config, dict) or not kv_config:
        return []
    subs = kv_config.get("connectors", [])
    return [kv_config] + [c for sub in subs for c in _kv_connectors(sub)]


def refuse_unsupported_draft_window(config) -> None:
    """Refuse, at startup, the deployments a windowed draft pool does not run in.

    Each would move or read draft rows through a path that only knows the
    target's block ids, which is the silent kind of wrong: the draft attends
    rows nobody wrote and only its acceptance says so.
    """
    why = None
    if config.pipeline_parallel_size > 1:
        why = "pipeline parallelism (the head schedules a pool its last stage sizes)"
    elif config.enable_rapidserve:
        why = "RapidServe (the decode side imports a pool laid out by block id)"
    for connector in _kv_connectors(getattr(config, "kv_transfer_config", None)):
        if why is not None:
            break
        if connector.get("kv_connector") == "moriio":
            why = "the MoRI-IO connector (it transfers draft rows by target block)"
        elif connector.get("enable_chunked_transfer"):
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
    """The context rows a windowed draft writes, and where.

    Row `j` of a forward over `seqlens` tokens per request belongs to request
    `r`; it is kept when its position falls in the blocks that request holds,
    `[first[r], extent[r])`, whose ids are `tables[r]` from `first[r]` on.
    Fills the kept rows' indices and slots and returns how many there are.
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

    With a `window`, the blocks are its own instead: a Pool.STATE class sized
    to the running requests' windows plus a cache of finished ones, addressed
    by the ids the scheduler's `DraftWindowPool` hands out. Each forward then
    stages its requests' held runs (`prepare_step`) for the context write and
    the block pass to address the pool by.
    """

    def __init__(self, model_runner, make_pool, window: DraftWindowGeometry = None):
        self.model_runner = model_runner
        self._make_pool = make_pool
        self._kv_pool = None
        self.num_blocks = 0  # set in allocate_kv_cache_tensors
        self.window = window

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
        contributions sum into one per-block cost instead of a second pool.

        A windowed pool is a STATE class instead: what it holds scales with the
        requests in flight, a window each, not with the cached history. Its
        extra entries are the finished requests' windows a prefix hit reuses.
        """
        if self.window is None:
            return [page_pool(self.kv_pool.entry_bytes)]
        per_req = self.window.blocks_per_req
        cached = round(
            envs.ATOM_DSPARK_DRAFT_KV_WINDOW_CACHE
            * self.model_runner.config.max_num_seqs
        )
        return [
            state_pool(
                DRAFT_WINDOW_CLASS,
                self.kv_pool.entry_bytes,
                entries_per_req=per_req,
                extra_entries=max(0, cached) * per_req,
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
                "Allocated draft KV pool: %d blocks, %d B of the paged allocation",
                blocks,
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

    def attach_transfer_tensors(self, transfer_tensors) -> None:
        """Add the draft's regions to the target's transfer description.

        Riding the target's ids, they are more PAGE units of the same blocks. A
        windowed pool's are addressed by its own ids, so they go in a list of
        their own, which no target-block path walks; a transfer moves them by
        each request's window (`draft_block_table`).
        """
        draft = self.get_kv_transfer_tensors()
        if self.window is None:
            transfer_tensors.merge_pages(draft)
            transfer_tensors.draft_layout = "full"
            return
        transfer_tensors.draft_pages = draft.pages
        transfer_tensors.num_draft_blocks = self.num_blocks
        transfer_tensors.draft_layout = "window"
        for page in draft.pages:
            region = page.region
            if region.total_bytes != region.unit_bytes * self.num_blocks:
                raise ValueError(
                    f"draft region {region.semantic_role} holds "
                    f"{region.total_bytes} B, not {self.num_blocks} blocks of "
                    f"{region.unit_bytes} B"
                )

    # ------------------------- windowed step staging ------------------------- #
    @property
    def window_table_width(self) -> int:
        """Blocks a forward's per-request window table holds."""
        return self.window.blocks_per_req + _WINDOW_TABLE_SLACK

    def step_buffers(self) -> dict[str, CpuGpuBuffer]:
        """The per-forward staging a windowed pool reads, for `forward_vars`.

        In `forward_vars` so a PP ring clones them per in-flight slot, though a
        windowed pool runs without PP today.
        """
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
            "draft_window_tables": CpuGpuBuffer(max_bs, self.window_table_width, **i32),
            "draft_window_first": CpuGpuBuffer(max_bs, **i32),
            # The block its table reaches, on the host: what the block pass
            # caps a request's reads and writes at.
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
        """Stage this forward's windows and the context rows they keep.

        `positions` are the scheduled tokens' own, `seqlens` how many each
        scheduled request has. A row past `len(seqlens)` -- the pad tail of a
        widened batch, or every row of a dummy forward -- holds one page of
        block 0 and writes nothing, like the block pass's pad rows.

        The kept rows ride on this forward's `attn_metadata`, so the context
        write can only ever read the ones staged for it (`kept_context`).
        """
        if self.window is None:
            return
        var = self.model_runner.forward_vars
        tables = var["draft_window_tables"]
        first = var["draft_window_first"]
        extent = var["draft_window_extent"]
        tables.np[:running_bs] = 0
        first.np[:running_bs] = 0
        extent.np[:running_bs] = 1
        attn_metadata.draft_context = (var["draft_context_rows"].gpu[:0], 0, False)
        attn_metadata.draft_slot_mapping = var["draft_context_slots"].gpu[:0]
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
                rows = var["draft_context_rows"].copy_to_gpu(n)
                slots = var["draft_context_slots"].copy_to_gpu(n)
                attn_metadata.draft_context = (rows, n, n == len(positions))
                attn_metadata.draft_slot_mapping = slots
        tables.copy_to_gpu(running_bs)
        first.copy_to_gpu(running_bs)

    def _check_decode_reads(
        self, positions: np.ndarray, seqlens: np.ndarray, first: np.ndarray
    ) -> None:
        """Refuse a decode whose block pass could read below a request's run.

        A request's verify rows start at its anchor's lower bound, and its
        block pass reads from `first_read_pos` of that anchor on. Below the run
        the window table holds nothing of the request's, so a read there takes
        another request's rows -- wrong, and only acceptance would say so.
        """
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
        """`(rows, count)` of the context rows this forward writes, or None.

        None for a pool riding the target's ids, whose rows go to the target's
        slots. `rows` is None when the forward's `num_rows` rows are all kept,
        in order. Their slots are `attn_metadata.draft_slot_mapping`, which the
        draft's context write reads; both were staged on this very metadata by
        `prepare_step`, and a forward that staged nothing is refused rather
        than handed the last one's.
        """
        if self.window is None:
            return None
        staged = getattr(attn_metadata, "draft_context", None)
        if staged is None:
            raise RuntimeError(
                "this forward staged no draft window (`DraftKvBuilder.prepare_step`)"
            )
        rows, n, all_rows = staged
        return (None if all_rows and n == num_rows else rows), n

    def block_window(self, running_bs: int):
        """`(tables, first, extent)` a block pass at `running_bs` rows reads
        through, or None for a pool riding the target's ids."""
        if self.window is None:
            return None
        var = self.model_runner.forward_vars
        return (
            var["draft_window_tables"].gpu[:running_bs],
            var["draft_window_first"].gpu[:running_bs],
            var["draft_window_extent"].np[:running_bs],
        )
