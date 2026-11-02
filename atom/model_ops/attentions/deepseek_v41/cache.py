# SPDX-License-Identifier: MIT
"""PAGE-backed global keys and complete, relocatable CSA2 request state."""

import numpy as np
import torch
from aiter.jit.utils.chip_info import get_gfx

from atom.model_ops.attentions.deepseek_v41.packed_rows import (
    gather_prefix_rows,
    pack_rows,
    write_packed_window,
)
from atom.model_ops.attentions.pool_layout.entry_arena import EntryMajorArena
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
    INDEX_FP8_SCALE_FMT,
    MAIN_FP4,
)
from atom.model_ops.blockscale import quantize_fp4, quantize_fp8
from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE
from atom.model_ops.deepseek_v41.candidate_table import bind_candidates
from atom.model_ops.deepseek_v41.compressor import compress_batch
from atom.model_ops.deepseek_v41.dspark import gather_window_rows
from atom.model_ops.deepseek_v41.index_plane import (
    IndexUnits,
    fp4_index_layout_for_arch,
)
from atom.model_ops.deepseek_v41.index_write import (
    write_index_rows,
    write_index_rows_fp4,
)
from atom.model_ops.deepseek_v41.rope_window import rope_quant_window
from atom.model_ops.deepseek_v41.unit_table import unit_table
from atom.model_ops.v4_kernels import make_compress_plans
from atom.model_ops.v4_kernels.state_writes import (
    swa_scatter_rows,
    swa_scatter_rows_reference,
    swa_write,
)
from atom.utils import CpuGpuBuffer, upload_numpy

from .indices import (
    build_indices,
    build_window_rows,
    fill_step_indptrs,
    splits_window,
)
from .metadata import prepare_batch_step
from .speculative import TentativeState


class PagedAttentionCache:
    def _allocate_backing(self, size, device):
        """This cache's whole pool, as one flat uint8 arena.

        The one seam a caller that already owns the bytes overrides. PAGE and
        STATE share a single address space here because
        :meth:`V41PoolGeometry.window` states a slot's ring as an offset past
        the absolute end of the paged region, and the ``_indices`` kernels
        dereference page rows and window rows against one base -- so the pool
        is handed over whole or not at all, and an override returns the whole
        arena rather than the paged part of it.

        What an override has to return is what this returns: zeroed, flat
        uint8, contiguous, on ``device``, and exactly ``size`` elements. Every
        region below is a `view` of it with its own strides, which neither a
        short arena nor a longer tail would divide, and the cursor reset plus
        every cold read assume the zeros.
        """
        return torch.zeros(size, dtype=torch.uint8, device=device)

    def __init__(
        self,
        geometry,
        pages,
        slots,
        device,
        max_tokens=0,
        workspace=None,
    ):
        if pages < 1 or slots < 1:
            raise ValueError("A paged cache needs positive PAGE and STATE capacities")
        self.geometry, self.num_pages, self.num_slots = geometry, pages, slots
        # The scorer's `ScoreWorkspace`, owned by the builder so the startup
        # profile's throwaway cache and the serving one share it.
        self.workspace = workspace
        self.packed = geometry.packed
        self.window_fp8 = geometry.window_fp8
        # MXFP8 rows everywhere, read by `sparse_attn_v41_paged_2src`
        self.kv_fp8 = geometry.kv_dtype == "fp8"
        self.indptr_device, self.max_tokens, self.indptr_buffers = device, 0, {}
        index_offsets, boundary = geometry.paged_extents(pages)
        size = boundary + slots * geometry.state_bytes
        self.backing = self._allocate_backing(size, device)
        main_bytes = self.num_pages * geometry.page_bytes
        self.pages = EntryMajorArena(
            geometry.page_fields,
            self.num_pages,
            device,
            buf=self.backing[:main_bytes],
            slot_stride=geometry.page_bytes,
        )
        # Each owner's `geometry.index_planes`, each `[pages, rows, width]` and
        # dense in its own rows: the stride a paged reader is handed is the
        # rows' and not the PAGE's. Both sides of the index plane take the
        # views from here.
        self.index_planes = {
            owner: tuple(
                self._index_plane(offset, geometry.rows_per_page(ratio), width)
                for offset, (_, width) in zip(
                    index_offsets[owner], geometry.index_planes
                )
            )
            for owner, ratio in geometry.owners
        }
        # The same bytes as `[tiles, tile rows, width]`, which is what a block
        # id addresses and what the scorer is handed.
        self.index_units = {
            owner: IndexUnits(
                *(
                    plane.view(-1, geometry.index_block_rows, plane.shape[-1])
                    for plane in planes
                ),
                layout=(
                    fp4_index_layout_for_arch(get_gfx())
                    if geometry.index_fp4
                    else "fp8"
                ),
            )
            for owner, planes in self.index_planes.items()
        }
        self.state = EntryMajorArena(
            geometry.state_fields,
            slots,
            device,
            buf=self.backing[boundary:],
            slot_stride=geometry.state_bytes,
        )
        self.state_bytes = self.backing[boundary:].view(slots, geometry.state_bytes)
        self.page_bytes = self.backing[:main_bytes].view(
            self.num_pages, geometry.page_bytes
        )
        self.cursor = self.state.view("cursor")[0]
        self.cursor[:, 1:].fill_(-1)
        # `[slot count, end + history]`, staged rather than built per step: a
        # fresh `torch.as_tensor` per forward is a fresh allocation and a fresh
        # pageable copy, and a batch can hold at most one request per slot.
        pinned = torch.device(device).type != "cpu"
        self._cursor_staging = CpuGpuBuffer(
            slots,
            self.cursor.shape[1],
            dtype=self.cursor.dtype,
            device=device,
            pin_memory=pinned,
        )
        # The same, for a verify step's candidate cursors: one row per prefix
        # a request could have accepted.
        self.tentative_staging = CpuGpuBuffer(
            slots,
            geometry.speculative_tokens + 1,
            self.cursor.shape[1],
            dtype=self.cursor.dtype,
            device=device,
            pin_memory=pinned,
        )
        # One H2D for every slot a forward recycles, so the two resets below are
        # two launches rather than two per fresh request. int64 because
        # `index_fill_` takes no other width, unlike the `index_select` above.
        self._reset_slots = CpuGpuBuffer(
            slots, dtype=torch.int64, device=device, pin_memory=pinned
        )
        # The Engram history each of those starts at (`BatchStep.start_histories`).
        self._reset_histories = CpuGpuBuffer(
            slots,
            self.cursor.shape[1] - 1,
            dtype=self.cursor.dtype,
            device=device,
            pin_memory=pinned,
        )
        # Where a caller that wants no history lands its cursor rows: the copy
        # is issued and judged a step later, so nothing waits for it. See
        # `prepare_state`.
        self._probe = torch.zeros(
            slots,
            self.cursor.shape[1],
            dtype=self.cursor.dtype,
            device="cpu",
            pin_memory=pinned,
        )
        self._probe_done = torch.cuda.Event() if pinned else None
        self._probe_claim = None
        if geometry.packed:
            self.pool = self.backing.view(-1, 1)
        elif self.kv_fp8:
            self.pool = self.backing.view(-1, geometry.row_bytes)
        else:
            self.pool = self.backing.view(torch.bfloat16).view(-1, geometry.head_dim)
        # Owner -> its row in the compressor rings, which hold the raw
        # projections a pool window reaching back before this forward needs.
        self.compress_indices = {
            owner: i for i, owner in enumerate(geometry.compress_owners)
        }
        self.pending = None
        # Last, after the pool itself: these are kilobytes against the pool's
        # gigabytes, and taking them first moves the base every reader of the
        # pool computes its offsets from.
        self._reserve_indptrs(max_tokens)

    def unit_regions(self):
        """`(base address, bytes)` of every region one PAGE unit owns.

        A unit is its main page and that page's rows in each index plane --
        `paged_bytes` in as many pieces as there are planes, since the two
        scale together but are laid out apart. Every plane is dense in pages,
        so a region's stride is its own size and unit `u` is at `base + u *
        bytes`. This is the destination stream a checkpoint image is cut into.
        """
        regions = [(self.page_bytes.data_ptr(), self.geometry.page_bytes)]
        regions += [
            (plane.data_ptr(), plane.stride(0) * plane.element_size())
            for planes in self.index_planes.values()
            for plane in planes
        ]
        return regions

    def unit_views(self, unit):
        """The same regions for one unit, named as `uint8` views."""
        return [self.page_bytes[unit]] + [
            plane[unit].flatten().view(torch.uint8)
            for planes in self.index_planes.values()
            for plane in planes
        ]

    def _index_plane(self, offset, rows, width):
        """`[pages, rows, width]` bytes at `offset`, untyped.

        A preshuffled row interleaves its bytes across the tile (and an FP8
        one carries its scale past the block's data), so a row has no element
        type to take a view in -- the writer and the scorer both address it as
        bytes.

        `as_strided`'s storage offset is absolute, so the retyped view's own has
        to be added -- omit it and every plane addresses from the front of the
        pool, over the main pages.
        """
        typed = self.backing.view(torch.uint8)
        return typed.as_strided(
            (self.num_pages, rows, width),
            (rows * width, width, 1),
            typed.storage_offset() + offset,
        )

    def _reserve_indptrs(self, tokens):
        """One `(prefix, extend)` pair per ratio, at an address that stays put.

        Every forward refills these rather than allocating its own, because a
        replay reruns no host code, so the kernels a capture recorded hold
        these addresses for good. Serving reserves its widest forward up
        front; only the isolated callers grow, and the guard says why they
        may.
        """
        if tokens <= self.max_tokens and self.indptr_buffers:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Indptr buffers cannot be reallocated under capture")
        self.max_tokens = max(tokens, self.max_tokens)
        self.indptr_buffers = {
            ratio: tuple(
                torch.empty(
                    self.max_tokens + 1, dtype=torch.int32, device=self.indptr_device
                )
                for _ in range(2)
            )
            for ratio in self.geometry.layer_ratios
        }

    def require_committed(self):
        if self.pending is not None:
            raise RuntimeError(
                "Commit the accepted prefix before reusing or checkpointing state"
            )

    def begin_step(
        self,
        requests,
        *,
        block_tables,
        tentative=False,
        is_prefill=False,
        buffers=None,
        running_bs=None,
        running_tokens=None,
        max_q_len=None,
        state_slot_out=None,
        plans=None,
        publication_group=None,
        query_prefix_ready=False,
        query_prefix_republish_reason=None,
        planners=(),
        floors=None,
        cached_until=None,
        finals=None,
        start_histories=None,
    ):
        """`floors`, `cached_until`, `finals`, `start_histories`: per request,
        on the host, `BatchStep`'s `request_*` and `start_histories`; None for
        a forward recomputing nothing, ending every prompt."""
        self.require_committed()
        requests = tuple(requests)
        if tentative and (
            self.geometry.speculative_tokens == 0
            or not requests
            or any(
                span.position == 0 or span.length > self.geometry.speculative_tokens + 1
                for span in requests
            )
        ):
            raise ValueError(
                "Tentative verification needs a prefix and sufficient window slack"
            )
        offset = 0
        seen = set()
        if len(block_tables) != len(requests):
            raise ValueError("One PAGE table is required per request")
        for span, row in zip(requests, block_tables):
            if span.length <= 0 or span.position < 0 or span.offset != offset:
                raise ValueError(
                    "Request spans must be nonempty and partition the token batch"
                )
            if not 0 <= span.slot < self.num_slots or span.slot in seen:
                raise ValueError("Each request needs its own valid STATE slot")
            needed = -(-span.end // self.geometry.block_size)
            if len(row) < needed:
                raise ValueError("Request PAGE table is incomplete or out of range")
            seen.add(span.slot)
            offset += span.length
        step = prepare_batch_step(
            requests,
            self.pool.device,
            block_tables=block_tables,
            page_limit=self.num_pages,
            tentative=tentative,
            is_prefill=is_prefill,
            buffers=buffers,
            running_bs=running_bs,
            running_tokens=running_tokens,
            max_q_len=max_q_len,
            state_slot_out=state_slot_out,
            ratios=tuple(ratio for ratio, _ in self.geometry.compress_ratios),
            publication_group=publication_group,
            query_prefix_ready=query_prefix_ready,
            query_prefix_republish_reason=query_prefix_republish_reason,
            planners=planners,
        )
        step.plans = (
            self._private_plans(requests, tentative) if plans is None else plans
        )
        count = len(requests)
        zeros = np.zeros(count, np.int32)
        step.request_floors = zeros if floors is None else floors
        step.request_cached_until = zeros if cached_until is None else cached_until
        step.request_finals = (
            np.ones(count, np.bool_) if finals is None else np.asarray(finals)
        )
        step.start_histories = start_histories
        # Per-forward and layer-invariant, so built here rather than by the
        # first layer to want one, exactly as V4 builds its own three. Triton,
        # like every reader of them, so a CPU pool has neither.

        # Held so a reader that cannot be handed this object still finds it.
        # Under vLLM's breakable capture an eager break is replayed with the
        # arguments it was captured with, so attention cannot take `step` as a
        # parameter and see this step -- it would see the one recorded with the
        # graph. The cache outlives every step, so reading it from here is live.
        self.current_step = step
        if not step.positions.is_cuda:
            return step
        self._reserve_indptrs(step.width)
        # Before the indptrs, which count each row's window from its floor.
        self.publish_replay_layout(step, count if running_bs is None else running_bs)
        return step

    def publish_replay_layout(self, step, running_bs, floors=None):
        """`step.swa_replay_start` and `swa_cached_until` from its host
        `request_floors` and `request_cached_until`, indexed by the step's own
        batch rows: the fixed zeros unless a request recomputes rows below
        where it resumed. ``floors``: the floors already published at fixed
        addresses, a bounded-replay tail's, whose rows no ring holds."""
        zeros = self._replay_start_zeros(running_bs)
        step.swa_replay_start = step.swa_cached_until = zeros
        if floors is not None:
            step.swa_replay_start = floors
        elif step.request_floors.any() and not step.decode:
            step.swa_replay_start, step.swa_cached_until = upload_numpy(
                np.stack((step.request_floors, step.request_cached_until)),
                step.positions.device,
            )

    def _replay_start_zeros(self, running_bs):
        """Every request's window from position 0: what a step that is not a
        bounded-replay tail reads as `swa_replay_start`, indexed by batch id up
        to `running_bs - 1`. One fixed buffer, sized for every slot up front,
        so the index kernels a capture records keep reading it on replay."""
        rows = max(running_bs, self.num_slots, 1)
        zeros = getattr(self, "_replay_zeros", None)
        if zeros is None or zeros.numel() < rows:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("The replay-start buffer cannot grow under capture")
            zeros = self._replay_zeros = torch.zeros(
                rows, dtype=torch.int32, device=self.indptr_device
            )
        return zeros

    def _private_plans(self, requests, tentative):
        """Plans into freshly allocated buffers, for a caller without any.

        The serving builder owns fixed-address ones and passes them in; this is
        the isolated path, alongside the private metadata buffers above.
        """
        if not requests:
            return {}
        lengths = np.asarray([span.length for span in requests], dtype=np.int32)
        rows = max(int(lengths.sum()) + len(requests), 1)
        device = self.pool.device

        def buffer(*shape, dtype=torch.int32):
            return CpuGpuBuffer(
                *shape, dtype=dtype, device=device, pin_memory=device.type != "cpu"
            )

        # The same three the serving builder declares, under the same keys.
        buffers = {
            ratio: {
                "compress": buffer(rows, 4),
                "write": buffer(rows, 4),
                "key_rope": buffer(rows, dtype=torch.int64),
            }
            for ratio, _ in self.geometry.compress_ratios
        }
        return make_compress_plans(
            lengths,
            np.asarray([span.end for span in requests], dtype=np.int32),
            self.geometry.compress_ratios,
            plan_buffers=buffers,
            extra_write=self.geometry.speculative_tokens if tentative else 0,
        )

    def _report_stale_state(self, cursors, starts, requests, fresh):
        """Raise on the first request whose state is not where it is wanted.

        Fresh requests are excluded: the slot they are about to reuse holds
        whatever the last tenant left, and the reset below is what makes it
        theirs.
        """
        wrong = np.flatnonzero(~fresh & (cursors[:, 0] != starts))
        if not wrong.size:
            return
        first = int(wrong[0])

        raise ValueError(
            f"Request {requests[first]} needs state at {starts[first]}, "
            f"found {int(cursors[first, 0])}; replay from a recoverable boundary"
        )

    def _judge_previous_probe(self):
        """Check the rows the last deferred `prepare_state` shipped, if they landed.

        Not ready yet means the verdict waits another step rather than blocking
        for it -- which is the whole point of having deferred it.
        """
        if self._probe_claim is None:
            return
        if self._probe_done is not None and not self._probe_done.query():
            return
        starts, requests, fresh = self._probe_claim
        self._probe_claim = None
        self._report_stale_state(
            self._probe[: starts.size].numpy(), starts, requests, fresh
        )

    def prepare_state(self, step, *, histories=True):
        """Read restored cursors; reset recycled slots before any layer writes.

        `histories=False` says the caller does not need this step's committed
        history -- the device path works it out of the cursor itself -- and so
        does not need this step's verdict either. The rows are then shipped
        into pinned memory without waiting and judged on the next call, which
        takes the one blocking D2H a decode step still had off the critical
        path. What it gives up is that a stale slot is named one step late; it
        is still a hard failure, and it is a "should never happen" bound, not
        an expected outcome.

        Whether a request is fresh and whether its state is where the scheduler
        thinks are both decided over the whole batch at once -- the per-request
        walk this replaced was 58 us of every decode step, spent branching on
        `span.position` one Python attribute at a time.
        """
        self.require_committed()
        self._judge_previous_probe()
        count = step.scheduled_bs
        starts = step.request_positions[:count]
        ids = [span.request_id for span in step.requests]
        # A slot starts over at its floor: 0, or where a request rebuilds its
        # state below a prefix hit (`BatchStep.request_floors`). One
        # recomputing rows below where its state is keeps the state, and its
        # cursor steps back to the forward's start.
        resumes = step.request_cached_until[:count]
        fresh = (starts == step.request_floors[:count]) & (resumes == 0)
        restarted = np.concatenate((np.flatnonzero(fresh), np.flatnonzero(resumes)))
        # where each state has to be
        wanted = np.where(resumes > 0, resumes, starts)
        if histories:
            cursors = (
                torch.index_select(self.cursor, 0, step.slots[:count]).cpu().numpy()
            )
            self._report_stale_state(cursors, wanted, ids, fresh)
        else:
            # Issued before the reset below, so a recycled slot is captured as
            # its old tenant left it -- which is why the judge skips fresh ones.
            self._probe[:count].copy_(
                torch.index_select(self.cursor, 0, step.slots[:count]),
                non_blocking=True,
            )
            if self._probe_done is not None:
                self._probe_done.record()
            self._probe_claim = (wanted, ids, fresh.copy())
            cursors = None
        if restarted.size:
            # The resets take one index, fresh slots first, so the slots cross
            # once.
            slots = self._reset_slots.np[: restarted.size]
            slots[:] = [step.requests[i].slot for i in restarted]
            index = self._reset_slots.copy_to_gpu(slots.size)
            self.state_bytes.index_fill_(0, index[: int(fresh.sum())], 0)
            if step.start_histories is None:
                self.cursor[:, 1:].index_fill_(0, index, -1)
                started = -1
            else:
                started = step.start_histories[:count][restarted]
                self._reset_histories.np[: restarted.size] = started
                self.cursor[:, 1:].index_copy_(
                    0, index, self._reset_histories.copy_to_gpu(restarted.size)
                )
            if cursors is not None:
                cursors[restarted, 0] = starts[restarted]
                cursors[restarted, 1:] = started
        if cursors is None:
            if step.tentative:
                self.pending = TentativeState(self, step)
            return None
        if step.tentative:
            self.pending = TentativeState(self, step, cursors[:, 1:])
        return cursors[:, 1:]

    def advance_cursor(self, step, histories):
        """Move every request's cursor to where this forward will leave it.

        Before the forward rather than after it: the model is a captured graph
        whose replay runs no host code, so a cursor written from Python inside
        it would be written once, at capture, and never again. Nothing between
        here and the next `prepare_state` reads the cursor -- checkpoint stores
        run ahead of the batch, so the image they take pairs the ring and the
        cursor of the step before this one, which is the pair that agrees.

        A tentative step has no business here: its cursor is the accepted
        prefix's, which only the sampler knows. `commit_tentative` writes it.
        """
        if step.tentative:
            raise RuntimeError("A tentative step's cursor is committed, not advanced")
        count = step.scheduled_bs
        if not count:
            return
        rows = self._cursor_staging.np[:count]
        rows[:, 0] = [span.end for span in step.requests]
        rows[:, 1:] = histories
        self.cursor[step.slots[:count].long()] = self._cursor_staging.copy_to_gpu(count)

    def commit_tentative(self, step, anchors):
        if self.pending is None or self.pending.step is not step:
            raise RuntimeError("Tentative state was not prepared for this step")
        self.pending.commit(anchors)
        self.pending = None

    def compress(self, owner, compressor, values, scores, step, rope):
        # A quantized pool interleaves values with their scales, which the
        # kernel's BF16 scatter cannot write; take the rotated echo and
        # quantize it here.
        scatter = (
            None
            if self.window_fp8
            else (self.pages.view(f"main_{owner}")[0], step.block_tables)
        )
        latent, rotated = compress_batch(
            self, owner, compressor, values, scores, step, rope, scatter=scatter
        )
        if rotated is not None:
            self._scatter_rows(
                self.pages.view(f"main_{owner}")[0],
                step,
                (
                    quantize_fp4(rotated, **MAIN_FP4)
                    if self.packed
                    else quantize_fp8(rotated)
                ),
                compressor.ratio,
            )
        return latent

    def write_index(self, owner, step, index, ratio):
        """One pass quantizes, preshuffles and scatters.

        The kernel resolves a row's address from the plan and the PAGE table
        itself, so nothing here computes one -- `ratio` is all it needs to
        turn the plan's position into a compressed row.
        """
        (plane,) = self.index_planes[owner]
        write_index_rows(
            index[0],
            plane,
            step.plans[ratio].compress_plan_gpu,
            step.block_tables,
            self.geometry.rows_per_page(ratio),
            ratio=ratio,
            rows_per_block=self.geometry.index_block_rows,
            scale_fmt=INDEX_FP8_SCALE_FMT,
        )

    def write_index_fp4(self, owner, step, keys, ratio, norm, rope):
        """`write_index` on the FP4 plane, from the keys before their norm:
        the same pass norms and rotates them as `Indexer.project_keys` does."""
        write_index_rows_fp4(
            keys,
            self.index_units[owner],
            step.plans[ratio].compress_plan_gpu,
            step.block_tables,
            self.geometry.rows_per_page(ratio),
            ratio=ratio,
            norm=norm,
            rope=rope,
        )

    def indptrs(self, step):
        """`step`'s `{ratio: (prefix indptr, extend indptr, reserved top-k)}`
        (`fill_step_indptrs`), into its fixed buffers: a TBO microbatch's own,
        else the cache's. Memoized as `unit_tiles` is, so a forward that builds
        no index (mono's) fills none."""
        if not step.indptrs:
            step.indptrs = fill_step_indptrs(
                step, self.geometry, step.indptr_buffers or self.indptr_buffers
            )
        return step.indptrs

    def unit_tiles(self, step, ratio):
        """Tile ids per query token: one table per ratio, shared by its owners.

        Memoized on the step and dropped by `begin_forward` rather than built
        with it: a table built outside the graph is one a replay reads without
        rebuilding. With a `workspace` the table is its
        ratio's fixed rows, rewritten each step; without one, a fresh
        allocation.
        """
        table = step.tiles.get(ratio)
        if table is None:
            block_tables = step.block_tables
            if not step.decode and step.requests:
                # Expand only pages visible to a nonempty prefill step,
                # including the cached prefix of a chunk/TBO microbatch. The
                # persistent table can hold a 1M context even for an 8k prompt.
                # Decode/verify and empty steps retain the full table width.
                end = max(request.end for request in step.requests)
                columns = (
                    end + self.geometry.block_size - 1
                ) // self.geometry.block_size
                block_tables = block_tables[:, :columns]
            table = step.tiles[ratio] = unit_table(
                block_tables,
                step.batch_ids,
                self.geometry.index_blocks_per_page(ratio),
                workspace=step.tile_workspace or self.workspace,
                ratio=ratio,
            )
        return table

    def candidate_blocks(self, step, source, ratio):
        """Layer `source`'s candidates bound to their requests' PAGEs at this
        ratio: one per forward, shared by every REINDEX layer they bound.
        Memoized on the step for the reason `unit_tiles` is."""
        bound = step.candidate_blocks.get(source)
        if bound is None:
            bound = step.candidate_blocks[source] = bind_candidates(
                step.candidates[source],
                step.block_tables,
                step.batch_ids,
                self.geometry.index_blocks_per_page(ratio),
                step.visible[ratio],
                rows_per_block=self.geometry.index_block_rows,
            )
        return bound

    def _scatter_rows(self, pages, step, value, ratio):
        """Scatter plan rows with V4's dtype-agnostic, sentinel-aware writer.

        PAGE fields have gaps between pages. Give the existing row scatter a
        zero-copy view whose row stride is one element, so destination indices
        are element offsets into this field's span. This retains the real page
        stride without flattening/copying the field or compacting live rows on
        the host during CUDA graph capture.
        """
        per_page = pages.shape[1]
        plan = step.plans[ratio].compress_plan_gpu
        batch = plan[:, 1].long()
        rows = plan[:, 2].long() // ratio
        page = step.block_tables[batch.clamp_min(0), (rows // per_page).clamp_min(0)]
        offsets = page.long() * pages.stride(0) + (rows % per_page) * pages.stride(1)
        last = (pages.shape[0] - 1) * pages.stride(0) + (per_page - 1) * pages.stride(1)
        pool = pages.as_strided((last + 1, pages.shape[-1]), (1, 1))
        rows_in = pack_rows(*value) if isinstance(value, tuple) else value
        scatter = swa_scatter_rows if pages.is_cuda else swa_scatter_rows_reference
        scatter(rows_in[0], offsets, batch, pool)

    def compress_state(self, owner):
        """This owner's `(kv_state, score_state)`, each `[slots, ring, dim]`.

        Straight out of the arena, so a relocated request carries its
        incomplete group with the rest of its state.
        """
        index = self.compress_indices[owner]
        return (
            self.state.view("compress_kv")[index],
            self.state.view("compress_score")[index],
        )

    def rope_positions(self, step):
        return step.positions

    def read_window(self, layer, slots):
        """Materialize only these requests' bounded context for block drafting."""
        self.require_committed()
        if not self.window_fp8:
            return self.state.view("window")[layer, slots.long()]
        return self._decode_windows(layer, slots)

    def read_windows(self, layers, slots):
        """Every stage's window for these requests, in one gather.

        The draft's stages read the same rows of the same tensor and differ
        only in which layer they read, so indexing them one at a time spent a
        gather and a `.long()` cast per stage on work that broadcasts.

        `layers` is a device tensor so it can be built once and kept; the
        result is `[len(layers), len(slots), ring_slots, head_dim]`.
        """
        self.require_committed()
        if self.window_fp8:
            return self._decode_windows(layers.long()[:, None, None], slots)
        # No `.long()`: the slot table is int32 and the kernel widens each
        # index itself, so casting the whole tensor first was a copy of it.
        return gather_window_rows(self.state.view("window"), layers, slots)

    def _decode_windows(self, layers, slots):
        """FP8 windows decoded to BF16: `[slots, ring, dim]` for an int layer,
        `[layers, slots, ring, dim]` for a `[layers, 1, 1]` device column.

        A layer's ring starts `layer` rings past layer 0's, so the layers move
        the addresses without a host read: a draft graph captures this.
        """
        first = self.geometry.window_bytes(0, self.num_pages)
        addresses = (
            first.ring_start
            + layers * (first.ring_slots * first.run_rows)
            + slots.long()[:, None] * first.slot_rows
            + torch.arange(first.ring_slots, device=slots.device) * first.run_rows
        )
        tagged = ((addresses << 1) | 1).flatten()
        # [0, rows] built on the device: no host copy inside a capture
        ptr = torch.arange(2, dtype=torch.int32, device=slots.device) * tagged.numel()
        dim = self.geometry.head_dim
        output = torch.empty(
            tagged.numel(), dim, dtype=torch.bfloat16, device=slots.device
        )
        gather_prefix_rows(self.backing, tagged, ptr, output, 0, 1)
        return output.view(*addresses.shape, dim)

    def query_fp8(self, query):
        """`(values, scales)` for `query`'s MXFP8 copy, which the FP8 pool's
        attention reads (`rope_quant_window`'s `query_fp8`); None otherwise."""
        if not self.kv_fp8:
            return None
        *rows, dim = query.shape
        return (
            torch.empty_like(query, dtype=FP8_DTYPE),
            torch.empty(
                (*rows, dim // 32), device=query.device, dtype=torch.float8_e8m0fnu
            ),
        )

    def rope_quant_window(self, layer, query, kv, rope, step, query_fp8=None):
        """Rotate the query, quantize the KV row, and store it where it can be.

        A decode stores the row and produces nothing: no query of a decode
        attends to it, so the window write folds in here. A prefill produces
        the row and stores nothing, because the queries in this very chunk have
        yet to read the rows a store would overwrite -- that write stays behind
        attention, in `write_window`. The cache layout only decides the
        payload, so both halves of that read the same either way.

        Returns the rows an extend pass attends to -- BF16, or on the FP8 pool
        `[tokens, row_bytes]` MXFP8 rows at the pool's row stride -- and
        whatever the window still owes, `None` for each the caller has no use
        for. `query` None: the KV row alone, for a prefill whose queries are
        not its rows (`rotate_query`). `query_fp8`: `query_fp8(query)`, the
        rotated query's MXFP8 destination.
        """
        dim, window_fp8 = self.geometry.head_dim, self.window_fp8

        if not step.width:
            return (None, None) if step.decode else (torch.empty_like(kv), None)
        seam = (
            (
                kv.new_empty(step.width, 0, dim)
                if query is None
                else query.view(step.width, -1, dim)
            ),
            kv.view(step.width, dim),
            rope.cos_cache,
            rope.sin_cache,
            step.positions,
        )
        if step.decode:
            # An FP8 window is stored as bytes, a BF16 one into the
            # `[rows, head_dim]` pool, which is the whole difference.
            ring = (
                (self.backing, step, self.geometry.window_bytes(layer, self.num_pages))
                if window_fp8
                else (self.pool, step, self.geometry.window(layer, self.num_pages))
            )
            rope_quant_window(
                *seam, rope_dim=rope.rope_dim, ring=ring, query_fp8=query_fp8
            )
            return None, None
        if self.kv_fp8:
            # Values then scales a row, the pool's own row: the attention's
            # extend source has to share the prefix's strides.
            rows = torch.empty(
                (step.width, self.geometry.row_bytes),
                device=kv.device,
                dtype=torch.uint8,
            )
            values = rows[:, :dim].view(FP8_DTYPE)
            scales = rows[:, dim:].view(torch.float8_e8m0fnu)
            rope_quant_window(
                *seam,
                rope_dim=rope.rope_dim,
                values=values,
                scales=scales,
                query_fp8=query_fp8,
            )
            return rows, (values, scales)
        qat = torch.empty_like(kv)
        values = torch.empty_like(kv, dtype=FP8_DTYPE) if window_fp8 else None
        scales = (
            torch.empty(
                (*kv.shape[:-1], dim // 32),
                device=kv.device,
                dtype=torch.float8_e8m0fnu,
            )
            if window_fp8
            else None
        )
        rope_quant_window(
            *seam, rope_dim=rope.rope_dim, qat=qat, values=values, scales=scales
        )
        return qat, (values, scales) if window_fp8 else qat

    def rotate_query(self, query, rope, positions, query_fp8=None):
        """`rope_quant_window`'s query half alone, at `positions`."""
        if positions.numel():
            rope_quant_window(
                query.view(positions.numel(), -1, self.geometry.head_dim),
                None,
                rope.cos_cache,
                rope.sin_cache,
                positions,
                rope_dim=rope.rope_dim,
                query_fp8=query_fp8,
            )

    def write_window(self, layer, kv, step):
        # `None` is `rope_quant_window` reporting that it already stored the
        # row, not an empty batch: those two differ in whether `step.width` is
        # zero, and only this one can reach here after a real forward.
        if kv is not None and step.width:
            window = self.geometry.window(layer, self.num_pages)
            # rows below where a request's ring already holds them stay
            cached = step.swa_cached_until if step.request_cached_until.any() else None
            if self.window_fp8:
                write_packed_window(
                    *kv,
                    self.backing,
                    step,
                    self.geometry.window_bytes(layer, self.num_pages),
                    self.geometry.head_dim,
                    cached,
                )
                return
            swa_write(
                kv.flatten(0, 1),
                step.positions,
                step.cu_seqlens_q,
                step.slots,
                self.pool,
                window,
                # The bucket, not the batch: this is the kernel's grid, and a
                # replay runs the one capture recorded. A padding request is
                # zero-length in `cu_seqlens_q`, which is what keeps it from
                # writing anything at all.
                min(step.max_q_len, window.ring_slots),
                cached_until=cached,
            )

    def attention_indices(self, spec, step):
        """This layer's `(prefix, pptr, second, sptr)`: rows into the pool, and
        rows into the attention's second source.

        Prefill and decode alike build their index group's in one launch:
        every layer's prefix, window included, or on a split decode
        (`splits_window`) the selection alone, one plane the group shares. A
        prefill's second source is its own KV (`extend`, one list for the
        group); a decode's is the layer's ring (`ring_view`), and a split
        decode's window rows one list for every layer.

        The group's first layer is also the layer that owns its selection, so
        by the time anyone asks, every input the run shares already exists.
        Later layers find theirs written and launch nothing.
        """
        # a spec that declares no group never names a start: its own
        first = spec.index_group_start if spec.index_group_size > 1 else spec.layer_id
        split = splits_window(step, self.geometry)
        built = step.group_indices.get(first)
        if built is None:
            layers = 1 if split else spec.index_group_size
            built = self._index_group(spec, step, first, layers)
            step.group_indices[first] = built
        prefix, pptr, extend, eptr = built
        if not split:
            return prefix[spec.layer_id - first], pptr, extend, eptr
        if step.window_rows is None:
            window = self.geometry.window(first, self.num_pages)
            step.window_rows = build_window_rows(step, self.geometry, window, eptr)
        return prefix[0], pptr, step.window_rows, eptr

    def ring_view(self, layer):
        """The row-addressed pool (BF16 or FP8 rows) from `layer`'s ring
        start: a split decode's second source, which its window rows index."""
        return self.pool[self.geometry.window(layer, self.num_pages).ring_start :]

    def _index_group(self, spec, step, first, layers):
        """One build covering `layers` consecutive layers from `first`.

        The per-layer ring offset is read off the geometry rather than
        rebuilt here, so the kernel's stride and `window()` cannot drift.
        """
        window = self.geometry.window(first, self.num_pages)
        stride = (
            0
            if layers == 1
            else self.geometry.window(first + 1, self.num_pages).ring_start
            - window.ring_start
        )
        return build_indices(
            step.selected[spec.topk_owner] if spec.ratio else None,
            self.indptrs(step)[spec.ratio],
            step,
            self.geometry,
            window,
            spec.kv_owner,
            spec.ratio,
            layers,
            stride,
        )
