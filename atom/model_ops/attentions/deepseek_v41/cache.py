# SPDX-License-Identifier: MIT
"""PAGE-backed global keys and complete, relocatable CSA2 request state."""

import numpy as np
import torch

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
from atom.model_ops.blockscale import quantize_fp4
from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE
from atom.model_ops.deepseek_v41.compressor import compress_batch
from atom.model_ops.deepseek_v41.dspark import gather_window_rows
from atom.model_ops.deepseek_v41.index_write import write_index_rows
from atom.model_ops.deepseek_v41.rope_window import rope_quant_window
from atom.model_ops.deepseek_v41.unit_table import unit_table
from atom.model_ops.v4_kernels import make_compress_plans
from atom.model_ops.v4_kernels.state_writes import (
    swa_scatter_rows,
    swa_scatter_rows_reference,
    swa_write,
)
from atom.utils import CpuGpuBuffer

from .indices import build_indices, fill_step_indptrs
from .metadata import prepare_batch_step
from .speculative import TentativeState


class PagedAttentionCache:
    def __init__(self, geometry, pages, slots, device, max_tokens=0):
        if pages < 1 or slots < 1:
            raise ValueError("A paged cache needs positive PAGE and STATE capacities")
        self.geometry, self.num_pages, self.num_slots = geometry, pages, slots
        self.packed = geometry.packed
        self.indptr_device, self.max_tokens, self.indptr_buffers = device, 0, {}
        index_offsets, boundary = geometry.paged_extents(pages)
        size = boundary + slots * geometry.state_bytes
        self.backing = torch.zeros(size, dtype=torch.uint8, device=device)
        main_bytes = self.num_pages * geometry.page_bytes
        self.pages = EntryMajorArena(
            geometry.page_fields,
            self.num_pages,
            device,
            buf=self.backing[:main_bytes],
            slot_stride=geometry.page_bytes,
        )
        # One plane per owner, `[pages, rows, width]`, dense in its own rows:
        # the stride a paged reader is handed is the rows' and not the PAGE's.
        # Both sides of the index plane take the view from here.
        self.index_planes = {
            owner: self._index_plane(
                index_offsets[owner], geometry.rows_per_page(ratio)
            )
            for owner, ratio in geometry.owners
        }
        # The same bytes as `[tiles, tile rows, width]`, which is what a block
        # id addresses and what the scorer is handed.
        self.index_units = {
            owner: plane.view(-1, geometry.index_block_rows, plane.shape[-1])
            for owner, plane in self.index_planes.items()
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
        self.pool = (
            self.backing.view(-1, 1)
            if geometry.packed
            else self.backing.view(torch.bfloat16).view(-1, geometry.head_dim)
        )
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
            for plane in self.index_planes.values()
        ]
        return regions

    def unit_views(self, unit):
        """The same regions for one unit, named as `uint8` views."""
        return [self.page_bytes[unit]] + [
            plane[unit].flatten().view(torch.uint8)
            for plane in self.index_planes.values()
        ]

    def _index_plane(self, offset, rows):
        """`[pages, rows, bytes]` at `offset`, untyped.

        A preshuffled FP8 row interleaves its bytes across the tile and carries
        its scale past the block's data, so a row has no element type to take a
        view in -- the writer and the scorer both address it as bytes.

        `as_strided`'s storage offset is absolute, so the retyped view's own has
        to be added -- omit it and every plane addresses from the front of the
        pool, over the main pages.
        """
        typed = self.backing.view(torch.uint8)
        width = self.geometry.index_row_bytes
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
        tentative=False,
        buffers=None,
        running_bs=None,
        running_tokens=None,
        max_q_len=None,
        state_slot_out=None,
        plans=None,
    ):
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
        for span in requests:
            if span.length <= 0 or span.position < 0 or span.offset != offset:
                raise ValueError(
                    "Request spans must be nonempty and partition the token batch"
                )
            if not 0 <= span.slot < self.num_slots or span.slot in seen:
                raise ValueError("Each request needs its own valid STATE slot")
            needed = -(-span.end // self.geometry.block_size)
            if len(span.block_ids) < needed or any(
                block < 0 or block >= self.num_pages for block in span.block_ids
            ):
                raise ValueError("Request PAGE table is incomplete or out of range")
            seen.add(span.slot)
            offset += span.length
        step = prepare_batch_step(
            requests,
            self.pool.device,
            tentative=tentative,
            buffers=buffers,
            running_bs=running_bs,
            running_tokens=running_tokens,
            max_q_len=max_q_len,
            state_slot_out=state_slot_out,
            ratios=tuple(ratio for ratio, _ in self.geometry.compress_ratios),
        )
        step.plans = (
            self._private_plans(requests, tentative) if plans is None else plans
        )
        # Per-forward and layer-invariant, so built here rather than by the
        # first layer to want one, exactly as V4 builds its own three. Triton,
        # like every reader of them, so a CPU pool has neither.
        if not step.positions.is_cuda:
            return step
        self._reserve_indptrs(step.width)
        step.indptrs = fill_step_indptrs(step, self.geometry, self.indptr_buffers)
        return step

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

    def _report_stale_state(self, cursors, starts, requests):
        """Raise on the first request whose state is not where it is wanted.

        Fresh requests are excluded by their own position: the slot they are
        about to reuse holds whatever the last tenant left, and the reset below
        is what makes it theirs.
        """
        wrong = np.flatnonzero((starts != 0) & (cursors[:, 0] != starts))
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
        starts, requests = self._probe_claim
        self._probe_claim = None
        self._report_stale_state(self._probe[: starts.size].numpy(), starts, requests)

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
        if histories:
            cursors = (
                torch.index_select(self.cursor, 0, step.slots[:count]).cpu().numpy()
            )
        else:
            # Issued before the reset below, so a recycled slot is captured as
            # its old tenant left it -- which is why the judge skips position 0.
            self._probe[:count].copy_(
                torch.index_select(self.cursor, 0, step.slots[:count]),
                non_blocking=True,
            )
            if self._probe_done is not None:
                self._probe_done.record()
            self._probe_claim = (starts.copy(), ids)
            cursors = None
        fresh = starts == 0
        if fresh.any():
            # Both resets take the same index, so the slots cross once.
            slots = self._reset_slots.np[: int(fresh.sum())]
            slots[:] = [span.slot for span, new in zip(step.requests, fresh) if new]
            index = self._reset_slots.copy_to_gpu(slots.size)
            self.state_bytes.index_fill_(0, index, 0)
            self.cursor[:, 1:].index_fill_(0, index, -1)
            if cursors is not None:
                cursors[fresh, 0] = 0
                cursors[fresh, 1:] = -1
        if cursors is None:
            if step.tentative:
                self.pending = TentativeState(self, step)
            return None
        self._report_stale_state(cursors, starts, ids)
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

    def commit_tentative(self, step, accepted_lengths):
        if self.pending is None or self.pending.step is not step:
            raise RuntimeError("Tentative state was not prepared for this step")
        self.pending.commit(accepted_lengths)
        self.pending = None

    def compress(self, owner, compressor, values, scores, step, rope):
        # The packed pool interleaves FP4 with its scales, which the kernel's
        # BF16 scatter cannot write; take the rotated echo and pack it here.
        scatter = (
            None
            if self.packed
            else (self.pages.view(f"main_{owner}")[0], step.block_tables)
        )
        latent, rotated = compress_batch(
            self, owner, compressor, values, scores, step, rope, scatter=scatter
        )
        if rotated is not None:
            packed = quantize_fp4(rotated, **MAIN_FP4)
            self._scatter_rows(
                self.pages.view(f"main_{owner}")[0],
                step,
                packed,
                compressor.ratio,
            )
        return latent

    def write_index(self, owner, step, index, ratio):
        """One pass quantizes, preshuffles and scatters.

        The kernel resolves a row's address from the plan and the PAGE table
        itself, so nothing here computes one -- `ratio` is all it needs to
        turn the plan's position into a compressed row.
        """
        write_index_rows(
            index[0],
            self.index_planes[owner],
            step.plans[ratio].compress_plan_gpu,
            step.block_tables,
            self.geometry.rows_per_page(ratio),
            ratio=ratio,
            rows_per_block=self.geometry.index_block_rows,
            scale_fmt=INDEX_FP8_SCALE_FMT,
        )

    def unit_tiles(self, step, ratio):
        """Tile ids per query token: one table per ratio, shared by its owners.

        Memoized on the step and dropped by `begin_forward` rather than built
        with it, unlike the indptrs: these rows are a fresh allocation, so a
        table built outside the graph is one a replay reads at the capture's
        address.
        """
        table = step.tiles.get(ratio)
        if table is None:
            table = step.tiles[ratio] = unit_table(
                step.block_tables,
                step.batch_ids,
                self.geometry.rows_per_page(ratio) // self.geometry.index_block_rows,
            )
        return table

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
        if not self.packed:
            return self.state.view("window")[layer, slots.long()]
        window = self.geometry.window(layer, self.num_pages)
        addresses = (
            window.ring_start
            + slots.long()[:, None] * window.slot_rows
            + torch.arange(window.ring_slots, device=slots.device) * window.run_rows
        )
        tagged = ((addresses << 1) | 1).flatten()
        ptr = torch.tensor([0, tagged.numel()], dtype=torch.int32, device=slots.device)
        output = torch.empty(
            tagged.numel(),
            self.geometry.head_dim,
            dtype=torch.bfloat16,
            device=slots.device,
        )
        gather_prefix_rows(self.backing, tagged, ptr, output, 0, 1)
        return output.view(slots.numel(), window.ring_slots, self.geometry.head_dim)

    def read_windows(self, layers, slots):
        """Every stage's window for these requests, in one gather.

        The draft's stages read the same rows of the same tensor and differ
        only in which layer they read, so indexing them one at a time spent a
        gather and a `.long()` cast per stage on work that broadcasts. The
        packed pool keeps the per-layer path: its rows are addressed through
        the ring geometry rather than indexed, and each layer's `ring_start`
        is its own.

        `layers` is a device tensor so it can be built once and kept; the
        result is `[len(layers), len(slots), ring_slots, head_dim]`.
        """
        self.require_committed()
        if self.packed:
            return torch.stack(
                [self.read_window(int(layer), slots) for layer in layers]
            )
        # No `.long()`: the slot table is int32 and the kernel widens each
        # index itself, so casting the whole tensor first was a copy of it.
        return gather_window_rows(self.state.view("window"), layers, slots)

    def rope_quant_window(self, layer, query, kv, rope, step):
        """Rotate the query, quantize the KV row, and store it where it can be.

        A decode stores the row and produces nothing: no query of a decode
        attends to it, so the window write folds in here. A prefill produces
        the row and stores nothing, because the queries in this very chunk have
        yet to read the rows a store would overwrite -- that write stays behind
        attention, in `write_window`. The cache layout only decides the
        payload, so both halves of that read the same either way.

        Returns the BF16 row an extend pass attends to and whatever the window
        still owes, `None` for each the caller has no use for.
        """
        dim, packed = self.geometry.head_dim, self.packed
        if not step.width:
            return (None, None) if step.decode else (torch.empty_like(kv), None)
        seam = (
            query.view(step.width, -1, dim),
            kv.view(step.width, dim),
            rope.cos_cache,
            rope.sin_cache,
            step.positions,
        )
        if step.decode:
            rope_quant_window(
                *seam,
                rope_dim=rope.rope_dim,
                # The packed pool is byte-addressed and the BF16 one is a
                # `[rows, head_dim]` view, which is the whole difference.
                ring=(
                    self.backing if packed else self.pool,
                    step,
                    self.geometry.window(layer, self.num_pages),
                    packed,
                ),
            )
            return None, None
        qat = torch.empty_like(kv)
        values = torch.empty_like(kv, dtype=FP8_DTYPE) if packed else None
        scales = (
            torch.empty(
                (*kv.shape[:-1], dim // 32),
                device=kv.device,
                dtype=torch.float8_e8m0fnu,
            )
            if packed
            else None
        )
        rope_quant_window(
            *seam, rope_dim=rope.rope_dim, qat=qat, values=values, scales=scales
        )
        return qat, (values, scales) if packed else qat

    def write_window(self, layer, kv, step):
        # `None` is `rope_quant_window` reporting that it already stored the
        # row, not an empty batch: those two differ in whether `step.width` is
        # zero, and only this one can reach here after a real forward.
        if kv is not None and step.width:
            window = self.geometry.window(layer, self.num_pages)
            if self.packed:
                write_packed_window(
                    *kv, self.backing, step, window, self.geometry.head_dim
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
            )

    def attention_indices(self, spec, step):
        """This layer's prefix rows, built with its whole group's on a decode.

        The group's first layer is also the layer that owns its selection, so
        by the time anyone asks, every input the run shares already exists.
        Later layers find theirs written and launch nothing.
        """
        first = spec.index_group_start
        if not step.decode or spec.index_group_size == 1:
            # A run of one, anchored on this layer's own ring, so a spec that
            # declares no group never has to name a start. A prefill takes this
            # too: it pays work rather than dispatch, and its `extend` plane is
            # this layer's alone.
            prefix, pptr, extend, eptr = self._index_group(spec, step, spec.layer_id, 1)
            return prefix[0], pptr, extend, eptr
        built = step.group_indices.get(first)
        if built is None:
            built = self._index_group(spec, step, first, spec.index_group_size)
            step.group_indices[first] = built
        prefix, pptr, extend, eptr = built
        return prefix[spec.layer_id - first], pptr, extend, eptr

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
            step,
            self.geometry,
            window,
            spec.kv_owner,
            spec.ratio,
            layers,
            stride,
        )
