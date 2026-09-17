# SPDX-License-Identifier: MIT
"""PAGE-backed global keys and complete, relocatable CSA2 request state."""

import numpy as np
import torch
import torch.nn.functional as F
from aiter.ops.cache import (
    cp_gather_indexer_k_quant_cache,
    indexer_k_quant_and_cache,
)
from atom.model_ops.attentions.deepseek_v41.packed_rows import (
    gather_index_rows,
    gather_prefix_rows,
    pack_rows,
    write_packed_window,
)
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
    INDEX_FP8_SCALE_FMT,
    MAIN_FP4,
)
from atom.model_ops.blockscale import quantize_fp4
from atom.model_ops.deepseek_v41.compressor import compress_batch
from atom.model_ops.deepseek_v41.indexer import TensorIndexKeys
from atom.model_ops.deepseek_v41.paged_scoring import unit_table

from atom.model_ops.attentions.pool_layout.entry_arena import EntryMajorArena
from atom.model_ops.attentions.pool_layout.v4_pool_fields import (
    MQA_LOGITS_PRESHUFFLE_ROWS,
)
from atom.model_ops.v4_kernels import make_compress_plans
from atom.model_ops.v4_kernels.state_writes import swa_write
from atom.utils import CpuGpuBuffer

from .indices import build_indices, fill_step_indptrs
from .metadata import prepare_batch_step
from .speculative import TentativeState


class PagedIndexKeys:
    """Read only a scoring tile or the selected candidate rows."""

    def __init__(self, pages, block_table, count, packed_dim=None):
        self.pages, self.block_table = pages, block_table
        self.packed_dim = packed_dim
        self.shape = (1, count, packed_dim or pages.shape[-1])

    def _read(self, ids):
        if self.packed_dim is not None:
            return gather_index_rows(self.pages, self.block_table, ids, self.packed_dim)
        rows = self.pages.shape[1]
        blocks = self.block_table[ids // rows].long()
        return self.pages[blocks, ids % rows]

    def tile(self, start, end):
        ids = torch.arange(start, end, device=self.pages.device)
        return self._read(ids).unsqueeze(0)

    def gather(self, ids):
        return self._read(ids)


class RequestCache:
    def __init__(self, cache, span, blocks):
        self.cache, self.span, self.blocks = cache, span, blocks
        self.packed = cache.geometry.packed
        self.index_dtype = cache.geometry.index_dtype

    def write_global(self, owner, begin, main, index):
        planes = (self.cache.pages.view(f"main_{owner}")[0], main), (
            self.cache.index_planes[owner],
            index,
        )
        for pages, value in planes:
            if isinstance(value, tuple):
                value = pack_rows(*value)
            rows = pages.shape[1]
            ids = torch.arange(begin, begin + value.shape[1], device=value.device)
            pages[self.blocks[ids // rows].long(), ids % rows] = value[0]

    def index_keys(self, owner, count):
        if self.index_dtype == "fp8":
            # Preshuffled rows are not addressable one at a time; aiter's own
            # inverse of the writer is what turns them back into a tile a
            # scorer can read.
            return TensorIndexKeys(self.cache.gather_index(owner, self.blocks, count))
        return PagedIndexKeys(
            self.cache.index_planes[owner],
            self.blocks,
            count,
            self.cache.geometry.index_dim if self.index_dtype == "fp4" else None,
        )


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
        self.index_ratios = dict(geometry.owners)
        self.index_planes = {
            owner: self._index_plane(
                index_offsets[owner], geometry.rows_per_page(ratio)
            )
            for owner, ratio in geometry.owners
        }
        # The same bytes as `[tiles, tile rows, width]`, which is what a block
        # id addresses. FP8 only, and not for want of generality: the other
        # formats are read a row at a time and their PAGE need not hold a whole
        # number of tiles, so there is no such view of them to take.
        self.index_units = (
            {
                owner: plane.view(-1, MQA_LOGITS_PRESHUFFLE_ROWS, plane.shape[-1])
                for owner, plane in self.index_planes.items()
            }
            if geometry.index_dtype == "fp8"
            else {}
        )
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
        # a request could have accepted, plus the length that bounds it.
        self.tentative_staging = CpuGpuBuffer(
            slots,
            geometry.speculative_tokens + 1,
            self.cursor.shape[1],
            dtype=self.cursor.dtype,
            device=device,
            pin_memory=pinned,
        )
        self.tentative_limits = CpuGpuBuffer(
            slots, dtype=self.cursor.dtype, device=device, pin_memory=pinned
        )
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
        """`[pages, rows, width]` at `offset`, in whatever the rows are stored as.

        `as_strided`'s storage offset is absolute, so the retyped view's own
        has to be added -- omit it and every plane addresses from the front of
        the pool, over the main pages.
        """
        dtype = torch.bfloat16 if self.geometry.index_dtype == "bf16" else torch.uint8
        typed = self.backing.view(dtype)
        width = self.geometry.index_row_bytes // dtype.itemsize
        return typed.as_strided(
            (self.num_pages, rows, width),
            (rows * width, width, 1),
            typed.storage_offset() + offset // dtype.itemsize,
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
        step.indptrs = fill_step_indptrs(
            step, self.geometry, self.indptr_buffers, step.longest
        )
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
        buffers = {
            ratio: {
                name: CpuGpuBuffer(
                    rows,
                    4,
                    dtype=torch.int32,
                    device=device,
                    pin_memory=device.type != "cpu",
                )
                for name in ("compress", "write")
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

    def prepare_state(self, step):
        """Read restored cursors once; reset recycled slots before any layer writes."""
        self.require_committed()
        cursors = self.cursor[step.slots[: step.scheduled_bs].long()].cpu().numpy()
        for i, span in enumerate(step.requests):
            if span.position == 0:
                self.state_bytes[span.slot].zero_()
                self.cursor[span.slot, 1:].fill_(-1)
                cursors[i, 0], cursors[i, 1:] = 0, -1
            elif cursors[i, 0] != span.position:
                raise ValueError(
                    f"Request {span.request_id} needs state at {span.position}, "
                    f"found {cursors[i, 0]}; replay from a recoverable boundary"
                )
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
        latent, rows, rotated = compress_batch(
            self, owner, compressor, values, scores, step, rope, scatter=scatter
        )
        if rotated is not None:
            packed = quantize_fp4(rotated, **MAIN_FP4)
            self._scatter_rows(
                self.pages.view(f"main_{owner}")[0],
                step,
                rows,
                packed,
                compressor.ratio,
            )
        return latent, rows

    def write_index(self, owner, step, rows, index, ratio):
        geometry = self.geometry
        per_page = geometry.rows_per_page(ratio)
        if geometry.index_dtype == "fp8":
            # One pass quantizes and preshuffles, addressing a row by its
            # position in the plane -- which a page and an offset into it come
            # to, the plane being dense.
            physical, offsets, _ = self._plan_destinations(step, rows, ratio, per_page)
            indexer_k_quant_and_cache(
                index[0],
                self.index_units[owner],
                physical * per_page + offsets,
                geometry.index_dim,
                INDEX_FP8_SCALE_FMT,
                preshuffle=True,
            )
            return
        if geometry.index_dtype == "fp4":
            index = quantize_fp4(index)
        self._scatter_rows(self.index_planes[owner], step, rows, index, ratio)

    @property
    def index_dtype(self):
        return self.geometry.index_dtype

    def scores_paged(self, step):
        """Whether this step's top-k can come from the plane, not from a tile."""
        return self.geometry.scores_paged(step.decode)

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
                self.geometry.rows_per_page(ratio) // MQA_LOGITS_PRESHUFFLE_ROWS,
            )
        return table

    def gather_index(self, owner, blocks, count):
        """This request's first `count` index rows, dequantized to BF16.

        aiter's own inverse of the preshuffling writer, so the two cannot
        disagree about the layout. The unit table is the PAGE table expanded:
        a page holds a fixed number of tiles, consecutively.
        """
        geometry = self.geometry
        ratio = self.index_ratios[owner]
        per_page = geometry.rows_per_page(ratio) // MQA_LOGITS_PRESHUFFLE_ROWS
        device = self.pool.device
        units = blocks.long()[:, None] * per_page + torch.arange(
            per_page, device=device
        )
        keys = torch.empty(
            (count, geometry.index_dim), dtype=torch.float8_e4m3fn, device=device
        )
        scales = torch.empty((count, 1), dtype=torch.float32, device=device)
        cp_gather_indexer_k_quant_cache(
            self.index_units[owner],
            keys,
            scales.view(torch.float8_e4m3fn),
            units.flatten().int()[None],
            torch.tensor([0, count], dtype=torch.int32, device=device),
            preshuffle=True,
        )
        # In FP32 and back, not BF16 times FP32: the product of a stored value
        # and its scale is what the scorer's own arithmetic uses, and a BF16
        # intermediate would round twice.
        return (keys.to(torch.float32) * scales).to(torch.bfloat16).unsqueeze(0)

    def _plan_destinations(self, step, rows, ratio, per_page):
        """Each plan row's physical page and its offset in that page.

        A plan's rows come from several requests at once, so the page comes
        from the row's own `batch_id` rather than one request's contiguous run.

        A plan cut to a CUDAGraph's fixed grid ends in sentinel rows, `-1` in
        both fields, and they stay negative: `-1 * per_page + (per_page - 1)`
        is again `-1`, the row index V4's writers already skip. `live` is for
        the one writer that cannot skip on its own -- torch advanced indexing,
        where a negative index is legal and lands on somebody's live row.
        """
        batch = step.plans[ratio].compress_plan_gpu[: rows.numel(), 1].long()
        live = batch >= 0
        pages = step.block_tables[batch.clamp_min(0), (rows // per_page).clamp_min(0)]
        return torch.where(live, pages.long(), -1), rows % per_page, live

    def _scatter_rows(self, pages, step, rows, value, ratio):
        """One destination per plan row, resolved from that same plan.

        A pair is native value and scale bytes, which is what the caller has
        when the plane it is writing is a quantized one -- the plane's own
        dtype is not asked, because the value already answered it.
        """
        physical, offsets, live = self._plan_destinations(
            step, rows, ratio, pages.shape[1]
        )
        rows_in = pack_rows(*value) if isinstance(value, tuple) else value
        pages[physical[live], offsets[live]] = rows_in[0][live]

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

    def requests(self, step):
        for i, (span, local_step) in enumerate(zip(step.requests, step.request_steps)):
            yield (
                RequestCache(self, span, step.block_tables[i]),
                local_step,
                span.token_slice,
            )

    def write_window(self, layer, kv, step):
        if step.width:
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
        selected = None
        if spec.ratio:
            if spec.topk_owner not in step.selected:
                parts = [local.indices[spec.topk_owner] for local in step.request_steps]
                # The width the indptr reserved, not the widest part this
                # batch produced: the reserve was taken before any scorer ran.
                topk = self.geometry.batch_topk(spec.ratio, step.longest, step.decode)
                selection = torch.cat(
                    [
                        F.pad(part, (0, topk - part.shape[-1]), value=-1)
                        for part in parts
                    ],
                    dim=1,
                )
                # Out to the forward's width: the tiled scorer only ran on the
                # rows a request owns, and `build_indices` reads one row per
                # row the forward runs. A padding row selects nothing.
                step.selected[spec.topk_owner] = F.pad(
                    selection, (0, 0, 0, step.width - selection.shape[1]), value=-1
                )
            selected = step.selected[spec.topk_owner]
        return build_indices(
            selected,
            step,
            self.geometry,
            self.geometry.window(spec.layer_id, self.num_pages),
            spec.kv_owner,
            spec.ratio,
        )
