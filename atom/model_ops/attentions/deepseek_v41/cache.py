# SPDX-License-Identifier: MIT
"""PAGE-backed global keys and complete, relocatable CSA2 request state."""

import numpy as np
import torch
import torch.nn.functional as F
from atom.model_ops.attentions.deepseek_v41.packed_rows import (
    gather_index_rows,
    gather_prefix_rows,
    pack_rows,
    write_packed_window,
)
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import MAIN_FP4
from atom.model_ops.blockscale import quantize_fp4
from atom.model_ops.deepseek_v41.compressor import compress_batch

from atom.model_ops.attentions.pool_layout.entry_arena import EntryMajorArena
from atom.model_ops.v4_kernels import make_compress_plans
from atom.model_ops.v4_kernels.state_writes import swa_write
from atom.utils import CpuGpuBuffer

from .indices import build_indices
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

    def write_global(self, owner, begin, main, index):
        for kind, value in (("main", main), ("index", index)):
            if self.packed:
                value = pack_rows(*value)
            pages = self.cache.pages.view(f"{kind}_{owner}")[0]
            rows = pages.shape[1]
            ids = torch.arange(begin, begin + value.shape[1], device=value.device)
            pages[self.blocks[ids // rows].long(), ids % rows] = value[0]

    def index_keys(self, owner, count):
        return PagedIndexKeys(
            self.cache.pages.view(f"index_{owner}")[0],
            self.blocks,
            count,
            self.cache.geometry.index_dim if self.packed else None,
        )


class PagedAttentionCache:
    def __init__(self, geometry, pages, slots, device):
        if pages < 1 or slots < 1:
            raise ValueError("A paged cache needs positive PAGE and STATE capacities")
        self.geometry, self.num_pages, self.num_slots = geometry, pages, slots
        self.packed = geometry.packed
        size = pages * geometry.page_bytes + slots * geometry.state_bytes
        self.backing = torch.zeros(size, dtype=torch.uint8, device=device)
        boundary = pages * geometry.page_bytes
        self.pages = EntryMajorArena(
            geometry.page_fields,
            pages,
            device,
            buf=self.backing[:boundary],
            slot_stride=geometry.page_bytes,
        )
        self.state = EntryMajorArena(
            geometry.state_fields,
            slots,
            device,
            buf=self.backing[boundary:],
            slot_stride=geometry.state_bytes,
        )
        self.state_bytes = self.backing[boundary:].view(slots, geometry.state_bytes)
        self.page_bytes = self.backing[:boundary].view(pages, geometry.page_bytes)
        self.cursor = self.state.view("cursor")[0]
        self.cursor[:, 1:].fill_(-1)
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
            if len(set(span.block_ids)) != len(span.block_ids):
                raise ValueError("A request cannot alias its own PAGE rows")
            seen.add(span.slot)
            offset += span.length
        step = prepare_batch_step(
            requests,
            self.pool.device,
            tentative=tentative,
            buffers=buffers,
            running_bs=running_bs,
            running_tokens=running_tokens,
            state_slot_out=state_slot_out,
        )
        step.plans = (
            self._private_plans(requests, tentative) if plans is None else plans
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
        cursors = self.cursor[step.slots.long()].cpu().numpy()
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
            self.pending = TentativeState(self, step)
        return cursors[:, 1:]

    def finish_step(self, step, histories):
        if step.tentative:
            if self.pending is None or self.pending.step is not step:
                raise RuntimeError("Tentative step was not prepared")
            self.pending.finish()
            return
        if not step.requests:
            return
        cursor = torch.as_tensor(histories, dtype=torch.int64, device=self.pool.device)
        ends = torch.tensor(
            [span.end for span in step.requests], device=self.pool.device
        )
        self.cursor[step.slots.long()] = torch.cat((ends[:, None], cursor), dim=1)

    def commit_tentative(self, accepted_lengths):
        if self.pending is None:
            raise RuntimeError("No tentative state to commit")
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
            self._scatter_rows("main", owner, step, rows, packed, compressor.ratio)
        return latent, rows

    def write_index(self, owner, step, rows, index, ratio):
        self._scatter_rows("index", owner, step, rows, index, ratio)

    def _scatter_rows(self, kind, owner, step, rows, value, ratio):
        """One destination per plan row, resolved from that same plan.

        A plan's rows come from several requests at once, so the page comes
        from the row's own `batch_id` rather than one request's contiguous run.
        """
        pages = self.pages.view(f"{kind}_{owner}")[0]
        per_block = pages.shape[1]
        batch = step.plans[ratio].compress_plan_gpu[: rows.numel(), 1].long()
        physical = step.block_tables[batch, rows // per_block].long()
        pages[physical, rows % per_block] = (
            pack_rows(*value) if self.packed else value[0]
        )

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
        if step.length:
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
                min(step.max_length, window.ring_slots),
            )

    def attention_indices(self, spec, step):
        selected = None
        if spec.ratio:
            if spec.topk_owner not in step.selected:
                parts = [local.indices[spec.topk_owner] for local in step.request_steps]
                width = max((part.shape[-1] for part in parts), default=0)
                step.selected[spec.topk_owner] = torch.cat(
                    [
                        F.pad(part, (0, width - part.shape[-1]), value=-1)
                        for part in parts
                    ],
                    dim=1,
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
