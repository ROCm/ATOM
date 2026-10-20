# SPDX-License-Identifier: MIT
"""Accepted-prefix selection for Engram history.

The window has physical slack, so rejecting rows only changes visibility, and
the compressor's ring is widened by the same slack -- a rejected round's writes
land past the window the next round reads, so neither needs a rollback. Engram
history is the one thing left that cannot be reconstructed from KV: it is
staged per input prefix and the sampler picks one per request before
checkpointing or drafting.

Every prefix is built on the host, out of the cursor image `prepare_state`
already read back and the compressed ids Engram already produced there, and
reaches the device as one pinned copy. Building them device-side instead cost
a blocking `hipMemcpyWithStream` per request -- `torch.as_tensor(list, device=)`
is pageable, so torch synchronizes the stream before it copies.
"""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _commit_cursors(
    cursor,
    staging,
    anchors,
    starts,
    slots,
    slot_stride,
    request_stride,
    prefix_stride,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One request per program: its accepted prefix's cursor into its slot."""
    i = tl.program_id(0)
    prefix = tl.load(anchors + i) - tl.load(starts + i)
    slot = tl.load(slots + i).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    keep = cols < WIDTH
    row = tl.load(
        staging + i * request_stride + prefix * prefix_stride + cols, mask=keep
    )
    tl.store(cursor + slot * slot_stride + cols, row, mask=keep)


class TentativeState:
    def __init__(self, cache, step, histories=None):
        self.cache, self.step = cache, step
        self.request_indices = {span.slot: i for i, span in enumerate(step.requests)}
        self.staging = cache.tentative_staging
        self.histories = histories
        self.written = set()
        # Set instead of `written` when a kernel filled the plane: one launch
        # over every request rather than a staged row per request, and nothing
        # for the host to have forgotten.
        self.staged_on_device = False

    def stage_history(self, span, compressed_ids):
        ids = np.asarray(compressed_ids, dtype=np.int64)
        if ids.shape != (span.length,):
            raise ValueError(
                "Tentative Engram history requires one compressed ID per token"
            )
        if self.histories is None:
            raise RuntimeError(
                "Host Engram staging needs the committed history; this step was "
                "prepared without it, so its candidates belong to the kernel"
            )
        i = self.request_indices[span.slot]
        rows = self.staging.np[i, : span.length]
        rows[:, 0] = np.arange(span.position + 1, span.end + 1)
        # Prefix `t` is the history after `t + 1` more ids: slide a window of
        # the history's own size along `[history | ids]`, dropping the one that
        # takes no id at all.
        history = self.histories[i]
        rows[:, 1:] = np.lib.stride_tricks.sliding_window_view(
            np.concatenate((history, ids)), history.size
        )[1:]
        self.written.add(span.slot)

    def commit(self, anchors):
        """`anchors`: each request's flat row of its last accepted token."""
        if not self.staged_on_device and self.written != set(self.request_indices):
            raise RuntimeError("Tentative state is missing an Engram prefix")

        count = self.step.scheduled_bs
        if anchors.shape != (count,):
            raise ValueError("Anchors must match the scheduled requests")
        cursors = (
            self.staging.gpu[:count]
            if self.staged_on_device
            else self.staging.copy_to_gpu(count)
        )
        # An anchor outside its own segment is silent here: it reads another
        # prefix's row, so a wrong anchor shows up as wrong tokens, not a fault.
        # The scheduled prefix, not the forward's width: a padding request owns
        # no slot, and the 0 standing in for one is a live request's.
        starts, slots = self.step.cu_seqlens_q[:count], self.step.slots[:count]
        cursor = self.cache.cursor
        if not cursor.is_cuda:
            cursor[slots.long()] = cursors[torch.arange(count), anchors - starts]
            return
        _commit_cursors[(count,)](
            cursor,
            cursors,
            anchors,
            starts,
            slots,
            cursor.stride(0),
            cursors.stride(0),
            cursors.stride(1),
            WIDTH=cursor.shape[1],
            BLOCK=triton.next_power_of_2(cursor.shape[1]),
        )
