# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Bounded GPU staging buffer, D2H/H2D, and the producer event.

A single-entry counterpart to `BlockGPUConnector`'s block-chunked staging: the
state tier moves one flat entry per transfer, so it needs the buffer, the copy
stream and the producer event, but none of the block/chunk orchestration. The
thread-state and buffer primitives are shared with the KV path.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

from atom.kv_transfer.offload.atom_lmcache_staging import (
    _NullCtx,
    _StagingBuffer,
    _ThreadTransferState,
    memory_object_as_uint8,
)

# ---------------------------------------------------------------------------
# StagedTransfer
# ---------------------------------------------------------------------------


class StagedTransfer:
    """Bounded GPU staging buffer, D2H/H2D, and the producer event.

    The half of the LMCache GPU connector that is not about chunks. KV and
    state both need a bounded device buffer, a copy stream, and an event the
    save worker synchronizes on; neither needs the other's orchestration. The
    chunk layer stays in `ATOMLMCacheGPUConnector` because it is genuinely
    KV-specific: `_iter_transfer_chunks` zips MemoryObjs against block-id
    groups with `strict=True` and sizes each from a startup per-block
    constant, so a single object of a different size breaks both invariants.
    State is not a member of that loop.

    **This class does not fence the producer; the caller keeps the source
    quiescent.** Nothing here waits on the forward's compute stream: `pack`
    issues its gather on a private `pack_stream`, and the two `_StagingBuffer`
    events only order this worker's own streams against each other. The two
    callers make the source safe by different means:

    * KV -- `connector.py` records the fence commit 7427e05e added to fix KV
      corruption on reload on the RPC thread and `synchronize()`s it through
      `save_ready_event` before handing the blocks over.
    * State -- `state_tier.submit_store` needs no event: the PAGE units are
      reserved out of the KV pool and engine-pinned for the whole transfer, so
      nothing on the compute stream is writing them and the gather reads them
      where they sit.

    Do not delete the KV `save_ready_event` believing this class covers it:
    without it the gather reads the staging entry's previous occupant, which is
    silent corruption.
    """

    def __init__(
        self,
        device: torch.device,
        staging_buffer_bytes: int,
        *,
        release_after_transfer: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self._staging_buffer_bytes = int(staging_buffer_bytes)
        self._release_after_transfer = release_after_transfer
        self._tls = threading.local()

    def _use_cuda(self) -> bool:
        return self.device.type == "cuda"

    def thread_state(self) -> _ThreadTransferState:
        states = getattr(self._tls, "states", None)
        if states is None:
            states = {}
            self._tls.states = states
        key = str(self.device)
        state = states.get(key)
        if state is None:
            state = _ThreadTransferState(
                self.device,
                self._use_cuda(),
            )
            states[key] = state
        return state

    def ensure_buffer(
        self,
        staging_buffer: _StagingBuffer,
        nbytes: int,
    ) -> torch.Tensor:
        nbytes = int(nbytes)
        if nbytes > self._staging_buffer_bytes:
            raise RuntimeError(
                "ATOM LMCache connector internal error: transfer group exceeds "
                "bounded GPU staging buffer: "
                f"nbytes={nbytes}, capacity={self._staging_buffer_bytes}"
            )
        if (
            staging_buffer.tensor is None
            or int(staging_buffer.tensor.numel()) != self._staging_buffer_bytes
        ):
            staging_buffer.tensor = torch.empty(
                (self._staging_buffer_bytes,),
                dtype=torch.uint8,
                device=self.device,
            )
        return staging_buffer.tensor[:nbytes]

    def release_buffer_if_requested(
        self,
        staging_buffer: _StagingBuffer,
    ) -> None:
        if not self._release_after_transfer:
            return
        staging_buffer.tensor = None

    # -- whole-entry transfer (state tier) --------------------------------

    @staticmethod
    def _segment_block_bytes(segments: list[torch.Tensor]) -> list[int]:
        return [int(seg.numel()) * seg.element_size() for seg in segments]

    def _device_ctx(self):
        if self._use_cuda():
            return torch.cuda.device(self.device)
        return _NullCtx()

    def pack(self, segments: list[torch.Tensor], dst: Any) -> None:
        """Gather `segments` into one contiguous object via the Triton packer.

        The existing kernel needs no modification: it is already a fully
        parameterized gather driven by segment_ptrs[] + segment_block_bytes[] +
        block_ids[]. State passes its own views as the segments with
        block_ids=[0] and chunk_block_counts=[1] -- a single "chunk" of one
        "block", which is what a whole-entry snapshot is. Per-segment sizes may
        differ (GDN's k-views and v-views do); `_build_meta` sums them into one
        `bytes_per_block`.

        `storage_manager.allocate` normally hands back *host* memory, and the
        packer requires a CUDA uint8 contiguous destination, so the general
        path packs into the bounded GPU staging buffer and D2H's from there.
        The event around that copy makes the *consumer* side safe: whoever
        reads the MemoryObj next must not see a D2H still in flight. It says
        nothing about the *producer* side -- this gather runs on a private
        stream and does not wait for the forward that wrote the entry. Keeping
        the source quiescent is the caller's job (KV via `save_ready_event`,
        state via reserved-and-pinned units in `state_tier.submit_store`); see
        the class docstring.
        """
        from atom.kv_transfer.offload.dense.triton_kv_staging import (
            fused_pack_chunk_major,
        )

        segments = list(segments)
        seg_bytes = self._segment_block_bytes(segments)
        nbytes = sum(seg_bytes)
        dst_tensor = memory_object_as_uint8(dst, nbytes)
        with self._device_ctx():
            state = self.thread_state()
            staging_buffer = state.staging_buffer
            if dst_tensor.is_cuda and dst_tensor.device == self.device:
                # Direct-to-device fast path: no staging hop, but the gather
                # still reads the source units on `pack_stream`. If the kernel
                # launch raises we must drain before propagating -- otherwise
                # the caller frees those units while a queued gather is still
                # about to read them (same hazard as the general path's except).
                try:
                    with state.stream_ctx(state.pack_stream):
                        fused_pack_chunk_major(
                            segments, seg_bytes, [1], [0], dst_tensor
                        )
                    if state.pack_stream is not None:
                        state.pack_stream.synchronize()
                except Exception:
                    self._drain_device()
                    raise
                return
            try:
                with state.stream_ctx(state.pack_stream):
                    # `ensure_buffer` may allocate device storage. Allocate it on
                    # the first consumer stream (the gather below), not the
                    # default stream, so a caching allocator cannot hand back
                    # storage whose previous default-stream user is still running
                    # and then let this side stream overwrite it early -- the CPU
                    # tier would otherwise hold a mix of the image and unrelated
                    # KV bytes under a valid prefix hash. Mirrors the hazard note
                    # on `atom_lmcache_staging.py`'s `ensure_buffer` call.
                    device_buf = self.ensure_buffer(staging_buffer, nbytes)
                    fused_pack_chunk_major(segments, seg_bytes, [1], [0], device_buf)
                self._handoff(state, state.pack_stream, state.copy_stream)
                with state.stream_ctx(state.copy_stream):
                    dst_tensor.copy_(
                        device_buf,
                        non_blocking=dst_tensor.device.type != "cpu",
                    )
                self._finish(state, state.copy_stream)
            except Exception:
                # The caller must release the source units on this path too --
                # otherwise a failed store holds a whole image out of the KV
                # pool. Draining the device first is what makes that safe: the
                # gather reads those units, and returning them while a kernel
                # is still queued would let the pool hand them to another
                # request whose writes the gather would then pick up.
                self._drain_device()
                raise
            finally:
                self.release_buffer_if_requested(staging_buffer)

    def _drain_device(self) -> None:
        """Wait out every kernel this device has queued. Failure paths only."""
        if self._use_cuda():
            torch.cuda.synchronize(self.device)

    def unpack(self, src: Any, segments: list[torch.Tensor]) -> None:
        """Scatter one packed object back over `segments` -- `pack`'s mirror.

        Same staging hop in reverse: H2D into the bounded device buffer, then
        the Triton unpack kernel writes the segments. The kernel's stream is
        the one that produces the observable result, so it is the one waited
        on before the segments are handed back to their owner.
        """
        from atom.kv_transfer.offload.dense.triton_kv_staging import (
            fused_unpack_chunk_major,
        )

        segments = list(segments)
        seg_bytes = self._segment_block_bytes(segments)
        nbytes = sum(seg_bytes)
        src_tensor = memory_object_as_uint8(src, nbytes)
        with self._device_ctx():
            state = self.thread_state()
            staging_buffer = state.staging_buffer
            if src_tensor.is_cuda and src_tensor.device == self.device:
                # Direct-from-device fast path: the unpack kernel writes the
                # destination `segments` (the request's state slots) on
                # `pack_stream`. If the launch raises we must drain before
                # propagating -- otherwise the caller frees those slots while a
                # queued kernel is still about to write them, corrupting the
                # next request that gets the slot (mirror of the general path).
                try:
                    with state.stream_ctx(state.pack_stream):
                        fused_unpack_chunk_major(
                            src_tensor, segments, seg_bytes, [1], [0]
                        )
                    if state.pack_stream is not None:
                        state.pack_stream.synchronize()
                except Exception:
                    self._drain_device()
                    raise
                return
            try:
                with state.stream_ctx(state.copy_stream):
                    # Allocate on the first consumer stream (the H2D copy below),
                    # not the default stream -- see the matching note in `pack`.
                    device_buf = self.ensure_buffer(staging_buffer, nbytes)
                    device_buf.copy_(
                        src_tensor,
                        non_blocking=src_tensor.device.type != "cpu",
                    )
                self._handoff(state, state.copy_stream, state.pack_stream)
                with state.stream_ctx(state.pack_stream):
                    fused_unpack_chunk_major(device_buf, segments, seg_bytes, [1], [0])
                self._finish(state, state.pack_stream)
            except Exception:
                # Mirror of pack()'s except: the unpack kernel scatters into the
                # caller's destination segments on pack_stream. Returning while
                # that kernel is still queued would let the owner reuse the
                # slots under a pending write -- silent state corruption. Drain
                # every queued kernel before propagating the failure.
                self._drain_device()
                raise
            finally:
                self.release_buffer_if_requested(staging_buffer)

    @staticmethod
    def _handoff(state: _ThreadTransferState, producer, consumer) -> None:
        """Record the producer event and make the consumer stream wait on it.

        No-op on a non-CUDA device: `_ThreadTransferState` leaves both streams
        and `_StagingBuffer.ready_event` at None there (see
        `atom_lmcache_staging.py`), and every copy on this path already ran
        synchronously on the calling thread, so there is nothing to fence. The
        event is created iff `use_cuda`, so `ready_event is None` is the exact
        non-CUDA test. Guarding here keeps `unpack`'s general path -- whose H2D
        leg is a plain `copy_` that succeeds on CPU -- from dereferencing that
        None and masking the meaningful `fused_unpack_chunk_major` (Triton needs
        CUDA) error that follows."""
        if state.staging_buffer.ready_event is None:
            return
        state.staging_buffer.ready_event.record(producer)
        consumer.wait_event(state.staging_buffer.ready_event)

    @staticmethod
    def _finish(state: _ThreadTransferState, producer) -> None:
        """Publish the result synchronously: block until `producer` drains, so
        the buffer is free and the caller may observe the bytes on return.

        Unlike `run_staged_pipeline`, this path does not hand a `free_event`
        to a later consumer -- `producer.synchronize()` is the whole fence -- so
        it deliberately leaves the shared buffer's `free_event`/
        `free_event_valid` untouched.

        No-op when `producer` is None (non-CUDA: `_ThreadTransferState` has no
        streams), matching `_handoff` -- the copies already completed
        synchronously, so there is nothing to drain."""
        if producer is None:
            return
        producer.synchronize()
