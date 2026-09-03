# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Whole-entry GPU staging for the Kimi-K3 state tier.

One state checkpoint moves as one flat byte stream, so this needs the bounded
device buffer, the two copy streams and the event handshake -- all borrowed from
``atom_lmcache_staging`` -- but none of ``BlockGPUConnector``'s chunk
orchestration, which is KV-specific: ``_iter_transfer_chunks`` zips MemoryObjs
against block-id groups and sizes each from a startup per-block constant, and a
single differently sized object breaks both invariants.
"""

from __future__ import annotations

import logging
import threading

import torch

from atom.kv_transfer.offload.atom_lmcache_staging import (
    _PipelineStage,
    _StagingBuffer,
    _ThreadTransferState,
    run_staged_pipeline,
)

logger = logging.getLogger("atom")


class StagedTransfer:
    """Move one whole state entry between GPU segments and a flat blob.

    Both directions run through ``run_staged_pipeline``, which owns the
    ``free_event`` handshake and, through ``recover_buffer``, the failure fence.
    Routing failures through that one callback is what keeps the directions
    symmetric: a slot may return to its pool only once the device has stopped
    reading (gather) or writing (scatter) it, so pack and unpack must both fence
    before propagating.

    Nothing here fences the *producer*. The gather runs on a private stream and
    does not wait for the forward that wrote the entry; keeping the source
    quiescent is the caller's job (the state tier reserves and pins the PAGE
    units for the whole transfer).
    """

    def __init__(
        self,
        device: torch.device,
        staging_buffer_bytes: int,
        *,
        release_after_transfer: bool = False,
    ) -> None:
        self.device = torch.device(device)
        # CUDA-only by construction: the Triton kernels require it and
        # ``run_staged_pipeline`` dereferences the stream events unconditionally.
        if self.device.type != "cuda":
            raise ValueError(f"K3 state staging needs a CUDA device, got {self.device}")
        self._staging_buffer_bytes = int(staging_buffer_bytes)
        if self._staging_buffer_bytes <= 0:
            raise ValueError("K3 state staging buffer bytes must be > 0")
        self._release_after_transfer = bool(release_after_transfer)
        self._tls = threading.local()
        # A buffer lands here only when a stream fence failed, so freeing it
        # could race still-running GPU work. Healthy devices never grow it.
        self._quarantined: list[torch.Tensor] = []
        self._quarantine_lock = threading.Lock()

    def thread_state(self) -> _ThreadTransferState:
        state = getattr(self._tls, "state", None)
        if state is None:
            state = _ThreadTransferState(self.device, True)
            self._tls.state = state
        return state

    def ensure_buffer(self, buf: _StagingBuffer, nbytes: int) -> torch.Tensor:
        nbytes = int(nbytes)
        if nbytes > self._staging_buffer_bytes:
            raise RuntimeError(
                "K3 state staging: entry exceeds bounded GPU staging buffer: "
                f"nbytes={nbytes}, capacity={self._staging_buffer_bytes}"
            )
        if buf.tensor is None or int(buf.tensor.numel()) != self._staging_buffer_bytes:
            buf.tensor = torch.empty(
                (self._staging_buffer_bytes,), dtype=torch.uint8, device=self.device
            )
            # The recorded free_event belongs to the tensor just dropped.
            buf.free_event_valid = False
        return buf.tensor[:nbytes]

    def _release_buffer(self, buf: _StagingBuffer) -> None:
        if not self._release_after_transfer:
            return
        buf.tensor = None
        buf.free_event_valid = False

    def _recover_buffer(
        self, buf: _StagingBuffer, stage_a: _PipelineStage, stage_b: _PipelineStage
    ) -> bool:
        """Fence both streams after a failure; report whether that succeeded.

        The caller frees the source units (pack) or the destination slot
        (unpack) as soon as we raise. Returning while a kernel is still queued
        would let the pool hand that memory to another request under a pending
        read or write -- silent corruption with no exception.
        """
        fenced = True
        for name, stream in (("stage_a", stage_a.stream), ("stage_b", stage_b.stream)):
            try:
                stream.synchronize()
            except Exception:
                fenced = False
                logger.exception(
                    "K3 state staging: %s fence failed; quarantining buffer", name
                )
        if not fenced:
            if buf.tensor is not None:
                with self._quarantine_lock:
                    self._quarantined.append(buf.tensor)
            buf.tensor = None
            buf.free_event_valid = False
        return fenced

    def _run(
        self, nbytes: int, stage_a: _PipelineStage, stage_b: _PipelineStage
    ) -> None:
        with torch.cuda.device(self.device):
            run_staged_pipeline(
                self.thread_state(),
                (int(nbytes),),
                stage_a=stage_a,
                stage_b=stage_b,
                ensure_buffer=self.ensure_buffer,
                group_nbytes=int,
                release_buffer=self._release_buffer,
                recover_buffer=self._recover_buffer,
            )

    @staticmethod
    def _segment_bytes(segments: list[torch.Tensor]) -> list[int]:
        return [int(seg.numel()) * seg.element_size() for seg in segments]

    def pack(self, segments, dst: torch.Tensor) -> None:
        """Gather ``segments`` into the flat uint8 ``dst``.

        The Triton kernel is already a parameterized gather over
        segment_ptrs/segment_block_bytes/block_ids, so a whole-entry snapshot is
        one "chunk" of one "block". Per-segment sizes may differ (GDN's k-views
        and v-views do); ``_build_meta`` sums them. The pipeline synchronizes the
        D2H stream before returning, so on a normal return the bytes are
        observable and no device work still reads the source.
        """
        from atom.kv_transfer.offload.dense.triton_kv_staging import (
            fused_pack_chunk_major,
        )

        segments = list(segments)
        seg_bytes = self._segment_bytes(segments)
        state = self.thread_state()
        self._run(
            sum(seg_bytes),
            _PipelineStage(
                stream=state.pack_stream,
                run=lambda _n, buf: fused_pack_chunk_major(
                    segments, seg_bytes, [1], [0], buf
                ),
            ),
            _PipelineStage(
                stream=state.copy_stream,
                run=lambda _n, buf: dst.copy_(
                    buf, non_blocking=dst.device.type != "cpu"
                ),
            ),
        )

    def unpack(self, src: torch.Tensor, segments) -> None:
        """Scatter the flat uint8 ``src`` over ``segments`` -- pack's mirror.

        The scatter is stage B, so the pipeline's terminal synchronize fences the
        stream that writes the caller's slot; only then may the slot be released.
        """
        from atom.kv_transfer.offload.dense.triton_kv_staging import (
            fused_unpack_chunk_major,
        )

        segments = list(segments)
        seg_bytes = self._segment_bytes(segments)
        state = self.thread_state()
        self._run(
            sum(seg_bytes),
            _PipelineStage(
                stream=state.copy_stream,
                run=lambda _n, buf: buf.copy_(
                    src, non_blocking=src.device.type != "cpu"
                ),
            ),
            _PipelineStage(
                stream=state.pack_stream,
                run=lambda _n, buf: fused_unpack_chunk_major(
                    buf, segments, seg_bytes, [1], [0]
                ),
            ),
        )
