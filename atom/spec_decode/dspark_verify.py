import logging
from typing import Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger("atom")

# Max in-flight async ell copies to keep queued. The CPU runs at most a step or
# two ahead of the GPU, so anything older is already superseded by the entries
# behind it; the cap just bounds the queue if the GPU falls far behind.
_MAX_ELL_INFLIGHT = 4


class VerifyScheduler:
    """Hardware-Aware Prefix Scheduler for confidence-scheduled block drafting.

    Owns the per-request verify-length (``ell``) machinery shared by any
    confidence-scheduled block drafter (DSpark today; e.g. a future Qwen block
    drafter next). Given the draft confidence head output it picks each request's
    verify length ``ell_r`` (paper Algorithm 1), then carries that ell across
    steps keyed by req_id (continuous batching reorders the batch between steps).

    Named for its product (the per-request verify length) rather than its input
    signal: the confidence head is one input to the schedule, but the class's job
    is to decide how many draft tokens each request verifies next step.

    The cost model inputs (``sps_table`` throughput profile, ``sts_temperatures``)
    are bound later by the runner's warmup/calibration; until then a synthetic
    monotone SPS stub keeps the path lossless.

    Kept sync-free on the decode hot path: ``record_ell`` fires an ASYNC D2H of
    ell and the {req_id: ell} map is adopted only once that copy has ALREADY
    completed -- the ell handoff is NEVER waited on.

    Why never waited on: ``record_ell`` runs at the end of a step, after the
    whole step (target forward + block draft) is queued on the default stream,
    and the copy stream waits on it -- so the copy completes only when the step
    drains. The map is read at the TOP of the next step
    (``ModelRunner.prepare_model`` -> ``_dspark_apply_q_bucket``). Blocking there
    would pin the host to the GPU's tail every step, collapsing the run-ahead the
    rest of this loop is built on (``recv_async_output`` and friends all consume
    a copy fired a step earlier, never the current one) and leaving the GPU idle
    for the whole of the next step's host-side prep. Measured as a large
    dspark -> next-decode bubble under ``confidence_schedule``.

    Consuming a step-or-two-stale ell instead is lossless: ell is only the
    PREDICTED accept count used to SIZE the next verify, a missing entry already
    falls back to full length (never under-verify), the hard anchor lower bound
    comes from the current step's ``num_bonus``, and any draft suffix dropped by
    a short ell is simply re-drafted next step.
    """

    def __init__(self, runner):
        # runner: provides the shared async D2H stream (tokenID_processor).
        self.runner = runner
        self.sps_table: Optional[torch.Tensor] = None
        self.sts_temperatures: Optional[torch.Tensor] = None
        self._last_ell: Optional[torch.Tensor] = None
        # FIFO of in-flight async D2H copies of ell, one entry per step:
        # (event, cpu_buf, req_ids). Drained non-blockingly by _drain_ell.
        self._ell_pending: list = []
        # Freshest RESOLVED {req_id: ell}, re-mapped onto each step's (possibly
        # reordered) batch by req_id. Lags the GPU by a step or two whenever the
        # CPU is running ahead -- see the class docstring.
        self._ell_map: dict = {}
        # Pinned landing ring for the D2H, [_MAX_ELL_INFLIGHT, max_num_seqs].
        # Allocated on first use (the runner's config is complete by then).
        self._ell_stage_ring: Optional[torch.Tensor] = None
        self._ell_stage_idx = 0
        # Event of the D2H that last landed in each ring slot (None = unused).
        # Checked before the slot is handed out again: evicting an entry from
        # `_ell_pending` does NOT cancel its in-flight DMA, so queue length
        # alone cannot tell us a slot is free.
        self._ell_slot_event: list = [None] * _MAX_ELL_INFLIGHT
        # Ring index returned by the last `_ell_stage` call, or None when it
        # fell back to a one-off buffer. Consumed by `record_ell`.
        self._ell_last_slot: Optional[int] = None

    def compute_ell(self, confidence: torch.Tensor) -> torch.Tensor:
        """Run the Hardware-Aware Prefix Scheduler (paper Algorithm 1) and return
        the per-request verify length ``ell`` as an int tensor [bs].

        This ONLY computes ell — it does not touch the draft tokens. The actual
        variable-length verification (Level B) consumes ell downstream to size
        each request's verification batch, which is where the throughput win
        comes from. Kept sync-free (no .item()/.tolist()) for the decode hot path.

        Args:
            confidence: [bs, L] per-position acceptance probs.
        """
        from atom.spec_decode.dspark_scheduler import schedule_prefix_lengths_tensor

        bs, L = confidence.shape
        sps_table = self.sps_table
        if sps_table is None:
            # Synthetic monotone-decreasing SPS stub until real calibration lands.
            # The stub's slope is ~1/steps, and the prefix scheduler indexes it by
            # B = R + m (R = LOCAL decode bs). Under DP-attention each rank holds
            # only 1/dp_size of the batch, so a local-bs ramp is dp_size x too
            # steep -> throughput looks to drop dp_size x faster per admitted draft
            # -> the scheduler early-stops sooner -> per-request ell is
            # over-truncated (the acceptance tail collapses; a DP-only regression
            # vs TP, which sees the same stub but with a large bs / shallow slope).
            # Scale steps by dp_size so the local B range sits in the shallow head
            # of the ramp, matching the TP curve. (Real DP-aware SPS calibration
            # under ragged is the proper long-term fix; this keeps the stub sane.)
            dp = max(1, self.runner.config.parallel_config.data_parallel_size)
            sps_table = torch.linspace(
                1.0, 0.1, steps=dp * bs * (L + 1) + 1, device=confidence.device
            )
        return schedule_prefix_lengths_tensor(
            confidence.detach(),
            sps_table,
            sts_temperatures=self.sts_temperatures,
        )

    def set_last_ell(self, ell: Optional[torch.Tensor]) -> None:
        """Stash the ell computed by this step's propose() (or None)."""
        self._last_ell = ell

    def _ell_stage(self, n: int) -> torch.Tensor:
        """Next pinned CPU slot to land an async ell D2H in, as an ``[n]`` view.

        Pinned matters here: a D2H into freshly allocated PAGEABLE memory is not
        reliably asynchronous, and this copy is issued behind
        ``wait_stream(default_stream)`` -- i.e. behind the whole step -- so a
        blocking one would stall the host on the GPU's tail, exactly what this
        class exists to avoid.

        A slot is handed out only once the D2H that last used it has LANDED,
        checked through that copy's event. Rotation alone is NOT enough: the
        ring has ``_MAX_ELL_INFLIGHT`` slots and ``record_ell`` caps
        ``_ell_pending`` at the same number, so as soon as the host runs that
        many steps ahead of the copy stream the index wraps onto the slot whose
        entry is being evicted -- and evicting the bookkeeping tuple does not
        cancel its DMA. Two copies would then write the same pinned buffer
        concurrently and the reader would see torn values (an ell that is
        neither current nor merely stale). While a slot is still busy we park
        the index and take a one-off buffer for this step instead.

        NOTE: ``device="cpu"`` is not optional. This ring is first allocated on
        the warmup forward, which ModelRunner.__init__ runs while the default
        device is still the GPU (it is only reset to "cpu" AFTER
        ``_maybe_warmup()`` returns), so an unqualified
        ``torch.zeros(..., pin_memory=True)`` allocates on the GPU and dies with
        "Only dense CPU tensors can be pinned". Other pin_memory sites in the
        engine get away with it only because they are unreachable from warmup.
        CpuGpuBuffer pins the same, explicitly-CPU way.
        """
        ring = self._ell_stage_ring
        width = ring.shape[1] if ring is not None else self.runner.config.max_num_seqs
        if n > width:
            # Should be unreachable (ell is [decode_bs] <= max_num_seqs); fall
            # back to a one-off pinned buffer rather than corrupt the ring.
            self._ell_last_slot = None
            return torch.zeros(n, dtype=torch.int64, device="cpu", pin_memory=True)
        if ring is None:
            ring = torch.zeros(
                _MAX_ELL_INFLIGHT,
                width,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            self._ell_stage_ring = ring
        idx = self._ell_stage_idx
        busy = self._ell_slot_event[idx]
        if busy is not None and not busy.query():
            # Previous D2H into this slot has not landed. Overwriting it now is
            # the torn-read race described above -- park the index (so the same
            # slot is retried next step) and stage into a one-off buffer.
            self._ell_last_slot = None
            return torch.zeros(n, dtype=torch.int64, device="cpu", pin_memory=True)
        slot = ring[idx]
        self._ell_last_slot = idx
        self._ell_stage_idx = (idx + 1) % _MAX_ELL_INFLIGHT
        return slot[:n]

    def record_ell(self, req_ids: Sequence) -> None:
        """Fire an ASYNC copy of this step's ell, keyed later by req_id.

        ell was computed in propose() ordered by THIS step's decode batch. We
        save {req_id: ell} so a LATER step can re-map it onto its own (possibly
        reordered) batch by req_id — batch position is not stable across steps
        under continuous batching.

        The copy is queued, never awaited; ``_drain_ell`` adopts it on whichever
        later step finds it already complete.
        """
        ell = self._last_ell
        if ell is None:
            self._ell_pending.clear()
            self._ell_map = {}
            return
        ell = ell.detach().reshape(-1)
        # Reuse the runner's shared async D2H stream
        copy_stream = self.runner.tokenID_processor.async_copy_stream
        default_stream = torch.cuda.current_stream()
        event = torch.cuda.Event()
        cpu_buf = self._ell_stage(ell.numel())
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(default_stream)
            cpu_buf.copy_(ell, non_blocking=True)
            event.record(copy_stream)
        # Bind the event to the ring slot this copy is landing in, so the slot
        # is not handed out again until the DMA completes. This outlives the
        # `_ell_pending` entry on purpose: the eviction below drops bookkeeping,
        # not the copy itself. None -> one-off buffer, nothing to guard.
        if self._ell_last_slot is not None:
            self._ell_slot_event[self._ell_last_slot] = event
        # Keep req_ids as a plain list snapshot (CPU-only, order-safe).
        self._ell_pending.append((event, cpu_buf, list(req_ids)))
        if len(self._ell_pending) > _MAX_ELL_INFLIGHT:
            del self._ell_pending[: -_MAX_ELL_INFLIGHT]

    def _drain_ell(self) -> None:
        """Adopt the freshest COMPLETED async ell copy. NEVER blocks.

        All copies are issued on one stream, so they complete in order: popping
        while the head's event is done leaves the newest finished entry in
        ``newest`` and keeps the still-in-flight ones queued for a later step.
        No completed copy -> keep the map we already have.
        """
        pending = self._ell_pending
        newest = None
        while pending and pending[0][0].query():
            newest = pending.pop(0)
        if newest is None:
            return
        _event, cpu_buf, req_ids = newest
        ell_np = cpu_buf.numpy().astype(np.int32)
        n = min(len(req_ids), ell_np.shape[0])
        self._ell_map = {req_ids[i]: int(ell_np[i]) for i in range(n)}

    @property
    def ell_by_req(self) -> dict:
        """{req_id: ell} for sizing this step's verify. Never syncs — see the
        class docstring for why a stale ell is the correct trade here."""
        self._drain_ell()
        return self._ell_map

    @ell_by_req.setter
    def ell_by_req(self, value: dict) -> None:
        # Direct assignment (e.g. reset to {}) drops the in-flight copies too.
        self._ell_map = value
        self._ell_pending.clear()

    def ell_nonblocking(self) -> dict:
        """Non-blocking read for the SAME-step postprocess path (carried back to
        the scheduler as fwd_output.dspark_ell).

        Identical policy to ``ell_by_req`` — this step's own copy was fired
        moments ago and will not be ready, so this returns the freshest map
        already resolved. Correctness: the scheduler only uses it to set
        seq.dspark_next_ell for NEXT-step sizing, and the worker's own
        _dspark_apply_q_bucket re-reads the map next step regardless.
        """
        self._drain_ell()
        return dict(self._ell_map)
