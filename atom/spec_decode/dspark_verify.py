import logging
from collections.abc import Sequence

import numpy as np
import torch

logger = logging.getLogger("atom")

# Bounds the queued async ell copies if the GPU falls far behind.
_MAX_ELL_INFLIGHT = 4

# Steps back the adopted ell comes from. >= 2 so its copy has landed (the wait is
# free), < _MAX_ELL_INFLIGHT so record_ell cannot reuse the slot being read.
_ELL_GENERATION = 2


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

    Sync-free by deferring: ell is adopted from a FIXED generation back, not from
    the latest copy (would pin the host to the GPU tail) nor from whichever has
    landed (per-rank, and ell is a shape under ragged). Stale ell is lossless --
    it only SIZES the next verify, and a short one is re-drafted next step.
    """

    def __init__(self, runner):
        # runner: provides the shared async D2H stream (tokenID_processor).
        self.runner = runner
        self.sps_table: torch.Tensor | None = None
        self.sts_temperatures: torch.Tensor | None = None
        self._last_ell: torch.Tensor | None = None
        # FIFO of in-flight async D2H copies of ell, one entry per step:
        # (event, cpu_buf, req_ids). Read by index in _resolve_ell, never popped.
        self._ell_pending: list = []
        # Freshest RESOLVED {req_id: ell}, re-mapped onto each step's (possibly
        # reordered) batch by req_id. Lags the GPU by a step or two whenever the
        # CPU is running ahead -- see the class docstring.
        self._ell_map: dict = {}
        # Pinned landing ring for the D2H, [_MAX_ELL_INFLIGHT, max_num_seqs].
        # Allocated on first use (the runner's config is complete by then).
        self._ell_stage_ring: torch.Tensor | None = None
        self._ell_stage_idx = 0
        # Event of the D2H that last landed in each ring slot (None = unused).
        # Checked before the slot is handed out again: evicting an entry from
        # `_ell_pending` does NOT cancel its in-flight DMA, so queue length
        # alone cannot tell us a slot is free.
        self._ell_slot_event: list = [None] * _MAX_ELL_INFLIGHT
        # Ring index returned by the last `_ell_stage` call, or None when it
        # fell back to a one-off buffer. Consumed by `record_ell`.
        self._ell_last_slot: int | None = None

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

    def set_last_ell(self, ell: torch.Tensor | None) -> None:
        """Stash the ell computed by this step's propose() (or None)."""
        self._last_ell = ell

    def _ell_stage(self, n: int) -> torch.Tensor:
        """Free pinned ell landing slot."""
        # pin_memory: a pageable D2H is not async, and this copy sits behind the
        # whole step -- blocking here stalls the host on the GPU's tail.
        ring = self._ell_stage_ring
        width = ring.shape[1] if ring is not None else self.runner.config.max_num_seqs
        if n > width:
            # Unreachable (ell is [decode_bs] <= max_num_seqs); one-off rather
            # than corrupt the ring.
            self._ell_last_slot = None
            return torch.zeros(n, dtype=torch.int64, device="cpu", pin_memory=True)
        if ring is None:
            # device="cpu" is not optional: warmup runs with a GPU default
            # device, where unqualified pin_memory=True allocates on GPU and
            # dies with "Only dense CPU tensors can be pinned".
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
            # Rotation alone does not free a slot: the ring and the _ell_pending
            # cap are the same size, so the index wraps onto an entry being
            # evicted -- and eviction does not cancel its DMA. Two writers, one
            # buffer, torn reads. Park the index and take a one-off buffer.
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

        The copy is queued here; ``_resolve_ell`` adopts it once it is
        ``_ELL_GENERATION`` steps old.
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
            del self._ell_pending[:-_MAX_ELL_INFLIGHT]

    def _resolve_ell(self) -> None:
        """Adopt the ell from a FIXED generation back (``_ELL_GENERATION``).

        The generation is fixed, not "whichever copy happens to have landed".
        Which copies have landed depends on how far THIS rank's CPU has run
        ahead, so an opportunistic pick makes different TP ranks adopt different
        generations. ell sizes the ragged layout, i.e. it is a shape, so the
        ranks then disagree on token counts and hang in aiter's symmetric
        all-reduce, each spinning for peer signals that never come (measured:
        7 ranks at 48 tokens, 1 at 24). A fixed index is identical on every rank
        because ``_ell_pending`` is appended to once per step, host-side.

        The ``synchronize`` is the guarantee, not the cost: generation N-2's copy
        was fired two steps ago and waits only on step N-2's GPU work, so with
        the CPU at most a step or two ahead it has long landed and this is a
        no-op. (N-1 would be the one that actually blocks -- it is queued behind
        the previous step's entire forward.)
        """
        pending = self._ell_pending
        if len(pending) < _ELL_GENERATION:
            self._ell_map = {}  # too early to have one -> full length
            return
        event, cpu_buf, req_ids = pending[-_ELL_GENERATION]
        event.synchronize()
        ell_np = cpu_buf.numpy().astype(np.int32)
        n = min(len(req_ids), ell_np.shape[0])
        self._ell_map = {req_ids[i]: int(ell_np[i]) for i in range(n)}

    @property
    def ell_by_req(self) -> dict:
        """{req_id: ell} from a fixed generation back, so every TP rank adopts
        the SAME one — see ``_resolve_ell``. Read this at the top of a step; the
        sync it performs is a no-op there and a real stall anywhere later."""
        self._resolve_ell()
        return self._ell_map

    @ell_by_req.setter
    def ell_by_req(self, value: dict) -> None:
        # Direct assignment (e.g. reset to {}) drops the in-flight copies too.
        self._ell_map = value
        self._ell_pending.clear()

    def ell_nonblocking(self) -> dict:
        """Non-blocking read for the SAME-step postprocess path (carried back to
        the scheduler as fwd_output.dspark_ell).

        Returns the map ``ell_by_req`` already resolved at the top of this step
        WITHOUT re-resolving: by now ``record_ell`` has appended this step's
        entry, so the fixed generation would land on a copy that is still in
        flight and the synchronize would really block. Correctness: the scheduler
        only uses it to set seq.dspark_next_ell for NEXT-step sizing, and the
        worker re-reads ell_by_req next step regardless.
        """
        return dict(self._ell_map)
