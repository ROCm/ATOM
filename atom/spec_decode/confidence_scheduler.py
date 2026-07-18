import logging
from typing import Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger("atom")


class ConfidenceScheduler:
    """Hardware-Aware Prefix Scheduler for confidence-scheduled block drafting.

    Owns the per-request verify-length (``ell``) machinery shared by any
    confidence-scheduled block drafter (DSpark today; e.g. a future Qwen block
    drafter next). Given the draft confidence head output it picks each request's
    verify length ``ell_r`` (paper Algorithm 1), then carries that ell across
    steps keyed by req_id (continuous batching reorders the batch between steps).

    The cost model inputs (``sps_table`` throughput profile, ``sts_temperatures``)
    are bound later by the runner's warmup/calibration; until then a synthetic
    monotone SPS stub keeps the path lossless.

    Kept sync-free on the decode hot path: ``record_ell`` fires an ASYNC D2H of
    ell to a pinned buffer and the {req_id: ell} map is materialized lazily.
    """

    def __init__(self, runner):
        # runner: provides the shared async D2H stream (tokenID_processor).
        self.runner = runner
        self.sps_table: Optional[torch.Tensor] = None
        self.sts_temperatures: Optional[torch.Tensor] = None
        self._last_ell: Optional[torch.Tensor] = None
        # req_id -> ell map from the PREVIOUS step's propose(), re-mapped onto the
        # next step's (possibly reordered) batch by req_id. Resolved lazily from
        # the async D2H fired by record_ell (event complete by next read).
        self._ell_map_cache: dict = {}
        self._ell_pending: Optional[tuple] = None  # (event, cpu_buf, req_ids)

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
            sps_table = torch.linspace(
                1.0, 0.1, steps=bs * (L + 1) + 1, device=confidence.device
            )
        return schedule_prefix_lengths_tensor(
            confidence.detach(),
            sps_table,
            sts_temperatures=self.sts_temperatures,
        )

    def set_last_ell(self, ell: Optional[torch.Tensor]) -> None:
        """Stash the ell computed by this step's propose() (or None)."""
        self._last_ell = ell

    def record_ell(self, req_ids: Sequence) -> None:
        """Fire an ASYNC copy of this step's ell, keyed later by req_id.

        ell was computed in propose() ordered by THIS step's decode batch. We
        save {req_id: ell} so the NEXT step can re-map it onto its own (possibly
        reordered) batch by req_id — batch position is not stable across steps
        under continuous batching.
        """
        ell = self._last_ell
        if ell is None:
            self._ell_pending = None
            self._ell_map_cache = {}
            return
        ell = ell.detach()
        # Reuse the runner's shared async D2H stream
        copy_stream = self.runner.tokenID_processor.async_copy_stream
        default_stream = torch.cuda.current_stream()
        event = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(default_stream)
            cpu_buf = ell.to("cpu", non_blocking=True)
            event.record(copy_stream)
        # Keep req_ids as a plain list snapshot (CPU-only, order-safe).
        self._ell_pending = (event, cpu_buf, list(req_ids))
        # Invalidate last step's resolved map; recomputed lazily on first read.
        self._ell_map_cache = None

    @property
    def ell_by_req(self) -> dict:
        """Lazily materialize {req_id: ell} from the async D2H fired by
        record_ell. Syncs the (already-complete) event on first read, then
        caches so repeated reads within a step are free."""
        cache = self._ell_map_cache
        if cache is not None:
            return cache
        pending = self._ell_pending
        if pending is None:
            self._ell_map_cache = {}
            return self._ell_map_cache
        event, cpu_buf, req_ids = pending
        event.synchronize()  # long done by next step; no hot-path stall
        ell_np = cpu_buf.numpy().astype(np.int32)
        n = min(len(req_ids), ell_np.shape[0])
        self._ell_map_cache = {req_ids[i]: int(ell_np[i]) for i in range(n)}
        return self._ell_map_cache

    @ell_by_req.setter
    def ell_by_req(self, value: dict) -> None:
        # Direct assignment (e.g. reset to {}) bypasses the pending copy.
        self._ell_map_cache = value
        self._ell_pending = None

    def ell_nonblocking(self) -> dict:
        """Non-blocking read of the ell map for the SAME-step postprocess path
        (carried back to the scheduler as fwd_output.dspark_ell). Must NOT sync:
        record_ell just fired this step's async D2H, and forcing it here would
        re-serialize CPU on GPU — the exact stall we removed. If the copy is
        already resolved (cache present) return it; if it's still pending (this
        step's fresh copy) query without waiting; else fall back to {}.

        Correctness: the scheduler only uses this to set seq.dspark_next_ell for
        NEXT-step sizing, and the worker's own _dspark_apply_q_bucket reads the
        (fully resolved) property next step regardless — so a same-step empty
        here never under-verifies."""
        cache = self._ell_map_cache
        if cache:
            return dict(cache)
        pending = self._ell_pending
        if pending is None:
            return {}
        event, cpu_buf, req_ids = pending
        if not event.query():  # not done yet — do NOT stall the hot path
            return {}
        ell_np = cpu_buf.numpy().astype(np.int32)
        n = min(len(req_ids), ell_np.shape[0])
        resolved = {req_ids[i]: int(ell_np[i]) for i in range(n)}
        self._ell_map_cache = resolved
        return dict(resolved)
