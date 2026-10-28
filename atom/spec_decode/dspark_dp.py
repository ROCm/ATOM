# SPDX-License-Identifier: MIT
"""DP verification budgets from fixed N-2 confidence and group execution costs.

The CPU planner runs on EVERY forward. Its payload rides ForwardMode's one DP
exchange (`sync_dp_metadata`), and every rank resolves the same plan from it.
Only delayed survival probabilities leave the GPU; request ids remain local.
The planner includes today's bonus floors before comparing executable graphs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from atom.spec_decode.dspark_delayed import DelayedRows
from atom.spec_decode.dspark_scheduler import calibrate_confidence, resolve_q_buckets


def executable_graph(rows, q, captured):
    return next((g for g in captured if g >= rows and g % q == 0), None)


def local_frontier(survival, floors, ceilings, candidates, qs, *, dummy=False):
    """Return [expected output, rows, q, valid] and allocations per candidate.

    Mandatory prefixes are charged first. Stable top-survival admission over
    the remaining prefixes is optimal at each (graph capacity, query cap).
    Dummy rows participate in shapes, but never in expected output tokens.
    """
    floors, ceilings = np.asarray(floors), np.asarray(ceilings)
    batch, gamma = survival.shape
    result = np.zeros((len(candidates), 4), dtype=np.float64)
    lengths = np.broadcast_to(ceilings, (len(candidates), batch)).copy()
    if (
        floors.shape != (batch,)
        or ceilings.shape != (batch,)
        or (floors < 1).any()
        or (ceilings < floors).any()
        or (ceilings > gamma + 1).any()
        or not np.isfinite(survival).all()
        or (survival < 0).any()
        or (survival > 1).any()
        or (np.diff(survival, axis=1) > 1e-6).any()
    ):
        return result, lengths
    pos = np.arange(gamma)[None, :]
    mandatory = pos < (floors - 1)[:, None]
    base_tau = float(batch + survival[mandatory].sum())
    base_rows = int(floors.sum())
    # One sort and vectorized admission per cap, not per captured graph.
    for cap in qs:
        if (floors > cap).any():
            continue
        indices = np.array(
            [
                i
                for i, (q, g) in enumerate(candidates)
                if q == cap and g % cap == 0 and g >= base_rows
            ],
            dtype=np.int64,
        )
        if not indices.size:
            continue
        allowed = (~mandatory) & (pos < (np.minimum(ceilings, cap) - 1)[:, None])
        eligible = np.flatnonzero(allowed)
        order = eligible[np.argsort(-survival.flat[eligible], kind="stable")]
        counts = np.minimum(
            np.array([candidates[i][1] for i in indices]) - base_rows, len(order)
        )
        priority = np.full(batch * gamma, len(order), dtype=np.int64)
        priority[order] = np.arange(len(order))
        lengths[indices] = floors + (
            priority.reshape(1, batch, gamma) < counts[:, None, None]
        ).sum(2)
        prefix = np.concatenate(([0.0], survival.flat[order].cumsum()))
        result[indices, 0] = 0.0 if dummy else base_tau + prefix[counts]
        result[indices, 1] = base_rows + counts
        result[indices, 2] = cap
        result[indices, 3] = 1
    return result, lengths


def choose_dp_candidate(
    frontiers, captured, full_rows, full_q, cost_ms, fallback_graph=None
):
    """The joint plan, or None to keep each rank's own ragged budget.

    Without a fallback: maximize sum(expected output) / time of the ACTUAL
    common graph, and only when that graph is smaller than the full one.

    With ``fallback_graph`` -- the graph the group runs anyway, set by its
    busiest rank's own budget -- the plan only FILLS that graph: among
    allocations whose common graph is exactly it, take the one with the most
    expected output. The step costs the same, and ranks that shrank on their
    own get back drafts the busiest rank is paying for regardless. Shrinking
    below it is left to each rank's own budget: measured, a further joint
    shrink lost more accepted tokens than the smaller graph saved.
    """
    full_graph = executable_graph(full_rows, full_q, captured)
    if full_graph is None:
        return None
    best, best_score = None, -1.0
    for i in range(frontiers.shape[1]):
        local = frontiers[:, i]
        if not (local[:, 3] == 1).all():
            continue
        rows, q = int(local[:, 1].max()), int(local[:, 2].max())
        graph = executable_graph(rows, q, captured)
        if graph is None:
            continue
        if fallback_graph is not None:
            if graph != fallback_graph:
                continue
            score = local[:, 0].sum()
        else:
            ms = cost_ms(graph)
            if not np.isfinite(ms) or ms <= 0:
                continue
            score = local[:, 0].sum() / ms
        # Prefer a larger graph on ties: do not discard free draft prefixes.
        if score > best_score or (
            score == best_score and best is not None and graph > best[1]
        ):
            best, best_score = (i, graph), score
    if best is None:
        return None
    if fallback_graph is None and best[1] >= full_graph:
        return None
    return best[0]


@dataclass
class DPState:
    batch: int
    context: int
    pure_decode: bool


class DPVerifyPlanner:
    """Joint DP plan, including TP leader publication and bounded async copies."""

    def __init__(self, scheduler, *, comm=None):
        self.scheduler = scheduler
        self.runner = scheduler.runner
        config = self.runner.config
        if comm is None:
            from atom.spec_decode.dspark_comm import DSparkComm

            comm = DSparkComm()
        self.comm = comm
        self.gamma = config.speculative_config.num_speculative_tokens
        self.qs = resolve_q_buckets(config.dspark.ragged_graph_sizes, self.gamma + 1)
        # The measured curve arrives through `bind` (offline profile at
        # construction, or the startup sweep after graph capture). Until then
        # no decode graph is plannable and every step keeps its full width.
        self.baseline_ms = None
        self.observe_only = config.dspark.online_sps_observe_only
        self.comm.agree(
            (
                self.qs,
                config.max_num_seqs,
                self.observe_only,
                config.dspark.online_sps,
                config.dspark.bucket_global,
                config.dspark.online_sps_sample_interval,
                config.dspark.online_sps_update_interval,
                config.dspark.online_sps_window_steps,
            )
        )
        self.epoch = 0
        self.copies = DelayedRows(
            lambda: self.runner.tokenID_processor.async_copy_stream, self.gamma
        )
        self.survival = None
        self._request_set = None
        self._local_context = 0
        self._group_context = None
        self.state = DPState(0, 0, False)
        self._captured = None
        self.candidates = []
        self.skipped_busy = 0
        self.planned = 0
        self.contracted = 0
        self.selected_q = None
        self._ids = ()
        self._lengths = None

    def bind(self, baseline_sps):
        # Every worker must plan with the same curve; a startup sweep reduces
        # its timings by MAX over TP and DP, so this only catches divergence.
        self.comm.agree([round(float(x), 6) for x in baseline_sps])
        self.baseline_ms = np.array([1000.0 / x for x in baseline_sps])
        self._captured = None

    def capture(self, confidence):
        if self.comm.tp_rank == 0:
            self.survival = (
                calibrate_confidence(
                    confidence.detach(), self.scheduler.sts_temperatures
                )
                .float()
                .cumprod(dim=1)
            )

    def record(self, req_ids):
        source, self.survival = self.survival, None
        if source is None:
            return
        # Every in-flight copy retains BOTH destination and source, even when
        # its epoch is no longer adoptable. Never recycle based on age alone.
        self.copies.reap(before=self.epoch - 2)
        if self.copies.busy():
            self.skipped_busy += 1
            return
        # Block drafters draft at least two requests (warmup does even with
        # max_num_seqs=1), as TokenIDProcessor's output capacity allows.
        capacity = max(self.runner.config.max_num_seqs, 2)
        if source.shape[0] > capacity or len(req_ids) != source.shape[0]:
            raise ValueError("DSpark DP confidence does not match request batch")
        self.copies.push(self.epoch, req_ids, source, rows=capacity)

    def _delayed(self, req_ids):
        # Same fixed N-2 barrier as VerifyScheduler; never choose a newer
        # generation based on rank-local event readiness.
        taken = self.copies.take(self.epoch - 2, req_ids, wait=True)
        return None if taken is None else taken[1]

    def invalidate(self):
        self.survival = None
        # Retain DMA ownership while making every old generation unadoptable.
        self.copies.invalidate()

    def prepare(self, batch):
        """This rank's planning payload, built on real, dummy, prefill and
        output-less forwards alike. It rides the step's one DP exchange
        (`sync_dp_metadata`); `resolve` turns every rank's payload into the plan.
        """
        from atom.utils.region_graph import TokenGraph

        self.epoch += 1
        self.selected_q = None
        self.survival = None
        self._lengths = None
        n = batch.total_seqs_num
        ids = tuple(batch.req_ids[:n])
        self._ids = ids
        if ids != self._request_set:
            self._local_context += 1
            self._request_set = ids
        captured = (
            tuple(
                g
                for g in TokenGraph.captured_sizes("decode")
                if g < len(self.baseline_ms)
            )
            if self.baseline_ms is not None
            else ()
        )
        if captured != self._captured:
            self.comm.agree(captured)
            self._captured = captured
            self.candidates = [(q, g) for q in self.qs for g in captured if g % q == 0]
        previous = getattr(self.runner.tokenID_processor, "prev_batch", None)
        prefill, dummy = bool(batch.total_tokens_num_prefill), bool(batch.is_dummy_run)
        ready = not prefill and previous is not None and tuple(previous.req_ids) == ids
        full_q = batch.num_spec_step + 1
        ready = ready and full_q == self.gamma + 1
        # One TP leader provides the delayed statistics, so TP workers cannot
        # select different allocations due to tiny confidence rounding changes.
        confidence = torch.zeros(1 + n * self.gamma, dtype=torch.float64, device="cpu")
        if self.comm.tp_rank == 0 and ready:
            delayed = self._delayed(ids)
            if delayed is not None:
                confidence[0] = 1
                confidence[1:] = torch.from_numpy(delayed.reshape(-1))
        self.comm.broadcast_tp(confidence)
        ready = bool(confidence[0]) or dummy
        survival = confidence[1:].numpy().reshape(n, self.gamma)
        ceilings = np.asarray(batch.num_scheduled_tokens[:n], dtype=np.int64)
        bonuses = getattr(batch, "num_bonus", None)
        floors = (
            ceilings
            if dummy or prefill
            else np.maximum(
                np.asarray(bonuses[:n]) if bonuses is not None else np.zeros(n), 0
            ).astype(np.int64)
            + 1
        )
        frontier, self._lengths = (
            local_frontier(
                survival, floors, ceilings, self.candidates, self.qs, dummy=dummy
            )
            if ready and not prefill
            else (np.zeros((len(self.candidates), 4)), None)
        )
        # Fixed-size header and frontier, no request identities on the wire.
        header = np.array(
            [
                int(prefill),
                0 if dummy else n,
                self._local_context,
                int(ready),
                int(ceilings.sum()),
                full_q,
                self.epoch,
                int(dummy),
            ],
            dtype=np.float64,
        )
        return np.concatenate((header, frontier.reshape(-1)))

    def resolve(self, gathered, fallback=None):
        """The group's plan from every rank's `prepare` payload ([dp, len]).

        Identical on every rank, since every rank holds the same rows.
        ``fallback``: ``(rows, q)`` the group runs without a plan -- every
        rank's own ragged budget, DP-maxed. Returns ``(q, rows per rank,
        {req_id: ell})`` for this rank, or None to keep the fallback.
        """
        headers = gathered[:, :8]
        if not (headers[:, 6] == self.epoch).all():
            raise RuntimeError("DSpark DP forward epochs diverged")
        context = tuple(headers[:, [0, 1, 2, 7]].reshape(-1))
        generation = self.state.context + int(context != self._group_context)
        self._group_context = context
        self.state = DPState(
            int(headers[:, 1].max()),
            generation,
            not headers[:, 0].any() and bool(headers[:, 1].sum()),
        )
        runtime = self.scheduler.runtime
        if runtime is not None:
            runtime.dp_state = self.state
            runtime.dp_planner = self
        if not self.state.pure_decode or not headers[:, 3].all() or not self._captured:
            return None
        model = runtime.model if runtime is not None and not self.observe_only else None
        cost = (
            (lambda g: model.estimate_ms(g, self.state.batch))
            if model is not None
            else (lambda g: self.baseline_ms[g])
        )
        self.planned += 1
        frontiers = gathered[:, 8:].reshape(self.comm.dp_size, -1, 4)
        fallback_graph = (
            executable_graph(int(fallback[0]), int(fallback[1]), self._captured)
            if fallback is not None
            else None
        )
        choice = choose_dp_candidate(
            frontiers,
            self._captured,
            int(headers[:, 4].max()),
            int(headers[:, 5].max()),
            cost,
            fallback_graph=fallback_graph,
        )
        if choice is None:
            return None
        self.contracted += 1
        self.selected_q = self.candidates[choice][0]
        plan = dict(zip(self._ids, (self._lengths[choice] - 1).tolist()))
        return self.selected_q, frontiers[:, choice, 1].astype(np.int64), plan
