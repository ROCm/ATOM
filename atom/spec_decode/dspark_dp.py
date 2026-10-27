# SPDX-License-Identifier: MIT
"""DP verification budgets from fixed N-2 confidence and group execution costs.

The CPU planner runs before ForwardMode's shape agreement on EVERY forward.
Only delayed survival probabilities leave the GPU; request ids remain local.
The planner includes today's bonus floors before comparing executable graphs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from atom.spec_decode.dspark_scheduler import calibrate_confidence, resolve_q_buckets


class SPSMesh:
    """Orthogonal TP/DP CPU groups; no new communicator or device collective."""

    def __init__(self, tp_group=None, dp_group=None):
        if tp_group is None:
            from aiter.dist.parallel_state import get_dp_group, get_tp_group

            tp_group, dp_group = get_tp_group().cpu_group, get_dp_group().cpu_group
        self.tp_group, self.dp_group = tp_group, dp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)
        self.dp_size = dist.get_world_size(dp_group)

    @staticmethod
    def gather(value, group, size):
        if size == 1:
            return value.unsqueeze(0)
        result = [torch.empty_like(value) for _ in range(size)]
        dist.all_gather(result, value, group=group)
        return torch.stack(result)

    def broadcast_tp(self, value):
        if self.tp_size > 1:
            dist.broadcast(
                value, src=dist.get_global_rank(self.tp_group, 0), group=self.tp_group
            )

    def gather_dp(self, value):
        return self.gather(value, self.dp_group, self.dp_size)

    def agree(self, value):
        """Fail on all workers before variable-size planner exchanges."""
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).digest()
        local = torch.tensor(list(digest), dtype=torch.uint8, device="cpu")
        tp = self.gather(local, self.tp_group, self.tp_size)
        both = self.gather_dp(tp)
        if not torch.all(both == local):
            raise ValueError(
                "DSpark DP workers disagree on profile or graph configuration"
            )


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


def choose_dp_candidate(frontiers, captured, full_rows, full_q, cost_ms):
    """Maximize sum(expected output) / group time at the ACTUAL common graph.

    The selected cap is passed to ForwardMode, even if all local maxima are
    smaller. This matters for non-nested divisors (e.g. q=2 and q=3): changing
    the cap could change the executable graph. Resolve rows after DP maxima.
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
        ms = cost_ms(graph)
        if not np.isfinite(ms) or ms <= 0:
            continue
        score = local[:, 0].sum() / ms
        # Prefer a larger graph on ties: do not discard free draft prefixes.
        if score > best_score or (
            score == best_score and best is not None and graph > best[1]
        ):
            best, best_score = (i, graph), score
    return best[0] if best is not None and best[1] < full_graph else None


@dataclass
class DPState:
    batch: int
    context: int
    pure_decode: bool


class DPVerifyPlanner:
    """Joint DP plan, including TP leader publication and bounded async copies."""

    def __init__(self, scheduler, *, mesh=None):
        self.scheduler = scheduler
        self.runner = scheduler.runner
        config = self.runner.config
        self.mesh = mesh or SPSMesh()
        self.gamma = config.speculative_config.num_speculative_tokens
        self.qs = resolve_q_buckets(config.dspark.ragged_graph_sizes, self.gamma + 1)
        profile = json.loads(Path(config.dspark.calibration_profile).read_text())
        self.baseline_ms = np.array([1000.0 / x for x in profile["sps_table"]])
        self.observe_only = config.dspark.online_sps_observe_only
        self.mesh.agree(
            (
                profile,
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
        self.pending = []
        self.free = []
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

    def capture(self, confidence):
        if self.mesh.tp_rank == 0:
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
        keep = []
        for entry in self.pending:
            epoch, event, cpu, _, _ = entry
            if epoch < self.epoch - 2 and event.query():
                self.free.append(cpu)
            else:
                keep.append(entry)
        self.pending = keep
        if len(self.pending) >= 4:
            self.skipped_busy += 1
            return
        capacity = (self.runner.config.max_num_seqs, self.gamma)
        cpu = (
            self.free.pop()
            if self.free
            else torch.empty(
                capacity, dtype=torch.float32, device="cpu", pin_memory=True
            )
        )
        n = source.shape[0]
        if n > capacity[0] or len(req_ids) != n:
            raise ValueError("DSpark DP confidence does not match request batch")
        stream = self.runner.tokenID_processor.async_copy_stream
        producer_stream = torch.cuda.current_stream()
        event = torch.cuda.Event()
        with torch.cuda.stream(stream):
            stream.wait_stream(producer_stream)
            cpu[:n].copy_(source, non_blocking=True)
            event.record(stream)
        self.pending.append((self.epoch, event, cpu, tuple(req_ids), source))

    def _delayed(self, req_ids):
        for epoch, event, cpu, ids, _ in self.pending:
            if epoch == self.epoch - 2 and ids == req_ids:
                # Same fixed N-2 barrier as VerifyScheduler; never choose a
                # newer generation based on rank-local event readiness.
                event.synchronize()
                return cpu[: len(ids)].numpy().copy()
        return None

    def invalidate(self):
        self.survival = None
        # Retain DMA ownership while making every old generation unadoptable.
        self.pending = [
            (-1, event, cpu, ids, source) for _, event, cpu, ids, source in self.pending
        ]

    def prepare(self, batch):
        """Called on real, dummy, prefill, and output-less forward paths alike."""
        from atom.utils.region_graph import TokenGraph

        self.epoch += 1
        self.selected_q = None
        self.survival = None
        n = batch.total_seqs_num
        ids = tuple(batch.req_ids[:n])
        if ids != self._request_set:
            self._local_context += 1
            self._request_set = ids
        captured = tuple(
            g for g in TokenGraph.captured_sizes("decode") if g < len(self.baseline_ms)
        )
        if captured != self._captured:
            self.mesh.agree(captured)
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
        if self.mesh.tp_rank == 0 and ready:
            delayed = self._delayed(ids)
            if delayed is not None:
                confidence[0] = 1
                confidence[1:] = torch.from_numpy(delayed.reshape(-1))
        self.mesh.broadcast_tp(confidence)
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
        frontier, lengths = (
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
        wire = torch.from_numpy(np.concatenate((header, frontier.reshape(-1))))
        gathered = self.mesh.gather_dp(wire).numpy()
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
        self.scheduler._ell_map = {}  # invalid generations never inherit a plan
        if not self.state.pure_decode or not headers[:, 3].all() or not captured:
            return
        model = runtime.model if runtime is not None and not self.observe_only else None
        cost = (
            (lambda g: model.estimate_ms(g, self.state.batch))
            if model is not None
            else (lambda g: self.baseline_ms[g])
        )
        self.planned += 1
        choice = choose_dp_candidate(
            gathered[:, 8:].reshape(self.mesh.dp_size, -1, 4),
            captured,
            int(headers[:, 4].max()),
            int(headers[:, 5].max()),
            cost,
        )
        if choice is not None:
            self.contracted += 1
            self.selected_q = self.candidates[choice][0]
            self.scheduler._ell_map = dict(zip(ids, (lengths[choice] - 1).tolist()))
