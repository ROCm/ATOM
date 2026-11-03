# SPDX-License-Identifier: MIT
"""Real decode timing and TP/DP-consistent publication for online DSpark costs."""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass

import torch

from atom.spec_decode.online_sps import OnlineSPS, common_rank_samples

logger = logging.getLogger("atom")
MAX_PENDING = 8


def validate_online_runtime(config):
    if not (config.dspark.online_sps or getattr(config.dspark, "bucket_global", False)):
        return
    if (
        config.pipeline_parallel_size != 1
        or config.prefill_context_parallel_size != 1
        or config.decode_context_parallel_size != 1
        or config.enable_tbo
        or config.enforce_eager
    ):
        raise ValueError(
            "Online DSpark SPS requires PP/PCP/DCP=1, graphs, and TBO disabled"
        )
    from atom.spec_decode.dspark_scheduler import resolve_q_buckets

    spec = getattr(config, "speculative_config", None)
    width = getattr(spec, "num_speculative_tokens", 5) + 1
    qs = resolve_q_buckets(getattr(config.dspark, "ragged_graph_sizes", ""), width)
    if any(b % a for a, b in itertools.pairwise(qs)):
        # The top-m search admits global survival prefixes per q cap; with a
        # pair like 2 and 3 a non-prefix allocation can reach a smaller
        # divisible graph that search never visits.
        raise ValueError(
            f"Online DSpark SPS / bucket_global need nested query buckets (each "
            f"divides the next), got {qs}; e.g. 1,3,6 or 1,2,6"
        )
    mode = config.compilation_config.cudagraph_mode
    if mode is None or not mode.uses_tr_hybrid():
        raise ValueError(
            "Online DSpark SPS requires TR_HYBRID token graphs: FULL graphs "
            "do not apply ragged verification lengths"
        )


@dataclass
class Timing:
    step: int
    batch: int
    rows: int
    start: object
    target_end: object
    end: object
    context: int = -1
    finished: bool = False


class DSparkRuntime:
    def __init__(self, runner, *, comm=None, event_factory=None):
        self.runner = runner
        cfg = runner.config.dspark
        # Bound to the measured curve by VerifyScheduler.bind_sps: from the
        # offline profile at construction, or after graph capture.
        self.model = None
        self._window = cfg.online_sps_window_steps
        self.sample_interval = cfg.online_sps_sample_interval
        self.update_interval = cfg.online_sps_update_interval
        if comm is None:
            from atom.spec_decode.dspark_comm import DSparkComm

            comm = DSparkComm()
        self.comm = comm
        # The joint DP planner, for its counters in the published snapshot.
        self.dp_planner = None
        self.event_factory = event_factory or (
            lambda: torch.cuda.Event(enable_timing=True)
        )
        self.step = 0
        self.cutoff = 0
        self.pending = []
        self.free = []
        self.skipped_busy = 0
        self.table_cache = {}
        self._request_set = None
        self._context = 0
        self._send = torch.zeros((MAX_PENDING, 6), dtype=torch.float64, device="cpu")

    def bind(self, baseline_sps):
        self.model = OnlineSPS(
            baseline_sps, self.runner.config.max_num_seqs, window=self._window
        )
        self.table_cache.clear()

    def begin(self, batch, forward_mode, dp_state=None):
        # Under DP the planner settles a group-wide decode predicate and a
        # request-set generation (`dp_state`) at shape agreement. Dummy ranks
        # take part in both timing and publication; mixed P/D steps take part
        # in neither.
        dp = dp_state
        if self.model is None:
            return None  # warmup and capture run before the curve exists
        if self.comm.dp_size > 1 and (dp is None or not dp.pure_decode):
            return None
        if (
            batch is None
            or (
                dp is None
                and (
                    batch.is_dummy_run
                    or batch.total_tokens_num_prefill
                    or not batch.total_tokens_num_decode
                )
            )
            or not forward_mode.token_graph
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        self.step += 1
        req_ids = getattr(batch, "req_ids", None)
        if req_ids is not None:
            current = tuple(req_ids[: batch.total_seqs_num])
            if current != self._request_set:
                self._context += 1
                self._request_set = current
        # The cadence depends only on the TP-shared batch stream, never on local
        # event readiness. Publication does no CUDA synchronize; the small CPU
        # collective at this boundary is intentionally synchronous.
        if self.step % self.update_interval == 0:
            self._publish()
        self._reap()
        if self.step % self.sample_interval:
            return None
        if len(self.pending) >= MAX_PENDING:
            self.skipped_busy += 1
            return None
        events = (
            self.free.pop()
            if self.free
            else tuple(self.event_factory() for _ in range(3))
        )
        timing = Timing(
            self.step,
            dp.batch if dp is not None else batch.total_seqs_num,
            forward_mode.running_tokens,
            *events,
            context=(
                dp.context
                if dp is not None
                else (self._context if req_ids is not None else -1)
            ),
        )
        timing.start.record()
        self.pending.append(timing)
        return timing

    @staticmethod
    def target_done(timing):
        if timing is not None:
            timing.target_end.record()

    @staticmethod
    def finish(timing):
        if timing is not None:
            timing.end.record()
            timing.finished = True

    def _reap(self):
        keep = []
        for sample in self.pending:
            if sample.step <= self.cutoff and sample.finished and sample.end.query():
                self.free.append((sample.start, sample.target_end, sample.end))
            else:
                keep.append(sample)
        self.pending = keep

    def _publish(self):
        self._send.zero_()
        available = [
            s
            for s in self.pending
            if s.step > self.cutoff
            and s.step <= self.step - 2
            and s.finished
            and s.end.query()
        ]
        for i, sample in enumerate(available):
            self._send[i] = torch.tensor(
                (
                    sample.step,
                    sample.batch,
                    sample.rows,
                    sample.start.elapsed_time(sample.target_end),
                    sample.target_end.elapsed_time(sample.end),
                    sample.context,
                ),
                dtype=torch.float64,
                device="cpu",
            )
        # All local fields describe the SAME group workload: a DP rank's local
        # bs/ids must not be used as the matching key here.
        rows = (
            self.comm.gather_dp(self.comm.gather_tp(self._send))
            .reshape(-1, MAX_PENDING, 6)
            .tolist()
        )
        matched = common_rank_samples(rows)
        contexts = {sample.step: sample.context for sample in available}
        for step, batch, executed_rows, target_ms, post_ms in matched:
            context = contexts[step]
            self.model.observe(
                batch,
                executed_rows,
                target_ms,
                post_ms,
                step=step,
                context=context if context >= 0 else None,
            )
        if self.model.publish(step=self.step):
            self.table_cache.clear()
            if self.comm.tp_leader and (
                self.model.version == 1 or self.model.version % 16 == 0
            ):
                snapshot = self.model.snapshot()
                if self.dp_planner is not None:
                    snapshot["dp_planned"] = self.dp_planner.planned
                    snapshot["dp_contracted"] = self.dp_planner.contracted
                logger.info(
                    "DSpark online SPS %s", json.dumps(snapshot, sort_keys=True)
                )
        # All ranks retire this epoch, including incomplete/unmatched samples.
        # Unfinished events stay owned until query() permits reuse.
        self.cutoff = self.step - 1

    def table(self, base_sps, batch):
        if self.model is None or self.model.version == 0:
            return base_sps
        if batch not in self.table_cache:
            rows = torch.arange(
                base_sps.numel(), device=base_sps.device, dtype=base_sps.dtype
            )
            latency = (
                1000.0 / base_sps
                + self.model.fixed_ms(batch)
                + rows * self.model.slope_ms_for(batch)
                + self.model.post_ms(batch)
            )
            self.table_cache[batch] = 1000.0 / latency
        return self.table_cache[batch]

    def snapshot(self):
        return {
            **(self.model.snapshot() if self.model is not None else {}),
            "decode_steps": self.step,
            "pending": len(self.pending),
            "skipped_busy": self.skipped_busy,
            "sample_interval": self.sample_interval,
            "update_interval": self.update_interval,
        }
