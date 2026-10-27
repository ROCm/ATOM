# SPDX-License-Identifier: MIT
"""Real decode timing and TP-consistent publication for online DSpark costs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from atom.spec_decode.online_sps import OnlineSPS, common_rank_samples

logger = logging.getLogger("atom")
MAX_PENDING = 8


def validate_online_runtime(config):
    if not (config.dspark.online_sps or getattr(config.dspark, "bucket_global", False)):
        return
    if (
        config.parallel_config.data_parallel_size != 1
        or config.pipeline_parallel_size != 1
        or config.prefill_context_parallel_size != 1
        or config.decode_context_parallel_size != 1
        or config.enable_expert_parallel
        or config.enable_tbo
        or config.enforce_eager
    ):
        raise ValueError(
            "Online DSpark SPS currently requires TP-only, DP/PP/PCP/DCP=1, graphs, and TBO disabled"
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
    def __init__(self, runner, *, group=None, event_factory=None):
        validate_online_runtime(runner.config)
        self.runner = runner
        cfg = runner.config.dspark
        profile = json.loads(Path(cfg.calibration_profile).read_text(encoding="utf-8"))
        self.model = OnlineSPS(
            profile["sps_table"],
            runner.config.max_num_seqs,
            window=cfg.online_sps_window_steps,
            cohort_steps=cfg.online_sps_update_interval,
        )
        self.sample_interval = cfg.online_sps_sample_interval
        self.update_interval = cfg.online_sps_update_interval
        if group is None:
            from aiter.dist.parallel_state import get_tp_group

            group = get_tp_group().cpu_group
        self.group = group
        self.world_size = dist.get_world_size(group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(group) if dist.is_initialized() else 0
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
        self._recv = [torch.empty_like(self._send) for _ in range(self.world_size)]

    def begin(self, batch, forward_mode):
        if (
            batch is None
            or batch.is_dummy_run
            or batch.total_tokens_num_prefill
            or not batch.total_tokens_num_decode
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
            batch.total_seqs_num,
            forward_mode.running_tokens,
            *events,
            context=self._context if req_ids is not None else -1,
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
        if self.world_size > 1:
            dist.all_gather(self._recv, self._send, group=self.group)
            rows = [r.tolist() for r in self._recv]
        else:
            rows = [self._send.tolist()]
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
            if self.rank == 0 and (
                self.model.version == 1 or self.model.version % 16 == 0
            ):
                logger.info(
                    "DSpark online SPS %s",
                    json.dumps(self.model.snapshot(), sort_keys=True),
                )
        # All ranks retire this epoch, including incomplete/unmatched samples.
        # Unfinished events stay owned until query() permits reuse.
        self.cutoff = self.step - 1

    def table(self, base_sps, batch):
        if self.model.version == 0:
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
            **self.model.snapshot(),
            "decode_steps": self.step,
            "pending": len(self.pending),
            "skipped_busy": self.skipped_busy,
            "sample_interval": self.sample_interval,
            "update_interval": self.update_interval,
        }
