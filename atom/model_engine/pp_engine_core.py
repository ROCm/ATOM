# SPDX-License-Identifier: MIT
# Pipeline-parallel EngineCore: one per PP stage.
# Head (stage 0) owns the Scheduler; downstream stages are stateless executors.
# Hidden states move over NCCL (pp_comm.py); batch metadata and sampled tokens
# cross stages via ZMQ (pp_transport.py).

import logging
import queue
from collections import deque

from atom.distributed.pp_transport import PPStageTransport
from atom.model_engine.engine_core import EngineCore

logger = logging.getLogger("atom")

# Collect poll timeout when the step made no other progress: bounds both
# new-request admission latency and busy-spinning while batches are in flight.
_PP_HEAD_IDLE_POLL_MS = 1


class PPEngineCoreProc(EngineCore):
    def __init__(self, config, input_address, output_address):
        pc = config.parallel_config
        self.pp_rank = pc.pipeline_parallel_rank
        self.pp_size = config.pipeline_parallel_size
        self.is_head = self.pp_rank == 0
        self.is_last = self.pp_rank == self.pp_size - 1
        super().__init__(config, input_address, output_address)
        self.pp_transport = PPStageTransport(
            self.pp_rank,
            self.pp_size,
            pc.pp_meta_addrs,
            pc.pp_token_addr,
        )
        self._in_flight: deque = deque()
        logger.info(
            f"{self.label}: PP stage {self.pp_rank}/{self.pp_size} "
            f"(head={self.is_head}, last={self.is_last}) ready"
        )

    def busy_loop(self):
        if self.is_head:
            self._head_busy_loop()
        else:
            self._downstream_busy_loop()

    def _head_busy_loop(self):
        shutdown = False
        try:
            while True:
                self.utility_handler.process_queue(self.utility_queue, self)
                shutdown = shutdown or self.pull_and_process_input_queue()
                if shutdown:
                    break
                if self._is_idle_rl_weights_offloaded():
                    continue
                if self._in_flight or not self.scheduler.is_finished():
                    self._pp_head_step()
        finally:
            try:
                self.runner_mgr.call_func("flush_pp_send", wait_out=True)
            except Exception:
                logger.exception("flush_pp_send during shutdown failed")
            try:
                self.scheduler.publish_kv_events()
            except Exception:
                logger.exception("KV event publish during shutdown failed")
            self.scheduler.shutdown_kv_events()

    def _pp_head_step(self):
        launched = 0
        while len(self._in_flight) < self.pp_size:
            result = self.scheduler.schedule()

            rejected = self.scheduler.take_rejected()
            if rejected:
                self.output_queue.put_nowait(rejected)

            if result is None:
                break
            scheduled_batch, seqs = result
            if scheduled_batch is None or len(scheduled_batch.req_ids) == 0:
                break

            needs_output = scheduled_batch.produces_output()
            if (
                self.kv_transfer_enabled
                and scheduled_batch.connector_meta_output is not None
            ):
                self.runner_mgr.call_func(
                    "process_kvconnector_output",
                    scheduled_batch.connector_meta_output,
                )
            self.pp_transport.send_metadata(scheduled_batch)
            self.runner_mgr.call_func("forward", scheduled_batch, wait_out=True)
            self.scheduler.mark_pp_inflight(scheduled_batch)
            self._in_flight.append((scheduled_batch, seqs, needs_output))
            launched += 1

        # Flush deferred send when idle — otherwise it dangles until next forward.
        if launched == 0:
            self.runner_mgr.call_func("flush_pp_send", wait_out=True)

        self._poll_kv_transfer_progress()

        poll_ms = 0 if launched else _PP_HEAD_IDLE_POLL_MS
        while self._in_flight:
            scheduled_batch, seqs, needs_output = self._in_flight[0]
            if not needs_output:
                self._in_flight.popleft()
                self.scheduler.release_pp_inflight(scheduled_batch)
                continue

            fwd_out = self.pp_transport.recv_tokens(timeout_ms=poll_ms)
            if fwd_out is None:
                break
            poll_ms = 0

            assert list(fwd_out.req_ids) == list(scheduled_batch.req_ids), (
                f"PP token ordering violated: received {list(fwd_out.req_ids)}, "
                f"expected FIFO head {list(scheduled_batch.req_ids)}"
            )

            self._in_flight.popleft()
            self.scheduler.release_pp_inflight(scheduled_batch)
            finished_seqs = self.scheduler.postprocess(
                seqs.values(),
                fwd_out,
                stream_output_queue=self.stream_output_queue,
                batch=scheduled_batch,
            )
            try:
                while not self.stream_output_queue.empty():
                    stream_outputs = self.stream_output_queue.get_nowait()
                    self.output_queue.put_nowait(("STREAM", stream_outputs))
            except queue.Empty:
                pass
            if finished_seqs:
                self.output_queue.put_nowait(finished_seqs)

    def _downstream_busy_loop(self):
        shutdown = False
        try:
            while True:
                self.utility_handler.process_queue(self.utility_queue, self)
                shutdown = shutdown or self.pull_and_process_input_queue()
                if shutdown:
                    break
                if self._is_idle_rl_weights_offloaded():
                    continue
                batch = self.pp_transport.recv_metadata(timeout_ms=100)
                if batch is None:
                    self.runner_mgr.call_func("flush_pp_send", wait_out=True)
                    continue
                fwd_out = self.runner_mgr.call_func("forward", batch, wait_out=True)
                if self.is_last and batch.produces_output():
                    self.pp_transport.send_tokens(fwd_out)
        finally:
            try:
                self.runner_mgr.call_func("flush_pp_send", wait_out=True)
            except Exception:
                logger.exception("flush_pp_send during shutdown failed")
            try:
                self.scheduler.publish_kv_events()
            except Exception:
                logger.exception("KV event publish during shutdown failed")
            self.scheduler.shutdown_kv_events()
