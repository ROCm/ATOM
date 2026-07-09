# SPDX-License-Identifier: MIT
# Pipeline-parallel EngineCore: one per PP stage (CPP A·P1b, 方案②).
#
# Each stage is an independent EngineCore process (reusing the DP multi-core
# infrastructure). Only the head (stage 0) owns the Scheduler and the request
# lifecycle; downstream stages are executors driven by the head's metadata.
#
#   head (stage 0):  schedule -> ZMQ metadata to downstream
#                             -> forward its layers (NCCL-send hidden downstream)
#                             -> ring=1: block for last stage's sampled tokens
#                             -> scheduler.postprocess(tokens) -> emit / next step
#   downstream:      recv metadata -> forward its layers (NCCL-recv hidden,
#                             middle stages NCCL-send onward)
#   last stage:      ... -> sample -> ZMQ tokens back to head
#
# Hidden states move GPU-to-GPU over NCCL inside ModelRunner.run_model
# (pp_comm.py); only the scheduled batch and the sampled output cross here
# (pp_transport.py).
#
# ring=1 (serial, this phase): the head blocks on the token feedback before
# scheduling the next batch. P2 (overlap) will let up to pp_size batches be
# in-flight — same code, the head just stops blocking.

import logging
import queue

from atom.distributed.pp_transport import PPStageTransport
from atom.model_engine.engine_core import EngineCore

logger = logging.getLogger("atom")


class PPEngineCoreProc(EngineCore):
    def __init__(self, config, input_address, output_address):
        pc = config.parallel_config
        self.pp_rank = pc.pipeline_parallel_rank
        self.pp_size = config.pipeline_parallel_size
        self.is_head = self.pp_rank == 0
        self.is_last = self.pp_rank == self.pp_size - 1
        # super().__init__ spawns this stage's workers, allocates its layers' KV
        # cache, builds the Scheduler (only the head uses it), and sends READY.
        super().__init__(config, input_address, output_address)
        self.pp_transport = PPStageTransport(
            self.pp_rank,
            self.pp_size,
            pc.pp_meta_addrs,
            pc.pp_token_addr,
        )
        logger.info(
            f"{self.label}: PP stage {self.pp_rank}/{self.pp_size} "
            f"(head={self.is_head}, last={self.is_last}) ready"
        )

    def busy_loop(self):
        if self.is_head:
            self._head_busy_loop()
        else:
            self._downstream_busy_loop()

    # ---- head (stage 0) -----------------------------------------------------
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
                if not self.scheduler.is_finished():
                    self._pp_head_step()
        finally:
            try:
                self.scheduler.publish_kv_events()
            except Exception:
                logger.exception("KV event publish during shutdown failed")
            self.scheduler.shutdown_kv_events()

    def _pp_head_step(self):
        result = self.scheduler.schedule()

        rejected = self.scheduler.take_rejected()
        if rejected:
            self.output_queue.put_nowait(rejected)

        if result is None:
            return
        scheduled_batch, seqs = result
        if scheduled_batch is None or len(scheduled_batch.req_ids) == 0:
            return

        # 1. Hand the scheduled batch to every downstream stage (CPU/ZMQ).
        self.pp_transport.send_metadata(scheduled_batch)
        # 2. Run this stage's layers; workers NCCL-send hidden downstream and
        #    (non-last) produce no logits. The returned output is empty here.
        self.runner_mgr.call_func("forward", scheduled_batch, wait_out=True)
        # 3. ring=1: block for the last stage's sampled tokens.
        fwd_out = self.pp_transport.recv_tokens()

        # 4. Head owns the lifecycle: append tokens, detect finish, emit.
        seqs = seqs.values()
        finished_seqs = self.scheduler.postprocess(
            seqs,
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

    # ---- downstream / last stage -------------------------------------------
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
                # Wait (with timeout, to stay responsive to shutdown) for the
                # head's scheduled batch, then run this stage's layers.
                batch = self.pp_transport.recv_metadata(timeout_ms=100)
                if batch is None:
                    continue
                fwd_out = self.runner_mgr.call_func("forward", batch, wait_out=True)
                if self.is_last:
                    # Feed the sampled tokens back to the head.
                    self.pp_transport.send_tokens(fwd_out)
        finally:
            try:
                self.scheduler.publish_kv_events()
            except Exception:
                logger.exception("KV event publish during shutdown failed")
            self.scheduler.shutdown_kv_events()
