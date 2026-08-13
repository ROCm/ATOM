# SPDX-License-Identifier: MIT
# Pipeline-parallel EngineCore: one per PP stage.
# Head (stage 0) owns the Scheduler; downstream stages are stateless executors.
# Hidden states move over NCCL (pp_comm.py); batch metadata and sampled tokens
# cross stages via ZMQ (pp_transport.py).

import logging
import queue
from collections import deque

from atom.distributed.pp_transport import PPStageTransport
from atom.kv_transfer.disaggregation.pp_kv_aggregator import PPKVAggregator
from atom.kv_transfer.disaggregation.types import KVConnectorOutput
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
            kv_status_addr=getattr(pc, "pp_kv_status_addr", ""),
        )
        self._in_flight: deque = deque()
        self._pending_prefix_hash: deque = deque()
        bm = self.scheduler.block_manager
        # Deferring used to be off under SWA: the sliding window published its
        # blocks into a content index in lockstep with the compressed ones, and
        # a deferred hash would have let the two disagree. The window is a
        # per-request ring now and publishes nothing, so the exception is gone.
        self._defer_prefix_hash: bool = bm.enable_prefix_caching
        self._pp_kv_aggregator: PPKVAggregator | None = None
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
            if scheduled_batch is None:
                break
            if len(scheduled_batch.req_ids) == 0:
                self._dispatch_connector_only_batch(scheduled_batch)
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
                if self._defer_prefix_hash:
                    self._pending_prefix_hash.append((scheduled_batch, seqs))
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
            self._flush_pending_prefix_hashes()
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

    def _flush_pending_prefix_hashes(self):
        while self._pending_prefix_hash:
            batch, seqs = self._pending_prefix_hash.popleft()
            try:
                self.scheduler.register_prefill_hashes(batch, seqs.values())
            except Exception:
                logger.exception(
                    "register_prefill_hashes failed for batch %s — "
                    "prefix-cache hits may degrade but inference continues",
                    list(batch.req_ids),
                )

    # -- KV transfer PP aggregation ------------------------------------------

    def _dispatch_connector_only_batch(self, batch) -> None:
        """Dispatch the KV connector metadata of a batch that has no requests.

        The metadata starts offload loads; dropping it strands parked
        sequences. Every stage must see it so ``PPKVAggregator`` reaches
        global completion.
        """
        if not self.kv_transfer_enabled:
            return
        meta = batch.connector_meta_output
        if meta is None or not getattr(meta, "requests", None):
            return
        self.runner_mgr.call_func("process_kvconnector_output", meta)
        self.pp_transport.send_metadata(batch)

    def _poll_kv_transfer_progress(self):
        """Aggregate KV transfer status from local TP workers AND downstream
        PP stages, then feed the result to the scheduler.

        For non-offload fields (finished_sending, finished_recving, etc.) the
        head's own TP-aggregated output goes directly to the scheduler — those
        are handled by mooncake's own PP-aware side-channel.

        For offload fields (finished_loading, failed_loading, finished_saving)
        the head's output is fed into :class:`PPKVAggregator` together with
        downstream stages' reports, and only globally-complete items reach the
        scheduler.
        """
        if not self.kv_transfer_enabled:
            return

        # Collect local TP-aggregated output.
        kvoutput = self.runner_mgr.call_func_with_aggregation("async_proc_aggregation")
        if kvoutput is None:
            kvoutput = KVConnectorOutput()

        # Non-offload fields go directly to scheduler.
        non_offload = KVConnectorOutput(
            finished_sending=kvoutput.finished_sending,
            finished_recving=kvoutput.finished_recving,
            failed_recving=kvoutput.failed_recving,
        )
        if not non_offload.is_empty():
            self.scheduler._update_from_kv_xfer_finished(non_offload)

        # Offload fields go through PP aggregator.
        has_offload = (
            kvoutput.finished_loading
            or kvoutput.failed_loading
            or kvoutput.finished_saving
        )
        pp_messages = self.pp_transport.recv_kv_status(timeout_ms=0)

        if not has_offload and not pp_messages:
            return

        if self._pp_kv_aggregator is None:
            self._pp_kv_aggregator = PPKVAggregator(self.pp_size)

        # Ingest head (stage 0) offload output.
        offload_local = KVConnectorOutput(
            finished_loading=kvoutput.finished_loading,
            failed_loading=kvoutput.failed_loading,
            finished_saving=kvoutput.finished_saving,
        )
        if not offload_local.is_empty():
            result = self._pp_kv_aggregator.ingest(0, offload_local)
            if not result.is_empty():
                self.scheduler._update_from_kv_xfer_finished(result)

        # Ingest downstream PP stages' offload output.
        for pp_rank, downstream_output in pp_messages:
            result = self._pp_kv_aggregator.ingest(pp_rank, downstream_output)
            if not result.is_empty():
                self.scheduler._update_from_kv_xfer_finished(result)

    # -- Downstream busy loop ------------------------------------------------

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
                    if self.kv_transfer_enabled:
                        self._poll_and_send_kv_status()
                    self.runner_mgr.call_func("flush_pp_send", wait_out=True)
                    continue

                if (
                    self.kv_transfer_enabled
                    and getattr(batch, "connector_meta_output", None) is not None
                ):
                    self.runner_mgr.call_func(
                        "process_kvconnector_output",
                        batch.connector_meta_output,
                    )

                if len(batch.req_ids) == 0:
                    if self.kv_transfer_enabled:
                        self._poll_and_send_kv_status()
                    continue

                fwd_out = self.runner_mgr.call_func("forward", batch, wait_out=True)

                if self.kv_transfer_enabled:
                    self._poll_and_send_kv_status()

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

    def _poll_and_send_kv_status(self):
        """Downstream: collect TP-aggregated KV status and send to head."""
        kvoutput = self.runner_mgr.call_func_with_aggregation("async_proc_aggregation")
        if kvoutput is not None and not kvoutput.is_empty():
            self.pp_transport.send_kv_status(kvoutput)
