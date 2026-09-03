# SPDX-License-Identifier: MIT
# Pipeline-parallel EngineCore: one per PP stage.
# Head (stage 0) owns the Scheduler; downstream stages are stateless executors.
# Hidden states move over NCCL (pp_comm.py); batch metadata and sampled tokens
# cross stages via ZMQ (pp_transport.py).

import logging
import queue
import time
from collections import deque

from atom.distributed.pp_transport import PPStageTransport
from atom.kv_transfer.disaggregation.pp_kv_aggregator import PPKVAggregator
from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    completion_req_key,
    connector_metadata_has_work,
)
from atom.model_engine.engine_core import EngineCore
from atom.model_engine.scheduler import ScheduledBatch

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
        self._held_sending: dict[str, tuple] = {}
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
                self.scheduler.heartbeat_throughput(time.monotonic())
                shutdown = shutdown or self.pull_and_process_input_queue()
                if shutdown:
                    break
                if self._is_idle_rl_weights_offloaded():
                    continue
                if self._in_flight or not self.scheduler.is_finished():
                    self._pp_head_step()
                elif self.has_pending_kv_work():
                    self._advance_idle_kv_transfer()
        finally:
            self._drain_kv_work_at_exit()
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

    def has_pending_kv_work(self) -> bool:
        """Extend the base predicate with the head's PP-only holding state.

        ``_held_sending`` pins a mooncake send until every stage has reported
        its save, and ``_pp_kv_aggregator`` holds the partial per-stage
        tallies that release it. Both outlive the scheduler queues, and both
        only drain from ``_poll_kv_transfer_progress``.
        """
        if super().has_pending_kv_work():
            return True
        if self._held_sending:
            return True
        return (
            self._pp_kv_aggregator is not None and self._pp_kv_aggregator.has_pending()
        )

    def _dispatch_idle_offload_work(self, dispatch_new: bool = True) -> None:
        """Override: fan the idle connector metadata out to every PP stage.

        ``Scheduler.schedule()`` returns None once waiting and running are
        both empty, so the connector-only batch it normally builds never
        materializes while draining. Build the metadata directly instead, and
        ship it downstream too — otherwise the stages never save their layers
        and ``PPKVAggregator`` cannot reach a quorum.

        ``dispatch_new`` exists only to match the base signature the shutdown
        drain calls through: this override never publishes new state loads/
        stores (the state tier refuses ``pp_size > 1`` outright, so there are
        none), so it has nothing to gate and the flag is inert here.
        """
        del dispatch_new
        if not self.kv_transfer_enabled:
            return
        connector = getattr(self.scheduler, "kv_connector", None)
        if connector is None or not getattr(connector, "is_offload", False):
            return
        self._dispatch_connector_only_batch(
            ScheduledBatch(
                seqs={},
                num_scheduled_tokens=[],
                total_tokens_num=0,
                connector_meta_output=connector.build_connector_meta(),
            )
        )

    def _dispatch_connector_only_batch(self, batch) -> None:
        """Dispatch the KV connector metadata of a batch that has no requests.

        The metadata starts offload loads; dropping it strands parked
        sequences. Every stage must see it so ``PPKVAggregator`` reaches
        global completion.
        """
        if not self.kv_transfer_enabled:
            return
        meta = batch.connector_meta_output
        if not connector_metadata_has_work(meta):
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

        # Reclaim any offload save whose completion report never came (worker
        # crash, dropped completion, LMCache force-unpin). This override fully
        # replaces the base `_poll_kv_transfer_progress`, whose getattr-guarded
        # call is the reclaimer's only caller repo-wide; without mirroring it
        # here, under `pp_size > 1` a stalled save stays in
        # `Scheduler.deferred_free_blocks` forever -- `has_pending_kv_work()`
        # stays True, so the engine busy-loops with every GPU idle, the blocks
        # never return to the pool, and `_drain_kv_work_at_exit` spins to
        # `KV_SHUTDOWN_DRAIN_TIMEOUT_S` on every shutdown. Self-throttled, so
        # calling it each poll is cheap; placed above the has_offload /
        # pp_messages early-returns so a quiet poll still reclaims.
        reconcile = getattr(self.scheduler, "_reconcile_stalled_deferred_saves", None)
        if callable(reconcile):
            reconcile()

        # Collect local TP-aggregated output.
        kvoutput = self.runner_mgr.call_func_with_aggregation("async_proc_aggregation")
        if kvoutput is None:
            kvoutput = KVConnectorOutput()

        # Recv/failed_recving go directly to scheduler.
        non_offload = KVConnectorOutput(
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
            # connector_completions are offload channel events (kimi_k3 state
            # index reports, dsv4 checkpoint boundaries). They too span all PP
            # stages, so they must reach the aggregator rather than the
            # scheduler directly -- and count as "offload work" so this poll
            # does not early-return and strand them.
            or kvoutput.connector_completions
        )
        pp_messages = self.pp_transport.recv_kv_status(timeout_ms=0)

        if not has_offload and not pp_messages and not kvoutput.finished_sending:
            return

        # No offload connector → finished_sending goes straight to scheduler.
        if self._pp_kv_aggregator is None and not has_offload and not pp_messages:
            if kvoutput.finished_sending:
                self.scheduler._update_from_kv_xfer_finished(
                    KVConnectorOutput(finished_sending=kvoutput.finished_sending)
                )
            return

        if self._pp_kv_aggregator is None:
            self._pp_kv_aggregator = PPKVAggregator(self.pp_size)

        # MultiConnector emits a send together with its stage-local saves, so
        # a send arriving alone belongs to a request no stage is saving —
        # holding it would strand it forever. A paired send waits for the
        # PP-wide quorum, which is per save generation, hence the whole set.
        # It is complete: mooncake sends only after the request's last chunk.
        local_saving = {
            completion_req_key(rid) for rid in kvoutput.finished_saving or ()
        }
        unpaired_sending = set()
        for rid in kvoutput.finished_sending or ():
            key = completion_req_key(rid)
            if key in local_saving:
                self._held_sending[key] = (
                    rid,
                    {
                        op
                        for op in kvoutput.finished_saving
                        if completion_req_key(op) == key
                    },
                )
            else:
                unpaired_sending.add(rid)
        if unpaired_sending:
            self.scheduler._update_from_kv_xfer_finished(
                KVConnectorOutput(finished_sending=unpaired_sending)
            )

        # Ingest head (stage 0) offload output.
        offload_local = KVConnectorOutput(
            finished_loading=kvoutput.finished_loading,
            failed_loading=kvoutput.failed_loading,
            finished_saving=kvoutput.finished_saving,
            connector_completions=kvoutput.connector_completions,
        )
        if not offload_local.is_empty():
            self._ingest_and_release(offload_local, 0)

        # Ingest downstream PP stages' offload output.
        for pp_rank, downstream_output in pp_messages:
            self._ingest_and_release(downstream_output, pp_rank)

    def _ingest_and_release(self, output: KVConnectorOutput, pp_rank: int):
        result = self._pp_kv_aggregator.ingest(pp_rank, output)
        if result.is_empty():
            return
        # Release held finished_sending whose global save is now complete.
        rel = set()
        for rid in result.finished_saving or ():
            key = completion_req_key(rid)
            held = self._held_sending.get(key)
            if held is None:
                continue
            raw, pending = held
            pending.discard(rid)
            if not pending:
                del self._held_sending[key]
                rel.add(raw)
        if rel:
            result.finished_sending = rel
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
            # One last report so the head's exit drain can still reach its
            # per-stage quorum for saves that landed after the final poll.
            try:
                if self.kv_transfer_enabled:
                    self._poll_and_send_kv_status()
            except Exception:
                logger.exception("final KV status report during shutdown failed")
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
