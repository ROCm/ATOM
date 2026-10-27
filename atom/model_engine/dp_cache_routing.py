# SPDX-License-Identifier: MIT
"""CoreManager integration for cache-affinity routing and safe retry."""

import json
import logging
import pickle
import queue
import time
import uuid
from threading import BoundedSemaphore, Thread

from atom.model_engine.dp_cache_balance import CacheBalanceConfig, CacheBalancer
from atom.utils import envs

logger = logging.getLogger("atom")


class DPCacheRouting:
    def _init_dp_cache_balance(self, config):
        self._cache_balancer = None
        self._cache_retry_payloads = {}
        if config.dp_load_balance != "cache_affinity":
            return
        if not config.enable_prefix_caching:
            raise ValueError("cache_affinity requires prefix caching")
        if getattr(config, "pipeline_parallel_size", 1) != 1:
            raise ValueError("cache_affinity currently requires PP=1")
        if getattr(config, "disagg_is_decode", False) or getattr(
            config, "disagg_is_prefill", False
        ):
            raise ValueError(
                "cache_affinity routes complete DP engines, not PD-only workers"
            )

        def has_mp(kvc):
            return isinstance(kvc, dict) and (
                kvc.get("kv_connector") == "lmcache_mp"
                or any(has_mp(c) for c in kvc.get("connectors", ()))
            )

        if not has_mp(getattr(config, "kv_transfer_config", None)):
            raise ValueError(
                "cache_affinity requires lmcache_mp and a shared cache tier; "
                "V4.1 must use the native PAGE/STATE connector"
            )
        hf = getattr(config, "hf_config", None)
        hf = getattr(hf, "text_config", hf)
        if getattr(hf, "model_type", None) == "deepseek_v41" and getattr(
            config, "enable_encoder_swa_bounded_replay", True
        ):
            raise ValueError(
                "V4.1 cache_affinity requires --no-encoder-swa-bounded-replay "
                "to retain native PAGE/STATE checkpoints for lmcache_mp"
            )
        values = json.loads(envs.ATOM_DP_CACHE_BALANCE_CONFIG)
        if isinstance(values, dict):
            values.setdefault(
                "checkpoint_tokens",
                getattr(config, "state_checkpoint_interval_tokens", 8192),
            )
        options = CacheBalanceConfig.parse(values)
        self._cache_balancer = CacheBalancer(self._routable_engine_count, options)
        self._cache_probe_slots = BoundedSemaphore(options.max_migrations)

    def _select_cache_rank_locked(self, seq, hint):
        balancer = self._cache_balancer
        chain = getattr(seq, "prompt_hash_chain", None)
        placement = balancer.place(
            seq.id,
            getattr(seq, "dp_session_id", None),
            seq.num_prompt_tokens,
            chain,
            self._rank_reqs,
            self._rank_tokens,
            time.monotonic(),
            explicit_rank=hint,
            can_migrate=True,
        )
        seq.dp_cache_balance = True
        seq.dp_cache_migration = placement.migration
        seq.dp_cache_source_rank = placement.source if placement.migration else None
        seq.dp_cache_minimum_hit = placement.minimum_hit
        seq.dp_cache_prepare_timeout = balancer.config.prepare_timeout_seconds
        if placement.migration:
            # The frontend Sequence is never modified by the target process.
            # ADD detaches its hash chain, so retain that separately for retry.
            self._cache_retry_payloads[seq.id] = (seq, chain)
        return placement.rank

    def _cache_admitted(self, rank, event):
        balancer = getattr(self, "_cache_balancer", None)
        if balancer is None:
            return
        sid, migration, hit = event
        with self._lb_lock:
            load = self._seq_load.get(sid)
            if load is None or load[0] != rank:
                return
            placement = balancer.requests.get(sid)
            if placement is None:
                return
            hit = max(0, min(int(hit), placement.prompt_tokens))
            if not balancer.admitted(sid, rank, migration, hit, time.monotonic()):
                return
            remaining = placement.prompt_tokens - hit
            self._rank_tokens[rank] += remaining - load[2]
            self._seq_load[sid] = (rank, load[1], remaining)
            self._cache_retry_payloads.pop(sid, None)

    def _retry_cache_migration(self, rank, returned_seq):
        """Consume target rejections, including stale replies after cancellation.

        Retries use the locked control socket: this is an output thread, not
        the single writer that owns the normal request socket.
        """
        if not getattr(returned_seq, "dp_cache_retry_reason", None):
            return False
        balancer = getattr(self, "_cache_balancer", None)
        if balancer is None:
            return False
        with self._lb_lock:
            sid = returned_seq.id
            payload = self._cache_retry_payloads.get(sid)
            load = self._seq_load.get(sid)
            if payload is None or load is None or load[0] != rank:
                return True
            migration = returned_seq.dp_cache_migration
            if (
                returned_seq.dp_cache_retry_reason == "recoverable_prefix_too_short"
                and hasattr(returned_seq, "dp_cache_recoverable_hit")
            ):
                if not balancer.begin_reconsider(sid, rank, migration):
                    return True
                if self._cache_probe_slots.acquire(blocking=False):
                    try:
                        Thread(
                            target=self._reconsider_cache_migration,
                            args=(
                                balancer,
                                rank,
                                returned_seq,
                                payload,
                                balancer.cache_epoch,
                            ),
                            name="CachePrefixReconsider",
                            daemon=True,
                        ).start()
                        return True
                    except Exception:
                        self._cache_probe_slots.release()
                        logger.exception("Could not start cache prefix probe")
                # No unbounded executor queue: under probe pressure compare
                # without giving A credit for its unverified historical hint.
                balancer.counters["migration_source_probe_busy"] += 1
                return self._apply_cache_recovery_locked(
                    balancer, rank, returned_seq, None, balancer.cache_epoch
                )
            source = balancer.retry(
                sid,
                rank,
                returned_seq.dp_cache_migration,
                time.monotonic(),
                reason=returned_seq.dp_cache_retry_reason,
            )
            if source is None:
                return True
            return self._resubmit_cache_placement_locked(balancer, returned_seq)

    def _cache_probe_active(self, balancer, sid, migration):
        p = balancer.requests.get(sid)
        return (
            self._cache_balancer is balancer
            and not getattr(self, "_closed", False)
            and sid in self._seq_load
            and p is not None
            and p.migration == migration
            and p.reconsidering
        )

    def _probe_cache_source(self, balancer, seq, payload):
        """Bounded source-only RPC; the output thread must remain free to reply."""
        import zmq

        from atom.model_engine.engine_core_protocol import EngineCoreRequestType

        deadline = time.monotonic() + balancer.config.source_probe_timeout_seconds
        request_id = uuid.uuid4().hex
        with self._lb_lock:
            if not self._cache_probe_active(balancer, seq.id, seq.dp_cache_migration):
                return None, "cancelled"
            source = balancer.requests[seq.id].source
        original, chain = payload
        frame = pickle.dumps(
            (
                EngineCoreRequestType.UTILITY,
                {
                    "cmd": "probe_local_prefix",
                    "request_id": request_id,
                    "seq": original,
                    "chain": chain,
                },
            )
        )
        with self._rpc_router.register(request_id) as replies:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._control_send_lock.acquire(timeout=remaining):
                return None, "timeout"
            try:
                # Probe congestion must not hold the control socket or consume
                # an unbounded part of the request's TTFT budget.
                self.control_sockets[source].send_multipart(
                    [self.control_identities[source], frame],
                    flags=zmq.DONTWAIT,
                    copy=True,
                )
            finally:
                self._control_send_lock.release()
            while True:
                with self._lb_lock:
                    if not self._cache_probe_active(
                        balancer, seq.id, seq.dp_cache_migration
                    ):
                        return None, "cancelled"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, "timeout"
                try:
                    rank, body = replies.get(timeout=min(0.05, remaining))
                except queue.Empty:
                    continue
                if rank != source or body.get("cmd") != "probe_local_prefix":
                    continue
                hit = body.get("result")
                if type(hit) is int and 0 <= hit < original.num_prompt_tokens:
                    return hit, "ok"
                return None, "unavailable"

    def _reconsider_cache_migration(self, balancer, rank, seq, payload, epoch):
        try:
            try:
                hit, status = self._probe_cache_source(balancer, seq, payload)
            except Exception:
                hit, status = None, "unavailable"
                logger.debug("Source prefix probe unavailable", exc_info=True)
            with self._lb_lock:
                if not self._cache_probe_active(
                    balancer, seq.id, seq.dp_cache_migration
                ):
                    return
                balancer.counters["migration_source_probe_" + status] += 1
                consumed = self._apply_cache_recovery_locked(
                    balancer, rank, seq, hit, epoch
                )
                if not consumed:
                    # Async retry has no ADD output loop to do terminal cleanup.
                    # Remove its charge under the same lock as retry/cancel.
                    r, n, t = self._seq_load.pop(seq.id)
                    self._rank_reqs[r] -= n
                    self._rank_tokens[r] -= t
                    self._finish_cache_placement_locked(seq.id, False)
            if not consumed:
                delivered = self._deliver_terminal_output(seq)
                if delivered and self._flush_stream_batch_fn is not None:
                    try:
                        self._flush_stream_batch_fn()
                    except Exception:
                        logger.exception("Cache retry terminal flush failed")
                self.outputs_queue.put_nowait([seq])
        finally:
            self._cache_probe_slots.release()

    def _apply_cache_recovery_locked(self, balancer, rank, seq, source_hit, epoch):
        load = self._seq_load[seq.id]
        requests, tokens = self._rank_reqs.copy(), self._rank_tokens.copy()
        requests[rank] -= load[1]
        tokens[rank] -= load[2]
        target_hit = seq.dp_cache_recoverable_hit
        if balancer.cache_epoch != epoch:
            # Clear/weight change invalidates both observations and fences the
            # owner commit; the request still needs to run on one engine.
            source_hit, target_hit = None, 0
            balancer.counters["migration_reconsider_invalidated"] += 1
        choice = balancer.reconsider(
            seq.id,
            rank,
            seq.dp_cache_migration,
            source_hit,
            target_hit,
            requests,
            tokens,
            time.monotonic(),
        )
        if choice is None:
            return True
        logger.debug(
            "Cache recovery seq=%s target=%d chosen=%d source_hit=%d target_hit=%d "
            "source_seconds=%.4f target_seconds=%.4f",
            seq.id,
            rank,
            choice.rank,
            choice.source_hit,
            choice.target_hit,
            choice.source_seconds,
            choice.target_seconds,
        )
        return self._resubmit_cache_placement_locked(balancer, seq)

    def _resubmit_cache_placement_locked(self, balancer, returned_seq):
        from atom.model_engine.engine_core_mgr import _add_frame

        sid = returned_seq.id
        original, chain = self._cache_retry_payloads[sid]
        placement = balancer.requests[sid]
        original.prompt_hash_chain = chain
        original.dp_cache_migration = placement.migration
        original.dp_cache_source_rank = placement.source
        original.dp_cache_minimum_hit = placement.minimum_hit
        if not placement.migration:
            self._cache_retry_payloads.pop(sid)
        rank, reqs, tokens = self._seq_load[sid]
        target = placement.rank
        remaining = original.num_prompt_tokens - placement.estimated_hit
        self._rank_reqs[rank] -= reqs
        self._rank_tokens[rank] -= tokens
        self._rank_reqs[target] += reqs
        self._rank_tokens[target] += remaining
        self._seq_load[sid] = (target, reqs, remaining)
        # Order sends with abort's load removal. Admission, never this estimate,
        # decides which tokens the engine may actually skip.
        try:
            self._send_control(target, _add_frame([original]))
        except Exception:
            returned_seq.leave_reason = "cache_migration_retry_failed"
            balancer.counters["migration_retry_send_failed"] += 1
            logger.exception("Cache recovery retry could not be sent")
            return False
        return True

    def _finish_cache_placement_locked(self, sid, succeeded):
        balancer = getattr(self, "_cache_balancer", None)
        if balancer is not None:
            balancer.finish(sid, time.monotonic(), succeeded=succeeded)
            self._cache_retry_payloads.pop(sid, None)

    def _invalidate_cache_predictions(self):
        balancer = getattr(self, "_cache_balancer", None)
        if balancer is not None:
            with self._lb_lock:
                balancer.invalidate()
