# SPDX-License-Identifier: MIT
"""CoreManager integration for cache-affinity routing and safe retry."""

import json
import logging
import time

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
        from atom.model_engine.engine_core_mgr import _add_frame

        balancer = getattr(self, "_cache_balancer", None)
        if balancer is None:
            return False
        with self._lb_lock:
            sid = returned_seq.id
            payload = self._cache_retry_payloads.get(sid)
            load = self._seq_load.get(sid)
            if payload is None or load is None or load[0] != rank:
                return True
            source = balancer.retry(
                sid,
                rank,
                returned_seq.dp_cache_migration,
                time.monotonic(),
                reason=returned_seq.dp_cache_retry_reason,
                recoverable_hit=getattr(returned_seq, "dp_cache_recoverable_hit", None),
            )
            if source is None:
                return True
            original, chain = self._cache_retry_payloads.pop(sid)
            original.prompt_hash_chain = chain
            original.dp_cache_migration = 0
            original.dp_cache_minimum_hit = 0
            self._rank_reqs[rank] -= load[1]
            self._rank_tokens[rank] -= load[2]
            self._rank_reqs[source] += load[1]
            self._rank_tokens[source] += original.num_prompt_tokens
            self._seq_load[sid] = (source, load[1], original.num_prompt_tokens)
            # Keep abort's load removal ordered after this send. The control
            # socket has its own lock and sending needs no response callback.
            try:
                self._send_control(source, _add_frame([original]))
            except Exception:
                # Let ADD's normal terminal path answer the client and release
                # the transferred load charge. One failed retry must not end
                # this output thread and strand every other request on it.
                returned_seq.leave_reason = "cache_migration_retry_failed"
                balancer.counters["migration_retry_send_failed"] += 1
                logger.exception("Cache migration fallback could not be sent")
                return False
        return True

    def _cache_work(self, rank, events):
        balancer = getattr(self, "_cache_balancer", None)
        if balancer is None:
            return
        with self._lb_lock:
            for sid, migration, revision, remaining in events:
                load = self._seq_load.get(sid)
                placement = balancer.requests.get(sid)
                if (
                    load is None
                    or load[0] != rank
                    or placement is None
                    or not placement.admitted
                    or placement.work_migration != migration
                    or revision <= placement.work_revision
                ):
                    continue
                placement.work_revision = revision
                remaining = max(0, int(remaining))
                self._rank_tokens[rank] += remaining - load[2]
                self._seq_load[sid] = (rank, load[1], remaining)
                balancer.counters["prefill_work_updates"] += 1

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
