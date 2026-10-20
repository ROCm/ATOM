"""Bounded host metadata for complete Prefill request and batch accounting.

No tensors, token payloads, GPU synchronization, or scheduling changes. Batch
residence may overlap under async scheduling and is not exclusive GPU time.
"""

import atexit
import hashlib
import json
import os
import time
from pathlib import Path

from pd_c48_timeline import batch_rows

STATS = (
    "num_prompt_tokens",
    "num_computed_tokens",
    "num_local_cached_tokens",
    "num_external_cached_tokens",
)


class RequestLedger:
    def __init__(
        self,
        scheduler,
        directory,
        *,
        max_records=100000,
        max_bytes=128 * 1024 * 1024,
        max_requests=20000,
    ):
        self.scheduler = scheduler
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"requests-{os.getpid()}.jsonl"
        self.stream = self.path.open("w", buffering=65536)
        self.max_records, self.max_bytes = max_records, max_bytes
        self.max_requests = max_requests
        self.records = self.bytes = self.dropped = 0
        self.admitted = self.untracked = self.finished = 0
        self.finished_externally = 0
        self.empty_schedules = self.sequence = 0
        self.observer_calls = self.observer_total_ns = self.observer_max_ns = 0
        self.states, self.pending = {}, {}
        self.seen = set()
        self.errors = []
        self.installed = False
        self.emit(
            "installed",
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scheduler=type(scheduler).__module__ + "." + type(scheduler).__name__,
            max_records=max_records,
            max_bytes=max_bytes,
            max_requests=max_requests,
        )
        atexit.register(self.close)

    def error(self, kind):
        if kind not in self.errors and len(self.errors) < 16:
            self.errors.append(kind)

    def emit(self, event, **fields):
        if (
            self.stream.closed
            or self.records >= self.max_records
            or self.bytes >= self.max_bytes
        ):
            self.dropped += 1
            return
        try:
            line = (
                json.dumps(dict(event=event, unix_ns=time.time_ns(), **fields)) + "\n"
            )
            self.stream.write(line)
            self.records += 1
            self.bytes += len(line.encode())
        except (OSError, ValueError) as exc:
            self.error(type(exc).__name__)

    def observe(self, callback, *args):
        # Evidence failures must not change serving results. The footer makes
        # this evidence unusable for acceptance rather than hiding the error.
        begin_ns = time.monotonic_ns()
        try:
            callback(*args)
        except Exception as exc:  # noqa: BLE001 - diagnostics must preserve serving.
            self.error("observer_" + type(exc).__name__)
        finally:
            elapsed_ns = time.monotonic_ns() - begin_ns
            self.observer_calls += 1
            self.observer_total_ns += elapsed_ns
            self.observer_max_ns = max(self.observer_max_ns, elapsed_ns)

    def pressure(self):
        return {
            name: len(getattr(self.scheduler, name, ()))
            for name in ("running", "waiting", "kv_holding_waiting")
        }

    def admitted_request(self, request):
        self.admitted += 1
        rid = request.request_id
        if rid in self.seen:
            self.error("reused_request_id")
            return
        if len(self.seen) >= self.max_requests:
            self.untracked += 1
            return
        self.seen.add(rid)
        self.states[rid] = {
            "request_id": rid,
            "arrival_time": request.arrival_time,
            "admitted_unix_ns": time.time_ns(),
            "prompt_tokens": request.num_prompt_tokens,
            "first_schedule_unix_ns": None,
            "first_emission_update_begin_unix_ns": None,
            "first_emission_update_end_unix_ns": None,
            "first_token_update_begin_unix_ns": None,
            "first_token_update_end_unix_ns": None,
            "first_computed_before": None,
            "last_computed_end": None,
            "scheduled_prompt_tokens": 0,
            "scheduled_total_tokens": 0,
            "scheduled_batches": 0,
            "mixed_batches": 0,
            "continuity_errors": 0,
            "missing_computed_before": 0,
            "prefill_stats_first": None,
            "prefill_stats_last": None,
            "prefill_stats_changes": 0,
            "prefill_stats_missing_before_output": 0,
            "prefill_stats_emitted": None,
            "prefill_stats_emission_count": 0,
        }

    def scheduled(self, output, begin_unix_ns, begin_ns, end_ns):
        self.sequence += 1
        if not output.num_scheduled_tokens:
            self.empty_schedules += 1
            return
        rows = [
            (rid, computed, count)
            for rid, computed, count in batch_rows(output)
            if count > 0
        ]
        if not rows:
            self.empty_schedules += 1
            return
        if id(output) in self.pending:
            self.error("duplicate_pending_output")
            return
        if len(self.pending) >= 4096:
            self.error("pending_output_limit")
            self.dropped += 1
            return
        record = {
            "sequence": self.sequence,
            "begin_unix_ns": begin_unix_ns,
            "begin_monotonic_ns": begin_ns,
            "end_monotonic_ns": end_ns,
            "schedule_host_ms": (end_ns - begin_ns) / 1e6,
            "rows": rows,
            "pressure_after_schedule": self.pressure(),
            "waiter_budget_retry_stats": dict(
                getattr(self.scheduler, "_c48_waiter_budget_stats", {})
            ),
        }
        self.pending[id(output)] = record
        for rid, computed, count in rows:
            state = self.states.get(rid)
            if state is None:
                self.error("schedule_without_admission")
                continue
            request = self.scheduler.requests.get(rid)
            stats = getattr(request, "prefill_stats", None)
            stats = {name: getattr(stats, name) for name in STATS} if stats else None
            if state["first_schedule_unix_ns"] is None:
                state["first_schedule_unix_ns"] = begin_unix_ns
                state["first_computed_before"] = computed
                state["prefill_stats_first"] = stats
            elif stats is not None and stats != state["prefill_stats_last"]:
                state["prefill_stats_changes"] += 1
            # take_prefill_stats() moves this object into the first output.
            # Retain the last non-null schedule snapshot; only disappearance
            # before an observed output is unexplained.
            if stats is not None:
                state["prefill_stats_last"] = stats
            elif state["prefill_stats_emission_count"] == 0:
                state["prefill_stats_missing_before_output"] += 1
            state["last_schedule_unix_ns"] = begin_unix_ns
            state["scheduled_total_tokens"] += count
            state["scheduled_batches"] += 1
            state["mixed_batches"] += len(rows) > 1
            if computed is None:
                state["missing_computed_before"] += 1
                continue
            previous = state["last_computed_end"]
            if previous is not None and previous != computed:
                state["continuity_errors"] += 1
            state["last_computed_end"] = computed + count
            prompt = state["prompt_tokens"]
            state["scheduled_prompt_tokens"] += max(
                0, min(prompt, computed + count) - min(prompt, computed)
            )

    def updated(self, output, result, begin_unix_ns, begin_ns, end_ns, end_unix_ns):
        record = self.pending.pop(id(output), None)
        scheduled_ids = {r[0] for r in record["rows"]} if record else set()
        emissions = []
        for batch in result.values():
            for item in batch.outputs:
                rid = item.request_id
                count = len(item.new_token_ids)
                emissions.append(
                    {
                        "request_id": rid,
                        "new_token_count": count,
                        "finish_reason": str(item.finish_reason),
                    }
                )
                state = self.states.get(rid)
                if state is None:
                    self.error("output_without_admission")
                    continue
                stats = item.prefill_stats
                if stats is not None:
                    state["prefill_stats_emission_count"] += 1
                    if state["prefill_stats_emitted"] is None:
                        state["prefill_stats_emitted"] = {
                            name: getattr(stats, name) for name in STATS
                        }
                if state["first_emission_update_begin_unix_ns"] is None:
                    state["first_emission_update_begin_unix_ns"] = begin_unix_ns
                    state["first_emission_update_end_unix_ns"] = end_unix_ns
                if count and state["first_token_update_begin_unix_ns"] is None:
                    state["first_token_update_begin_unix_ns"] = begin_unix_ns
                    state["first_token_update_end_unix_ns"] = end_unix_ns
                    state["first_token_from_scheduled_batch"] = rid in scheduled_ids
                if item.finish_reason is not None:
                    self.finish(rid, "finished", str(item.finish_reason))
        if record:
            self.emit(
                "batch",
                **record,
                update_begin_unix_ns=begin_unix_ns,
                update_begin_monotonic_ns=begin_ns,
                update_end_monotonic_ns=end_ns,
                update_end_unix_ns=end_unix_ns,
                schedule_to_update_ms=(begin_ns - record["end_monotonic_ns"]) / 1e6,
                update_host_ms=(end_ns - begin_ns) / 1e6,
                emissions=emissions,
                pressure_after_update=self.pressure(),
            )
        elif emissions:
            self.emit(
                "unassociated_output",
                update_begin_unix_ns=begin_unix_ns,
                emissions=emissions,
            )

    def finish(self, rid, disposition, finish_reason=None):
        state = self.states.pop(rid)
        self.finished += disposition == "finished"
        self.finished_externally += disposition == "finished_externally"
        stats = state["prefill_stats_first"]
        state["prompt_work_complete"] = bool(
            disposition == "finished"
            and state["first_token_update_begin_unix_ns"] is not None
            and stats is not None
            and state["prefill_stats_last"] == stats
            and state["prefill_stats_changes"] == 0
            and state["prefill_stats_emitted"] == stats
            and state["prefill_stats_emission_count"] == 1
            and state["prefill_stats_missing_before_output"] == 0
            and state["missing_computed_before"] == 0
            and state["continuity_errors"] == 0
            and state["first_computed_before"]
            == stats["num_local_cached_tokens"] + stats["num_external_cached_tokens"]
            and state["scheduled_prompt_tokens"] == stats["num_computed_tokens"]
            and state["first_computed_before"] + state["scheduled_prompt_tokens"]
            == state["prompt_tokens"]
            == stats["num_prompt_tokens"]
        )
        self.emit(
            "request", **state, disposition=disposition, finish_reason=finish_reason
        )

    def external_finishes(self, requests):
        for request in requests:
            if request.request_id in self.states:
                self.finish(
                    request.request_id, "finished_externally", str(request.status)
                )

    def close(self):
        if self.stream.closed:
            return
        for rid in list(self.states):
            self.observe(self.finish, rid, "pending_at_close")
        try:
            # The footer bypasses the record cap so truncation remains visible.
            self.stream.write(
                json.dumps(
                    {
                        "event": "closed",
                        "unix_ns": time.time_ns(),
                        "records": self.records,
                        "bytes": self.bytes,
                        "dropped": self.dropped,
                        "errors": self.errors,
                        "admitted": self.admitted,
                        "untracked": self.untracked,
                        "finished": self.finished,
                        "finished_externally": self.finished_externally,
                        "pending_batches": len(self.pending),
                        "empty_schedules": self.empty_schedules,
                        "observer_calls": self.observer_calls,
                        "observer_total_ns": self.observer_total_ns,
                        "observer_max_ns": self.observer_max_ns,
                        "waiter_budget_retry_stats": dict(
                            getattr(self.scheduler, "_c48_waiter_budget_stats", {})
                        ),
                    }
                )
                + "\n"
            )
        except (OSError, ValueError):
            pass
        finally:
            try:
                self.stream.close()
            except (OSError, ValueError) as exc:
                self.error(type(exc).__name__)

    def install(self):
        if self.installed:
            return self
        self.installed = True
        add = self.scheduler.add_request
        schedule = self.scheduler.schedule
        update = self.scheduler.update_from_output
        finish_requests = self.scheduler.finish_requests
        shutdown = self.scheduler.shutdown

        def observed_add(request, *args, **kwargs):
            result = add(request, *args, **kwargs)
            self.observe(self.admitted_request, request)
            return result

        def observed_schedule(*args, **kwargs):
            begin_ns, begin_unix_ns = time.monotonic_ns(), time.time_ns()
            result = schedule(*args, **kwargs)
            self.observe(
                self.scheduled, result, begin_unix_ns, begin_ns, time.monotonic_ns()
            )
            return result

        def observed_update(output, *args, **kwargs):
            begin_ns, begin_unix_ns = time.monotonic_ns(), time.time_ns()
            result = update(output, *args, **kwargs)
            end_ns, end_unix_ns = time.monotonic_ns(), time.time_ns()
            self.observe(
                self.updated,
                output,
                result,
                begin_unix_ns,
                begin_ns,
                end_ns,
                end_unix_ns,
            )
            return result

        def observed_finish_requests(*args, **kwargs):
            result = finish_requests(*args, **kwargs)
            self.observe(self.external_finishes, result)
            return result

        def observed_shutdown(*args, **kwargs):
            try:
                return shutdown(*args, **kwargs)
            finally:
                self.close()

        self.scheduler.add_request = observed_add
        self.scheduler.schedule = observed_schedule
        self.scheduler.update_from_output = observed_update
        self.scheduler.finish_requests = observed_finish_requests
        self.scheduler.shutdown = observed_shutdown
        return self


def install_request_ledger(scheduler, directory):
    """Install once and persist the footer before EngineCore executor teardown."""
    existing = getattr(scheduler, "_c48_request_ledger", None)
    if existing is not None:
        return existing
    from pd_c48_cache_lifecycle import install_engine_shutdown_flush
    from vllm.v1.engine.core import EngineCore

    ledger = RequestLedger(scheduler, directory).install()
    scheduler._c48_request_ledger = ledger
    install_engine_shutdown_flush(EngineCore, scheduler, ledger.close)
    return ledger
