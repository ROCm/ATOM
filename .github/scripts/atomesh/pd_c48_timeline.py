"""Bounded, observational Prefill probes for the C48 performance experiment.

No token payloads, tensor reads, GPU synchronization, or scheduling decisions.
The original scheduler object/method results are preserved. Times are host wall
times, not exclusive GPU durations; async output residence can overlap batches.
"""

import atexit
import faulthandler
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


def batch_rows(output):
    computed = {r.req_id: r.num_computed_tokens for r in output.scheduled_new_reqs}
    cached = output.scheduled_cached_reqs
    computed.update(zip(cached.req_ids, cached.num_computed_tokens))
    return [
        (rid, computed.get(rid), count)
        for rid, count in sorted(output.num_scheduled_tokens.items())
    ]


def batch_key(output):
    return hashlib.sha256(json.dumps(batch_rows(output)).encode()).hexdigest()[:24]


@contextmanager
def startup_stage(worker, stage):
    directory = Path(os.environ["C48_DIAG_DIR"]).parent / "startup-stacks"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f'{os.environ["C48_DIAG_ROLE"]}-rank-{worker.rank}-{stage}'
    begin = time.time_ns()
    record = {
        "stage": stage,
        "begin_unix_ns": begin,
        "pid": os.getpid(),
        "stack_after_seconds": 300,
        "repeat": False,
    }
    record["meminfo_before"] = Path("/proc/meminfo").read_text()
    path = directory / (prefix + ".json")
    path.write_text(json.dumps(record) + "\n")
    print(
        "[C48-STARTUP] "
        + json.dumps({"stage": stage, "rank": worker.rank, "begin_unix_ns": begin}),
        flush=True,
    )
    with (directory / (prefix + ".stack.txt")).open("w") as stack:
        faulthandler.dump_traceback_later(300, repeat=False, file=stack, exit=False)
        try:
            yield
        except BaseException as exc:
            record["exception_type"] = type(exc).__name__
            raise
        finally:
            faulthandler.cancel_dump_traceback_later()
            record["end_unix_ns"] = time.time_ns()
            record["elapsed_seconds"] = (record["end_unix_ns"] - begin) / 1e9
            record["meminfo_after"] = Path("/proc/meminfo").read_text()
            path.write_text(json.dumps(record) + "\n")
            print(
                "[C48-STARTUP] "
                + json.dumps(
                    {
                        "stage": stage,
                        "rank": worker.rank,
                        "elapsed_seconds": record["elapsed_seconds"],
                        "exception_type": record.get("exception_type"),
                    }
                ),
                flush=True,
            )


class Timeline:
    def __init__(self, scheduler, directory):
        self.scheduler = scheduler
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / f"scheduler-{os.getpid()}.jsonl").open(
            "w", buffering=65536
        )
        self.plan = None
        self.next_check = 0
        self.next_summary = 0
        self.sequence = 0
        self.records = 0
        self.bytes = 0
        self.dropped = 0
        self.pending = {}
        atexit.register(self.close)
        self.emit(
            "installed",
            scheduler=type(scheduler).__module__ + "." + type(scheduler).__name__,
            max_records=100000,
            max_bytes=64 * 1024 * 1024,
        )

    def emit(self, event, **fields):
        if self.records >= 100000 or self.bytes >= 64 * 1024 * 1024:
            self.dropped += 1
            return
        line = (
            json.dumps(
                dict(
                    event=event,
                    unix_ns=time.time_ns(),
                    monotonic_ns=time.monotonic_ns(),
                    **fields,
                )
            )
            + "\n"
        )
        self.stream.write(line)
        self.records += 1
        self.bytes += len(line)

    def close(self):
        if not self.stream.closed:
            self.stream.write(
                json.dumps(
                    {
                        "event": "closed",
                        "records": self.records,
                        "bytes": self.bytes,
                        "dropped": self.dropped,
                    }
                )
                + "\n"
            )
            self.stream.close()

    def active(self):
        now = time.monotonic()
        if self.plan is None and now >= self.next_check:
            self.next_check = now + 5
            path = self.directory / "windows.json"
            if path.exists():
                self.plan = json.loads(path.read_text())
                self.emit("plan_observed", plan=self.plan)
        if self.plan is None:
            return False, False
        elapsed = (time.time_ns() - self.plan["phase_start_unix_ns"]) / 1e9
        phase = 0 <= elapsed <= self.plan["phase_seconds"]
        detail = phase and any(
            lo <= elapsed <= hi for lo, hi in self.plan["detail_windows"]
        )
        return phase, detail

    def pressure(self):
        return {
            "running": len(self.scheduler.running),
            "waiting": len(self.scheduler.waiting),
            "kv_holding_waiting": len(self.scheduler.kv_holding_waiting),
        }

    def install(self):
        schedule = self.scheduler.schedule
        update = self.scheduler.update_from_output
        add = self.scheduler.add_request

        def observed_schedule(*args, **kwargs):
            phase, detail = self.active()
            begin = time.monotonic_ns()
            begin_unix = time.time_ns()
            output = schedule(*args, **kwargs)
            end = time.monotonic_ns()
            self.sequence += 1
            if detail:
                key = batch_key(output)
                self.pending[id(output)] = (self.sequence, key, end)
                rows = []
                for rid, computed, count in batch_rows(output):
                    request = self.scheduler.requests.get(rid)
                    stats = getattr(request, "prefill_stats", None)
                    rows.append(
                        {
                            "request_id": rid,
                            "computed_before": computed,
                            "scheduled_tokens": count,
                            "prompt_tokens": getattr(
                                request, "num_prompt_tokens", None
                            ),
                            "arrival_time": getattr(request, "arrival_time", None),
                            "prefill_stats": (
                                {
                                    name: getattr(stats, name)
                                    for name in (
                                        "num_prompt_tokens",
                                        "num_computed_tokens",
                                        "num_local_cached_tokens",
                                        "num_external_cached_tokens",
                                    )
                                }
                                if stats is not None
                                else None
                            ),
                            "status": str(getattr(request, "status", None)),
                        }
                    )
                self.emit(
                    "schedule",
                    sequence=self.sequence,
                    batch_key=key,
                    begin_unix_ns=begin_unix,
                    begin_monotonic_ns=begin,
                    end_monotonic_ns=end,
                    schedule_host_ms=(end - begin) / 1e6,
                    total_tokens=output.total_num_scheduled_tokens,
                    has_sync_kv_loads=output.has_sync_kv_loads,
                    requests=rows,
                    **self.pressure(),
                )
            if phase and time.monotonic() >= self.next_summary:
                self.next_summary = time.monotonic() + 1
                self.emit("pressure", sequence=self.sequence, **self.pressure())
                self.stream.flush()
            return output

        def observed_update(output, *args, **kwargs):
            entry = self.pending.pop(id(output), None)
            begin = time.monotonic_ns()
            begin_unix = time.time_ns()
            result = update(output, *args, **kwargs)
            end = time.monotonic_ns()
            if entry:
                sequence, key, scheduled_end = entry
                emissions = [
                    {
                        "request_id": o.request_id,
                        "new_token_count": len(o.new_token_ids),
                        "finish_reason": str(o.finish_reason),
                    }
                    for batch in result.values()
                    for o in batch.outputs
                ]
                self.emit(
                    "output",
                    sequence=sequence,
                    batch_key=key,
                    begin_unix_ns=begin_unix,
                    begin_monotonic_ns=begin,
                    end_monotonic_ns=end,
                    schedule_to_update_ms=(begin - scheduled_end) / 1e6,
                    update_host_ms=(end - begin) / 1e6,
                    emissions=emissions,
                    **self.pressure(),
                )
            return result

        def observed_add(request, *args, **kwargs):
            result = add(request, *args, **kwargs)
            phase, _ = self.active()
            if phase:
                self.emit(
                    "request_admitted",
                    request_id=request.request_id,
                    prompt_tokens=request.num_prompt_tokens,
                    arrival_time=request.arrival_time,
                )
            return result

        self.scheduler.schedule = observed_schedule
        self.scheduler.update_from_output = observed_update
        self.scheduler.add_request = observed_add
        return self
