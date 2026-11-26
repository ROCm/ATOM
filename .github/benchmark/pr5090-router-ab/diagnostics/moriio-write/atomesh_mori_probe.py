"""Bounded observations of real submitted writes and native completion."""

import functools
import json
import os
import sys
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_submitted = {}
_successes = 0


def _emit(event):
    directory = os.environ.get("ATOMESH_MORI_PROBE_DIR")
    if directory:
        path = Path(directory) / f"producer-{os.getpid()}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(dict(event, pid=os.getpid(), time_ns=time.time_ns())) + "\n"
            )


def install(writer_class):
    original = writer_class._do_layer_write

    @functools.wraps(original)
    def observed(self, plan, sessions):
        statuses = original(self, plan, sessions)
        if os.environ.get("ATOMESH_MORI_PROBE_DIR") and _successes < 16:
            with _lock:
                if _successes < 16 and (
                    plan.transfer_id in _submitted or len(_submitted) < 256
                ):
                    row = _submitted.setdefault(plan.transfer_id, [0, 0])
                    row[0] += sum(int(size) for size in plan.transfer_sizes)
                    row[1] += len(statuses)
        return statuses

    writer_class._do_layer_write = observed


def completed(worker, transfer_id, statuses):
    try:
        _completed(worker, transfer_id, statuses)
    except Exception as error:
        print(f"[DEBUG-mori-proof] observation failed: {error!r}", file=sys.stderr)


def _completed(worker, transfer_id, statuses):
    global _successes
    if not os.environ.get("ATOMESH_MORI_PROBE_DIR"):
        return
    with _lock:
        counts = _submitted.pop(transfer_id, None)
        if counts is None or _successes >= 16:
            return
        _emit(
            {
                "event": "completed_native_write",
                "pp_rank": worker.pp_rank,
                "tp_rank": worker.tp_rank,
                "dp_rank": worker.dp_rank,
                "transfer_id": transfer_id,
                "bytes": counts[0],
                "submitted_statuses": counts[1],
                "completed_statuses": len(statuses),
                "all_succeeded": bool(statuses)
                and all(status.Succeeded() for status in statuses),
            }
        )
        _successes += 1
        if _successes == 16:
            _submitted.clear()
