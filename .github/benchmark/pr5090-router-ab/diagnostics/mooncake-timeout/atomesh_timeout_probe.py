"""Bounded metadata-only diagnostics, after registration or a failed native write."""

import json
import os
import sys
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_failures = 0


def covered(address, length, ranges):
    end = address + length
    cursor = address
    for base, size in sorted(ranges):
        if base > cursor:
            break
        cursor = max(cursor, base + size)
        if cursor >= end:
            return True
    return length == 0


def _metadata(worker):
    return {
        "pid": os.getpid(),
        "tid": threading.get_native_id(),
        "time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "role": "prefill" if worker.is_kv_producer else "decode",
        "pp_rank": worker.pp_rank,
        "session": f"{worker.hostname}:{worker.rpc_port}",
        "device_id": str(worker.device_id),
        "sender_workers": worker.num_sender_workers,
        "async_sender_tasks": worker.num_sender_tasks,
        "ranges": list(zip(worker.seen_base_addresses, worker._kv_data_lens)),
    }


def _write(data):
    directory = os.environ["ATOMESH_MOONCAKE_PROBE_DIR"]
    path = Path(directory) / f"connector-{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, separators=(",", ":")) + "\n")


def record_registration(worker, event):
    if not os.environ.get("ATOMESH_MOONCAKE_PROBE_DIR"):
        return
    try:
        with _lock:
            data = _metadata(worker)
            data.update(event=event, registered=not worker._kv_unregistered)
            _write(data)
    except Exception as error:  # noqa: BLE001 - diagnostics must preserve workload results
        print(
            f"[DEBUG-ppdep-timeout] Registration probe failed: {error!r}",
            file=sys.stderr,
        )


def record_failure(worker, peer, sources, targets, lengths, return_code, duration):
    global _failures
    if not os.environ.get("ATOMESH_MOONCAKE_PROBE_DIR"):
        return
    try:
        with _lock:
            if _failures >= 3:
                return
            _failures += 1
            data = _metadata(worker)
            descriptors = list(zip(sources, targets, lengths))
            data.update(
                event="failed_native_write",
                peer=peer,
                return_code=return_code,
                duration_s=duration,
                registered=not worker._kv_unregistered,
                native_call_start_monotonic_ns=data["monotonic_ns"]
                - int(duration * 1e9)
                - 1_000_000,
                descriptor_count=len(descriptors),
                total_bytes=sum(lengths),
                source_outside_ranges=[
                    i
                    for i, (a, _, n) in enumerate(descriptors)
                    if not covered(a, n, data["ranges"])
                ],
                descriptors=descriptors[:65536],
                descriptors_omitted=max(0, len(descriptors) - 65536),
            )
            _write(data)
    except Exception as error:  # noqa: BLE001 - diagnostics must preserve workload results
        print(f"[DEBUG-ppdep-timeout] Failure probe failed: {error!r}", file=sys.stderr)
