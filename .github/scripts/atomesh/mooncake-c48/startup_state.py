"""Read-only startup observations for the unresolved C48 HIP 209 failure.

This module performs no extra kernel loads, cache eviction, or retries.
Observations are incremental so worker teardown cannot lose the entire trace.
"""

from datetime import datetime, timezone
import functools
import json
import os
from pathlib import Path
import socket


def snapshot(label, error=None):
    directory = os.environ.get("MOONCAKE_KERNEL_EVIDENCE_DIR")
    if not directory:
        return
    try:
        import torch

        row = {
            "at": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "cuda_initialized": torch.cuda.is_initialized(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        }
        if error is not None:
            row["error"] = repr(error)
        if row["cuda_initialized"]:
            # Query the current device only; touching peers initializes contexts.
            device = torch.cuda.current_device()
            stats = torch.cuda.memory_stats(device)
            row.update(
                device=device,
                free_total=list(torch.cuda.mem_get_info(device)),
                torch_memory={key: stats.get(key) for key in (
                    "allocated_bytes.all.current", "reserved_bytes.all.current",
                    "active_bytes.all.current", "inactive_split_bytes.all.current",
                    "num_alloc_retries", "num_ooms",
                )},
            )
        row["physical_vram"] = {
            str(path): int(path.read_text())
            for path in Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")
        }
        root = Path(directory) / f"{socket.gethostname()}-{os.getpid()}"
        root.mkdir(parents=True, exist_ok=True)
        with (root / "startup-state.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    except Exception as collection_error:
        print(f"[mooncake-state-evidence] {label}: {collection_error}", flush=True)


def trace_stage(label):
    def decorate(function):
        @functools.wraps(function)
        def traced(*args, **kwargs):
            snapshot(label + ":before")
            try:
                result = function(*args, **kwargs)
            except BaseException as error:
                snapshot(label + ":failed", error)
                raise
            snapshot(label + ":after")
            return result
        return traced
    return decorate
