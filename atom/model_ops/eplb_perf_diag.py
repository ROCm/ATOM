"""Temporary EPLB host-side diagnostics; no added device synchronization."""

import faulthandler
import functools
import gc
import json
import os
from pathlib import Path
import resource
import threading
import time


def install(namespace):
    root = Path(os.environ["ATOM_EPLB_DIAGNOSTICS_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    state = threading.local()
    gc_start = {}

    def gc_event(phase, info):
        generation = info["generation"]
        if phase == "start":
            gc_start[generation] = time.perf_counter()
        elif getattr(state, "row", None) is not None:
            elapsed = time.perf_counter() - gc_start.pop(generation, time.perf_counter())
            state.row["gc_seconds"] += elapsed

    gc.callbacks.append(gc_event)

    def snapshot():
        r = resource.getrusage(resource.RUSAGE_SELF)
        return {name: getattr(r, name) for name in (
            "ru_utime", "ru_stime", "ru_minflt", "ru_majflt", "ru_nvcsw", "ru_nivcsw"
        )}

    def wrap_phase(name):
        original = namespace[name]

        @functools.wraps(original)
        def measured(*args, **kwargs):
            row = getattr(state, "row", None)
            if row is None:
                return original(*args, **kwargs)
            if name == "_execute_batched_p2p_ops":
                operations = kwargs["ops_by_logical"]
                ops = [op for batch in operations.values() for op in batch]
                row["p2p_ops"] = len(ops)
                row["p2p_bytes_send_plus_receive"] = sum(
                    op.tensor.numel() * op.tensor.element_size() for op in ops
                )
            before = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                row["phase_host_seconds"][name] = (
                    row["phase_host_seconds"].get(name, 0.0)
                    + time.perf_counter() - before
                )

        namespace[name] = measured

    for name in (
        "_plan_single_layer_migration", "_execute_batched_p2p_ops",
        "_migrate_single_layer", "move_from_buffer", "commit_layer",
    ):
        wrap_phase(name)

    original = namespace["migrate_and_commit_chunk"]

    @functools.wraps(original)
    def measured_chunk(*args, **kwargs):
        row = {
            "pid": os.getpid(), "rank": kwargs["rank"],
            "layers": kwargs["layer_ids"], "start_ns": time.time_ns(),
            "phase_host_seconds": {}, "gc_seconds": 0.0,
        }
        before = snapshot()
        started = time.perf_counter()
        state.row = row
        # File per process, opened after spawning. Only a long individual layer
        # triggers a stack dump; ordinary forwards have no sampling overhead.
        with (root / f"stacks-{os.getpid()}.log").open("a") as stack:
            faulthandler.dump_traceback_later(10, repeat=True, file=stack)
            try:
                return original(*args, **kwargs)
            finally:
                faulthandler.cancel_dump_traceback_later()
                row["end_ns"] = time.time_ns()
                row["host_seconds"] = time.perf_counter() - started
                row["resource_delta"] = {
                    k: v - before[k] for k, v in snapshot().items()
                }
                state.row = None
                with (root / f"migration-{os.getpid()}.jsonl").open("a") as out:
                    out.write(json.dumps(row) + "\n")

    namespace["migrate_and_commit_chunk"] = measured_chunk
