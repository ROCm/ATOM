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
    torch = namespace["torch"]
    root = Path(os.environ["ATOM_EPLB_DIAGNOSTICS_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    state = threading.local()
    gc_start = {}

    def capture_layout(old_list, new_list):
        row = getattr(state, "row", None)
        if row is not None:
            row["old_physical_to_logical"] = old_list
            row["new_physical_to_logical"] = new_list

    namespace["_capture_perf_layout"] = capture_layout

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
                row["p2p_operations"] = [
                    {"op": op.op.__name__, "peer": op.peer,
                     "bytes": op.tensor.numel() * op.tensor.element_size()}
                    for op in ops
                ]
            if name == "_migrate_single_layer":
                row["weight_layout"] = [
                    {"shape": list(w.shape), "stride": list(w.stride()),
                     "dtype": str(w.dtype), "element_size": w.element_size()}
                    for w in kwargs["routed_experts_weights"]
                ]
            before = time.perf_counter()
            events = None
            if name in ("_execute_batched_p2p_ops", "move_from_buffer"):
                events = (torch.cuda.Event(enable_timing=True),
                          torch.cuda.Event(enable_timing=True))
                events[0].record()
            try:
                return original(*args, **kwargs)
            finally:
                if events is not None:
                    events[1].record()
                    state.events[name] = events
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
        state.events = {}
        # File per process, opened after spawning. Only a long individual layer
        # triggers a stack dump; ordinary forwards have no sampling overhead.
        with (root / f"stacks-{os.getpid()}.log").open("a") as stack:
            faulthandler.dump_traceback_later(0.5, repeat=False, file=stack)
            try:
                return original(*args, **kwargs)
            finally:
                faulthandler.cancel_dump_traceback_later()
                row["end_ns"] = time.time_ns()
                row["host_seconds"] = time.perf_counter() - started
                row["resource_delta"] = {
                    k: v - before[k] for k, v in snapshot().items()
                }
                # The existing tensor-equality assertion normally drains these
                # events. Never synchronize for measurement: retain null if a
                # phase is still pending. P2P spans include stream dependencies.
                row["phase_gpu_ms"] = {
                    name: begin.elapsed_time(end) if end.query() else None
                    for name, (begin, end) in state.events.items()
                }
                state.row = None
                with (root / f"migration-{os.getpid()}.jsonl").open("a") as out:
                    out.write(json.dumps(row) + "\n")

    namespace["migrate_and_commit_chunk"] = measured_chunk
