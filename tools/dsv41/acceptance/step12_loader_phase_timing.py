"""Uninstalled CPU timing of iterator materialization and weight dispatch.

Only a future named startup may install this. It does not read tensors, alter
loader settings, add a synchronization, or wrap any model forward.
"""

import functools
import inspect
import json
import os
import threading
import time
from pathlib import Path


class Timings:
    def __init__(self, clock):
        self.clock = clock
        self.stages = {}

    def call(self, stage, fn, *args, **kwargs):
        wall, cpu = self.clock()
        try:
            return fn(*args, **kwargs)
        finally:
            end_wall, end_cpu = self.clock()
            record = self.stages.setdefault(
                stage, {"calls": 0, "wall_ns": 0, "thread_cpu_ns": 0}
            )
            record["calls"] += 1
            record["wall_ns"] += end_wall - wall
            record["thread_cpu_ns"] += end_cpu - cpu


class TimedIterator:
    def __init__(self, iterable, timing):
        self.iterator = iter(iterable)
        self.timing = timing

    def __iter__(self):
        return self

    def __next__(self):
        return self.timing.call("iterator_next", next, self.iterator)


def clock():
    return time.perf_counter_ns(), time.thread_time_ns()


def bind(loader, dispatcher_class, publish, rank_tag, timer=clock):
    """Wrap the actual loader alias and dispatcher; host-testable without Torch.

    Iterator time includes native reader/materialization work and StopIteration.
    Dispatch time includes synchronous writes with the current one-thread loader.
    With multiple loader threads it includes submission, not asynchronous work.
    These are CPU intervals; they do not identify individual GPU transfer times.
    """
    original = loader.load_weights_into_model
    dispatch = dispatcher_class.dispatch
    if getattr(original, "_step12_loader_phase_timing", False) or getattr(
        dispatch, "_step12_loader_phase_timing", False
    ):
        raise RuntimeError("Loader timing already installed")
    signature = inspect.signature(original)
    if (
        signature.parameters["weights_iterator"].kind
        is not inspect.Parameter.KEYWORD_ONLY
    ):
        raise RuntimeError("Native loader iterator ABI changed")
    local = threading.local()
    sequence = 0

    @functools.wraps(dispatch)
    def observed_dispatch(self, orig_ckpt_name, name, tensor):
        timing = getattr(local, "active", None)
        if timing is None:
            return dispatch(self, orig_ckpt_name, name, tensor)
        kind = "expert" if ".experts." in orig_ckpt_name else "other"
        return timing.call(
            "dispatch_" + kind, dispatch, self, orig_ckpt_name, name, tensor
        )

    @functools.wraps(original)
    def observed_load(*args, **kwargs):
        nonlocal sequence
        bound = signature.bind(*args, **kwargs)
        if getattr(local, "active", None) is not None:
            raise RuntimeError("Unexpected nested checkpoint load")
        timing = Timings(timer)
        factory = bound.arguments["weights_iterator"]

        def iterator(*iterator_args, **iterator_kwargs):
            def create():
                return TimedIterator(factory(*iterator_args, **iterator_kwargs), timing)

            return timing.call("iterator_creation", create)

        bound.arguments["weights_iterator"] = iterator
        ordinal = sequence
        sequence += 1
        started, started_cpu = timer()
        local.active = timing
        failure = None
        try:
            return original(*bound.args, **bound.kwargs)
        except BaseException as error:
            failure = type(error).__name__
            raise
        finally:
            ended, ended_cpu = timer()
            local.active = None
            report = {
                "status": "load_returned" if failure is None else "load_raised",
                "failure_type": failure,
                "ordinal": ordinal,
                "rank": rank_tag(),
                "pid": os.getpid(),
                "model_path": str(bound.arguments.get("model_name_or_path", "")),
                "loader_threads_environment": os.environ.get("ATOM_LOADER_NUM_THREADS"),
                "total_wall_ns": ended - started,
                "total_thread_cpu_ns": ended_cpu - started_cpu,
                "stages": timing.stages,
                "scope": "Startup CPU intervals only. Iterator includes opening/materializing/skipping and terminal iteration. Dispatch includes synchronous work or queue submission, according to native loader settings. No tensor copy/read, GPU event/synchronize, threading change or inference hook was introduced. Not isolated I/O/GPU durations or model speedup evidence.",
            }
            try:
                publish(report)
            except Exception:
                if failure is None:
                    raise
                # Preserve the native load failure if its diagnostic cannot be saved.

    observed_load._step12_loader_phase_timing = True
    observed_dispatch._step12_loader_phase_timing = True
    loader.load_weights_into_model = observed_load
    dispatcher_class.dispatch = observed_dispatch


def install():
    if os.environ.get("DSV41_STEP12_LOADER_TIMING") != "1":
        raise RuntimeError("Use only a future explicitly named startup diagnostic")
    from atom.model_loader import loader
    from atom.model_loader.weight_dispatch import WeightDispatcher

    root = Path(os.environ["DSV41_REBASE_ATOM_SOURCE"]).resolve()
    if not Path(loader.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Wrong model loader source imported")
    output = Path(os.environ["DSV41_OWNER_SERVING_OUTPUT"])

    def publish(report):
        if report["ordinal"] >= 8:
            raise RuntimeError("More than eight startup loads in one worker")
        raw = json.dumps(report, indent=2) + "\n"
        if len(raw.encode()) > 16_000:
            raise RuntimeError("Startup timing exceeded its per-record bound")
        path = output / f"loader_phases_pid{os.getpid()}_{report['ordinal']}.json"
        path.open("x").write(raw)

    bind(loader, WeightDispatcher, publish, loader.rank_tag)
