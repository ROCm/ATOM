"""CPU-only diagnostics for the isolated K3 prefill experiment."""

import functools
import itertools
import os
import sys
import threading
import time
from pathlib import Path


def emit(message):
    try:
        os.write(2, (message + "\n").encode())
    except OSError:
        pass


def memory_status():
    result = {}
    for name in ("memory.current", "memory.peak", "memory.max", "memory.events"):
        try:
            result[name] = Path("/sys/fs/cgroup", name).read_text().strip()
        except OSError as exc:
            result[name] = type(exc).__name__
    return result


def start(label, interval=120):
    def sample():
        while True:
            time.sleep(interval)
            frames = sys._current_frames()
            frame = None
            try:
                names = {thread.ident: thread.name for thread in threading.enumerate()}
                lines = [
                    (
                        f"K3_STACKS label={label} pid={os.getpid()} "
                        f"time_ns={time.time_ns()} mono_ns={time.monotonic_ns()}"
                    )
                ]
                for ident, frame in frames.items():
                    lines.append(f"Thread {ident} {names.get(ident, 'unknown')}")
                    for _ in range(80):
                        if frame is None:
                            break
                        code = frame.f_code
                        lines.append(
                            f"  File {code.co_filename!r}, line {frame.f_lineno}, "
                            f"in {code.co_name}"
                        )
                        frame = frame.f_back
            except Exception as exc:  # noqa: BLE001 - diagnostics must not affect serving
                lines = [f"K3_STACKS_ERROR label={label} type={type(exc).__name__}"]
            finally:
                frame = None
                frames.clear()
            lines.append(f"K3_MEMORY pid={os.getpid()} {memory_status()}")
            emit("\n".join(lines))

    threading.Thread(target=sample, daemon=True, name="K3StackProbe").start()
    emit(f"K3_STACKS label={label} pid={os.getpid()} Python-frame sampler started")


def trace_backend(cls):
    calls = itertools.count()

    def wrap(method):
        queries = itertools.count()

        @functools.wraps(method)
        def traced(self, *args, **kwargs):
            logged = method.__name__ != "query_event" or next(queries) % 1024 == 0
            call = next(calls)
            prefix = (
                f"K3_IPC pid={os.getpid()} tid={threading.get_native_id()} "
                f"call={call} op={method.__name__}"
            )
            start_ns = time.monotonic_ns()
            if logged:
                emit(f"{prefix} ENTER time_ns={time.time_ns()} mono_ns={start_ns}")
            try:
                result = method(self, *args, **kwargs)
            except BaseException as exc:
                emit(
                    f"{prefix} ERROR type={type(exc).__name__} "
                    f"duration_ns={time.monotonic_ns() - start_ns}"
                )
                raise
            duration_ns = time.monotonic_ns() - start_ns
            if logged or duration_ns >= 1_000_000_000:
                emit(
                    f"{prefix} EXIT duration_ns={duration_ns} "
                    f"mono_ns={start_ns} enter_logged={logged}"
                )
            return result

        return traced

    for name in (
        "create_event",
        "export_event",
        "import_event",
        "record_event",
        "wait_event",
        "query_event",
        "synchronize_event",
    ):
        setattr(cls, name, wrap(getattr(cls, name)))


def report_exit(proc):
    # The sentinel can become readable before a nonblocking waitpid sees exit.
    try:
        proc.join(timeout=1)
        emit(
            f"K3_WORKER_EXIT name={proc.name} pid={proc.pid} "
            f"exitcode={proc.exitcode} time_ns={time.time_ns()} memory={memory_status()}"
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must not affect serving
        emit(f"K3_WORKER_EXIT diagnostic_error={type(exc).__name__}")
