"""Opt-in initialization progress and stacks; no signal or process attachment.

Install only in a new named attempt. Output uses that attempt's bounded stderr
drain. Timers and stage logging stop when ModelRunner construction finishes.
"""

import faulthandler
import functools
import os
import sys
import time


class InitializationProgress:
    def __init__(self, *, timeout=120, stream=None, fault=None):
        self.timeout = timeout
        self.stream = stream if stream is not None else sys.stderr
        self.fault = fault if fault is not None else faulthandler
        self.active = False
        self.constructing = False
        self.emitted_bytes = 0

    def emit(self, message):
        raw = (f"OWNER_INIT pid={os.getpid()} {message}\n").encode()
        if self.emitted_bytes + len(raw) <= 64_000:
            self.stream.write(raw.decode())
            self.stream.flush()
            self.emitted_bytes += len(raw)

    def progress(self, stage):
        if not self.active:
            return
        self.emit(stage)
        # A native watchdog can report Python stacks even while the main
        # thread is blocked in a compiler. It does not register a signal.
        self.fault.dump_traceback_later(
            self.timeout, repeat=False, file=self.stream, exit=False
        )

    def start(self):
        self.active = True
        self.progress("warmup entered")

    def stop(self):
        if self.active:
            self.fault.cancel_dump_traceback_later()
            self.emit("initialization exited; watchdog disabled")
        self.active = False

    def tracked(self, function, stage):
        @functools.wraps(function)
        def call(*args, **kwargs):
            if not self.active:
                return function(*args, **kwargs)
            started = time.monotonic()
            self.progress(stage + " enter")
            try:
                return function(*args, **kwargs)
            finally:
                self.progress(f"{stage} exit elapsed={time.monotonic() - started:.3f}s")

        return call


def bind_initialization_lifetime(runner_class, progress):
    original_init = runner_class.__init__
    original_warmup = runner_class.warmup_model

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        progress.constructing = True
        try:
            return original_init(self, *args, **kwargs)
        finally:
            progress.stop()
            progress.constructing = False

    @functools.wraps(original_warmup)
    def warmup(self, *args, **kwargs):
        if progress.constructing:
            progress.start()
        return original_warmup(self, *args, **kwargs)

    runner_class.__init__ = initialize
    runner_class.warmup_model = warmup


def install():
    if os.environ.get("DSV41_OWNER_INIT_PROGRESS") != "1":
        return None
    from atom.model_engine.model_runner import ModelRunner
    from atom.utils import backends

    if getattr(ModelRunner, "_owner_init_progress", None) is not None:
        return ModelRunner._owner_init_progress
    progress = InitializationProgress()

    backends.split_graph = progress.tracked(backends.split_graph, "split_graph")
    for cls, method in (
        (backends.PiecewiseCompileInterpreter, "call_module"),
        (backends.CompilerManager, "compile"),
        (backends.CompilerManager, "load"),
    ):
        setattr(
            cls,
            method,
            progress.tracked(getattr(cls, method), cls.__name__ + "." + method),
        )
    bind_initialization_lifetime(ModelRunner, progress)
    ModelRunner._owner_init_progress = progress
    return progress
