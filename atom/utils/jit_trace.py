# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Explicit accounting for kernel compilation on the serving path.

A JIT compile that happens inside a forward is charged to that forward, and
nothing today says so. The only surviving trace is indirect -- LLVM occupancy
warnings in the server log, from which a compile can be *inferred* but never
timed -- so a prefill that stalled for seconds while FlyDSL lowered the GEMM it
had just selected is indistinguishable from one that was merely large.

Time the compiles instead. The hooks below sit exclusively on cold paths:

  * ``MlirCompiler.compile`` runs only when there was nothing to reuse.
  * ``JitCacheManager.get`` runs only when the in-process cache missed, i.e.
    once per kernel variant per process. It is worth timing on its own -- it
    takes a shared ``FileLock`` and unpickles from disk, so with eight ranks
    starting together the wait, not the compile, can be what stalls.
  * ``aiter.jit.core.build_module`` shells out to hipcc; nothing about it is
    hot.

None of them is reached per launch: the per-launch path returns from
``JitFunction``'s call-state cache without entering any of these. So with a
warm cache this module writes nothing and costs nothing, which is also what
makes it the readout for a cache-seeding change -- an empty trace is the
result, not the absence of one.

Installation never fails a server. A version whose attributes do not match is
reported once and left alone; serving is not worth a stack trace about
telemetry.
"""

import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_tracer = None


class _JitTracer:
    """Writes one JSON line per compile-ish event.

    Line-buffered and written inline, which is affordable precisely because
    every hooked call is rare and long: a compile costs seconds, the record
    costs a write of a couple hundred bytes. Buffering it to amortize a cost
    this lopsided would only risk losing the tail.
    """

    def __init__(self, path):
        self.path = path
        self._file = open(path, "a", buffering=1)
        self._lock = threading.Lock()
        self.events = 0

    def emit(self, event, name, elapsed_ns, **extra):
        record = {
            "event": event,
            "name": name,
            "ms": round(elapsed_ns / 1e6, 3),
            "t_ns": time.monotonic_ns(),
            "wall": time.time(),
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            **extra,
        }
        line = json.dumps(record, default=str) + "\n"
        # Compiles can run on more than one thread (aiter builds under a
        # process lock, FlyDSL under a file lock); interleaved partial writes
        # would corrupt lines that are meant to be read with json.loads.
        with self._lock:
            self._file.write(line)
            self.events += 1

    def close(self):
        try:
            self._file.close()
        except Exception:  # pragma: no cover - nothing to do at teardown
            pass


def _hook_flydsl(tracer):
    from flydsl.compiler import jit_function

    compiler = jit_function.MlirCompiler
    original_compile = compiler.compile.__func__

    def compile(cls, module, **kwargs):
        start = time.monotonic_ns()
        failed = False
        try:
            return original_compile(cls, module, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            tracer.emit(
                "flydsl_compile",
                kwargs.get("func_name") or "",
                time.monotonic_ns() - start,
                arch=kwargs.get("arch") or "",
                failed=failed,
            )

    compiler.compile = classmethod(compile)

    manager = jit_function.JitCacheManager
    original_get = manager.get

    def get(self, cache_key):
        start = time.monotonic_ns()
        value = original_get(self, cache_key)
        # The key is a content hash and is long; the head is enough to pair a
        # probe with the compile that follows it.
        tracer.emit(
            "flydsl_cache_probe",
            str(cache_key)[:64],
            time.monotonic_ns() - start,
            hit=value is not None,
            cache_dir=str(getattr(self, "cache_dir", "")),
        )
        return value

    manager.get = get


def _hook_aiter(tracer):
    from aiter.jit import core

    original_build = core.build_module

    def build_module(md_name, *args, **kwargs):
        start = time.monotonic_ns()
        failed = False
        try:
            return original_build(md_name, *args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            tracer.emit(
                "aiter_build_module",
                md_name,
                time.monotonic_ns() - start,
                failed=failed,
            )

    core.build_module = build_module


def install(path):
    """Hook the compile paths, writing events to `path`. Idempotent.

    Returns the names of the backends that were hooked, so a caller can log
    that FlyDSL was instrumented but AITER was not, rather than leave the
    reader to conclude from an empty trace that nothing ever compiled.
    """
    global _tracer
    if _tracer is not None:
        return ()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tracer = _JitTracer(path)
    hooked = []
    for name, hook in (("flydsl", _hook_flydsl), ("aiter", _hook_aiter)):
        try:
            hook(tracer)
            hooked.append(name)
        except Exception as exc:
            # An install failure must cost the trace, never the server: these
            # reach into another package's internals, and a version that moved
            # them is a missing measurement, not a broken deployment.
            logger.warning("JIT trace: could not hook %s (%s)", name, exc)
    if not hooked:
        tracer.close()
        return ()
    _tracer = tracer
    logger.info("JIT trace: hooked %s, writing to %s", ", ".join(hooked), path)
    return tuple(hooked)


def close():
    global _tracer
    if _tracer is not None:
        _tracer.close()
        _tracer = None
