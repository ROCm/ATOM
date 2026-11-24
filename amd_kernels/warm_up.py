# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entries register their warm-up and warm themselves up on their first call (root README, rule 4: every variant
compiles before serving).

``@entry(name, mode=, sizes=, inputs=)`` registers an entry's public function (``REGISTRY``) and wraps it. An entry
declares data, this module runs the warm-up:

- ``mode(*args, **kwargs)``: what a call fixes about its kernels besides the token count (plain or prepared input,
  reduced or partial output, which weights are loaded). A warm-up never crosses modes.
- ``sizes(mode)``: the token counts that reach every variant of the mode, up to the entry's maximum.
- ``inputs(mode, n, *args, **kwargs)``: ``(args, kwargs)`` of one call at ``n`` tokens, built from the triggering
  call's arguments (zeros: the dispatch depends on shapes only).

The first call of each mode in a process calls the entry once per size in FlyDSL's compile-only mode (every
variant compiles into FlyDSL's artifact cache, no kernel launches; the torch glue still runs on each size's
buffers), then makes the call itself. "No launches" includes the handles ``flyc.compile`` returned to earlier real
calls (``_run_compiled``'s ``CompiledFunction``): FlyDSL checks compile-only mode on a ``@flyc.jit`` call only, and
such a handle would run on buffers whose producer the warm-up only compiled (``_compile_only``). A later real call
of a variant finds it in that cache. After the warm-up, a FlyDSL compile under the entry is a variant the warm-up
missed: it logs a warning (``late_compiles``).

The compiles run in parallel processes (``set_workers``; default one per compile, up to the CPUs this process may
use): a first pass runs the warm-up calls with ``flyc.compile`` recording each compile instead of doing it (the
launcher and the argument types), and worker processes rebuild each recorded launcher from its factory
(``@launcher``; a module-level ``@flyc.jit`` function by its name) and compile it with CPU placeholder tensors into
FlyDSL's disk cache. The workers take the items in size order from a queue; as soon as every item of a size is
compiled, this process runs that size's warm-up call once more with ``flyc.compile`` making handles from CPU
placeholders (``_HandleMaker``: FlyDSL 0.5.0 traces the launcher, finds the binary in the disk cache and returns a
``CompiledFunction`` without launching), so the hosts hold every variant's handle before serving and no request
traces (0.5.0 traces on every cache lookup). Sizes that could not be loaded so (a launcher whose factory is not
registered, arguments that do not pickle, a worker error) run once more in compile-only mode in this process,
which compiles the rest. A worker sees only this process's GPU.

The caller does nothing: it builds the weights, runs its dummy forwards and serves (vLLM runs each CUDA graph size
eagerly once before it captures it). A first call inside CUDA graph capture does not warm up (its torch work would
be captured) and compiles lazily, with a warning. Compile-only mode is process-wide (FlyDSL's ``COMPILE_ONLY``):
while a warm-up runs, another thread's FlyDSL launches would be skipped too. Test harnesses that call an entry at a
single size turn the automatic warm-up off (``set_auto(False)``, tests/__init__.py); the warm-up tests turn it on.
"""

import contextlib
import contextvars
import functools
import logging
import os
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass

import torch

log = logging.getLogger("amd_kernels")


@dataclass(frozen=True)
class Entry:
    """A registered entry (``@entry``): its unwrapped function and its warm-up declaration."""

    name: str
    fn: Callable
    mode: Callable
    sizes: Callable
    inputs: Callable


REGISTRY: dict[str, Entry] = {}

_PACKAGE = os.path.dirname(os.path.abspath(__file__)) + os.sep
_THIS = os.path.abspath(__file__)
_auto = True
_warmed: set = set()
_lock = threading.RLock()
_in_warm_up = contextvars.ContextVar("amd_kernels_in_warm_up", default=False)
_current = contextvars.ContextVar("amd_kernels_entry", default=None)
_hooked = False
_late = 0


_workers = None


def set_auto(enabled: bool) -> None:
    """Turn the first-call warm-up on or off for this process (default on)."""
    global _auto
    _auto = enabled


def set_workers(n: int | None) -> None:
    """At most ``n`` worker processes for the warm-up's compiles (1: compile in this process; None, the default:
    one per compile, up to the CPUs this process may use)."""
    global _workers
    _workers = n


def _max_workers() -> int:
    if _workers is not None:
        return max(1, _workers)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def launcher(factory):
    """Register a launcher factory (a module-level function with picklable arguments): what it returns records how
    to rebuild it, so a warm-up worker process can compile the same launcher."""
    assert "<locals>" not in factory.__qualname__, factory.__qualname__
    ref = (factory.__module__, factory.__qualname__)

    @functools.wraps(factory)
    def build(*args, **kwargs):
        exe = factory(*args, **kwargs)
        if getattr(exe, "_akr_rebuild", None) is None:
            try:
                exe._akr_rebuild = (ref, args, kwargs)
            except AttributeError:  # an object without a __dict__: compiled in-process
                pass
        return exe

    return build


def warmed() -> frozenset:
    """The (entry name, mode) pairs warmed up in this process."""
    return frozenset(_warmed)


def late_compiles() -> int:
    """FlyDSL compiles under an entry after its warm-up, in this process."""
    return _late


def entry(name: str, *, mode: Callable, sizes: Callable, inputs: Callable):
    """Register ``fn`` as an entry that warms each of its modes up on the first call of that mode (module doc)."""

    def wrap(fn):
        spec = Entry(name, fn, mode, sizes, inputs)
        assert name not in REGISTRY, f"entry {name} registered twice"
        REGISTRY[name] = spec

        @functools.wraps(fn)
        def call(*args, **kwargs):
            if _in_warm_up.get():  # a warm-up's own calls (another entry's inputs)
                return fn(*args, **kwargs)
            key = (name, mode(*args, **kwargs))
            if _auto and key not in _warmed:
                _warm(spec, key, args, kwargs)
            token = _current.set(key)
            try:
                return fn(*args, **kwargs)
            finally:
                _current.reset(token)

        call.entry = spec
        return call

    return wrap


def _flydsl_compile_only() -> bool:
    from flydsl.utils import env

    return bool(env.compile.compile_only)


@contextlib.contextmanager
def _compile_only():
    from flydsl.compiler.jit_function import CompiledFunction

    prev = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"  # FlyDSL reads it at every @flyc.jit call (flydsl.utils.env)
    # A CompiledFunction (what flyc.compile returned to an earlier real call, ``_run_compiled``'s ``exe._cf``) does
    # not read it: in a warm-up it would launch on buffers whose producer was only compiled (M3.1-Flash TP4 SP
    # prefill: the compact MoE mode's warm-up at 17 rows runs the GEMM2 that the BF16 mode loaded at 97-128 rows on
    # a sort it has not, and faults with HSA_STATUS_ERROR_MEMORY_FAULT on every rank). Only the warm-up's own calls
    # skip: another thread's held handles keep launching.
    launch = CompiledFunction.__call__

    def call(self, *args):
        return None if _in_warm_up.get() else launch(self, *args)

    CompiledFunction.__call__ = call
    try:
        yield
    finally:
        CompiledFunction.__call__ = launch
        if prev is None:
            os.environ.pop("COMPILE_ONLY", None)
        else:
            os.environ["COMPILE_ONLY"] = prev


def _warm(spec: Entry, key, args, kwargs) -> None:
    _hook_compiler()
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        log.warning("amd_kernels: first call of %s (mode %s) inside CUDA graph capture: no warm-up, its variants "
                    "compile lazily", *key)  # fmt: skip
        _warmed.add(key)
        return
    with _lock:
        if key in _warmed:
            return
        token = _in_warm_up.set(True)
        t0 = time.perf_counter()
        sizes = list(spec.sizes(key[1]))
        stats = {}
        # a process already in compile-only mode keeps no handles: they would launch after the warm-up
        load = not _flydsl_compile_only()
        try:
            with _compile_only(), torch.no_grad():
                workers = _max_workers()
                if workers > 1 and os.environ.get("FLYDSL_RUNTIME_ENABLE_CACHE", "1") != "0":
                    stats = _compile_in_workers(spec, key[1], sizes, args, kwargs, workers, load)
                if not stats.get("complete"):  # compiles here what the workers did not
                    t1 = time.perf_counter()
                    for n in sizes:
                        a, k = spec.inputs(key[1], n, *args, **kwargs)
                        spec.fn(*a, **k)
                    stats["final_s"] = round(time.perf_counter() - t1, 1)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # the warm-up sizes' torch buffers
        finally:
            _in_warm_up.reset(token)
        _warmed.add(key)
        log.info("amd_kernels: %s (mode %s) warmed up at %d sizes in %.1f s %s", key[0], key[1], len(sizes),
                 time.perf_counter() - t0, stats)  # fmt: skip


class _Unreplayable(Exception):
    pass


def _spec(a):
    """A picklable description of one launch argument, rebuilt by ``_build`` in a worker."""
    if isinstance(a, torch.Tensor):
        return ("tensor", a.dtype, tuple(a.shape), tuple(a.stride()))
    if a is None or isinstance(a, (bool, int, float, str)):
        return ("value", a)
    if isinstance(a, torch.cuda.Stream) or type(a).__name__ == "Stream":
        return ("stream",)
    raise _Unreplayable(type(a).__name__)


def _build(s):
    if s[0] == "tensor":  # a CPU tensor is a compile placeholder: same cache key, nothing launches
        return torch.empty_strided(s[2], s[3], dtype=s[1], device="cpu")
    return s[1] if s[0] == "value" else None


class _Recorder:
    """``flyc.compile`` during the recording pass: notes each (launcher, argument types) compile and compiles
    nothing (returns None, as in compile-only mode, so a caller keeps no handle)."""

    def __init__(self, real, calls, hints=None):
        self.real, self.calls, self.hints = real, calls, hints

    def __getitem__(self, hints):
        return _Recorder(self.real, self.calls, hints)

    def __call__(self, func, *args):
        if not args:  # flyc.compile[hints](launch): the hinted function itself
            return (self.real[self.hints] if self.hints else self.real)(func)
        rebuild = getattr(func, "_akr_rebuild", None) or _module_level(func)
        fn = getattr(func, "func", func)  # a JitFunction's Python function
        name = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', type(func).__name__)}"
        try:  # a name in place of an item: a compile no worker can replay
            item = (rebuild, self.hints, tuple(_spec(a) for a in args)) if rebuild is not None else name
        except _Unreplayable as e:
            item = f"{name} ({e} argument)"
        self.calls.append(item)


def _placeholder(a):
    """A CPU stand-in for a launch argument: same dtype, shape and strides (FlyDSL compiles it, never launches)."""
    if isinstance(a, torch.Tensor):
        return torch.empty_strided(tuple(a.shape), tuple(a.stride()), dtype=a.dtype, device="cpu")
    return a


class _HandleMaker:
    """``flyc.compile`` during the load pass: the real compile on CPU placeholders outside compile-only mode, so
    FlyDSL traces, takes the binary from the disk cache and returns a ``CompiledFunction`` that has not launched;
    the host keeps it as it keeps any first-call handle."""

    def __init__(self, real, hints=None):
        self.real, self.hints = real, hints

    def __getitem__(self, hints):
        return _HandleMaker(self.real, hints)

    def __call__(self, func, *args):
        real = self.real[self.hints] if self.hints else self.real
        if not args:
            return real(func)
        prev = os.environ.get("COMPILE_ONLY")
        os.environ["COMPILE_ONLY"] = "0"  # read at every call (flydsl.utils.env)
        try:
            return real(func, *[_placeholder(a) for a in args])
        finally:
            os.environ["COMPILE_ONLY"] = prev if prev is not None else "1"


def _module_level(func):
    """``("attr", module, name)`` for a ``@flyc.jit`` function defined at module level: a worker takes it by name."""
    import sys

    fn = getattr(func, "func", None)
    mod, name = getattr(fn, "__module__", None), getattr(fn, "__qualname__", "")
    if mod and name and "." not in name and getattr(sys.modules.get(mod), name, None) is func:
        return ("attr", mod, name)
    return None


def _compile_in_workers(spec: Entry, mode, sizes, args, kwargs, workers, load=True) -> dict:
    """Record the mode's compiles, compile them in up to ``workers`` processes into FlyDSL's disk cache, and load
    each size's handles in this process as soon as its items are compiled. ``complete`` in the result: every size
    was loaded so."""
    import pickle

    import flydsl.compiler as flyc

    t0 = time.perf_counter()
    calls, per_size = [], []
    real = flyc.compile
    flyc.compile = _Recorder(real, calls)
    try:
        for n in sizes:
            start = len(calls)
            a, k = spec.inputs(mode, n, *args, **kwargs)
            spec.fn(*a, **k)
            per_size.append((n, start, len(calls)))
    except Exception as e:  # noqa: BLE001 - a host that cannot run on the recorder: everything compiles here
        log.warning("amd_kernels: %s warm-up recording failed: %s: %s", spec.name, type(e).__name__, e)
        return {"record_error": type(e).__name__}
    finally:
        flyc.compile = real
    items, index, call_item, missed = [], {}, [], set()
    for item in calls:
        if isinstance(item, str):
            missed.add(item)
            call_item.append(None)
            continue
        try:
            blob = pickle.dumps(item)
        except Exception:  # noqa: BLE001 - an argument that does not pickle
            missed.add(f"{item[0][-1] if isinstance(item[0][0], str) else item[0][0][1]} (arguments do not pickle)")
            call_item.append(None)
            continue
        if blob not in index:
            index[blob] = len(items)
            items.append(blob)
        call_item.append(index[blob])
    # a size loads once all its items are compiled; one an item of which cannot be replayed is left to the final pass
    needs = [(n, {call_item[c] for c in range(lo, hi)}) for n, lo, hi in per_size]
    stats = {"recorded": len(calls), "replayed": len(items), "record_s": round(time.perf_counter() - t0, 1)}
    loaded, failed = [], set()

    def load_ready(done):
        while len(loaded) < len(needs):
            n, need = needs[len(loaded)]
            if None in need or need & failed:
                loaded.append(False)
                continue
            if not need <= done:
                return
            loaded.append(load and _load_size(spec, mode, n, args, kwargs, real))

    if items:
        t1 = time.perf_counter()
        n_workers = min(workers, len(items))
        errors = _run_workers(items, n_workers, load_ready, failed)
        stats.update(workers=n_workers, workers_s=round(time.perf_counter() - t1, 1), errors=len(errors))
        for e in errors[:3]:
            log.warning("amd_kernels: %s warm-up worker: %s", spec.name, e)
    else:
        load_ready(set())
    stats["loaded"] = sum(loaded)
    if missed:
        stats["not_replayed"] = sorted(missed)
    stats["complete"] = all(loaded) and len(loaded) == len(needs) and not missed and not stats.get("errors")
    return stats


def _load_size(spec: Entry, mode, n, args, kwargs, real) -> bool:
    """One size's warm-up call with ``flyc.compile`` making handles (``_HandleMaker``): every variant of the size
    is traced, taken from the disk cache and held by its host. False (logged) where the call fails."""
    import flydsl.compiler as flyc

    flyc.compile = _HandleMaker(real)
    try:
        a, k = spec.inputs(mode, n, *args, **kwargs)
        spec.fn(*a, **k)
        return True
    except Exception as e:  # noqa: BLE001 - the final pass compiles this size in compile-only mode
        log.warning("amd_kernels: %s warm-up load at %d tokens failed: %s: %s", spec.name, n, type(e).__name__, e)
        return False
    finally:
        flyc.compile = real


def _run_workers(items, workers, on_progress, failed) -> list:
    """Compile the pickled items in fresh interpreters that see only this process's GPU (``_worker_main``; not
    multiprocessing, whose spawn imports the caller's main module again and runs its top-level code). Worker ``i``
    gets item numbers ``i, i + workers, ...`` (item order = size order) on its stdin at once, so it never waits for
    this process; ``on_progress(done)`` runs here after each batch of finished items (the load pass of every size
    whose items are all compiled) while the workers go on."""
    import pickle
    import selectors
    import subprocess
    import sys
    import tempfile

    dev = torch.cuda.current_device()
    env = dict(os.environ, COMPILE_ONLY="1")
    env["FLYDSL_GPU_ARCH"] = torch.cuda.get_device_properties(dev).gcnArchName.split(":")[0]
    var = next((v for v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES") if os.environ.get(v)), None)
    sel = os.environ[var].split(",")[dev] if var else str(dev)
    env[var or "HIP_VISIBLE_DEVICES"] = sel
    # vLLM mirrors HIP_VISIBLE_DEVICES into CUDA_VISIBLE_DEVICES; a worker that narrows only one of the two sees
    # them disagree and HIP aborts it ("Conflicting visibility of agent-N"), so narrow the mirror too
    other = "CUDA_VISIBLE_DEVICES" if var == "HIP_VISIBLE_DEVICES" else "HIP_VISIBLE_DEVICES"
    if var and os.environ.get(other) == os.environ[var]:
        env[other] = sel
    root = os.path.dirname(os.path.dirname(_THIS))  # the directory that holds amd_kernels
    env["PYTHONPATH"] = os.pathsep.join([root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    cmd = [sys.executable, "-c", "from amd_kernels.warm_up import _worker_main; _worker_main()"]
    errors, done = [], set()
    try:
        with tempfile.TemporaryDirectory(prefix="akr-warm-up-") as d, selectors.DefaultSelector() as poll:
            path = os.path.join(d, "items.pkl")
            with open(path, "wb") as f:
                pickle.dump(items, f)
            procs = []
            for i in range(workers):
                err = open(os.path.join(d, f"{i}.err"), "w")  # a file: no pipe to drain
                p = subprocess.Popen(cmd + [path], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=err)  # fmt: skip
                procs.append((p, err))
                poll.register(p.stdout.fileno(), selectors.EVENT_READ, [b""])  # unbuffered: os.read per event
                p.stdin.write("".join(f"{num}\n" for num in range(i, len(items), workers)).encode())
                p.stdin.close()
            open_procs = len(procs)
            while open_procs:
                for key, _ in poll.select():
                    chunk = os.read(key.fd, 65536)
                    if not chunk:  # the worker exited
                        poll.unregister(key.fd)
                        open_procs -= 1
                        continue
                    *lines, key.data[0] = (key.data[0] + chunk).split(b"\n")
                    for line in lines:
                        kind, _, rest = line.decode(errors="replace").partition(" ")
                        num, _, msg = rest.partition(" ")
                        if kind not in ("DONE", "ERROR"):
                            continue
                        done.add(int(num))
                        if kind == "ERROR":
                            failed.add(int(num))
                            errors.append(msg)
                on_progress(done)
            for p, err in procs:
                rc = p.wait()
                err.close()
                if rc:
                    with open(err.name) as f:
                        errors.append(f"worker exited {rc}: {f.read().strip()[-300:]}")
    except Exception as e:  # noqa: BLE001 - no workers: the final pass compiles everything here
        errors.append(f"workers failed: {type(e).__name__}: {e}")
    on_progress(done)
    return errors


def _worker_main() -> None:
    """A warm-up worker (``_run_workers``): compile the items of ``sys.argv[1]`` whose numbers arrive on stdin."""
    import pickle
    import sys

    with open(sys.argv[1], "rb") as f:
        items = pickle.load(f)
    for line in sys.stdin:
        num = int(line)
        errs = _replay(items[num])
        print(f"ERROR {num} {errs[0]}" if errs else f"DONE {num}", flush=True)


def _replay(blob) -> list:
    """In a worker: rebuild one recorded launcher from its factory and compile it with CPU placeholders."""
    import importlib
    import pickle

    import flydsl.compiler as flyc

    rebuild, hints, specs = pickle.loads(blob)
    qualname = rebuild[2] if isinstance(rebuild[0], str) else rebuild[0][1]
    try:
        if isinstance(rebuild[0], str):  # ("attr", module, name): a module-level @flyc.jit function
            exe = getattr(importlib.import_module(rebuild[1]), rebuild[2])
        else:  # ((module, factory), args, kwargs): rebuilt by its @launcher factory
            (module, factory), fargs, fkwargs = rebuild
            exe = getattr(importlib.import_module(module), factory)(*fargs, **fkwargs)
        (flyc.compile[hints] if hints else flyc.compile)(exe, *[_build(s) for s in specs])
    except Exception as e:  # noqa: BLE001
        return [f"{qualname}: {type(e).__name__}: {str(e)[:200]}"]
    return []


def _hook_compiler() -> None:
    """Count and report FlyDSL compiles under a warmed entry (a disk or memory cache hit does not compile)."""
    global _hooked
    with _lock:
        if _hooked:
            return
        from flydsl.compiler.jit_function import MlirCompiler

        real = MlirCompiler.compile  # a classmethod: bound to MlirCompiler

        def compile_and_report(cls, *a, **k):
            key = _current.get()
            if key is not None and key in _warmed and not _in_warm_up.get():
                _report(key)
            return real(*a, **k)

        MlirCompiler.compile = classmethod(compile_and_report)
        _hooked = True


def _report(key) -> None:
    global _late
    with _lock:
        _late += 1
    frames = [
        f
        for f in traceback.extract_stack()
        if os.path.abspath(f.filename).startswith(_PACKAGE) and os.path.abspath(f.filename) != _THIS
    ]
    site = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "?"
    log.warning("amd_kernels: %s (mode %s) compiled a variant after its warm-up, from %s", key[0], key[1], site)

