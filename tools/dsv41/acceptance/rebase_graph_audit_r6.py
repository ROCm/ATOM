"""Bounded, opt-in runtime evidence for the rebased service's warmup boundary.

Install before model construction. Hooks live at host/compiler entry points,
never in a traced layer. Counters are attempts (including cache loads/waits),
not isolated compile times. Native replay counters require GPU trace evidence
before being called fast-path execution. Unsupported compiler APIs are fatal.
"""

import collections
import contextlib
import functools
import hashlib
import inspect
import json
import os
import threading
import time
from pathlib import Path

REQUIRED_HOOKS = {
    "triton_specialization",
    "flydsl_compile_lock",
    "flydsl_ir_context",
    "flydsl_codegen",
    "aiter_native_build",
    "aiter_ctypes_build",
    "atom_backend_request",
    "hip_graph_capture",
    "dynamo_trace",
}


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    if len(raw) > 1_000_000:
        raise RuntimeError("Graph audit receipt exceeds 1 MB")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


class Audit:
    def __init__(self, phase_file):
        self.phase_file = Path(phase_file)
        self.lock = threading.RLock()
        self.counts = collections.Counter()
        self.late = collections.deque(maxlen=32)
        self.hooks = {}
        self.graphs = {}
        self.sealed = False
        self.active = 0
        self.failure = None
        self.process_identity = (os.getpid(), time.time_ns())

    def phase(self):
        value = self.phase_file.read_text().strip()
        if value not in ("warmup", "sealed"):
            raise ValueError("Graph phase must be warmup or sealed")
        self.sealed |= value == "sealed"
        return "sealed" if self.sealed else "warmup"

    def event(self, name, detail="", *, cold=True):
        with self.lock:
            phase = self.phase()
            self.counts[f"{phase}:{name}"] += 1
            if cold and phase == "sealed":
                full = str(detail)
                self.late.append(
                    {
                        "event": name,
                        "detail": full[:16384],
                        "detail_complete": len(full) <= 16384,
                        "detail_characters": len(full),
                        "detail_sha256": hashlib.sha256(full.encode()).hexdigest(),
                        "ns": time.time_ns(),
                    }
                )

    @contextlib.contextmanager
    def forward(self):
        with self.lock:
            self.active += 1
        try:
            yield
        except BaseException as error:
            with self.lock:
                self.failure = repr(error)[:1024]
            raise
        finally:
            with self.lock:
                self.active -= 1

    def snapshot(self, request_id, rank):
        with self.lock:
            if self.active:
                return None
            return {
                "request_id": request_id,
                "rank": rank,
                "process_identity": self.process_identity,
                "phase": self.phase(),
                "counts": dict(self.counts),
                "late_examples": list(self.late),
                "hooks": dict(self.hooks),
                "graphs": dict(self.graphs),
                "failure": self.failure,
                "scope": "Host attempts and replay calls; not isolated JIT duration",
            }


def hook(audit, owner, name, label):
    """Preserve Python descriptors and instrument only cold entry points."""
    raw = inspect.getattr_static(owner, name)
    descriptor = type(raw) if isinstance(raw, (classmethod, staticmethod)) else None
    original = raw.__func__ if descriptor else raw
    if not callable(original):
        raise TypeError(f"Unsupported audit API: {label}")
    source = inspect.unwrap(original)
    code = inspect.getsource(source)
    signature = inspect.signature(source)
    audit.hooks[label] = {
        "file": inspect.getsourcefile(source),
        "sha256": hashlib.sha256(code.encode()).hexdigest(),
    }

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs).arguments
        detail = {
            key: value
            for key, value in bound.items()
            if key in ("func_name", "key", "cache_key", "specialization_key")
        }
        audit.event(label, detail)
        return original(*args, **kwargs)

    setattr(owner, name, descriptor(wrapped) if descriptor else wrapped)


def install():
    import torch
    from aiter.dist.parallel_state import get_tp_group, model_parallel_is_initialized
    from aiter.jit import core
    from flydsl.compiler import jit_function, kernel_function
    from triton.runtime.jit import JITFunction

    from atom.model_engine.model_runner import ModelRunner
    from atom.utils.backends import CompilerManager
    from atom.utils.region_graph import TokenGraph

    if getattr(ModelRunner, "_rebase_graph_audit", False):
        raise RuntimeError("Install graph auditing once, before model construction")
    output = Path(os.environ["DSV41_OWNER_SERVING_OUTPUT"])
    audit = Audit(os.environ["DSV41_OWNER_GRAPH_PHASE_FILE"])
    # _do_compile is Triton's specialization-miss path, including disk loading.
    # FlyDSL's lock is past memory/disk hits; context starts IR construction,
    # while MlirCompiler is codegen. These overlap and must not be summed.
    for owner, name, label in (
        (JITFunction, "_do_compile", "triton_specialization"),
        (jit_function.JitCacheManager, "compile_lock", "flydsl_compile_lock"),
        (kernel_function.CompilationContext, "__init__", "flydsl_ir_context"),
        (jit_function.MlirCompiler, "compile", "flydsl_codegen"),
        (core, "build_module", "aiter_native_build"),
        (core, "build_ctypes_module", "aiter_ctypes_build"),
        (CompilerManager, "compile", "atom_backend_request"),
        (torch.cuda.CUDAGraph, "capture_begin", "hip_graph_capture"),
    ):
        hook(audit, owner, name, label)

    # A re-trace may happen before an ATOM backend request reaches its cache.
    torch._dynamo.on_compile_start(lambda *a, **kw: audit.event("dynamo_trace"))
    audit.hooks["dynamo_trace"] = {"api": "torch._dynamo.on_compile_start"}

    record, replay = TokenGraph.record, TokenGraph.replay

    @functools.wraps(record)
    def audited_record(self, forward, args=(), kwargs=None, key=None, phase=None):
        label = getattr(forward, "__qualname__", type(forward).__name__)
        if label.startswith("DecoderGraph.run."):
            self._rebase_audit_kind = "native_decoder"
        kind = getattr(self, "_rebase_audit_kind", type(self).__name__)
        audit.event("native_record", (kind, key, phase))
        result = record(self, forward, args, kwargs, key, phase)
        actual_key = self.key(args, kwargs or {}) if key is None else key
        inputs = self.inputs(actual_key, phase)
        bindings = [
            (x.data_ptr(), tuple(x.shape), tuple(x.stride()), str(x.dtype))
            for x in inputs
            if isinstance(x, torch.Tensor)
        ]
        with audit.lock:
            identity = repr((id(self), actual_key, phase))
            audit.graphs[identity] = {"kind": kind, "bindings": bindings}
        return result

    @functools.wraps(replay)
    def audited_replay(self, *args, **kwargs):
        result = replay(self, *args, **kwargs)
        kind = getattr(self, "_rebase_audit_kind", type(self).__name__)
        # phase() is read at the outer forward boundary; no filesystem reads
        # per layer/graph replay and no timestamp when the path is warm.
        with audit.lock:
            phase = "sealed" if audit.sealed else "warmup"
            audit.counts[f"{phase}:{kind}_replay"] += 1
        return result

    TokenGraph.record, TokenGraph.replay = audited_record, audited_replay
    original = ModelRunner.forward

    @functools.wraps(original)
    def forward(self, batch):
        with audit.forward():
            with audit.lock:
                audit.phase()
            result = original(self, batch)
            if not batch.is_dummy_run:
                audit.event("real_forward", cold=False)
            return result

    ModelRunner.forward = forward
    ModelRunner._rebase_graph_audit = audit

    def snapshots():
        previous = None
        membership = WorkerMembership()
        request = output / "graph_audit_request.json"
        while True:
            time.sleep(0.2)
            try:
                if not request.exists():
                    continue
                request_id = json.loads(request.read_text())["id"]
                if request_id == previous:
                    continue
                rank = membership.rank(model_parallel_is_initialized, get_tp_group)
                if rank is None:
                    continue
                snapshot = audit.snapshot(request_id, rank)
                if snapshot is not None:
                    write_json(output / f"graph_audit_rank{rank}.json", snapshot)
                    previous = request_id
            except BaseException as error:  # noqa: BLE001 - publish failed observer
                write_json(
                    output / f"graph_audit_failure_pid{os.getpid()}.json",
                    {"error": repr(error)},
                )
                return

    threading.Thread(target=snapshots, daemon=True).start()
    return audit


class WorkerMembership:
    """API/controller processes never publish TP receipts; workers cannot vanish."""

    def __init__(self):
        self.initialized = False

    def rank(self, is_initialized, get_group):
        if not is_initialized():
            if self.initialized:
                raise RuntimeError("Previously initialized audit worker lost TP")
            return None
        self.initialized = True
        return get_group().rank_in_group


def validate_interval(before, after, *, require_decoder=True):
    """Never accept missing ranks, restarted workers or a late cold operation."""
    if sorted(row["rank"] for row in before) != list(range(4)) or sorted(
        row["rank"] for row in after
    ) != list(range(4)):
        raise ValueError("Need exactly four distinct rank snapshots")
    deltas = []
    for old, new in zip(
        sorted(before, key=lambda r: r["rank"]),
        sorted(after, key=lambda r: r["rank"]),
        strict=True,
    ):
        if (
            new["failure"]
            or old["hooks"] != new["hooks"]
            or not REQUIRED_HOOKS.issubset(new["hooks"])
            or old["process_identity"] != new["process_identity"]
        ):
            raise ValueError("Worker failed or audit hooks changed/missing")
        if old["phase"] != "sealed" or new["phase"] != "sealed":
            raise ValueError("Both measurement boundaries must be sealed")
        delta = {
            key: new["counts"].get(key, 0) - old["counts"].get(key, 0)
            for key in old["counts"].keys() | new["counts"].keys()
        }
        allowed = {
            "real_forward",
            "native_decoder_replay",
            "ReqGraph_replay",
            "TokenGraph_replay",
        }
        if any(value < 0 for value in delta.values()) or any(
            value and key.split(":", 1)[1] not in allowed
            for key, value in delta.items()
        ):
            raise ValueError(f"Cold operation/restart in measurement: {delta}")
        if delta.get("sealed:real_forward", 0) <= 0:
            raise ValueError("No real measured forwards")
        if require_decoder and delta.get("sealed:native_decoder_replay", 0) <= 0:
            raise ValueError("Native decoder replay was not observed")
        if old["graphs"] != new["graphs"]:
            raise ValueError("Native graph inventory changed in measurement")
        deltas.append(delta)
    return deltas
