# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Keep a model's processed weights on the GPU across server restarts.

A *daemon* is an ordinary ATOM launch with ``ATOM_WEIGHT_CACHE=daemon``: its
runners load and post-process their shard from disk as usual, then stop there.
Each rank exports what the load produced as CUDA IPC handles and serves them on
a Unix socket named after its GPU. A server started with
``ATOM_WEIGHT_CACHE=client`` builds its model on the meta device and maps those
tensors instead of reading the checkpoint, with no copy.

What the daemon serves is worked out by diffing the model (modules and the
ATOM/AITER objects hanging off them) before and after ``load_model``:

* every parameter after loading;
* buffers and tensor attributes that loading created or changed (compared by
  content), for example MLA's absorbed W_K/W_V or MegaMoE's repacked weights;
* plain values, containers of tensors and plain values, and module-level
  functions/classes that loading set.

If loading created any other kind of state (a new object, a new or removed
submodule), nothing is served. MoE ``init_prepare_finalize`` builds
communication and kernel objects for serving, so the daemon skips it and the
client runs it after mapping the weights.

The weights are a function of the load code and of what that code read, so a
client takes the cache only if all of these match the daemon's:

* identity: checkpoint headers, hf_config, rank, GPU arch, library versions;
* what the daemon's load read (input_recorder.py): the environment variables
  and config fields that loading read, the construction-time attributes it
  read (so a setting read by a constructor and used by post-processing, such
  as a quant method's layout flag, counts), the globals the loading code
  referenced, and the files it opened. Anything loading did not read (serving
  flags such as --max-model-len, run-time kernel toggles) does not count;
* code: an AST hash of every ATOM/AITER function that ran during load and the
  cache key of every Triton kernel launched meanwhile. The client re-hashes
  them from its own sources; both sides refuse sources that changed after
  their process started. Editing code that does not run during load (a
  forward, a constructor) keeps the cache;
* structure: the module tree, module types, parameter shapes, and the shapes
  of the buffers the daemon exports.

There is no silent fallback once a daemon serves a GPU: a mismatch on any rank
fails the launch with the reason (the daemon's copy leaves no room for a second
one), and all ranks decide together. Without a daemon, the server loads from
disk as usual. The daemon also checks before every hand-out that no client
has written into the shared weights.

``ATOM_PRELOAD=1`` on a normal server launch manages the daemon: with none on
its GPUs it starts one with its own command line and environment (detached, so
it outlives the server) and maps its weights once loaded; with one that does
not match (new load code, other settings or model) it stops it and starts a
new one, once. A daemon exits after ``ATOM_PRELOAD_IDLE_TIMEOUT_S`` without a
connected server. ``python -m atom.model_loader.weight_cache status|stop``
shows or stops this user's daemons.
"""

from __future__ import annotations

import ast
import atexit
import contextlib
import copy
import dataclasses
import enum
import errno
import fcntl
import gc
import glob
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import pickle
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
import types
from typing import Any

import numpy as np
import torch
from torch import nn

from atom.distributed.group_agreement import group_agree, group_max, group_min
from atom.utils import envs

logger = logging.getLogger("atom")

PROTOCOL_VERSION = 3
DAEMON_RUNNER = "atom.model_loader.weight_cache.WeightCacheDaemonRunner"
_TAG = "[weight-cache]"
# Set by the API server (record_launch) for its engine's processes.
_LAUNCH_ENV = "ATOM_PRELOAD_LAUNCH"  # ATOM_PRELOAD: the saved launch
_DAEMON_PID_ENV = "ATOM_WEIGHT_CACHE_DAEMON_PID"  # daemon: its main process
_DAEMON_LOG_ENV = "ATOM_WEIGHT_CACHE_DAEMON_LOG"  # daemon started by ATOM_PRELOAD

# Engine configs referenced from modules carry per-launch fields; config
# fields enter the key through what loading reads.
_CONFIG_CLASSES = frozenset({"Config", "CompilationConfig", "ParallelConfig"})
# Runtime infrastructure whose state changes for reasons other than weights.
_NO_RECURSE = (
    "atom.distributed",
    "atom.model_engine",
    "aiter.dist",
    "torch",
    "triton",
    "mori",
    "logging",
    "threading",
    "multiprocessing",
    "concurrent",
    "queue",
    "asyncio",
    "socket",
    "zmq",
    "transformers.tokenization",
    "tokenizers",
)
_DIFF_PACKAGES, _DIFF_DEPTH = ("atom.", "aiter.", "transformers."), 2


class _Miss(Exception):
    """The cache cannot be used; the message says why."""


class _SourceChanged(Exception):
    """A source file changed after this process started."""


class _Unshippable(Exception):
    """A value that cannot be sent to another process."""


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _is_native_enum(v: Any) -> bool:
    """A pybind11 enum value (e.g. aiter's QuantType): pickles by value."""
    t = type(v)
    return (
        hasattr(t, "__members__")
        and hasattr(v, "name")
        and any(b.__name__ == "pybind11_object" for b in t.__mro__)
    )


def _ref_name(v: Any) -> str | None:
    """``module:qualname`` of a module-level function or class (pickles by reference)."""
    if not isinstance(v, (types.FunctionType, types.BuiltinFunctionType, type)):
        return None
    mod, qual = getattr(v, "__module__", None), getattr(v, "__qualname__", None)
    if not mod or not qual or "<" in qual:
        return None
    obj = sys.modules.get(mod)
    for part in qual.split("."):
        obj = getattr(obj, part, None)
    return f"{mod}:{qual}" if obj is v else None


def _is_plain(v: Any, depth: int = 0) -> bool:
    """A value that pickles by value and means the same in another process."""
    if (
        v is None
        or isinstance(
            v,
            (
                bool,
                int,
                float,
                complex,
                str,
                bytes,
                torch.dtype,
                torch.device,
                torch.Size,
                enum.Enum,
                np.generic,
            ),
        )
        or _is_native_enum(v)
    ):
        return True
    if isinstance(v, np.ndarray):
        return v.dtype != object
    if depth >= 32:
        return False
    if isinstance(v, (list, tuple, set, frozenset)):
        return all(_is_plain(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return all(
            _is_plain(k, depth + 1) and _is_plain(x, depth + 1) for k, x in v.items()
        )
    return False


def _norm(v: Any, depth: int = 0) -> Any:
    """JSON-able, process-independent rendering of a value for comparison."""
    if isinstance(v, enum.Enum) or _is_native_enum(v):
        return f"{type(v).__name__}.{v.name}"
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (torch.dtype, torch.device)):
        return str(v)
    if isinstance(v, torch.Size):
        return list(v)
    if isinstance(v, bytes):
        return hashlib.sha256(v).hexdigest()
    if isinstance(v, np.ndarray):
        return f"ndarray:{v.dtype}:{v.shape}:{hashlib.sha256(v.tobytes()).hexdigest()[:16]}"
    if isinstance(v, np.generic):
        return v.item()
    ref = _ref_name(v)
    if ref:
        return f"ref:{ref}"
    if depth >= 32:
        return f"<{type(v).__qualname__}>"
    if isinstance(v, (list, tuple)):
        return [_norm(x, depth + 1) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted(str(_norm(x, depth + 1)) for x in v)
    if isinstance(v, dict):
        return {
            str(k): _norm(x, depth + 1)
            for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))
        }
    compute_hash = getattr(v, "compute_hash", None)
    if callable(compute_hash):
        with contextlib.suppress(Exception):
            return f"{type(v).__qualname__}:{compute_hash()}"
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {
            "__dc__": type(v).__qualname__,
            **{
                f.name: _norm(getattr(v, f.name, None), depth + 1)
                for f in dataclasses.fields(v)
                if not f.name.startswith("_")
            },
        }
    return f"<{type(v).__module__}.{type(v).__qualname__}>"


def _json(v: Any) -> str:
    return json.dumps(_norm(v), sort_keys=True, default=str)


_DIGEST_CHUNK = 1 << 24
_DIGEST_WEIGHTS: dict[str, torch.Tensor] = {}


def _digest(t: torch.Tensor) -> str:
    """Exact content digest (two position-weighted sums over the raw bytes)."""
    if t.is_meta:
        return "meta"
    if t.numel() == 0:
        return "empty"
    flat = t.detach()
    if not flat.is_contiguous():
        flat = flat.contiguous()
    flat = flat.reshape(-1).view(torch.uint8)
    n = flat.numel()
    x = flat
    for dt, size in ((torch.int64, 8), (torch.int32, 4)):
        if n % size == 0 and flat.storage_offset() % size == 0:
            with contextlib.suppress(RuntimeError):
                x = flat.view(dt)
                break
    key = str(x.device)
    w = _DIGEST_WEIGHTS.get(key)
    if w is None:
        w = (
            torch.arange(1, _DIGEST_CHUNK + 1, device=x.device, dtype=torch.int64)
            % 2147483629
        )
        _DIGEST_WEIGHTS[key] = w
    s1 = s2 = 0
    for i in range(0, x.numel(), _DIGEST_CHUNK):
        c = x[i : i + _DIGEST_CHUNK]
        if c.dtype != torch.int64:
            c = c.to(torch.int64)
        s1 += int(c.sum())
        s2 += int((c * w[: c.numel()]).sum())
    mask = (1 << 64) - 1
    return f"{n}:{s1 & mask:x}:{s2 & mask:x}"


# ---------------------------------------------------------------------------
# Cache key: configuration
# ---------------------------------------------------------------------------


def _checkpoint_digest(model_path: str) -> str:
    """Hash of every shard's safetensors header (names, dtypes, shapes, offsets)."""
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        return f"path:{os.path.realpath(model_path)}"
    h = hashlib.sha256()
    for path in files:
        with open(path, "rb") as f:
            raw = f.read(8)
            (n,) = struct.unpack("<Q", raw)
            h.update(os.path.basename(path).encode())
            h.update(raw + f.read(n))
    return h.hexdigest()


def _git_state(path: str) -> str:
    try:
        head = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if head.returncode != 0:
            return f"nogit:{head.stderr.strip()[:80]}"
        # Compiled sources are not covered by the code trace; their local edits are.
        diff = subprocess.run(
            ["git", "-C", path, "diff", "HEAD", "--", "csrc", "hsa", "3rdparty"],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if diff.returncode != 0:
            return f"{head.stdout.strip()}+diff-failed"
        dirty = hashlib.sha256(diff.stdout).hexdigest()[:12] if diff.stdout else "clean"
        return f"{head.stdout.strip()}+{dirty}"
    except Exception as e:  # noqa: BLE001 - git missing: a fixed marker
        return f"nogit:{type(e).__name__}"


def _versions() -> dict:
    out = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "hip": str(torch.version.hip),
    }
    for name, dists in (
        ("triton", ("triton",)),
        ("flydsl", ("flydsl", "FlyDSL")),
        ("mori", ("mori", "amd-mori")),
    ):
        version = None
        for dist in dists:
            with contextlib.suppress(Exception):
                version = importlib.metadata.version(dist)
                break
        if version is None and name in sys.modules:
            version = getattr(sys.modules[name], "__version__", "?")
        out[name] = version or "absent"
    return out


def compute_key(config, rank: int, device: torch.device) -> dict:
    """The identity part of the key: which checkpoint, model, rank, GPU and
    libraries. Settings enter through what loading reads (input_recorder)."""
    import aiter

    hf = config.hf_config
    try:
        hf_dict = hf.to_dict()
    except Exception:  # noqa: BLE001
        hf_dict = vars(hf)
    props = torch.cuda.get_device_properties(device)
    return {
        "protocol": PROTOCOL_VERSION,
        "model": os.path.realpath(config.model),
        "checkpoint": _checkpoint_digest(config.model),
        "hf_config": hashlib.sha256(_json(hf_dict).encode()).hexdigest(),
        "rank": rank,
        "arch": getattr(props, "gcnArchName", ""),
        "versions": _versions(),
        "aiter": _git_state(
            os.path.dirname(os.path.dirname(os.path.realpath(aiter.__file__)))
        ),
    }


def _diff_key(daemon: dict, client: dict, prefix: str = "") -> list[str]:
    diffs = []
    for k in sorted(set(daemon) | set(client)):
        a, b = daemon.get(k, "<missing>"), client.get(k, "<missing>")
        if isinstance(a, dict) and isinstance(b, dict):
            diffs += _diff_key(a, b, f"{prefix}{k}.")
        elif a != b:
            diffs.append(f"{prefix}{k}: daemon={str(a)[:120]} client={str(b)[:120]}")
    return diffs


# ---------------------------------------------------------------------------
# Cache key: code
# ---------------------------------------------------------------------------

_SOURCE_STATS: dict[str, tuple[int, int]] | None = None
_SKIP_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        "target",
        "build",
        "csrc",
        "hsa",
        "3rdparty",
        "node_modules",
    }
)


def _package_roots() -> dict[str, str]:
    roots = {}
    for name in ("atom", "aiter"):
        with contextlib.suppress(Exception):
            roots[name] = os.path.dirname(
                os.path.realpath(importlib.import_module(name).__file__)
            )
    return roots


def snapshot_sources() -> None:
    """Record each ATOM/AITER source file's (mtime, size) as this process
    starts, so a file edited after its code was imported is caught later."""
    global _SOURCE_STATS
    if _SOURCE_STATS is not None:
        return
    stats = {}
    for root in _package_roots().values():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fn in filenames:
                if fn.endswith(".py"):
                    path = os.path.join(dirpath, fn)
                    with contextlib.suppress(OSError):
                        st = os.stat(path)
                        stats[os.path.realpath(path)] = (st.st_mtime_ns, st.st_size)
    _SOURCE_STATS = stats


class _CodeTracer:
    """Collect the Python code objects that run while tracing (all threads).

    Uses sys.monitoring (3.12+), disabling each code object after its first
    start, so tracing costs about one callback per function."""

    def __init__(self):
        self.codes: set = set()
        self._mon = getattr(sys, "monitoring", None)
        self._tool = None

    def _on_start(self, code, offset):
        self.codes.add(code)
        return self._mon.DISABLE

    def _on_trace(self, frame, event, arg):
        if event == "call":
            self.codes.add(frame.f_code)

    def resume(self):
        if self._tool is not None:
            self._mon.set_events(self._tool, self._mon.events.PY_START)
        else:
            sys.settrace(self._on_trace)
            threading.settrace(self._on_trace)

    def pause(self):
        if self._tool is not None:
            self._mon.set_events(self._tool, 0)
        else:
            sys.settrace(None)
            threading.settrace(None)

    def __enter__(self):
        if self._mon is not None:
            # Not COVERAGE_ID/PROFILER_ID: leave those to coverage tools and profilers.
            self._tool = next(
                (t for t in (3, 4) if self._mon.get_tool(t) is None), None
            )
        if self._tool is not None:
            self._mon.use_tool_id(self._tool, "atom-weight-cache")
            self._mon.register_callback(
                self._tool, self._mon.events.PY_START, self._on_start
            )
        else:
            logger.warning(
                f"{_TAG} no free sys.monitoring tool id; tracing with sys.settrace"
            )
        self.resume()
        return self

    def __exit__(self, *exc):
        self.pause()
        if self._tool is not None:
            self._mon.register_callback(self._tool, self._mon.events.PY_START, None)
            self._mon.free_tool_id(self._tool)
        return False


@contextlib.contextmanager
def _record_jit(kernels: dict):
    """Record every Triton (and Gluon) kernel launched in the block."""
    try:
        from triton.runtime.jit import JITFunction
    except Exception:  # noqa: BLE001 - no Triton, nothing to record
        yield
        return
    orig = JITFunction.__dict__.get("run")
    if orig is None:
        logger.warning(
            f"{_TAG} cannot hook triton JITFunction.run; Triton kernels are not keyed"
        )
        yield
        return

    def run(self, *args, **kwargs):
        kernels.setdefault(id(self), self)
        return orig(self, *args, **kwargs)

    JITFunction.run = run
    try:
        yield
    finally:
        JITFunction.run = orig


def _jit_key(module: str, qual: str) -> str:
    try:
        obj = importlib.import_module(module)
        for part in qual.split("."):
            obj = getattr(obj, part)
        while not hasattr(obj, "cache_key") and hasattr(obj, "fn"):
            obj = obj.fn  # autotuner / heuristics wrappers
        return str(obj.cache_key)
    except Exception as e:  # noqa: BLE001
        return f"<unresolved {type(e).__name__}>"


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _strip


def _dump(node) -> str:
    node = _StripDocstrings().visit(copy.deepcopy(node))
    return ast.dump(node, annotate_fields=False, include_attributes=False)


class _SourceHasher:
    """AST hashes of functions in the atom/aiter packages."""

    def __init__(self):
        self.roots = _package_roots()
        self._trees: dict[str, ast.AST | None] = {}

    def locate(self, filename: str) -> tuple[str, str] | None:
        fn = os.path.realpath(filename)
        for pkg, root in self.roots.items():
            if fn.startswith(root + os.sep):
                return pkg, os.path.relpath(fn, root)
        return None

    def _tree(self, pkg: str, rel: str):
        key = f"{pkg}/{rel}"
        if key not in self._trees:
            path = os.path.realpath(
                os.path.join(self.roots.get(pkg, "/nonexistent"), rel)
            )
            try:
                st = os.stat(path)
            except OSError:
                self._trees[key] = None
                return None
            # Files created after start (e.g. JIT-generated) have nothing to compare with.
            known = _SOURCE_STATS.get(path) if _SOURCE_STATS is not None else None
            if known is not None and known != (st.st_mtime_ns, st.st_size):
                raise _SourceChanged(key)
            with open(path, encoding="utf-8") as f:
                self._trees[key] = ast.parse(f.read())
        return self._trees[key]

    @staticmethod
    def _find_all(node, parts: list[str]) -> list:
        for i, part in enumerate(parts):
            found = []
            stack = list(getattr(node, "body", []))
            while stack:
                child = stack.pop(0)
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    if child.name == part:
                        found.append(child)
                elif isinstance(child, (ast.If, ast.Try, ast.With)):
                    stack.extend(getattr(child, "body", []))
                    stack.extend(getattr(child, "orelse", []))
                    stack.extend(getattr(child, "finalbody", []))
                    for h in getattr(child, "handlers", []):
                        stack.extend(h.body)
            if not found:
                return [node]  # not resolvable: the enclosing scope stands in
            if i == len(parts) - 1:
                return found  # every same-named definition (property setters, if/else)
            node = found[0]
        return [node]

    def digest(self, pkg: str, rel: str, qual: str) -> str:
        tree = self._tree(pkg, rel)
        if tree is None:
            return "<unreadable>"
        parts = [] if qual == "<module>" else qual.split(".")
        nodes = self._find_all(tree, parts)
        return hashlib.sha256("\n".join(_dump(n) for n in nodes).encode()).hexdigest()[
            :16
        ]


def _outer_qualname(code) -> str:
    qual = getattr(code, "co_qualname", code.co_name).split(".<locals>")[0]
    parts = [p for p in qual.split(".") if not p.startswith("<")]
    return ".".join(parts) if parts else "<module>"


def code_fingerprint(codes, kernels) -> list[tuple]:
    """What ran during load: functions and launched Triton kernels. (Module
    constants and bound functions the code used are keyed as globals by
    input_recorder.)"""
    hasher = _SourceHasher()
    own = {
        os.path.realpath(__file__),
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "input_recorder.py"),
    }
    fns = set()
    for code in codes:
        if os.path.realpath(code.co_filename) in own:
            continue
        where = hasher.locate(code.co_filename)
        if where is not None:
            fns.add((*where, _outer_qualname(code)))
    out = [
        ("fn", pkg, rel, qual, hasher.digest(pkg, rel, qual)) for pkg, rel, qual in fns
    ]
    for jf in kernels:
        fn = getattr(jf, "fn", None)
        module, qual = getattr(fn, "__module__", "?"), getattr(fn, "__qualname__", "?")
        try:
            key = str(jf.cache_key)
        except Exception as e:  # noqa: BLE001 - an unkeyable kernel never matches
            key = f"<error {type(e).__name__} pid {os.getpid()}>"
        out.append(("jit", module, qual, "", key))
    return sorted(set(out))


def verify_code(fingerprint) -> list[str]:
    """Entries whose code differs in this process (raises _SourceChanged)."""
    hasher = _SourceHasher()
    changed = []
    for kind, a, b, c, digest in fingerprint:
        now = hasher.digest(a, b, c) if kind == "fn" else _jit_key(a, b)
        if now != digest:
            changed.append(f"{a}/{b}:{c}" if kind == "fn" else f"Triton kernel {a}.{b}")
    return changed


# ---------------------------------------------------------------------------
# Model state: snapshots (before/after load), structure (construction check)
# ---------------------------------------------------------------------------

_MODULE_INTERNALS = frozenset(vars(nn.Module()).keys()) | {
    "_parameters",
    "_buffers",
    "_modules",
}
_FUNCTION_TYPES = (
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
)


def _recursable(v: Any, packages: tuple) -> bool:
    t = type(v)
    mod = t.__module__ or ""
    if isinstance(
        v,
        (nn.Module, torch.Tensor, enum.Enum, type, types.ModuleType) + _FUNCTION_TYPES,
    ):
        return False
    if (
        not hasattr(v, "__dict__")
        or not mod.startswith(packages)
        or mod.startswith(_NO_RECURSE)
    ):
        return False
    return not (mod == "atom.config" and t.__name__ in _CONFIG_CLASSES)


def _walk_objects(model: nn.Module, depth: int, packages: tuple):
    """(path, obj, is_module) for every module and the objects reachable from
    their attributes (``depth`` levels). Paths: ``mod.path`` for modules,
    ``mod.path::attr::attr`` below them. Each object appears once."""
    seen: set[int] = set()
    for path, mod in model.named_modules(remove_duplicate=False):
        yield path, mod, True
        queue = [
            (f"{path}::{k}", v, 1)
            for k, v in vars(mod).items()
            if k not in _MODULE_INTERNALS
        ]
        while queue:
            p, v, d = queue.pop(0)
            if id(v) in seen or not _recursable(v, packages):
                continue
            seen.add(id(v))
            yield p, v, False
            if d < depth:
                queue += [(f"{p}::{k}", x, d + 1) for k, x in vars(v).items()]


def _attrs(obj: Any, is_module: bool) -> dict:
    return {
        k: v
        for k, v in vars(obj).items()
        if not (is_module and k in _MODULE_INTERNALS) and not isinstance(v, nn.Module)
    }


def _callable_name(v: Any) -> str | None:
    if isinstance(v, types.MethodType):
        v = v.__func__
    if isinstance(v, _FUNCTION_TYPES):
        return f"{getattr(v, '__module__', '?')}.{getattr(v, '__qualname__', getattr(v, '__name__', '?'))}"
    return None


def _fp(v: Any, depth: int = 0, *, content: bool = True):
    """Fingerprint for the before/after diff (``content``: tensor bytes, ids)
    or, with ``content=False``, for comparing construction across processes."""
    if isinstance(v, torch.Tensor):
        if not content:
            return ("T", tuple(v.shape), str(v.dtype))
        return (
            "T",
            id(v),
            v.data_ptr() if not v.is_meta else 0,
            tuple(v.shape),
            tuple(v.stride()),
            str(v.dtype),
            _digest(v),
        )
    if _is_plain(v):
        return ("V", _json(v))
    ref = _ref_name(v)
    if ref:
        return ("R", ref)
    name = _callable_name(v)
    if name:
        return ("F", name) if not content else ("F", name, id(v))
    if depth < 8 and isinstance(v, (list, tuple)):
        return (
            "C",
            type(v).__name__,
            tuple(_fp(x, depth + 1, content=content) for x in v),
        )
    if depth < 8 and isinstance(v, dict) and all(_is_plain(k) for k in v):
        return (
            "D",
            tuple(
                sorted(
                    (_json(k), _fp(x, depth + 1, content=content)) for k, x in v.items()
                )
            ),
        )
    t = f"{type(v).__module__}.{type(v).__qualname__}"
    return ("O", t) if not content else ("O", id(v), t)


def snapshot(model: nn.Module) -> dict:
    """Everything loading can change, with tensor contents (daemon only)."""
    snap = {}
    for path, obj, is_module in _walk_objects(model, _DIFF_DEPTH, _DIFF_PACKAGES):
        entry = {"attrs": {k: _fp(v) for k, v in _attrs(obj, is_module).items()}}
        if is_module:
            # Parameters are always exported, so their identity is enough.
            entry["params"] = {
                k: ("P", id(p), tuple(p.shape), str(p.dtype))
                for k, p in obj._parameters.items()
                if p is not None
            }
            entry["buffers"] = {
                k: _fp(b) for k, b in obj._buffers.items() if b is not None
            }
        snap[path] = entry
    return snap


def structure(model: nn.Module) -> dict:
    """The module tree, module types, parameter shapes/dtypes and buffer
    shapes/dtypes, comparable between a real and a meta-built model.
    (Construction-time settings that loading read are compared separately.)"""
    return {
        path: {
            "type": f"{type(mod).__module__}.{type(mod).__qualname__}",
            "params": {
                k: _fp(p, content=False)
                for k, p in mod._parameters.items()
                if p is not None
            },
            "buffers": {
                k: _fp(b, content=False)
                for k, b in mod._buffers.items()
                if b is not None
            },
        }
        for path, mod in model.named_modules(remove_duplicate=False)
    }


def _exported_structure(built: dict, export: dict) -> dict:
    """``built`` with only the buffers the daemon exports or deletes: the
    client builds the others itself, e.g. caches sized by --max-model-len."""
    out = {}
    for path, entry in built.items():
        item = export["plan"].get(path, {})
        keep = set(item.get("buffers", {})) | {
            k for kind, k in item.get("delete", []) if kind == "buffer"
        }
        out[path] = {
            **entry,
            "buffers": {k: v for k, v in entry["buffers"].items() if k in keep},
        }
    return out


def _diff_structure(daemon: dict, client: dict, limit: int = 20) -> list[str]:
    diffs = []
    for path in sorted(set(daemon) | set(client)):
        a, b = daemon.get(path), client.get(path)
        if a is None or b is None:
            diffs.append(
                f"{path or '<root>'}: {'missing in client' if b is None else 'new in client'}"
            )
            continue
        if a["type"] != b["type"]:
            diffs.append(f"{path}: type {a['type']} -> {b['type']}")
        for k in sorted(set(a["params"]) | set(b["params"])):
            if a["params"].get(k) != b["params"].get(k):
                diffs.append(
                    f"{path or '<root>'}.{k} (param): daemon={a['params'].get(k)} client={b['params'].get(k)}"
                )
        for k, fp in a["buffers"].items():
            if b["buffers"].get(k) != fp:
                diffs.append(
                    f"{path or '<root>'}.{k} (buffer): daemon={fp} client={b['buffers'].get(k)}"
                )
        if len(diffs) >= limit:
            break
    return diffs[:limit]


# ---------------------------------------------------------------------------
# Export (daemon) / import (client)
# ---------------------------------------------------------------------------


class _Exporter:
    def __init__(self):
        self.storages: dict[int, tuple] = {}
        self.views: dict[int, torch.Tensor] = {}  # whole storages as uint8, for digests
        self.bytes = 0

    def tensor(self, t: torch.Tensor) -> dict:
        t = t.detach()
        if not t.is_cuda:
            return {"value": t.clone()}
        st = t.untyped_storage()
        if st.nbytes() == 0:
            # No allocation, so no IPC handle: rebuilt locally.
            return {
                "empty": True,
                "shape": tuple(t.shape),
                "stride": tuple(t.stride()),
                "dtype": t.dtype,
            }
        key = st.data_ptr()
        if key not in self.storages:
            self.storages[key] = st._share_cuda_()
            self.views[key] = torch.empty(0, dtype=torch.uint8, device=t.device).set_(
                st, 0, (st.nbytes(),), (1,)
            )
            self.bytes += st.nbytes()
        return {
            "storage": key,
            "offset": t.storage_offset(),
            "shape": tuple(t.shape),
            "stride": tuple(t.stride()),
            "dtype": t.dtype,
        }


def _pack(v: Any, ex: _Exporter, depth: int = 0):
    if isinstance(v, torch.Tensor):
        return ("T", ex.tensor(v))
    if _is_plain(v) or _ref_name(v):
        return ("V", v)  # by value, or by reference for module-level functions/classes
    if depth < 8 and isinstance(v, (list, tuple)):
        return ("C", type(v) is tuple, [_pack(x, ex, depth + 1) for x in v])
    if depth < 8 and isinstance(v, dict) and all(_is_plain(k) for k in v):
        return ("D", [(k, _pack(x, ex, depth + 1)) for k, x in v.items()])
    raise _Unshippable(f"{type(v).__module__}.{type(v).__qualname__}")


def _unpack(p, make):
    kind = p[0]
    if kind == "T":
        return make(p[1])
    if kind == "V":
        return p[1]
    if kind == "C":
        items = [_unpack(x, make) for x in p[2]]
        return tuple(items) if p[1] else items
    return {k: _unpack(x, make) for k, x in p[1]}


def build_export(model: nn.Module, before: dict, after: dict) -> dict:
    """The tensors and attributes ``load_model`` produced, as an import plan."""
    ex = _Exporter()
    objects = {
        path: (obj, is_module)
        for path, obj, is_module in _walk_objects(model, _DIFF_DEPTH, _DIFF_PACKAGES)
    }
    plan: dict[str, dict] = {}
    unshippable: list[str] = []
    unshippable += [
        f"{p or '<root>'}: submodule removed during load"
        for p in before
        if "::" not in p and p not in after
    ]
    unshippable += [
        f"{p}: submodule created during load"
        for p in after
        if "::" not in p and p not in before
    ]
    for path, entry in after.items():
        if path not in before:
            continue  # a new object: the attribute holding it decides below
        obj, is_module = objects[path]
        prev = before[path]
        item: dict[str, Any] = {"params": {}, "buffers": {}, "attrs": {}, "delete": []}
        if is_module:
            for k, p in obj._parameters.items():
                if p is None:
                    continue
                meta = ex.tensor(p)
                meta["pattrs"] = {a: v for a, v in vars(p).items() if _is_plain(v)}
                item["params"][k] = meta
            for k, b in obj._buffers.items():
                if b is not None and prev["buffers"].get(k) != entry["buffers"].get(k):
                    item["buffers"][k] = ex.tensor(b)
            item["delete"] += [
                ("param", k) for k in prev["params"] if k not in entry["params"]
            ]
            item["delete"] += [
                ("buffer", k) for k in prev["buffers"] if k not in entry["buffers"]
            ]
        attrs = _attrs(obj, is_module)
        for k, fp in entry["attrs"].items():
            if prev["attrs"].get(k) == fp:
                continue
            try:
                item["attrs"][k] = _pack(attrs[k], ex)
            except _Unshippable as e:
                unshippable.append(f"{path or '<root>'}.{k}: {e}")
        item["delete"] += [
            ("attr", k) for k in prev["attrs"] if k not in entry["attrs"]
        ]
        if any(item.values()):
            plan[path] = item
    return {
        "plan": plan,
        "storages": ex.storages,
        "views": ex.views,
        "bytes": ex.bytes,
        "unshippable": unshippable,
    }


def _resolve(model: nn.Module, path: str):
    mod_path, _, attr_path = path.partition("::")
    obj = model.get_submodule(mod_path) if mod_path else model
    for name in filter(None, attr_path.split("::")):
        obj = vars(obj)[name]
    return obj


def _install_param(obj: nn.Module, k: str, t: torch.Tensor, pattrs: dict) -> None:
    """Put ``t`` into parameter ``k``, keeping the Parameter object the model
    was built with (other references to it see the new data)."""
    new = nn.Parameter(t, requires_grad=False)
    old = obj._parameters.get(k)
    attrs = dict(vars(old)) if old is not None else {}
    attrs.update(pattrs)
    if old is not None and type(old) is nn.Parameter:
        try:
            torch.utils.swap_tensors(old, new)
            old.__dict__.clear()
            old.__dict__.update(attrs)
            return
        except Exception:  # noqa: BLE001, S110 - extra references: install a new object
            pass
    new.__dict__.update(attrs)
    vars(obj).pop(k, None)
    obj._buffers.pop(k, None)
    obj._parameters[k] = new


def apply_export(model: nn.Module, export: dict, device: torch.device) -> int:
    """Point ``model`` (built on meta) at the daemon's tensors. Returns bytes mapped."""
    storages = {}
    for key, share in export["storages"].items():
        args = list(share)
        args[0] = device.index  # the daemon's index for this GPU may differ
        storages[key] = torch.UntypedStorage._new_shared_cuda(*args)

    def make(meta: dict) -> torch.Tensor:
        if "value" in meta:
            return meta["value"]
        if meta.get("empty"):
            return torch.empty_strided(
                meta["shape"], meta["stride"], dtype=meta["dtype"], device=device
            )
        t = torch.empty(0, dtype=meta["dtype"], device=device)
        t.set_(storages[meta["storage"]], meta["offset"], meta["shape"], meta["stride"])
        return t

    for path, item in export["plan"].items():
        obj = _resolve(model, path)
        is_module = isinstance(obj, nn.Module)
        d = vars(obj)
        for kind, k in item["delete"]:
            if kind == "param":
                obj._parameters.pop(k, None)
            elif kind == "buffer":
                obj._buffers.pop(k, None)
            else:
                d.pop(k, None)
        for k, meta in item["params"].items():
            _install_param(obj, k, make(meta), meta["pattrs"])
        for k, meta in item["buffers"].items():
            obj._buffers[k] = make(meta)
        for k, packed in item["attrs"].items():
            if is_module and (
                k in obj._parameters or k in obj._buffers or k in obj._modules
            ):
                raise RuntimeError(
                    f"{path}.{k}: attribute shadows a parameter, buffer or submodule"
                )
            d[k] = _unpack(packed, make)
    return sum(s.nbytes() for s in storages.values())


def _leftover_meta(model: nn.Module) -> list[str]:
    left = []

    def check(name, v, depth=0):
        if isinstance(v, torch.Tensor):
            if v.is_meta:
                left.append(name)
        elif depth < 4 and isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                check(f"{name}[{i}]", x, depth + 1)
        elif depth < 4 and isinstance(v, dict):
            for k, x in v.items():
                check(f"{name}[{k!r}]", x, depth + 1)

    for path, obj, is_module in _walk_objects(model, _DIFF_DEPTH, _DIFF_PACKAGES):
        if is_module:
            for k, p in obj._parameters.items():
                check(f"{path}.{k}", p)
            for k, b in obj._buffers.items():
                check(f"{path}.{k}", b)
        for k, v in _attrs(obj, is_module).items():
            check(f"{path}.{k}", v)
    return left


@contextlib.contextmanager
def params_on_meta():
    """Build a model with every nn.Parameter on the meta device; buffers and
    other tensors are built normally. The registered Parameter stays the
    object the constructor holds (its data is swapped to meta), so attributes
    set after registration (set_weight_attrs) land on it."""
    orig = nn.Module.register_parameter

    def register_parameter(self, name, param):
        orig(self, name, param)
        p = self._parameters.get(name)
        if p is None or p.is_meta:
            return
        meta = nn.Parameter(p.detach().to("meta"), requires_grad=p.requires_grad)
        attrs = dict(vars(p))
        if type(p) is nn.Parameter:
            try:
                torch.utils.swap_tensors(p, meta)
                p.__dict__.clear()
                p.__dict__.update(attrs)
                return
            except Exception:  # noqa: BLE001, S110 - extra references: replace instead
                pass
        meta.__dict__.update(attrs)
        self._parameters[name] = meta

    nn.Module.register_parameter = register_parameter
    try:
        yield
    finally:
        nn.Module.register_parameter = orig


@contextlib.contextmanager
def _skip_prepare_finalize(model: nn.Module):
    """The daemon never serves: leave MoE comm/kernel setup to the client."""
    from atom.model_ops.moe import FusedMoEMethodBase

    patched = []
    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, FusedMoEMethodBase) and "init_prepare_finalize" not in vars(
            qm
        ):
            qm.init_prepare_finalize = lambda layer: None
            patched.append(qm)
    try:
        yield
    finally:
        for qm in patched:
            vars(qm).pop("init_prepare_finalize", None)


def _run_prepare_finalize(model: nn.Module) -> int:
    from atom.model_ops.moe import FusedMoEMethodBase

    n = 0
    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, FusedMoEMethodBase):
            qm.init_prepare_finalize(mod)
            n += 1
    return n


def dump_state_digest(model: nn.Module, rank_name: str, prefix: str) -> None:
    """Debug oracle: write every tensor's content digest and every plain
    attribute to ``{prefix}.{rank_name}.json``, to compare a disk load with
    a cache hit."""
    t0 = time.perf_counter()
    out = {}
    for path, obj, is_module in _walk_objects(model, _DIFF_DEPTH, _DIFF_PACKAGES):
        if is_module:
            for k, p in obj._parameters.items():
                if p is not None:
                    out[f"{path}.{k} [param]"] = [
                        list(p.shape),
                        str(p.dtype),
                        _digest(p),
                    ]
            for k, b in obj._buffers.items():
                if b is not None:
                    out[f"{path}.{k} [buffer]"] = [
                        list(b.shape),
                        str(b.dtype),
                        _digest(b),
                    ]
        for k, v in _attrs(obj, is_module).items():
            fp = _fp(v, content=False)
            if isinstance(v, torch.Tensor):
                fp = (*fp, _digest(v))
            out[f"{path}.{k}"] = str(fp)
    path = f"{prefix}.{rank_name}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=0, sort_keys=True)
    logger.info(
        f"{_TAG} {rank_name}: state digest ({len(out)} entries) -> {path} in {time.perf_counter() - t0:.1f}s"
    )


# ---------------------------------------------------------------------------
# Unix socket transport, daemon lock
# ---------------------------------------------------------------------------


def socket_dir() -> str:
    d = envs.ATOM_WEIGHT_CACHE_DIR or f"/dev/shm/atom_weight_cache_{os.getuid()}"
    os.makedirs(d, mode=0o700, exist_ok=True)
    st = os.lstat(d)
    if stat.S_ISLNK(st.st_mode) or st.st_uid != os.getuid():
        raise PermissionError(f"{d} is a symlink or not owned by uid {os.getuid()}")
    os.chmod(d, 0o700)
    return d


def gpu_identity(device: torch.device) -> str:
    p = torch.cuda.get_device_properties(device)
    pci = f"{getattr(p, 'pci_domain_id', 0):04x}:{getattr(p, 'pci_bus_id', 0):02x}:{getattr(p, 'pci_device_id', 0):02x}"
    return f"{getattr(p, 'uuid', '')}|{pci}"


def socket_path(device: torch.device) -> str:
    gid = hashlib.sha256(gpu_identity(device).encode()).hexdigest()[:16]
    return os.path.join(socket_dir(), f"gpu_{gid}.sock")


def _send(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("<Q", len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("weight cache peer closed the connection")
        buf += chunk
    return bytes(buf)


def _recv(sock: socket.socket) -> Any:
    (n,) = struct.unpack("<Q", _recv_exact(sock, 8))
    return pickle.loads(_recv_exact(sock, n))


def _check_peer(sock: socket.socket) -> None:
    cred = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    _, uid, _ = struct.unpack("3i", cred)
    if uid != os.getuid():
        raise PermissionError(f"weight cache peer uid {uid} != {os.getuid()}")


def daemon_alive(path: str) -> bool:
    """Whether a daemon (loading or serving) holds this GPU's lock."""
    try:
        fd = os.open(path + ".lock", os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EACCES):
                return True
            raise
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


class _DaemonLock:
    """Held for the daemon process's lifetime, from before loading starts."""

    def __init__(self, path: str, **info):
        self.path = path
        self.fd = os.open(path + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.monotonic() + 10
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.monotonic() > deadline:  # clients probe with brief shared locks
                    os.close(self.fd)
                    raise RuntimeError(
                        f"another weight cache daemon holds this GPU ({path}.lock)"
                    ) from None
                time.sleep(0.2)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)  # stale socket from a dead daemon
        # Read by `status` and by servers that stop this daemon.
        self.info = {
            "pid": os.getpid(),
            # The launch's main process: SIGTERM there stops the whole daemon.
            "daemon_pid": int(os.environ.get(_DAEMON_PID_ENV) or 0) or None,
            "log": os.environ.get(_DAEMON_LOG_ENV),
            "started": time.time(),
            **info,
        }
        self.set_state("loading")

    def set_state(self, state: str, **info) -> None:
        self.info.update(info, state=state, since=time.time())
        tmp = f"{self.path}.state.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(self.info, f)
        os.replace(tmp, self.path + ".state")

    def release(self) -> None:
        for p in (self.path, self.path + ".state"):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(p)
        with contextlib.suppress(OSError):
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)


class _Server:
    def __init__(
        self,
        path: str,
        payload: dict,
        views: dict,
        label: str,
        idle_timeout_s: int = 0,
        on_idle=None,
    ):
        self.path, self.payload, self.label = path, payload, label
        self.views = views
        t0 = time.perf_counter()
        self.digests = {k: _digest(v) for k, v in views.items()}
        logger.info(
            f"{_TAG} {label}: digested {len(views)} shared storages in {time.perf_counter() - t0:.2f}s"
        )
        self.poisoned: list[int] = []
        self._verify_lock = threading.Lock()
        # Connected clients: a server keeps its connection for its lifetime.
        self.clients = 0
        self.idle_since = time.monotonic()
        self._clients_lock = threading.Lock()
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(path)
        os.chmod(path, 0o600)
        self.sock.listen(64)
        threading.Thread(
            target=self._accept, name="weight-cache-server", daemon=True
        ).start()
        if idle_timeout_s > 0 and on_idle is not None:
            threading.Thread(
                target=self._idle_watch,
                args=(idle_timeout_s, on_idle),
                name="weight-cache-idle",
                daemon=True,
            ).start()

    def _idle_watch(self, timeout_s: int, on_idle) -> None:
        while True:
            time.sleep(min(30.0, timeout_s / 4))
            with self._clients_lock:
                idle_for = (
                    time.monotonic() - self.idle_since if self.clients == 0 else 0.0
                )
            if idle_for >= timeout_s:
                logger.info(
                    f"{_TAG} {self.label}: no server connected for {idle_for:.0f}s "
                    f"(ATOM_PRELOAD_IDLE_TIMEOUT_S={timeout_s}); stopping the daemon"
                )
                on_idle()
                return

    def _client(self, delta: int) -> None:
        with self._clients_lock:
            self.clients += delta
            if self.clients == 0:
                self.idle_since = time.monotonic()

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.close()

    def _verify(self) -> list[int]:
        """Storages whose bytes changed since export (a client wrote into them)."""
        with self._verify_lock:
            if not self.poisoned:
                self.poisoned = [
                    k for k, v in self.views.items() if _digest(v) != self.digests[k]
                ]
            return self.poisoned

    def _accept(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return  # closed
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket):
        peer = "?"
        try:
            _check_peer(conn)
            req = _recv(conn)
            peer = req.get("who", "?")
            if req.get("cmd") == "ping":
                with self._clients_lock:
                    clients = self.clients
                    idle_s = time.monotonic() - self.idle_since if clients == 0 else 0.0
                _send(
                    conn,
                    {
                        "summary": self.payload["summary"],
                        "poisoned": len(self.poisoned),
                        "clients": clients,
                        "idle_s": idle_s,
                    },
                )
                return
            if req.get("cmd") != "get":
                _send(conn, {"error": f"unknown command {req.get('cmd')!r}"})
                return
            changed = self._verify()
            if changed:
                logger.error(
                    f"{_TAG} {self.label}: {len(changed)} shared storages changed since export "
                    "(something wrote into the cached weights); refusing to serve them"
                )
                _send(
                    conn,
                    {
                        "error": f"{len(changed)} of the daemon's shared storages changed since export; restart the daemon"
                    },
                )
                return
            self._client(+1)
            try:
                _send(conn, self.payload)
                logger.info(f"{_TAG} {self.label}: served {peer}")
                # The connection stays open while the client lives, so each side
                # sees the other go away.
                with contextlib.suppress(OSError):
                    while conn.recv(1 << 12):
                        pass
            finally:
                self._client(-1)
            logger.info(f"{_TAG} {self.label}: {peer} disconnected")
        except Exception as e:  # noqa: BLE001 - one bad client must not stop serving
            logger.warning(f"{_TAG} {self.label}: request from {peer} failed: {e}")
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Daemon management: ATOM_PRELOAD and the status/stop tool
# ---------------------------------------------------------------------------

# Starts the daemon in a new session and exits at once, so the daemon is no
# server's descendant: a server's shutdown kills its whole process tree, and a
# terminal's Ctrl-C signals its whole process group.
_DETACH = (
    "import subprocess, sys\n"
    "with open(sys.argv[1], 'ab') as log:\n"
    "    p = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)\n"
    "print(p.pid)\n"
)


def _initial_environ() -> dict[str, str]:
    """This process's environment as it was at exec, before imports changed it."""
    try:
        with open("/proc/self/environ", "rb") as f:
            raw = f.read()
    except OSError:
        return dict(os.environ)
    env = {}
    for item in raw.split(b"\0"):
        k, sep, v = item.partition(b"=")
        if sep and k:
            env[os.fsdecode(k)] = os.fsdecode(v)
    return env


def _proc_stat(pid: int) -> list[bytes] | None:
    """Fields of /proc/<pid>/stat after the command name (state, ppid, ...)."""
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as f:
            return f.read().rsplit(b")", 1)[1].split()
    except (OSError, IndexError, ValueError):
        return None


def _pid_running(pid) -> bool:
    st = _proc_stat(pid) if pid else None
    return bool(st) and st[0] not in (b"Z", b"X")


def _ancestors(pid: int, depth: int = 8) -> list[int]:
    out = []
    for _ in range(depth):
        st = _proc_stat(pid)
        if not st:
            break
        pid = int(st[1])
        if pid <= 1:
            break
        out.append(pid)
    return out


def _remove(*paths: str) -> None:
    for p in paths:
        with contextlib.suppress(OSError):
            os.unlink(p)


def record_launch() -> None:
    """Called first thing by the API server. A daemon tells its runners its
    main process (stopping that stops the daemon). An ATOM_PRELOAD server
    saves its command line and environment, to start a daemon with them."""
    if envs.ATOM_WEIGHT_CACHE == "daemon":
        os.environ[_DAEMON_PID_ENV] = str(os.getpid())
        return
    if not envs.ATOM_PRELOAD:
        return
    d = socket_dir()
    for old in glob.glob(os.path.join(d, "launch_*.json")):
        with contextlib.suppress(ValueError, IndexError):
            if not _pid_running(int(os.path.basename(old).split("_")[1])):
                _remove(old, old + ".daemon")
    path = os.path.join(d, f"launch_{os.getpid()}_{time.time_ns()}.json")
    launch = {
        "argv": [sys.executable, *sys.orig_argv[1:]],
        "cwd": os.getcwd(),
        "env": _initial_environ(),
    }
    fd = os.open(
        path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
    )  # holds the environment
    with os.fdopen(fd, "w") as f:
        json.dump(launch, f)
    os.environ[_LAUNCH_ENV] = path
    atexit.register(_remove, path, path + ".daemon")


def _spawn_daemon() -> dict:
    """Start a daemon with this server's command line and environment. Every
    rank calls this; the first one starts it, the others get its pid and log."""
    launch_path = os.environ.get(_LAUNCH_ENV)
    if not launch_path:
        raise _Miss(
            "ATOM_PRELOAD starts the daemon with the API server's command line, "
            "but this engine was not started by the API server"
        )
    marker = launch_path + ".daemon"
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            with contextlib.suppress(OSError, ValueError):
                with open(marker) as f:
                    info = json.load(f)
                if "error" in info:
                    raise _Miss(info["error"])
                return info
            time.sleep(0.2)
        raise _Miss("the rank that starts the daemon did not report it")
    try:
        with open(launch_path) as f:
            launch = json.load(f)
        env = {
            k: v
            for k, v in launch["env"].items()
            if k not in ("ATOM_PRELOAD", _LAUNCH_ENV)
        }
        log = os.path.join(
            socket_dir(), f"daemon_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}.log"
        )
        env.update({"ATOM_WEIGHT_CACHE": "daemon", _DAEMON_LOG_ENV: log})
        out = subprocess.run(
            [sys.executable, "-c", _DETACH, log, *launch["argv"]],
            cwd=launch["cwd"],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            start_new_session=True,
            timeout=120,
            check=False,
        )
        if out.returncode != 0:
            raise RuntimeError(
                f"launcher exited with {out.returncode}: {out.stderr.strip()[-2000:]}"
            )
        info = {"pid": int(out.stdout.strip()), "log": log}
    except Exception as e:  # noqa: BLE001 - the other ranks wait for the marker
        info = {"error": f"could not start a weight cache daemon: {e!r}"}
    with os.fdopen(fd, "w") as f:
        json.dump(info, f)
    if "error" in info:
        raise _Miss(info["error"])
    return info


def _wait_started(path: str, started: dict) -> None:
    """Wait until the daemon this launch started holds this GPU."""
    t0 = time.monotonic()
    while not daemon_alive(path):
        if not _pid_running(started["pid"]):
            raise _Miss(
                f"the weight cache daemon (pid {started['pid']}) exited while starting; see {started['log']}"
            )
        if time.monotonic() - t0 > envs.ATOM_WEIGHT_CACHE_WAIT_S:
            raise _Miss(
                f"the weight cache daemon (pid {started['pid']}) did not take this GPU within "
                f"{envs.ATOM_WEIGHT_CACHE_WAIT_S:.0f}s; see {started['log']}"
            )
        time.sleep(0.5)


def daemon_state(path: str) -> dict | None:
    try:
        with open(path + ".state") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _ping(path: str, timeout: float = 10) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(path)
        _check_peer(sock)
        _send(sock, {"cmd": "ping", "protocol": PROTOCOL_VERSION})
        return _recv(sock)


def stop_daemon(path: str, timeout_s: float = 180) -> int | None:
    """Stop the daemon holding this GPU (SIGTERM to its main process) and wait
    until it lets go. Returns that pid; None when no daemon holds the GPU."""
    deadline = time.monotonic() + 5
    while True:
        if not daemon_alive(path):
            return None
        state = daemon_state(path) or {}
        if state.get("pid") or time.monotonic() > deadline:
            break
        time.sleep(0.2)  # the lock is taken just before the state is written
    runner, main = state.get("pid"), state.get("daemon_pid")
    if not _pid_running(runner):
        raise RuntimeError(
            f"the daemon holding {path}.lock (pid {runner}) does not run in this PID "
            "namespace (another container?); stop it where it runs"
        )
    if not main or main not in _ancestors(runner):
        raise RuntimeError(
            f"cannot find the main process of the daemon holding {path}.lock (runner pid {runner})"
        )
    with contextlib.suppress(ProcessLookupError):
        os.kill(main, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    while daemon_alive(path):
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"the weight cache daemon (pid {main}) still holds {path}.lock {timeout_s:.0f}s after SIGTERM"
            )
        time.sleep(0.5)
    return main


def _in_use_by_others(path: str) -> str | None:
    """Why this GPU's daemon must not be replaced (None: it may be)."""
    others, deadline = 0, time.monotonic() + 3
    while True:  # this server has just closed its own connection
        try:
            others = _ping(path).get("clients", 0)
        except OSError:
            return None  # still loading, or gone
        except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
            return f"could not ask the daemon whether servers use it: {e!r}"
        if others == 0 or time.monotonic() > deadline:
            break
        time.sleep(0.1)
    if others:
        return (
            f"{others} other running server(s) use the daemon's weights on this GPU; they keep "
            "them in memory, so a new daemon would not fit. Stop them first"
        )
    return None


# ---------------------------------------------------------------------------
# Daemon runner
# ---------------------------------------------------------------------------


def _daemon_runner_class():
    from atom.model_engine.model_runner import ModelRunner
    from atom.model_loader.loader import load_model

    class WeightCacheDaemonRunner(ModelRunner):
        """Loads its shard like a serving runner and serves it to clients over
        the socket; sets up nothing for serving requests (weights_only), so
        its EngineCore skips the KV cache, warmup and graphs and reports ready."""

        weights_only = True

        def _build_and_load_model(self, model_class):
            config = self.config
            t0 = time.perf_counter()
            label = f"rank {self.rank}"
            path = socket_path(self.device)
            self._wc_lock = _DaemonLock(
                path,
                rank=self.rank,
                model=config.model,
                gpu=gpu_identity(self.device),
                idle_timeout_s=envs.ATOM_PRELOAD_IDLE_TIMEOUT_S,
            )
            key = compute_key(config, self.rank, self.device)
            recorder = self._input_recorder  # started in ModelRunner.__init__
            recorder.enter_construct(config)
            self.model = model_class(config)
            fused_fn = getattr(self.model, "load_fused_expert_weights", None)
            torch.set_default_device(None)
            built = structure(self.model)
            before = snapshot(self.model)
            kernels: dict = {}
            with _CodeTracer() as tracer, _record_jit(kernels):
                recorder.enter_load(self.model)
                with _skip_prepare_finalize(self.model):
                    load_model(
                        self.model,
                        config.model,
                        config.hf_config,
                        config.load_dummy,
                        load_fused_expert_weights_fn=fused_fn,
                    )
            recorder.finish_load(tracer.codes)  # before anything else reads the model
            reads = recorder.key_reads(config.model)
            if envs.ATOM_WEIGHT_CACHE_READS:
                recorder.dump(f"{envs.ATOM_WEIGHT_CACHE_READS}.{self.rank_name}.json")
            after = snapshot(self.model)
            try:
                codes = code_fingerprint(tracer.codes, kernels.values())
            except _SourceChanged as e:
                raise RuntimeError(
                    f"{_TAG} {label}: {e} changed after this daemon started, so the weights it "
                    "loaded may not match the file; restart the daemon"
                ) from None
            export = build_export(self.model, before, after)
            views = export.pop("views")
            self._release_serving_state(export)
            summary = {
                "pid": os.getpid(),
                "daemon_pid": self._wc_lock.info.get("daemon_pid"),
                "rank": self.rank,
                "model": config.model,
                "gpu": gpu_identity(self.device),
                "bytes": export["bytes"],
                "tensors": sum(
                    len(i["params"]) + len(i["buffers"]) + len(i["attrs"])
                    for i in export["plan"].values()
                ),
                "functions": len(codes),
                "unshippable": export["unshippable"][:20],
                "loaded_at": time.time(),
            }
            payload = {
                "protocol": PROTOCOL_VERSION,
                "key": key,
                "reads": reads,
                "code": codes,
                "structure": _exported_structure(built, export),
                "export": export,
                "summary": summary,
            }
            logger.info(
                f"{_TAG} {label}: load read {len(reads['env'])} env vars, {len(reads['config'])} config "
                f"fields, {len(reads['attrs'])} construction attributes, {len(reads['classattrs'])} class "
                f"attributes, {len(reads['globals'])} globals, {len(reads['files'])} files"
            )
            self._wc_server = _Server(
                path,
                payload,
                views,
                label,
                idle_timeout_s=self._wc_lock.info["idle_timeout_s"],
                on_idle=self._stop_daemon_launch,
            )
            self._wc_lock.set_state("serving", bytes=export["bytes"])
            logger.info(
                f"{_TAG} {label}: serving {summary['tensors']} entries ({export['bytes'] / (1 << 30):.2f} GiB, "
                f"{len(codes)} code entries) at {path} after {time.perf_counter() - t0:.1f}s"
            )
            if export["unshippable"]:
                logger.warning(
                    f"{_TAG} {label}: loading created state that cannot be shared, so clients will "
                    f"refuse this cache: {export['unshippable'][:10]}"
                )

        def _stop_daemon_launch(self):
            """Idle timeout: shut the whole daemon down through its main
            process, as SIGTERM to the launch would."""
            main_pid = self._wc_lock.info.get("daemon_pid")
            if main_pid:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(main_pid, signal.SIGTERM)
            else:
                self.exit()  # main process unknown: at least stop serving

        def exit(self):
            """Stop serving: called by the engine's shutdown (or on SIGTERM).
            Clients that mapped the weights keep them until they exit."""
            if not getattr(self, "still_running", False):
                return
            self.still_running = False
            server = getattr(self, "_wc_server", None)
            if server is not None:
                server.close()
            lock = getattr(self, "_wc_lock", None)
            if lock is not None:
                lock.release()
            logger.info(f"{_TAG} rank {self.rank}: stopped serving")

        def _release_serving_state(self, export: dict) -> None:
            """Everything but the weights is a client's KV cache lost: drop the
            communicators and runner buffers that only serving needs."""
            from aiter import destroy_dist_env

            def used() -> float:
                free, total = torch.cuda.mem_get_info(self.device)
                return (total - free) / (1 << 30)

            before = used()
            # Tensors the clients build for themselves (unchanged buffers such
            # as RoPE caches) need no device memory here.
            shared = export["storages"]
            for _, obj, is_module in _walk_objects(
                self.model, _DIFF_DEPTH, _DIFF_PACKAGES
            ):
                if is_module:
                    for k, b in list(obj._buffers.items()):
                        if (
                            b is not None
                            and b.is_cuda
                            and b.untyped_storage().data_ptr() not in shared
                        ):
                            obj._buffers[k] = b.to("meta")
                for k, v in list(_attrs(obj, is_module).items()):
                    if (
                        isinstance(v, torch.Tensor)
                        and not isinstance(v, nn.Parameter)
                        and v.is_cuda
                        and v.untyped_storage().data_ptr() not in shared
                    ):
                        vars(obj)[k] = v.to("meta")
            with contextlib.suppress(Exception):
                destroy_dist_env()
            if torch.distributed.is_initialized():
                with contextlib.suppress(Exception):
                    torch.distributed.destroy_process_group()
            for name in (
                "tokenID_processor",
                "sampler",
                "_dynamic_chunking",
                "vision_embeddings",
            ):
                if hasattr(self, name):
                    setattr(self, name, None)
            gc.collect()
            torch.cuda.empty_cache()
            after = used()
            weight_bytes = export["bytes"]
            logger.info(
                f"{_TAG} rank {self.rank}: device memory {after:.2f} GiB after releasing "
                f"{before - after:.2f} GiB of serving state; weights {weight_bytes / (1 << 30):.2f} GiB, "
                f"other allocated {(torch.cuda.memory_allocated() - weight_bytes) / (1 << 30):.2f} GiB, "
                f"allocator reserved-but-free {(torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / (1 << 30):.2f} GiB"
            )

    return WeightCacheDaemonRunner


def __getattr__(name: str):
    # Resolved lazily: model_runner imports this module's client hook.
    if name == "WeightCacheDaemonRunner":
        cls = _daemon_runner_class()
        globals()[name] = cls
        return cls
    raise AttributeError(name)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _agree_group():
    """The CPU group of every runner in this launch (they all load together)."""
    try:
        from aiter.dist.parallel_state import get_world_group

        return get_world_group().cpu_group
    except Exception:  # noqa: BLE001 - single process
        return None


def _fetch(path: str, label: str) -> tuple[socket.socket, dict]:
    """Wait for a loading daemon, then get its payload (raises _Miss)."""
    t0 = last = time.monotonic()
    while not os.path.exists(path):
        if not daemon_alive(path):
            raise _Miss("the daemon exited while loading")
        waited = time.monotonic() - t0
        if waited > envs.ATOM_WEIGHT_CACHE_WAIT_S:
            raise _Miss(f"the daemon is still loading after {waited:.0f}s")
        if time.monotonic() - last > 30:
            logger.info(
                f"{_TAG} {label}: waiting for the daemon to finish loading ({waited:.0f}s)"
            )
            last = time.monotonic()
        time.sleep(1)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(30)
        sock.connect(path)
        _check_peer(sock)
        _send(
            sock,
            {
                "cmd": "get",
                "protocol": PROTOCOL_VERSION,
                "who": f"pid {os.getpid()} {label}",
            },
        )
        sock.settimeout(600)  # the daemon digests its storages first
        payload = _recv(sock)
        sock.settimeout(None)
    except Exception as e:  # noqa: BLE001
        sock.close()
        raise _Miss(f"daemon at {path} unreachable: {e}") from None
    if "error" in payload:
        sock.close()
        raise _Miss(payload["error"])
    return sock, payload


def _check_payload(payload: dict, runner) -> list[str]:
    """Why this payload cannot be used here (empty: it can)."""
    if payload.get("protocol") != PROTOCOL_VERSION:
        return [f"protocol {payload.get('protocol')} != {PROTOCOL_VERSION}"]
    from atom.model_loader.input_recorder import diff_static_reads

    diffs = _diff_key(
        payload["key"], compute_key(runner.config, runner.rank, runner.device)
    )
    if diffs:
        return ["checkpoint, model, rank, GPU or library versions differ:"] + diffs
    diffs = diff_static_reads(payload["reads"], runner.config)
    if diffs:
        return ["settings the daemon's load read differ:"] + diffs
    try:
        changed = verify_code(payload["code"])
    except _SourceChanged as e:
        return [f"{e} changed after this server started; restart it"]
    if changed:
        return [
            f"{len(changed)} function(s) or kernel(s) that ran while the daemon loaded have changed:"
        ] + changed
    if payload["export"]["unshippable"]:
        return ["the daemon's load created state that cannot be shared:"] + payload[
            "export"
        ]["unshippable"]
    return []


_STOP_HINT = "python -m atom.model_loader.weight_cache stop"


def _fail(
    label: str, reasons: list[str], pid=None, preload: bool = False
) -> RuntimeError:
    detail = (
        "\n  ".join(reasons[:25])
        if reasons
        else "another rank could not use it; see that rank's log"
    )
    if preload:
        hint = (
            "ATOM_PRELOAD could not set up a weight cache daemon for this launch. Fix the "
            f"cause above, or stop the daemons ({_STOP_HINT}) and launch without ATOM_PRELOAD "
            "to load from disk."
        )
    else:
        hint = (
            "The daemon keeps its copy on these GPUs, so this server will not load a second "
            "one from disk. Restart the daemon with this code and configuration (ATOM_PRELOAD=1 "
            f"has servers do that), or stop it ({_STOP_HINT}) to load from disk."
        )
    return RuntimeError(
        f"{_TAG} {label}: cannot use the weight cache daemon{f' (pid {pid})' if pid else ''}: {detail}\n{hint}"
    )


def _probe(path: str, label: str, runner, started: dict | None):
    """This rank's daemon: (connection, payload, why it cannot be used, whether
    there is one). Waits for a daemon that is starting or loading."""
    sock, payload = None, None
    try:
        if started is not None:
            _wait_started(path, started)
        if not daemon_alive(path):
            return None, None, [], False
        sock, payload = _fetch(path, label)
        reasons = _check_payload(payload, runner)
    except _Miss as e:
        reasons = [str(e)]
    except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
        reasons = [f"weight cache error: {e!r}"]
    if reasons and sock is not None:
        sock.close()
        sock = None
    return sock, payload, reasons, True


def try_load_from_cache(runner, model_class) -> bool:
    """Build ``runner.model`` from a daemon's weights; False: load from disk.

    Client mode uses the daemon that is there: False when none of this
    launch's GPUs has one, an error when one is there but cannot be used.
    ATOM_PRELOAD manages the daemon: it starts one when there is none, and
    when the one there cannot be used, stops it and starts a new one (once).
    Every rank of the launch takes the same decisions."""
    from atom.model_loader.input_recorder import diff_attr_reads

    t0 = time.perf_counter()
    label = f"rank {runner.rank}"
    group = _agree_group()
    auto = envs.ATOM_PRELOAD
    path = socket_path(runner.device)
    started = None  # the daemon this launch started
    model, t_meta = None, 0.0
    while True:
        sock, payload, reasons, present = _probe(path, label, runner, started)
        status = 0 if not present else (2 if reasons else 1)
        lo, hi = group_min(status, group), group_max(status, group)
        summary = (payload or {}).get("summary", {})
        pid = summary.get("daemon_pid") or summary.get("pid")  # as `status` shows it
        if hi == 0 and not auto:
            logger.info(
                f"{_TAG} {label}: no weight cache daemon on this launch's GPUs; loading from disk"
            )
            return False
        if lo == hi == 1:
            if model is None:  # built once: a model registers its layers globally
                t_build = time.perf_counter()
                with params_on_meta():
                    model = model_class(runner.config)
                torch.set_default_device(None)
                t_meta = time.perf_counter() - t_build
            diffs = _diff_structure(payload["structure"], structure(model))
            diffs += diff_attr_reads(payload["reads"], model)
            if group_agree(not diffs, group):
                break
            reasons = (
                (["model construction differs where it matters to loading:"] + diffs)
                if diffs
                else []
            )
        elif lo == 0 and hi > 0 and not reasons:
            reasons = (
                ["a daemon serves only some of this launch's GPUs"]
                if present
                else [
                    "no daemon serves this GPU, but other GPUs of this launch have one"
                ]
            )
        if sock is not None:
            sock.close()
            sock = None
        if not auto:
            raise _fail(label, reasons, pid)
        if started is not None:
            # The daemon this launch just started does not fit it either.
            with contextlib.suppress(Exception):
                stop_daemon(path)
            raise _fail(
                label,
                (
                    reasons
                    + [
                        f"(this launch started that daemon and has stopped it again; its log: {started['log']})"
                    ]
                    if reasons
                    else []
                ),
                pid,
                preload=True,
            )
        if hi > 0:
            err = _in_use_by_others(path)
            if not group_agree(err is None, group):
                why = (
                    ["it would need a restart, since it cannot be used here:"] + reasons
                    if reasons
                    else []
                )
                raise _fail(label, ([err] if err else []) + why, pid, preload=True)
            if reasons:
                logger.warning(
                    f"{_TAG} {label}: restarting the weight cache daemon{f' (pid {pid})' if pid else ''}, "
                    "it cannot be used here:\n  " + "\n  ".join(reasons[:25])
                )
            else:
                logger.info(
                    f"{_TAG} {label}: restarting the weight cache daemon (another rank cannot use its daemon)"
                )
            err = None
            try:
                stop_daemon(path)
            except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
                err = f"could not stop the daemon: {e}"
            if not group_agree(err is None, group):
                raise _fail(label, [err] if err else [], pid, preload=True)
        else:
            logger.info(
                f"{_TAG} {label}: no weight cache daemon on this launch's GPUs; starting one (ATOM_PRELOAD)"
            )
        err = None
        try:
            started = _spawn_daemon()
        except _Miss as e:
            err = str(e)
        if not group_agree(err is None, group):
            raise _fail(label, [err] if err else [], preload=True)
        logger.info(
            f"{_TAG} {label}: started weight cache daemon pid {started['pid']} (log {started['log']}); "
            "waiting for it to load"
        )

    t_map = time.perf_counter()
    error = None
    try:
        mapped = apply_export(model, payload["export"], runner.device)
        left = _leftover_meta(model)
        if left:
            error = f"{len(left)} tensors still on meta after mapping: {left[:8]}"
    except Exception as e:  # noqa: BLE001
        error = f"mapping the daemon's tensors failed: {e!r}"
    if not group_agree(error is None, group):
        sock.close()
        raise _fail(label, [error] if error else [], pid)
    torch.cuda.reset_peak_memory_stats()
    # Collective on EP setups; a failure here must not be swallowed.
    n_moe = _run_prepare_finalize(model)
    runner.model = model
    runner._weight_cache_conn = sock
    threading.Thread(
        target=_watch_daemon, args=(sock, label, path), daemon=True
    ).start()
    logger.info(
        f"{_TAG} {label}: cache hit, mapped {mapped / (1 << 30):.2f} GiB from daemon pid {pid} "
        f"in {time.perf_counter() - t0:.2f}s ({'started the daemon and waited for it, ' if started else ''}"
        f"fetch+checks {t_map - t0 - t_meta:.2f}s, meta build {t_meta:.2f}s, "
        f"map {time.perf_counter() - t_map:.2f}s, {n_moe} MoE prepare/finalize)"
    )
    return True


def _watch_daemon(sock: socket.socket, label: str, path: str) -> None:
    with contextlib.suppress(OSError):
        while sock.recv(1 << 12):
            pass
    # The mapping stays valid (the memory is freed when its last user exits),
    # so this server keeps working; only the next restart loses the cache.
    logger.warning(
        f"{_TAG} {label}: the weight cache daemon ({path}) exited; this server keeps "
        "its mapped weights, but its next restart needs a new daemon (ATOM_PRELOAD starts one)"
    )


# ---------------------------------------------------------------------------
# python -m atom.model_loader.weight_cache status|stop
# ---------------------------------------------------------------------------


def _running_daemons() -> dict:
    """{main pid: [(socket path, state), ...]} of this user's daemons here."""
    launches: dict = {}
    for lock in sorted(glob.glob(os.path.join(socket_dir(), "gpu_*.sock.lock"))):
        path = lock[: -len(".lock")]
        if daemon_alive(path):
            st = daemon_state(path) or {}
            launches.setdefault(
                st.get("daemon_pid") or f"? (runner {st.get('pid')})", []
            ).append((path, st))
    return launches


def _age(seconds: float) -> str:
    s = max(int(seconds), 0)
    return (
        f"{s // 3600}h{s % 3600 // 60:02d}m"
        if s >= 3600
        else f"{s // 60}m{s % 60:02d}s"
    )


def _status(launches: dict) -> None:
    now = time.time()
    for main_pid, gpus in launches.items():
        st0 = gpus[0][1]
        timeout = st0.get("idle_timeout_s")
        print(f"daemon pid {main_pid}: {st0.get('model', '?')}")
        print(
            f"  up {_age(now - st0.get('started', now))}, exits after "
            f"{f'{timeout}s idle' if timeout else 'no idle timeout'}; log {st0.get('log') or '-'}"
        )
        for path, st in sorted(gpus, key=lambda g: g[1].get("rank", 0)):
            line = f"  rank {st.get('rank', '?')}: {st.get('state', '?')}"
            if st.get("state") == "serving":
                try:
                    r = _ping(path, timeout=5)
                    line += f", {r['summary']['bytes'] / (1 << 30):.1f} GiB, {r.get('clients', '?')} server(s) connected"
                    if r.get("clients") == 0:
                        line += f", idle {_age(r.get('idle_s', 0))}"
                    if r.get("poisoned"):
                        line += ", WEIGHTS MODIFIED by a client (will refuse to serve)"
                except Exception as e:  # noqa: BLE001
                    line += f", not answering ({e})"
            else:
                line += f" for {_age(now - st.get('since', now))}"
            print(f"{line}; gpu {st.get('gpu', '?')}")


def _stop(launches: dict) -> int:
    rc = 0
    for main_pid, gpus in launches.items():
        servers = 0  # each server connects once per GPU
        for path, _ in gpus:
            with contextlib.suppress(Exception):
                servers = max(servers, _ping(path, timeout=5).get("clients", 0))
        try:
            for path, _ in gpus:
                stop_daemon(path)  # the first one stops the whole launch
        except Exception as e:  # noqa: BLE001
            print(f"daemon pid {main_pid}: {e}")
            rc = 1
            continue
        note = (
            f"; {servers} running server(s) keep its weights mapped until they exit"
            if servers
            else ""
        )
        print(f"stopped daemon pid {main_pid} ({len(gpus)} GPU(s)){note}")
    return rc


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m atom.model_loader.weight_cache",
        description="Show or stop this user's ATOM weight cache daemons on this machine.",
    )
    parser.add_argument("command", choices=("status", "stop"))
    args = parser.parse_args(argv)
    launches = _running_daemons()
    if not launches:
        print(f"no weight cache daemon running (in {socket_dir()})")
        return 0
    if args.command == "status":
        _status(launches)
        return 0
    return _stop(launches)


if __name__ == "__main__":
    sys.exit(main())
