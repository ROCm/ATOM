# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The cache key, apart from what loading read (recorder.py): identity
(checkpoint, model, rank, GPU, libraries), the code that ran during load,
and the model structure."""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib
import importlib.metadata
import os
import subprocess
import sys
import threading
import time

import torch
from torch import nn

from atom.model_loader.weight_cache.values import (
    _TAG,
    _fp,
    _import_qual,
    _json,
    logger,
)
from atom.model_loader.weight_iterator import _MAX_SAFETENSORS_HEADER_BYTES


class _SourceChanged(Exception):
    """A source file changed after this process started."""


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


_SMALL_FILE = 1 << 20


def file_identity(path: str) -> str:
    """A file as the key sees it: a small one (a config) by content, a large
    one (a weight shard) by size, mtime and inode, as rewriting it changes."""
    try:
        st = os.stat(path)
        if st.st_size > _SMALL_FILE:
            return f"{st.st_size}:{st.st_mtime_ns}:{st.st_ino}"
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError as e:
        return f"<unreadable {type(e).__name__}>"


def _checkpoint_digest(model_path: str) -> str:
    """Every file in the checkpoint directory by name and file_identity, with
    the safetensors headers (as bytes: parsing them takes ~1 s on V4-Pro)."""
    root = os.path.realpath(model_path)
    if not os.path.isdir(root):
        return f"path:{root}"
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            h.update(f"{os.path.relpath(path, root)}\0{file_identity(path)}\n".encode())
            if name.endswith(".safetensors"):
                with contextlib.suppress(OSError), open(path, "rb") as f:
                    n = int.from_bytes(f.read(8), "little")
                    h.update(f.read(min(n, _MAX_SAFETENSORS_HEADER_BYTES)))
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
        # Installed metadata only: whether a module is imported yet differs
        # between processes.
        version = None
        for dist in dists:
            with contextlib.suppress(Exception):
                version = importlib.metadata.version(dist)
                break
        out[name] = version or "absent"
    return out


def compute_key(config, rank: int, device: torch.device) -> dict:
    """Which checkpoint, model, drafter, shard, GPU and libraries."""
    import aiter

    hf = config.hf_config
    try:
        hf_dict = hf.to_dict()
    except Exception:  # noqa: BLE001
        hf_dict = vars(hf)
    from atom.model_loader.weight_utils import local_model_dir

    props = torch.cuda.get_device_properties(device)
    pc = config.parallel_config
    model_dir = local_model_dir(config.model)  # a hub id names a cache snapshot
    spec = config.speculative_config
    return {
        "model": os.path.realpath(model_dir),
        "checkpoint": _checkpoint_digest(model_dir),
        # The daemon holds this drafter's weights too.
        "draft": (
            {
                "method": spec.method,
                "checkpoint": spec.model
                and _checkpoint_digest(local_model_dir(spec.model)),
            }
            if spec
            else None
        ),
        "hf_config": hashlib.sha256(_json(hf_dict).encode()).hexdigest(),
        "rank": rank,
        # The shard a rank holds depends on the whole layout (EP spans DP).
        "parallel": {
            "tp": config.tensor_parallel_size,
            "tp_world": config.tp_world_size,
            "pcp": config.pcp_group_size,
            "dp": pc.data_parallel_size,
            "dp_rank": pc.data_parallel_rank,
            "dp_attention": bool(config.enable_dp_attention),
            "ep": bool(config.enable_expert_parallel),
        },
        "arch": getattr(props, "gcnArchName", ""),
        "versions": _versions(),
        "aiter": _git_state(
            os.path.dirname(os.path.dirname(os.path.realpath(aiter.__file__)))
        ),
    }


def _diff_key(daemon: dict, client: dict, prefix: str = "") -> list[str]:
    """Differences between two nested dicts (key, structure)."""
    diffs = []
    for k in sorted(set(daemon) | set(client)):
        a, b = daemon.get(k, "<missing>"), client.get(k, "<missing>")
        if isinstance(a, dict) and isinstance(b, dict):
            diffs += _diff_key(a, b, f"{prefix}{k}.")
        elif a != b:
            diffs.append(f"{prefix}{k}: daemon={str(a)[:120]} client={str(b)[:120]}")
    return diffs


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------

_OWN_DIR = os.path.dirname(os.path.realpath(__file__))
# Set by the API server for its engine's processes (preload.record_launch).
_LAUNCH_T0_ENV = "ATOM_WEIGHT_CACHE_LAUNCH_T0"


def _launch_start() -> float:
    """A time before this process imported its code: when its API server
    started (preload.record_launch), else when this process started."""
    with contextlib.suppress(TypeError, ValueError):
        return float(os.environ.get(_LAUNCH_T0_ENV))
    return _process_start()


def _process_start() -> float:
    """This process's start in wall-clock time, to the clock tick. (psutil's
    create_time adds the boot time, truncated to the second, so it can be up
    to a second early: a file edited just before the launch would count as
    edited after it.)"""
    with open("/proc/self/stat", "rb") as f:
        ticks = int(f.read().rsplit(b")", 1)[1].split()[19])
    since_boot = ticks / os.sysconf("SC_CLK_TCK")
    return time.time() - (time.clock_gettime(time.CLOCK_BOOTTIME) - since_boot)


def _package_roots() -> dict[str, str]:
    roots = {}
    for name in ("atom", "aiter"):
        with contextlib.suppress(Exception):
            roots[name] = os.path.dirname(
                os.path.realpath(importlib.import_module(name).__file__)
            )
    return roots


def _code_location(filename: str, roots: dict[str, str]) -> tuple[str, str] | None:
    """(package, path in it) of ATOM/AITER code outside this package."""
    fn = os.path.realpath(filename)
    if fn.startswith(_OWN_DIR + os.sep):
        return None
    for pkg, root in roots.items():
        if fn.startswith(root + os.sep):
            return pkg, os.path.relpath(fn, root)
    return None


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
            self._mon.set_events(self._tool, self._mon.events.PY_START)
            self._mon.restart_events()  # what an earlier trace disabled runs again
        else:
            logger.warning(
                f"{_TAG} no free sys.monitoring tool id; tracing with sys.settrace"
            )
            sys.settrace(self._on_trace)
            threading.settrace(self._on_trace)
        return self

    def __exit__(self, *exc):
        if self._tool is not None:
            self._mon.set_events(self._tool, 0)
            self._mon.register_callback(self._tool, self._mon.events.PY_START, None)
            self._mon.free_tool_id(self._tool)
        else:
            sys.settrace(None)
            threading.settrace(None)
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


def _kernel_key(jf) -> str:
    """A Triton kernel's cache key, or a hash of its source if it has none."""
    try:
        return str(jf.cache_key)
    except Exception:  # noqa: BLE001
        src = getattr(jf, "src", None)
        if not isinstance(src, str):
            return "<unkeyable>"
        return "src:" + hashlib.sha256(src.encode()).hexdigest()


def _jit_key(module: str, qual: str) -> str:
    try:
        obj = _import_qual(module, qual)
    except Exception as e:  # noqa: BLE001
        return f"<unresolved {type(e).__name__}>"
    while not hasattr(obj, "cache_key") and hasattr(obj, "fn"):
        obj = obj.fn  # autotuner / heuristics wrappers
    return _kernel_key(obj)


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


class _Outline(ast.NodeTransformer):
    """Function bodies left out: a module or class body that ran only defined
    its functions, and those that ran are keyed on their own."""

    def visit_FunctionDef(self, node):
        node.body = [ast.Pass()]
        return node

    visit_AsyncFunctionDef = visit_FunctionDef


class _SourceHasher:
    """AST hashes (docstrings stripped) of functions in the atom/aiter packages.

    A source file changed (ctime) after this launch started may not be what
    the process imported, so reading one raises _SourceChanged."""

    def __init__(self):
        self.roots = _package_roots()
        self.t0 = _launch_start()
        self._sources: dict[str, bytes | None] = {}
        self._trees: dict[str, ast.AST | None] = {}

    def source(self, pkg: str, rel: str) -> bytes | None:
        key = f"{pkg}/{rel}"
        if key not in self._sources:
            path = os.path.realpath(
                os.path.join(self.roots.get(pkg, "/nonexistent"), rel)
            )
            try:
                changed = os.stat(path).st_ctime > self.t0
                if not changed:
                    with open(path, "rb") as f:
                        self._sources[key] = f.read()
            except OSError:
                self._sources[key] = None
            else:
                if changed:
                    raise _SourceChanged(key)
        return self._sources[key]

    def file_hash(self, pkg: str, rel: str) -> str | None:
        src = self.source(pkg, rel)
        return None if src is None else hashlib.sha256(src).hexdigest()

    def _tree(self, pkg: str, rel: str):
        key = f"{pkg}/{rel}"
        if key not in self._trees:
            src = self.source(pkg, rel)
            self._trees[key] = (
                None
                if src is None
                else _StripDocstrings().visit(ast.parse(src.decode("utf-8")))
            )
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
        if not parts or all(
            isinstance(n, ast.ClassDef) and n.name == parts[-1] for n in nodes
        ):
            nodes = [_Outline().visit(copy.deepcopy(n)) for n in nodes]
        dumps = (
            ast.dump(n, annotate_fields=False, include_attributes=False) for n in nodes
        )
        return hashlib.sha256("\n".join(dumps).encode()).hexdigest()[:16]


def _outer_qualname(code) -> str:
    qual = getattr(code, "co_qualname", code.co_name).split(".<locals>")[0]
    parts = [p for p in qual.split(".") if not p.startswith("<")]
    return ".".join(parts) if parts else "<module>"


def code_fingerprint(codes, kernels) -> dict:
    """What ran during load: functions (AST hashes, plus the hash of each
    file, so a client with the same bytes skips parsing) and launched Triton
    kernels. (Module constants and bound functions the code used are keyed
    as globals by the recorder.)"""
    hasher = _SourceHasher()
    fns = set()
    for code in codes:
        where = _code_location(code.co_filename, hasher.roots)
        if where is not None:
            fns.add((*where, _outer_qualname(code)))
    entries = [
        ("fn", pkg, rel, qual, hasher.digest(pkg, rel, qual)) for pkg, rel, qual in fns
    ]
    for jf in kernels:
        fn = getattr(jf, "fn", None)
        module, qual = getattr(fn, "__module__", "?"), getattr(fn, "__qualname__", "?")
        entries.append(("jit", module, qual, "", _kernel_key(jf)))
    files = {f"{pkg}/{rel}": hasher.file_hash(pkg, rel) for pkg, rel, _ in fns}
    return {"entries": sorted(set(entries)), "files": files}


def verify_code(fingerprint: dict) -> list[str]:
    """Entries whose code differs in this process (raises _SourceChanged)."""
    hasher = _SourceHasher()
    changed = []
    for kind, a, b, c, digest in fingerprint["entries"]:
        if kind == "fn":
            same_file = hasher.file_hash(a, b) == fingerprint["files"].get(f"{a}/{b}")
            if not same_file and hasher.digest(a, b, c) != digest:
                changed.append(f"{a}/{b}:{c}")
        elif _jit_key(a, b) != digest:
            changed.append(f"Triton kernel {a}.{b}")
    return changed


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def structure(model: nn.Module) -> dict:
    """The module tree, module types, parameter shapes/dtypes and buffer
    shapes/dtypes, comparable between a real and a meta-built model."""
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


def exported_structure(built: dict, export: dict) -> dict:
    """``built`` with only the buffers the daemon exports or deletes: the
    client builds the others itself, e.g. caches sized by --max-model-len.
    Both sides filter with the daemon's export plan before comparing."""
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
