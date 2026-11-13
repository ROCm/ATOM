# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Record the inputs Python code reads while the daemon builds and loads its
model; what loading read becomes part of the cache key.

Recorded: environment variables and config fields that loading read,
construction-time attributes (instance and class) that loading read, globals
the loading code referenced, and files it opened. Phases: ``init`` (process
setup), ``construct`` and ``load`` (checkpoint load and
post-processing); recording stops there. Native code (C++ getenv, files opened
by C++) is invisible here.
"""

from __future__ import annotations

import builtins
import importlib
import io
import os
import sys
import threading
import types
from typing import Any

from atom.model_loader.weight_cache.key import (
    _code_location,
    _package_roots,
    file_identity,
)
from atom.model_loader.weight_cache.values import (
    _MODULE_INTERNALS,
    _attrs,
    _import_qual,
    _is_plain,
    _ref_name,
    _resolve,
    _walk_objects,
    key_fp,
)

_REC: InputRecorder | None = None
_ENV_ORIG = None
_OPEN_ORIG = None
_SKIP_FRAME_FILES = (
    "/os.py",
    "/_collections_abc.py",
    "/atom/utils/envs.py",
    "/weight_cache/recorder.py",
)
_BULK_THRESHOLD = 25  # distinct variables read from one line: a copy of the environment
# Environment reads that count for the cache key: made by these packages'
# code. Others (a progress bar reading COLUMNS) do not decide weights.
KEY_ENV_PREFIXES = ("atom/", "aiter/", "flydsl/", "triton/", "mori/")
_ROOTS_CACHE: list[str] | None = None
_REALPATHS: dict[str, str] = {}


def _roots() -> list[str]:
    global _ROOTS_CACHE
    if _ROOTS_CACHE is None:
        out = []
        for name in ("atom", "aiter", "torch"):
            mod = sys.modules.get(name)
            if mod is not None and getattr(mod, "__file__", None):
                out.append(
                    os.path.dirname(os.path.dirname(os.path.realpath(mod.__file__)))
                )
        _ROOTS_CACHE = out
    return _ROOTS_CACHE


def _where(frame) -> tuple[str, bool]:
    """(location of the first caller outside os/envs plumbing, whether it is a bulk copy)."""
    bulk = False
    while frame is not None:
        fn = frame.f_code.co_filename
        # os/_collections_abc are frozen modules ("<frozen os>") on 3.12.
        if fn.startswith("<frozen") or fn.endswith(_SKIP_FRAME_FILES):
            if frame.f_code.co_name in (
                "copy",
                "__iter__",
                "items",
                "values",
                "keys",
                "__repr__",
            ):
                bulk = True
            frame = frame.f_back
            continue
        path = _REALPATHS.get(fn)
        if path is None:
            path = os.path.realpath(fn)
            for root in _roots():
                if path.startswith(root + os.sep):
                    path = os.path.relpath(path, root)
                    break
            _REALPATHS[fn] = path
        return f"{path}:{frame.f_lineno} {frame.f_code.co_name}", bulk
    return "?", bulk


def _env_getitem(self, key):
    try:
        value = _ENV_ORIG(self, key)
    except KeyError:
        rec = _REC
        if rec is not None:
            rec._env_read(key, None)
        raise
    rec = _REC
    if rec is not None:
        rec._env_read(key, value)
    return value


def _recording_open(file, mode="r", *args, **kwargs):
    rec = _REC
    if (
        rec is not None
        and rec.files_on
        and isinstance(file, (str, bytes, os.PathLike))
        and not rec._busy()
        and not any(c in mode for c in "wax+")
    ):
        rec._file_read(os.fsdecode(file))
    return _OPEN_ORIG(file, mode, *args, **kwargs)


def _raw_class_attr(cls: type, name: str):
    for c in cls.__mro__:
        if name in c.__dict__:
            return c.__dict__[name]
    raise KeyError(name)


def _safe_key_fp(v: Any):
    try:
        return key_fp(v)
    except Exception as e:  # noqa: BLE001
        # Never matches: the client's side raises too and compares as missing.
        return ("<error>", type(e).__name__)


class InputRecorder:
    def __init__(self):
        self.phase = "init"
        self.env: dict[str, dict] = {}
        self.env_by_line: dict[str, set] = {}
        self.attrs: dict[tuple, dict] = {}  # (kind, path, name) -> info
        self.paths: dict[int, tuple[str, str]] = {}  # id(obj) -> (kind, path)
        self.pre_load_fps: dict[tuple[str, str], Any] = {}
        self.pre_load_globals: dict[str, dict] = {}  # module -> name -> fingerprint
        self.files: dict[str, set] = {}
        self.globals: dict[tuple, Any] = {}
        self.files_on = False
        self.attrs_on = False
        self._patched: list[tuple[type, Any, bool]] = []
        self._local = threading.local()
        # (object id, name, phase) already handled: loading reads the same
        # attributes millions of times, and only the first read counts.
        self._seen: set[tuple[int, str, str]] = set()

    # -- lifecycle -----------------------------------------------------------

    @classmethod
    def start(cls) -> InputRecorder:
        global _REC, _ENV_ORIG
        rec = cls()
        if _ENV_ORIG is None:
            _ENV_ORIG = os._Environ.__getitem__
            os._Environ.__getitem__ = _env_getitem
        _REC = rec
        return rec

    def enter_construct(self, config) -> None:
        global _OPEN_ORIG
        self.phase = "construct"
        self._register_config(config)
        if _OPEN_ORIG is None:
            _OPEN_ORIG = builtins.open
            builtins.open = io.open = _recording_open
        self.files_on = True
        self.attrs_on = True

    def enter_load(self, model) -> None:
        self.phase = "load"
        self.attrs_on = True
        # The construction-time values: what a client builds, and what loading
        # started from. (The first value loading reads may be one it wrote.)
        self.pre_load_globals = _module_globals()
        values = {}
        for path, obj, is_module in _walk_objects(model):
            self.paths[id(obj)] = ("model", path)
            for name, v in _attrs(obj, is_module).items():
                values[(path, name)] = v
            if is_module:
                for name, b in obj._buffers.items():
                    if b is not None:
                        values[(path, f"[buffer]{name}")] = b
            self._patch_class(type(obj))
        # Fingerprint now, before loading mutates anything in place.
        self.pre_load_fps = {k: _safe_key_fp(v) for k, v in values.items()}

    def finish_load(self, codes) -> None:
        """Stop recording; ``codes``: the code objects that ran during load."""
        global _REC, _ENV_ORIG, _OPEN_ORIG
        self.attrs_on = False
        self.files_on = False
        self._unpatch()
        if _OPEN_ORIG is not None:
            builtins.open = io.open = _OPEN_ORIG
            _OPEN_ORIG = None
        if _ENV_ORIG is not None:
            os._Environ.__getitem__ = _ENV_ORIG
            _ENV_ORIG = None
        _REC = None
        self._collect_globals(codes)

    # -- recording -----------------------------------------------------------

    def _busy(self) -> bool:
        return getattr(self._local, "busy", False)

    def _env_read(self, key: str, value) -> None:
        if self._busy():
            return
        self._local.busy = True
        try:
            where, bulk = _where(sys._getframe(2))
            if bulk:
                where = "bulk: " + where
            self.env_by_line.setdefault(where, set()).add(key)
            info = self.env.setdefault(key, {"load_value": None, "load_where": set()})
            if self.phase == "load":
                if not info["load_where"]:
                    info["load_value"] = value
                info["load_where"].add(where)
        finally:
            self._local.busy = False

    def _attr_read(self, kind: str, path: str, name: str, value) -> None:
        if self._busy():
            return
        self._local.busy = True
        try:
            key = (kind, path, name)
            info = self.attrs.get(key)
            if info is None:
                # The first read is the value the load consumed.
                info = self.attrs[key] = {"phases": set(), "fp": key_fp(value)}
            info["phases"].add(self.phase)
        except Exception:  # noqa: BLE001, S110 - recording must never break loading
            pass
        finally:
            self._local.busy = False

    def _file_read(self, path: str) -> None:
        if path.endswith((".py", ".pyc")) or path.startswith(("/proc", "/sys", "/dev")):
            return
        self.files.setdefault(os.path.realpath(path), set()).add(self.phase)

    # -- class patching --------------------------------------------------------

    def _register_config(self, config) -> None:
        roots = [("config", config)]
        hf = getattr(config, "hf_config", None)
        if hf is not None:
            roots.append(("hf_config", hf))
        seen = set()
        while roots:
            path, obj = roots.pop()
            if id(obj) in seen or not hasattr(obj, "__dict__") or isinstance(obj, type):
                continue
            seen.add(id(obj))
            self.paths.setdefault(id(obj), ("config", path))
            self._patch_class(type(obj))
            for k, v in vars(obj).items():
                mod = type(v).__module__ or ""
                if (
                    hasattr(v, "__dict__")
                    and not isinstance(v, (type, types.ModuleType))
                    and mod.startswith(("atom.", "transformers"))
                    and "tokeniz" not in mod
                    and path.count(".") < 3
                ):
                    roots.append((f"{path}.{k}", v))

    def _patch_class(self, cls: type) -> None:
        if any(c is cls for c, _, _ in self._patched):
            return
        own = "__getattribute__" in cls.__dict__
        orig_own = cls.__dict__.get("__getattribute__")
        current = cls.__getattribute__
        if getattr(current, "_atom_input_recorder", False) and not own:
            return  # inherits a recording parent
        rec = self
        skip = _MODULE_INTERNALS

        def __getattribute__(obj, name, _orig=current):
            try:
                value = _orig(obj, name)
            except AttributeError:
                # nn.Module buffers/parameters are served by __getattr__
                # after this raises.
                if rec.attrs_on and name[:2] != "__" and name not in skip:
                    hit = rec.paths.get(id(obj))
                    if hit is not None:
                        rec._on_missing(obj, hit, name)
                raise
            if rec.attrs_on and name[:2] != "__" and name not in skip:
                hit = rec.paths.get(id(obj))
                if hit is not None:
                    rec._on_attr(obj, hit, name, value)
            return value

        __getattribute__._atom_input_recorder = True
        try:
            cls.__getattribute__ = __getattribute__
        except (TypeError, AttributeError):
            return  # builtin / extension type
        self._patched.append((cls, orig_own, own))

    def _first(self, obj, name: str) -> bool:
        key = (id(obj), name, self.phase)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def _on_attr(self, obj, hit, name: str, value) -> None:
        if not self._first(obj, name):
            return
        kind, path = hit
        try:
            d = object.__getattribute__(obj, "__dict__")
        except Exception:  # noqa: BLE001
            d = {}
        if name in d:
            self._attr_read(kind, path, name, value)
            return
        # A class attribute (layout constants and the like), not a method or property.
        try:
            raw = _raw_class_attr(type(obj), name)
        except KeyError:
            return
        if _is_plain(raw) or _ref_name(raw):
            cls = type(obj)
            self._attr_read("class", f"{cls.__module__}:{cls.__qualname__}", name, raw)

    def _on_missing(self, obj, hit, name: str) -> None:
        if not self._first(obj, f"[buffer]{name}"):
            return
        try:
            buffers = object.__getattribute__(obj, "__dict__").get("_buffers")
        except Exception:  # noqa: BLE001
            return
        if isinstance(buffers, dict) and buffers.get(name) is not None:
            self._attr_read(hit[0], hit[1], f"[buffer]{name}", buffers[name])

    def _unpatch(self) -> None:
        for cls, orig, own in reversed(self._patched):
            try:
                if own:
                    cls.__getattribute__ = orig
                else:
                    del cls.__getattribute__
            except (TypeError, AttributeError):
                pass
        self._patched.clear()

    # -- globals -------------------------------------------------------------

    def _collect_globals(self, codes) -> None:
        """The globals the loading code referenced, as the load found them (one
        it set itself travels with process_export)."""
        by_file = {}
        for mod in list(sys.modules.values()):
            f = getattr(mod, "__file__", None)
            if f:
                by_file[os.path.realpath(f)] = mod
        roots = _package_roots()
        for code in codes:
            if _code_location(code.co_filename, roots) is None:
                continue
            mod = by_file.get(os.path.realpath(code.co_filename))
            if mod is None:
                continue
            g = vars(mod)
            before = self.pre_load_globals.get(mod.__name__)
            for name in code.co_names:
                if name not in g or name.startswith("__") or not _keyed(g[name]):
                    continue
                if before is None:  # imported by the load: as importing builds it
                    self.globals[(mod.__name__, name)] = key_fp(g[name])
                elif name in before:
                    self.globals[(mod.__name__, name)] = before[name]

    # -- results -------------------------------------------------------------

    def key_reads(self, checkpoint_dir: str | None = None) -> dict:
        """What the load read, for the cache key (load phase only)."""
        bulk = {
            w
            for w, keys in self.env_by_line.items()
            if w.startswith("bulk") or len(keys) >= _BULK_THRESHOLD
        }
        env = {}
        for k, info in self.env.items():
            where = [w for w in info["load_where"] if w not in bulk]
            if any(w.startswith(KEY_ENV_PREFIXES) for w in where):
                env[k] = info["load_value"]
        config, attrs, classattrs = {}, {}, {}
        for (kind, path, name), info in self.attrs.items():
            if "load" not in info["phases"]:
                continue
            if kind == "config":
                config[f"{path}.{name}"] = info["fp"]
            elif kind == "class":
                classattrs[f"{path}\t{name}"] = info["fp"]
            elif (path, name) in self.pre_load_fps:
                attrs[f"{path}\t{name}"] = self.pre_load_fps[(path, name)]
        ckpt = os.path.realpath(checkpoint_dir) + os.sep if checkpoint_dir else None
        files = {}
        for p, phases in self.files.items():
            if (
                "load" in phases
                and not (ckpt and p.startswith(ckpt))
                and os.path.isfile(p)
            ):
                files[p] = file_identity(p)
        return {
            "env": env,
            "config": config,
            "attrs": attrs,
            "classattrs": classattrs,
            "globals": {f"{m}\t{n}": fp for (m, n), fp in self.globals.items()},
            "files": files,
        }


def _keyed(v) -> bool:
    """A global whose value enters the key."""
    if isinstance(v, (types.ModuleType, type)):
        return False
    return (
        _is_plain(v) or bool(_ref_name(v)) or isinstance(v, types.BuiltinFunctionType)
    )


def _module_globals() -> dict[str, dict]:
    """The keyed globals of the ATOM/AITER modules there are, fingerprinted."""
    out = {}
    for name, mod in list(sys.modules.items()):
        if mod is not None and name.split(".")[0] in ("atom", "aiter"):
            out[name] = {
                k: _safe_key_fp(v)
                for k, v in list(vars(mod).items())
                if not k.startswith("__") and _keyed(v)
            }
    return out


# ---------------------------------------------------------------------------
# Client side: compare a daemon's reads with this process
# ---------------------------------------------------------------------------

STATIC_READS = ("env", "config", "classattrs", "globals", "files")
_MISSING = ("<missing>",)


def _now(kind: str, key: str, config, model):
    """This process's value of one read, rendered like the daemon's."""
    if kind == "env":
        return os.environ.get(key)
    if kind == "files":
        return file_identity(key)
    if kind == "config":
        root, *rest = key.split(".")
        obj = config if root == "config" else getattr(config, "hf_config", None)
        for part in rest:
            obj = getattr(obj, part)
        return key_fp(obj)
    a, name = key.split("\t")
    if kind == "classattrs":
        return key_fp(_raw_class_attr(_import_qual(*a.split(":")), name))
    if kind == "globals":
        return key_fp(vars(importlib.import_module(a))[name])
    obj = _resolve(model, a)  # attrs: on the freshly built (meta) model
    if name.startswith("[buffer]"):
        return key_fp(obj._buffers[name[len("[buffer]") :]])
    return key_fp(vars(obj)[name])


def _describe(kind: str, key: str, then, now) -> str:
    if kind == "files":
        return f"file {key} changed"
    if kind == "env":
        return f"env {key}: daemon={then} client={now}"
    if kind == "config":
        what = key
    else:
        a, name = key.split("\t")
        what = {
            "classattrs": f"class attribute {a.split(':')[-1]}.{name}",
            "globals": f"global {a}.{name}",
            "attrs": f"{a or '<root>'}.{name}",
        }[kind]
    return f"{what}: daemon={str(then)[:80]} client={str(now)[:80]}"


def diff_reads(reads: dict, kinds, config=None, model=None) -> list[str]:
    """What the daemon's load read (of these kinds) that differs here."""
    out = []
    for kind in kinds:
        for key, then in sorted(reads[kind].items()):
            try:
                now = _now(kind, key, config, model)
            except Exception:  # noqa: BLE001
                now = _MISSING
            if now != then:
                out.append(_describe(kind, key, then, now))
    return out
