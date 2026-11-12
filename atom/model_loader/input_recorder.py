# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Record the inputs Python code reads while a runner builds and loads its model.

Used two ways:

* by the weight cache daemon: what its load read becomes part of the cache
  key (``key_reads``), so a client is compared on exactly what decided the
  weights: environment variables and config fields that loading read,
  construction-time attributes (instance and class) that loading read,
  globals the loading code referenced, and files it opened;
* for diagnosis, with ``ATOM_WEIGHT_CACHE_READS=<prefix>``: every phase is
  written to ``<prefix>.<rank>.json``, with where each read happened.

Phases: ``init`` (runner start to model construction), ``construct`` and
``load`` (checkpoint load and post-processing). Recording stops there:
warmup, graph capture and serving cannot change the loaded weights.

Native code (C++ getenv, files opened by C++) is invisible here.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import logging
import os
import sys
import threading
import time
import types
from typing import Any

logger = logging.getLogger("atom")

_REC: InputRecorder | None = None
_ENV_ORIG = None
_OPEN_ORIG = None
_SKIP_FRAME_FILES = (
    "/os.py",
    "/_collections_abc.py",
    "/atom/utils/envs.py",
    "/input_recorder.py",
)
_BULK_THRESHOLD = 25  # distinct variables read from one line: a copy of the environment
# Environment reads that count for the cache key: made by these packages'
# code. Others (a progress bar reading COLUMNS) do not decide weights.
KEY_ENV_PREFIXES = ("atom/", "aiter/", "flydsl/", "triton/", "mori/")
_ROOTS_CACHE: list[str] | None = None


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
        path = os.path.realpath(fn)
        for root in _roots():
            if path.startswith(root + os.sep):
                path = os.path.relpath(path, root)
                break
        return f"{path}:{frame.f_lineno} {frame.f_code.co_name}", bulk
    return "?", bulk


def key_fp(v: Any):
    """Process-independent fingerprint of a value a load read: plain values
    by value, functions by name, objects by type; integer tensors (index
    maps) by content, floating ones (often uninitialized workspaces) by
    shape and dtype."""
    import torch

    from atom.model_loader import weight_cache as wc

    if isinstance(v, torch.Tensor):
        if v.is_floating_point() or v.is_complex() or v.is_meta:
            return ("T", tuple(v.shape), str(v.dtype))
        return ("T", tuple(v.shape), str(v.dtype), wc._digest(v))
    return wc._fp(v, content=False)


def _brief(v: Any) -> str:
    try:
        s = str(key_fp(v))
    except Exception as e:  # noqa: BLE001 - recording must never break loading
        s = f"<unprintable {type(e).__name__}>"
    return s if len(s) <= 140 else s[:137] + "..."


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


class InputRecorder:
    def __init__(self):
        self.phase = "init"
        self.t0 = time.time()
        self.env: dict[str, dict] = {}
        self.env_by_line: dict[str, set] = {}
        self.attrs: dict[tuple, dict] = {}  # (kind, path, name) -> info
        self.paths: dict[int, tuple[str, str]] = {}  # id(obj) -> (kind, path)
        self.pre_load_attrs: set[tuple[str, str]] = set()
        self.pre_load_fps: dict[tuple[str, str], Any] = {}
        self.files: dict[str, set] = {}
        self.globals: dict[tuple, Any] = {}
        self.files_on = False
        self.attrs_on = False
        self.functions_in_load = 0
        self._patched: list[tuple[type, Any, bool]] = []
        self._local = threading.local()

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
        self.config_roots = {
            "config": config,
            "hf_config": getattr(config, "hf_config", None),
        }
        self._register_config(config)
        if _OPEN_ORIG is None:
            _OPEN_ORIG = builtins.open
            builtins.open = io.open = _recording_open
        self.files_on = True
        self.attrs_on = True

    def enter_load(self, model) -> None:
        from atom.model_loader import weight_cache as wc

        self.phase = "load"
        self.attrs_on = True
        # The construction-time values: what a client builds, and what loading
        # started from. (The first value loading reads may be one it wrote.)
        self.pre_load_fps = {}
        for path, obj, is_module in wc._walk_objects(
            model, wc._DIFF_DEPTH, wc._DIFF_PACKAGES
        ):
            self.paths[id(obj)] = ("model", path)
            for name, v in wc._attrs(obj, is_module).items():
                self.pre_load_attrs.add((path, name))
                self.pre_load_fps[(path, name)] = v
            if is_module:
                for name, b in obj._buffers.items():
                    if b is not None:
                        self.pre_load_attrs.add((path, f"[buffer]{name}"))
                        self.pre_load_fps[(path, f"[buffer]{name}")] = b
            self._patch_class(type(obj))
        # Fingerprint now, before loading mutates anything in place.
        fps = {}
        for k, v in self.pre_load_fps.items():
            try:
                fps[k] = key_fp(v)
            except Exception as e:  # noqa: BLE001 - unfingerprintable: never matches
                fps[k] = ("<error>", type(e).__name__, os.getpid())
        self.pre_load_fps = fps

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
        self.functions_in_load = len(codes)
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
            info = self.env.setdefault(
                key,
                {
                    "values": set(),
                    "phases": {},
                    "where": {},
                    "load_value": None,
                    "load_where": set(),
                },
            )
            info["values"].add(value)
            info["phases"][self.phase] = info["phases"].get(self.phase, 0) + 1
            if len(info["where"]) < 6:
                info["where"].setdefault(where, self.phase)
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
                info = self.attrs[key] = {"phases": {}, "fp": key_fp(value)}
            info["phases"][self.phase] = info["phases"].get(self.phase, 0) + 1
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
        skip = _module_internals()

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

    def _on_attr(self, obj, hit, name: str, value) -> None:
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
        from atom.model_loader import weight_cache as wc

        if wc._is_plain(raw) or wc._ref_name(raw):
            cls = type(obj)
            self._attr_read("class", f"{cls.__module__}:{cls.__qualname__}", name, raw)

    def _on_missing(self, obj, hit, name: str) -> None:
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
        from atom.model_loader import weight_cache as wc

        by_file = {}
        for mod in list(sys.modules.values()):
            f = getattr(mod, "__file__", None)
            if f:
                by_file[os.path.realpath(f)] = mod
        roots = wc._package_roots()
        own = {os.path.realpath(wc.__file__), os.path.realpath(__file__)}
        for code in codes:
            fn = os.path.realpath(code.co_filename)
            if fn in own or not any(fn.startswith(r + os.sep) for r in roots.values()):
                continue
            mod = by_file.get(fn)
            if mod is None:
                continue
            g = vars(mod)
            for name in code.co_names:
                if name not in g or name.startswith("__"):
                    continue
                v = g[name]
                if isinstance(v, (types.ModuleType, type)):
                    continue
                if (
                    wc._is_plain(v)
                    or wc._ref_name(v)
                    or isinstance(v, types.BuiltinFunctionType)
                ):
                    self.globals[(mod.__name__, name)] = key_fp(v)

    # -- results -------------------------------------------------------------

    def _bulk_lines(self) -> set:
        return {
            w
            for w, keys in self.env_by_line.items()
            if w.startswith("bulk") or len(keys) >= _BULK_THRESHOLD
        }

    def key_reads(self, checkpoint_dir: str | None = None) -> dict:
        """What the load read, for the cache key (load phase only)."""
        bulk = self._bulk_lines()
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
            elif (path, name) in self.pre_load_attrs:
                attrs[f"{path}\t{name}"] = self.pre_load_fps[(path, name)]
        ckpt = os.path.realpath(checkpoint_dir) + os.sep if checkpoint_dir else None
        files = {}
        for p, phases in self.files.items():
            if (
                "load" in phases
                and not (ckpt and p.startswith(ckpt))
                and os.path.isfile(p)
            ):
                files[p] = _file_digest(p)
        return {
            "env": env,
            "config": config,
            "attrs": attrs,
            "classattrs": classattrs,
            "globals": {f"{m}\t{n}": fp for (m, n), fp in self.globals.items()},
            "files": files,
        }

    def dump(self, out: str) -> None:
        """Full record of every phase, for diagnosis."""
        bulk = self._bulk_lines()
        env = {}
        for k, info in sorted(self.env.items()):
            where = {w: p for w, p in info["where"].items() if w not in bulk}
            if not where:
                continue  # only seen in bulk copies
            env[k] = {
                "value": sorted(str(v) for v in info["values"]),
                "phases": info["phases"],
                "where": where,
            }
        attrs = {}
        for (kind, path, name), info in sorted(self.attrs.items()):
            attrs[f"{kind}:{path}.{name}"] = {
                "phases": info["phases"],
                "value": str(info["fp"])[:140],
                "set_by_construction": (
                    (path, name) in self.pre_load_attrs if kind == "model" else None
                ),
            }
        data = {
            "elapsed_s": round(time.time() - self.t0, 1),
            "env": env,
            "env_bulk": {w: len(self.env_by_line[w]) for w in sorted(bulk)},
            "attrs": attrs,
            "globals": {
                f"{m}.{n}": str(v)[:140] for (m, n), v in sorted(self.globals.items())
            },
            "files": {p: sorted(ph) for p, ph in sorted(self.files.items())},
            "functions_in_load": self.functions_in_load,
        }
        try:
            with open(out, "w") as f:
                json.dump(data, f, indent=1, sort_keys=True)
            logger.info(
                f"[weight-cache] input record -> {out} ({len(env)} env vars, {len(attrs)} attributes)"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[weight-cache] cannot write input record {out}: {e}")


def _file_digest(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        return f"<unreadable {type(e).__name__}>"


def _module_internals():
    from atom.model_loader import weight_cache as wc

    return wc._MODULE_INTERNALS


# ---------------------------------------------------------------------------
# Client side: compare a daemon's reads with this process
# ---------------------------------------------------------------------------

_MISSING = ("<missing>",)


def _resolve_config(config, path: str):
    root, *rest = path.split(".")
    obj = config if root == "config" else getattr(config, "hf_config", None)
    for part in rest:
        obj = getattr(obj, part)
    return obj


def diff_static_reads(reads: dict, config) -> list[str]:
    """Env vars, config fields, class attributes, globals and files the
    daemon's load read, compared with this process (before building a model)."""
    import importlib

    out = []
    for k, v in sorted(reads["env"].items()):
        now = os.environ.get(k)
        if now != v:
            out.append(f"env {k}: daemon={v} client={now}")
    for path, fp in sorted(reads["config"].items()):
        try:
            now = key_fp(_resolve_config(config, path))
        except Exception:  # noqa: BLE001
            now = _MISSING
        if now != fp:
            out.append(f"{path}: daemon={str(fp)[:80]} client={str(now)[:80]}")
    for key, fp in sorted(reads["classattrs"].items()):
        cls_name, name = key.split("\t")
        mod, qual = cls_name.split(":")
        try:
            obj = importlib.import_module(mod)
            for part in qual.split("."):
                obj = getattr(obj, part)
            now = key_fp(_raw_class_attr(obj, name))
        except Exception:  # noqa: BLE001
            now = _MISSING
        if now != fp:
            out.append(
                f"class attribute {qual}.{name}: daemon={str(fp)[:80]} client={str(now)[:80]}"
            )
    for key, fp in sorted(reads["globals"].items()):
        mod, name = key.split("\t")
        try:
            now = key_fp(vars(importlib.import_module(mod))[name])
        except Exception:  # noqa: BLE001
            now = _MISSING
        if now != fp:
            out.append(
                f"global {mod}.{name}: daemon={str(fp)[:80]} client={str(now)[:80]}"
            )
    for path, digest in sorted(reads["files"].items()):
        if _file_digest(path) != digest:
            out.append(f"file {path} changed")
    return out


def diff_attr_reads(reads: dict, model) -> list[str]:
    """Construction-time attributes the daemon's load read, compared with
    this process's freshly built (meta) model."""
    from atom.model_loader import weight_cache as wc

    out = []
    for key, fp in sorted(reads["attrs"].items()):
        path, name = key.split("\t")
        try:
            obj = wc._resolve(model, path)
            if name.startswith("[buffer]"):
                now = key_fp(obj._buffers[name[len("[buffer]") :]])
            else:
                now = key_fp(vars(obj)[name])
        except Exception:  # noqa: BLE001
            now = _MISSING
        if now != fp:
            out.append(
                f"{path or '<root>'}.{name}: daemon={str(fp)[:80]} client={str(now)[:80]}"
            )
    return out
