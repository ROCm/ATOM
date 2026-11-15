# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What the daemon serves: the model state ``load_model`` produced, found by
diffing the model before and after loading, exported as CUDA IPC handles and
applied to a client's meta-built model."""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import types
from typing import Any

import torch
from torch import nn

from atom.model_engine.model_setup import swap_param
from atom.model_loader.weight_cache.values import (
    _MODULE_INTERNALS,
    _attrs,
    _entries,
    _fp,
    _is_plain,
    _recursable,
    _ref_name,
    _resolve,
    _Unshippable,
    _walk_objects,
    config_objects,
    safe_fp,
)

_DEEP_DEPTH = 6  # how far below a module the fingerprints of unwalked objects go
_STATE = ("atom", "aiter")  # the packages whose globals process_snapshot covers


def snapshot(model: nn.Module) -> dict:
    """Everything loading can change in ``model``, with tensor contents
    (daemon only): what the walk exports, and fingerprints of the objects it
    does not reach, which loading must leave alone."""
    walk = {}
    for path, obj, is_module in _walk_objects(model):
        entry = {"attrs": {k: _fp(v) for k, v in _attrs(obj, is_module).items()}}
        if is_module:
            entry["type"] = f"{type(obj).__module__}.{type(obj).__qualname__}"
            # Parameters are always exported, so their identity is enough.
            entry["params"] = {
                k: None if p is None else ("P", id(p), tuple(p.shape), str(p.dtype))
                for k, p in obj._parameters.items()
            }
            entry["buffers"] = {k: _fp(b) for k, b in obj._buffers.items()}
        walk[path] = entry
    return {"walk": walk, "deep": _unwalked(model)}


def _unwalked(model: nn.Module) -> dict:
    """Fingerprints of the attributes of objects the walk does not export:
    deeper than it goes, or held in containers."""
    walked = {id(obj) for _, obj, _ in _walk_objects(model)}
    out, seen = {}, set()
    for mpath, mod in model.named_modules():
        queue = [
            (f"{mpath}::{k}", v, 1)
            for k, v in vars(mod).items()
            if k not in _MODULE_INTERNALS and not isinstance(v, nn.Module)
        ]
        while queue:
            path, v, depth = queue.pop()
            if depth > _DEEP_DEPTH or id(v) in seen:
                continue
            if isinstance(v, (list, tuple, dict)):
                seen.add(id(v))
                items = v.items() if isinstance(v, dict) else enumerate(v)
                queue += [(f"{path}[{k!r}]", x, depth + 1) for k, x in items]
            elif _recursable(v):
                seen.add(id(v))
                for k, x in vars(v).items():
                    if id(v) not in walked:
                        out[f"{path}::{k}"] = _fp(x)
                    if not isinstance(x, (nn.Module, types.ModuleType)):
                        queue.append((f"{path}::{k}", x, depth + 1))
    return out


_MAX_HOST_BYTES = 256 << 20  # host tensors travel in the payload


class _Exporter:
    """Tensors as views of whole storages, so aliasing survives: device ones
    shared over IPC (``views``), host ones sent by value (``host``)."""

    def __init__(self):
        self.views: dict[int, torch.Tensor] = {}  # device storages as uint8
        self.host: dict[str, tuple[torch.Tensor, bool]] = {}  # (bytes, pinned)

    def unshippable(self) -> list[str]:
        n = sum(b.numel() for b, _ in self.host.values())
        return [f"{n >> 20} MiB of host tensors"] if n > _MAX_HOST_BYTES else []

    def tensor(self, t: torch.Tensor) -> dict:
        t = t.detach()
        st = t.untyped_storage()
        if st.nbytes() == 0:
            # No allocation, so no IPC handle: rebuilt locally.
            return {
                "empty": True,
                "shape": tuple(t.shape),
                "stride": tuple(t.stride()),
                "dtype": t.dtype,
            }
        whole = torch.empty(0, dtype=torch.uint8, device=t.device).set_(
            st, 0, (st.nbytes(),), (1,)
        )
        if t.is_cuda:
            key = st.data_ptr()
            self.views.setdefault(key, whole)
        else:
            key = f"host:{st.data_ptr()}"
            if key not in self.host:
                pinned = torch.cuda.is_available() and t.is_pinned()
                self.host[key] = (whole.clone(), pinned)
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
    objects = {path: (obj, is_module) for path, obj, is_module in _walk_objects(model)}
    plan: dict[str, dict] = {}
    deep0, deep1 = before["deep"], after["deep"]
    unshippable = [
        f"{p}: changed during load, out of the cache's reach"
        for p in sorted(set(deep0) | set(deep1))
        if deep0.get(p) != deep1.get(p)
    ]
    before, after = before["walk"], after["walk"]
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
        if prev.get("type") != entry.get("type"):
            unshippable.append(f"{path or '<root>'}: replaced by a {entry['type']}")
            continue
        item: dict[str, Any] = {"params": {}, "buffers": {}, "attrs": {}, "delete": []}
        if is_module:
            for k, p in obj._parameters.items():
                if p is None:
                    if prev["params"].get(k) is not None:
                        item["params"][k] = None  # loading set it to None
                    continue
                meta = ex.tensor(p)
                meta["replaced"] = prev["params"].get(k, ("P", id(p)))[1] != id(p)
                meta["pattrs"] = {}
                for a, v in vars(p).items():
                    # Loaders (bound methods) are the client's own.
                    with contextlib.suppress(_Unshippable):
                        meta["pattrs"][a] = _pack(v, ex)
                item["params"][k] = meta
            for k, b in obj._buffers.items():
                if prev["buffers"].get(k) != entry["buffers"].get(k):
                    item["buffers"][k] = None if b is None else ex.tensor(b)
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
        "views": ex.views,
        "host": ex.host,
        "unshippable": unshippable + ex.unshippable(),
    }


def leave_out(export: dict, paths: set[str]) -> None:
    """Drop the modules at ``paths`` from ``export``, with the storages
    nothing left in it uses."""
    plan = export["plan"]
    for p in [p for p in plan if p.partition("::")[0] in paths]:
        del plan[p]
    used = set()

    def note(meta: dict) -> None:
        used.add(meta.get("storage"))

    for item in plan.values():
        for meta in (*item["params"].values(), *item["buffers"].values()):
            if meta is not None:
                note(meta)
                for packed in meta.get("pattrs", {}).values():
                    _unpack(packed, note)
        for packed in item["attrs"].values():
            _unpack(packed, note)
    export["views"] = {k: v for k, v in export["views"].items() if k in used}
    export["host"] = {k: v for k, v in export["host"].items() if k in used}


def process_snapshot(config=None) -> dict:
    """What loading can change outside the model (daemon only): ATOM/AITER
    module globals, their classes' attributes, the attributes of the objects
    the globals hold, the config's objects and the environment."""
    own = __name__.rpartition(".")[0]  # this package's state is the daemon's
    modules, state = set(), {}
    for path, obj in config_objects(config) if config is not None else ():
        for a, x in list(vars(obj).items()):
            state[("config", path, a)] = safe_fp(_fp, x)
    for name, mod in list(sys.modules.items()):
        if mod is None or name.startswith(own) or name.split(".")[0] not in _STATE:
            continue
        modules.add(name)
        for k, v in list(vars(mod).items()):
            if k.startswith("__") or isinstance(v, types.ModuleType):
                continue
            if isinstance(v, type):
                if v.__module__ == name:  # by the name the module holds it under
                    for a, x in list(vars(v).items()):
                        if not a.startswith("__"):
                            state[("class", name, k, a)] = safe_fp(_fp, x)
                continue
            state[("global", name, k)] = safe_fp(_fp, v)
            if _recursable(v):
                for a, x in list(vars(v).items()):
                    state[("object", name, k, a)] = safe_fp(_fp, x)
    return {"modules": modules, "state": state, "env": os.environ.copy()}


def process_export(states: list[dict]) -> dict:
    """What loading changed outside the model, as changes a client applies:
    module globals and class attributes by value, and environment variables.
    ``states``: process_snapshot before and after each load. A module a load
    imported is left out: importing it again builds it again."""
    changed, env = set(), set()
    for before, after in zip(states[::2], states[1::2]):
        s0, s1 = before["state"], after["state"]
        changed.update(
            k
            for k in set(s0) | set(s1)
            if (k[0] == "config" or k[1] in before["modules"])
            and s0.get(k) != s1.get(k)
        )
        e0, e1 = before["env"], after["env"]
        env.update(k for k in set(e0) | set(e1) if e0.get(k) != e1.get(k))
    ex = _Exporter()
    changes, unshippable = [], []
    for key in sorted(changed, key=str):
        kind, module, *rest = key
        where = f"{module}:{'.'.join(rest)}"
        if kind in ("object", "config"):
            unshippable.append(
                f"{where}: changed during load, out of the cache's reach"
            )
            continue
        owner = sys.modules[module]
        if kind == "class":
            owner = getattr(owner, rest[0])
        if rest[-1] not in vars(owner):
            changes.append((kind, module, rest, None))  # deleted
            continue
        try:
            changes.append((kind, module, rest, _pack(vars(owner)[rest[-1]], ex)))
        except _Unshippable as e:
            unshippable.append(f"{where}: {e}")
    return {
        "changes": changes,
        "env": {k: os.environ.get(k) for k in sorted(env)},
        "views": ex.views,
        "host": ex.host,
        "unshippable": unshippable + ex.unshippable(),
    }


def share_storages(views: dict) -> dict:
    """IPC handles of the exported storages (``views``). Share last: torch
    never frees a shared storage in this process, even with no consumer."""
    return {key: v.untyped_storage()._share_cuda_() for key, v in views.items()}


def _install_param(obj, k: str, t: torch.Tensor, meta: dict, make, filled: dict):
    """Put ``t`` into parameter ``k``, swapping it into the Parameter object
    the model was built with (other references see it). ``filled``: objects
    swapped already; one tied to a parameter loading gave other data (it
    untied them) gets a Parameter of its own."""
    built = obj._parameters.get(k)
    where = (t.untyped_storage().data_ptr(), t.storage_offset(), t.shape, t.stride())
    tied = built is not None and id(built) in filled
    if tied and filled[id(built)] == where:
        return  # still tied: it holds this data already
    pattrs = {a: _unpack(x, make) for a, x in meta["pattrs"].items()}
    # A Parameter loading replaced has only the attributes it was given.
    keep = {} if built is None or meta["replaced"] else vars(built)
    attrs = {**keep, **pattrs}
    new = nn.Parameter(t, requires_grad=False)
    if not tied and swap_param(built, new, attrs):
        filled[id(built)] = where
        return
    new.__dict__.update(attrs)
    vars(obj).pop(k, None)
    obj._buffers.pop(k, None)
    obj._parameters[k] = new


def _maker(export: dict, device: torch.device):
    """(the function that rebuilds an exported tensor, bytes mapped)."""
    storages = {}
    for key, share in export["storages"].items():
        args = list(share)
        args[0] = device.index  # the daemon's index for this GPU may differ
        storages[key] = torch.UntypedStorage._new_shared_cuda(*args)
    mapped = sum(s.nbytes() for s in storages.values())
    for key, (data, pinned) in export["host"].items():
        storages[key] = (data.pin_memory() if pinned else data).untyped_storage()

    def make(meta: dict) -> torch.Tensor:
        if meta.get("empty"):
            return torch.empty_strided(
                meta["shape"], meta["stride"], dtype=meta["dtype"], device=device
            )
        st = storages[meta["storage"]]
        t = torch.empty(0, dtype=meta["dtype"], device=st.device)
        t.set_(st, meta["offset"], meta["shape"], meta["stride"])
        return t

    return make, mapped


def apply_process(export: dict, device: torch.device) -> int:
    """Make the changes ``process_export`` found. Returns bytes mapped."""
    make, mapped = _maker(export, device)
    for kind, module, rest, packed in export["changes"]:
        owner = importlib.import_module(module)
        if kind == "class":
            owner = getattr(owner, rest[0])
        if packed is None:
            with contextlib.suppress(AttributeError):
                delattr(owner, rest[-1])
        else:
            setattr(owner, rest[-1], _unpack(packed, make))
    for k, v in export["env"].items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return mapped


def apply_export(model: nn.Module, export: dict, device: torch.device) -> int:
    """Point ``model`` (built on meta) at the daemon's tensors. Returns bytes mapped."""
    make, mapped = _maker(export, device)
    filled = {}  # id(Parameter the model was built with) -> where its data is
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
            if meta is None:
                obj._parameters[k] = None
                continue
            _install_param(obj, k, make(meta), meta, make, filled)
        for k, meta in item["buffers"].items():
            obj._buffers[k] = None if meta is None else make(meta)
        for k, packed in item["attrs"].items():
            if is_module and (
                k in obj._parameters or k in obj._buffers or k in obj._modules
            ):
                raise RuntimeError(
                    f"{path}.{k}: attribute shadows a parameter, buffer or submodule"
                )
            d[k] = _unpack(packed, make)
    return mapped


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

    for path, _, _, k, v in _entries(model):
        check(f"{path}.{k}", v)
    return left


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
