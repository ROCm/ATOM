# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What the daemon serves: the model state ``load_model`` produced, found by
diffing the model before and after loading, exported as CUDA IPC handles and
applied to a client's meta-built model."""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch import nn

from atom.model_loader.weight_cache.values import (
    _attrs,
    _entries,
    _fp,
    _is_plain,
    _ref_name,
    _resolve,
    _Unshippable,
    _walk_objects,
)


def snapshot(model: nn.Module) -> dict:
    """Everything loading can change, with tensor contents (daemon only)."""
    snap = {}
    for path, obj, is_module in _walk_objects(model):
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


class _Exporter:
    def __init__(self):
        self.views: dict[int, torch.Tensor] = {}  # whole storages as uint8
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
        if key not in self.views:
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
    objects = {path: (obj, is_module) for path, obj, is_module in _walk_objects(model)}
    plan: dict[str, dict] = {}
    unshippable = [
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
        "views": ex.views,
        "bytes": ex.bytes,
        "unshippable": unshippable,
    }


def share_storages(views: dict) -> dict:
    """IPC handles of the exported storages (``views``). Share last: torch
    never frees a shared storage in this process, even with no consumer."""
    return {key: v.untyped_storage()._share_cuda_() for key, v in views.items()}


def _swap_param(old, new: nn.Parameter, attrs: dict) -> bool:
    """Move ``new``'s data into ``old``, keeping the Parameter object other
    references hold, with attributes ``attrs``. False: it cannot be swapped
    (another type, or extra references to its data)."""
    if type(old) is not nn.Parameter:
        return False
    try:
        torch.utils.swap_tensors(old, new)
    except Exception:  # noqa: BLE001 - the caller installs ``new`` instead
        return False
    old.__dict__.clear()
    old.__dict__.update(attrs)
    return True


def _install_param(obj: nn.Module, k: str, t: torch.Tensor, pattrs: dict) -> None:
    """Put ``t`` into parameter ``k``, keeping the Parameter object the model
    was built with (other references to it see the new data)."""
    new = nn.Parameter(t, requires_grad=False)
    old = obj._parameters.get(k)
    attrs = {**(vars(old) if old is not None else {}), **pattrs}
    if _swap_param(old, new, attrs):
        return
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

    for path, _, _, k, v in _entries(model):
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
        if not _swap_param(p, meta, attrs):
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
