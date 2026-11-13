# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Values: what pickles by value, process-independent renderings and
fingerprints, tensor content digests, and the walk over a model's state."""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import hashlib
import importlib
import json
import logging
import sys
import types
from typing import Any

import numpy as np
import torch
from torch import nn

logger = logging.getLogger("atom")

_TAG = "[weight-cache]"

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
# The objects hanging off modules that the walk follows, and how deep.
_DIFF_PACKAGES, _DIFF_DEPTH = ("atom.", "aiter.", "transformers."), 2

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


class _Miss(Exception):
    """The cache cannot be used; the message says why."""


class _Unshippable(Exception):
    """A value that cannot be sent to another process."""


def _is_native_enum(v: Any) -> bool:
    """A pybind11 enum value (e.g. aiter's QuantType): pickles by value."""
    t = type(v)
    return (
        hasattr(t, "__members__")
        and hasattr(v, "name")
        and any(b.__name__ == "pybind11_object" for b in t.__mro__)
    )


def _import_qual(module: str, qual: str) -> Any:
    obj = importlib.import_module(module)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


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


_PLAIN_TYPES = (
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
)


def _is_plain(v: Any, depth: int = 0) -> bool:
    """A value that pickles by value and means the same in another process."""
    if v is None or isinstance(v, _PLAIN_TYPES) or _is_native_enum(v):
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
    # hf_config.to_dict() can hold whatever ATOM set on it.
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


def _digest_sums(t: torch.Tensor):
    """(bytes, s1, s2) of a tensor's raw bytes with the sums left on its
    device, or a fixed string for meta and empty tensors."""
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
    for k, i in enumerate(range(0, x.numel(), _DIGEST_CHUNK)):
        c = x[i : i + _DIGEST_CHUNK]
        if c.dtype != torch.int64:
            c = c.to(torch.int64)
        s1 = s1 + c.sum()
        # Weighted by the position in the chunk and by the chunk, so data
        # moved between chunks changes the digest too.
        s2 = s2 + (c * w[: c.numel()]).sum() * (k + 1)
    return n, s1, s2


def _digests(tensors) -> list[str]:
    """Exact content digests (two position-weighted sums over the raw bytes),
    with one host sync per device."""
    sums = [_digest_sums(t) for t in tensors]
    out = [s if isinstance(s, str) else "" for s in sums]
    by_device: dict = {}
    for i, s in enumerate(sums):
        if not isinstance(s, str):
            by_device.setdefault(s[1].device, []).append(i)
    mask = (1 << 64) - 1
    for idx in by_device.values():
        vals = torch.stack([v for i in idx for v in sums[i][1:]]).tolist()
        for j, i in enumerate(idx):
            out[i] = f"{sums[i][0]}:{vals[2 * j] & mask:x}:{vals[2 * j + 1] & mask:x}"
    return out


def _digest(t: torch.Tensor) -> str:
    return _digests([t])[0]


def release_digest_buffers() -> None:
    """Drop the digest weights (128 MiB per device); the caller empties the cache."""
    _DIGEST_WEIGHTS.clear()


# ---------------------------------------------------------------------------
# The walk over a model's state
# ---------------------------------------------------------------------------


def _recursable(v: Any) -> bool:
    t = type(v)
    mod = t.__module__ or ""
    if isinstance(
        v,
        (nn.Module, torch.Tensor, enum.Enum, type, types.ModuleType) + _FUNCTION_TYPES,
    ):
        return False
    if (
        not hasattr(v, "__dict__")
        or not mod.startswith(_DIFF_PACKAGES)
        or mod.startswith(_NO_RECURSE)
    ):
        return False
    return not (mod == "atom.config" and t.__name__ in _CONFIG_CLASSES)


def config_objects(config) -> list[tuple[str, Any]]:
    """(path, object) for the config and the config objects under it."""
    roots = [("config", config)]
    hf = getattr(config, "hf_config", None)
    if hf is not None:
        roots.append(("hf_config", hf))
    out, seen = [], set()
    while roots:
        path, obj = roots.pop()
        if id(obj) in seen or not hasattr(obj, "__dict__") or isinstance(obj, type):
            continue
        seen.add(id(obj))
        out.append((path, obj))
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
    return out


def _walk_objects(model: nn.Module):
    """(path, obj, is_module) for every module and the objects reachable from
    their attributes (``_DIFF_DEPTH`` levels). Paths: ``mod.path`` for
    modules, ``mod.path::attr::attr`` below them. Each object appears once."""
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
            if id(v) in seen or not _recursable(v):
                continue
            seen.add(id(v))
            yield p, v, False
            if d < _DIFF_DEPTH:
                queue += [(f"{p}::{k}", x, d + 1) for k, x in vars(v).items()]


def _attrs(obj: Any, is_module: bool) -> dict:
    return {
        k: v
        for k, v in vars(obj).items()
        if not (is_module and k in _MODULE_INTERNALS) and not isinstance(v, nn.Module)
    }


def _entries(model: nn.Module):
    """(path, obj, kind, name, value) for every parameter, buffer and
    attribute in the walk; kind is "param", "buffer" or "attr"."""
    for path, obj, is_module in _walk_objects(model):
        if is_module:
            for k, p in list(obj._parameters.items()):
                yield path, obj, "param", k, p
            for k, b in list(obj._buffers.items()):
                yield path, obj, "buffer", k, b
        for k, v in _attrs(obj, is_module).items():
            yield path, obj, "attr", k, v


def _resolve(model: nn.Module, path: str):
    mod_path, _, attr_path = path.partition("::")
    obj = model.get_submodule(mod_path) if mod_path else model
    for name in filter(None, attr_path.split("::")):
        obj = vars(obj)[name]
    return obj


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
    if not content:
        return ("O", t)
    if isinstance(v, (dict, set, list, tuple)):  # one changed in place changes size
        return ("O", id(v), t, len(v))
    return ("O", id(v), t)


def key_fp(v: Any):
    """Process-independent fingerprint of a value a load read: plain values
    by value, functions by name, objects by type; integer tensors (index
    maps) by content, floating ones (often uninitialized workspaces) by
    shape and dtype."""
    if isinstance(v, torch.Tensor):
        if v.is_floating_point() or v.is_complex() or v.is_meta:
            return ("T", tuple(v.shape), str(v.dtype))
        return ("T", tuple(v.shape), str(v.dtype), _digest(v))
    return _fp(v, content=False)
