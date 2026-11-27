# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Plugin host: named model seams and modules that enabled plugins may serve
(HOOK_API 3).

A seam is an op a model calls directly or a region of native code whose
consumer stays the model's. The model writes each seam's contract where it
calls it (``seam(SeamSpec(...))``) and calls it through a ``bind`` made at
build (hot paths) or through ``call`` (module-level functions). A whole
``nn.Module`` forward, or one of its methods, is taken over by path or by
class (``register_module``) with no seam at all. Ops AITER routes itself stay
on ``aiter.plugin``; one plan drives both halves.

Rules:
- Default deny. Installing a plugin makes it available; only a plan
  (``--plugin-recipe``) enables it. ``ATOM_PLUGIN_HOST=0``
  only disables; ``ATOM_PLUGIN_HOST_STRICT=1`` only adds checks.
- Native is untouched: a seam no plan owner registered binds to the native
  object itself, and a ``call`` costs one dict lookup.
- A seam may have ordered owners for disjoint cells. Their ``select`` is a
  pure metadata predicate, asked once per route-cache key (the seam's key
  inputs, the call's keyword arguments and the graph state); the first that
  accepts serves. Strict mode asks them all and refuses two accepts. Once
  ``impl`` runs its errors propagate; native is never retried.
- What ATOM knows about a call reaches a plugin only as a ``Capability``; every
  impl also gets ``valid_rows``, the real row count on the device (pad-M), or
  None when all rows are real. A decision made inside CUDA-graph capture is the
  capture's: replays reuse it for any real row count up to the bucket.
- Traced Python (``torch.compiler.is_compiling()``) is native; a seam inside a
  compiled region with an owner runs as one opaque op (``atom_plugin::span``)
  that decides on every execution.
- A plan is fixed for the process. To switch plans, restart the server
  (``ATOM_PRELOAD=1`` keeps the loaded weights on the GPUs across it).
- Decisions, not launches, are counted: a CUDA-graph replay runs no Python.
"""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import importlib
import logging
import re
import sys
import threading
import time
import types
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

logger = logging.getLogger("atom")

HOOK_API = 3
ENTRY_POINT_GROUP = "atom.layer_plugins"
AITER_SOURCE = "atom plugin plan"
GRAPHS = ("eager", "capture_safe")  # what a plugin may claim
MX = "mxfp8_e8m0_g32"  # e4m3 values, row-major group-32 e8m0 scales


@dataclass(frozen=True)
class SeamSpec:
    """One seam: its contract and how ATOM reaches it.

    ``inputs``/``kwargs`` are what the native and the impl both receive;
    ``returns`` names the result (one value for one name, nothing checked for
    none: in place); ``key`` the inputs whose layout (shape, dtype, strides,
    16-byte alignment; a scalar's value) keys the route cache; ``graph``
    where it runs (``eager``, ``capture_safe`` or ``opaque_op``: inside a
    compiled region, as one opaque op); ``parent`` the region it is nested in.
    A select may tell calls apart only by key inputs, keyword arguments and
    the capability; an unkeyed input is a per-call-site constant (weights) or
    a per-step index that must not decide the route.
    """

    name: str
    contract: int
    kind: str  # "op" | "region"
    form: str  # "bind" | "call"
    inputs: tuple[str, ...]
    kwargs: tuple[str, ...]
    returns: tuple[str, ...]
    key: tuple[str, ...]
    graph: str
    target: str
    parent: str | None = None
    boundary: str = "tensor"
    doc: str = ""
    boot_only: bool = False  # called once, while the model is built
    # Collectives an owner of this span must run itself. None declared: an
    # owner's ``owns_collectives`` is recorded, never required, so a plugin
    # written for either convention registers alike.
    collectives: tuple[str, ...] = ()
    # The module path pattern the call site is in (``layers.*.ffn.shared_experts``):
    # a recipe that takes over an enclosing module must plan this seam too.
    scope: str | None = None
    # Seams this span makes unreachable when its owner serves (alternative,
    # overlapping spans of one region that are not nested as ``parent``).
    covers: tuple[str, ...] = ()
    # Seams that must stay native while this one has an owner (a window that
    # spans another seam's native code, e.g. a cross-layer transition).
    excludes: tuple[str, ...] = ()


COLLECTIVES = ("tp_all_reduce", "ep_all_to_all", "dp_gather")


# No seam registry: a seam's contract is written where the model calls it
# (``bind(..., spec=)``, ``site(name, spec=)``, or ``seam(spec)`` next to a
# ``call``), so a model hard-codes its own seams in its own code. The host
# keeps what the built model declared and checks the plan against it once the
# model is built (``check_plan``).
_SEAM_SPECS: dict[str, SeamSpec] = {}
_SEAM_SOURCES: dict[str, str] = {}  # seam -> the call site that declared it
SEAMS = MappingProxyType(_SEAM_SPECS)
_NAME_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
# The host's own model-agnostic seam (frozen HOOK_API surface).
MODEL_INSTALL = "model.install"


def _core_specs() -> list[SeamSpec]:
    return [
        SeamSpec(
            MODEL_INSTALL,
            1,
            "region",
            "call",
            ("model", "atom_config", "drafter"),
            ("metadata_builder",),
            ("model",),
            (),
            "eager",
            "atom/model_engine/model_runner.py",
            doc=(
                "boot hook: after the load, before KV sizing and capture, the "
                "model to serve (an owner wraps or replaces it; native: the "
                "model's own installers)"
            ),
            boot_only=True,
        )
    ]


def boot_only() -> frozenset[str]:
    return frozenset(n for n, s in _SEAM_SPECS.items() if s.boot_only)


_TUPLE_FIELDS = (
    "inputs",
    "kwargs",
    "returns",
    "key",
    "collectives",
    "covers",
    "excludes",
)


def _spec_of(raw: Any) -> SeamSpec:
    if isinstance(raw, SeamSpec):
        return raw
    if isinstance(raw, Mapping):
        fields_ = dict(raw)
        for name in _TUPLE_FIELDS:
            if name in fields_:
                fields_[name] = tuple(fields_[name])
        return SeamSpec(**fields_)
    raise TypeError(f"a seam spec is a SeamSpec or a mapping, not {type(raw).__name__}")


def _spec_error(spec: SeamSpec) -> str | None:
    if not isinstance(spec.name, str) or not _NAME_RE.match(spec.name):
        return f"name {spec.name!r} is not dotted lower-case"
    if not isinstance(spec.contract, int) or spec.contract < 1:
        return "contract must be a positive int"
    if spec.kind not in ("op", "region"):
        return f"kind {spec.kind!r}"
    if spec.form not in ("bind", "call"):
        return f"form {spec.form!r}"
    if spec.graph not in ("eager", "capture_safe", "opaque_op"):
        return f"graph {spec.graph!r}"
    if not set(spec.key) <= set(spec.inputs):
        return "key must name inputs"
    if not set(spec.collectives) <= set(COLLECTIVES):
        return f"collectives {spec.collectives!r} not in {COLLECTIVES}"
    return None


def seam(spec: Any, *, source: str | None = None) -> str:
    """Declare a seam where the model calls it (frozen HOOK_API surface) and
    return its name. The same spec again (every layer's call site) is a no-op;
    another spec under the same name is refused (``SeamConflict``): a seam has
    one contract per process. Owners registered before the build are checked
    against it now."""
    spec = _spec_of(spec)
    where = source or spec.target or "?"
    error = _spec_error(spec)
    if error is not None:
        raise SeamConflict(f"{where}: seam {spec.name!r}: {error}")
    with _lock:
        have = _SEAM_SPECS.get(spec.name)
        if have is not None:
            if have != spec:
                raise SeamConflict(
                    f"{where}: seam {spec.name!r} conflicts with the spec "
                    f"{_SEAM_SOURCES.get(spec.name)} declared"
                )
            return spec.name
        _SEAM_SPECS[spec.name] = spec
        _SEAM_SOURCES[spec.name] = where
        for owner, route in _registered.get(spec.name, {}).items():
            _check_route(spec, owner, route)
    return spec.name


def _check_route(spec: SeamSpec, owner: str, route) -> None:
    """An owner registered before its seam was declared (the model is built
    after configure): its contract and form are checked against the spec."""
    if route.contract != spec.contract:
        raise PluginContractError(
            f"{owner}: registered {spec.name} contract {route.contract}, "
            f"the model declares {spec.contract}"
        )
    if set(spec.collectives) - set(route.collectives):
        raise PluginContractError(
            f"{owner}: {spec.name} contains {sorted(spec.collectives)}"
        )


@dataclass(frozen=True)
class Capability:
    """What ATOM tells a plugin about a call; it replaces private flags."""

    seam: str
    quant: str | None = None
    rows: int = 0  # padded leading dim of the first key input (M)
    captured: bool = False  # decided inside CUDA-graph capture
    bucket: int | None = None  # the captured pass's padded rows
    flags: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class Registration:
    seam: str
    owner: str
    accepted: bool
    reason: str


@dataclass(frozen=True)
class Decision:
    seam: str
    route: str  # "plugin" | "native"
    reason: str  # selected, declined, select-error, graph-unsafe, identity, ...
    owner: str | None = None
    declined: int = 0  # owners asked before it (all of them, when native)
    captured: bool = False
    bucket: int | None = None


class RegistrationRefused(ValueError):
    """A registration that can never be valid; ``reason`` is its short code."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


class SeamConflict(ValueError):
    """Two declarations of one seam disagree, or a spec is malformed."""


class PluginConflict(RuntimeError):
    """Strict mode: two owners accepted the same call."""


class PluginContractError(RuntimeError):
    """An owner's result does not have its seam's declared shape."""


class PluginDeclined(RuntimeError):
    """``on_decline: error``: every owner of a planned seam declined a call."""


@dataclass(frozen=True)
class _Route:
    owner: str
    impl: Callable[..., Any]
    select: Callable[..., Any] | None
    graph: str
    collectives: tuple[str, ...] = ()
    contract: int = 1


_lock = threading.RLock()
_plan = None  # atom.plugin_recipe.Plan
_requested: str | None = None  # the plan id configure was asked for
_kill_switch = False
_strict = False
_on_decline = "native"
_arm = "plugin"
_loading: str | None = None  # the entry point whose register() is running
_registered: dict[str, dict[str, _Route]] = {}  # seam -> owner -> route
_tables: dict[Any, dict[str, tuple[_Route, ...]]] = {}  # table -> seam -> routes
_active: dict[Any, dict[str, tuple[_Route, ...]]] = {}  # the same, for the arm
_owned: frozenset[str] = frozenset()  # seams with an owner in some table
_bound: set[str] = set()
_cache: dict[tuple, tuple[Decision, _Route | None, Capability]] = {}
_CACHE_LIMIT = 4096
_counts: Counter = Counter()
_refusals: list[Registration] = []
_last: Decision | None = None
_aiter_applied: tuple[str, ...] | None = None  # the enable list AITER last got


def api_satisfies(requires: str, api: int = HOOK_API) -> bool:
    """``">=3,<4"``-style clauses (``>=``, ``<``, ``==``) against ``api``."""
    for clause in (c.strip() for c in str(requires).split(",") if c.strip()):
        for op, test in ((">=", api.__ge__), ("==", api.__eq__), ("<", api.__lt__)):
            if clause.startswith(op):
                try:
                    if not test(int(clause[len(op) :])):
                        return False
                except ValueError:
                    return False
                break
        else:
            return False
    return True


def _refuse(seam, owner, reason, detail):
    _refusals.append(Registration(seam, owner, False, reason))
    logger.warning("plugin_host: %s refused for %s: %s", owner, seam, detail)
    raise RegistrationRefused(reason, detail)


def _planned(seam: str, owner: str) -> bool:
    if _plan is None:
        return False
    for table in _plan.tables.values():
        if table is None or owner in table.get(seam, ()):
            return True
    return False


def register(
    *,
    seam: str,
    contract: int,
    impl: Callable[..., Any],
    owner: str,
    select: Callable[..., Any] | None = None,
    graph: str = "capture_safe",
    requires_api: str = f">={HOOK_API},<{HOOK_API + 1}",
    owns_collectives: Iterable[str] = (),
) -> Registration:
    """Offer ``owner`` for ``seam``. Refusals that are bugs raise; an owner the
    plan does not enable here gets ``accepted=False``.

    ``owns_collectives``: the collectives the owner runs itself inside the
    span. A seam's own ``collectives`` must be covered; no seam declares any,
    so the argument is accepted and recorded from either kind of plugin."""
    with _lock:
        unknown = set(owns_collectives) - set(COLLECTIVES)
        if unknown:
            _refuse(seam, owner, "unknown-collective", f"{sorted(unknown)}")
        if not isinstance(seam, str) or not _NAME_RE.match(seam):
            _refuse(seam, owner, "unknown-seam", f"{seam!r} is not a seam name")
        # The model declares its seams when it is built, after this: a known
        # spec is checked here, an unknown one when the model declares it.
        spec = SEAMS.get(seam)
        if spec is not None:
            if set(spec.collectives) - set(owns_collectives):
                _refuse(
                    seam,
                    owner,
                    "collectives-not-owned",
                    f"{seam} contains {sorted(spec.collectives)}",
                )
            if contract != spec.contract:
                _refuse(
                    seam, owner, "contract-mismatch", f"{contract} != {spec.contract}"
                )
        if not api_satisfies(requires_api):
            _refuse(seam, owner, "api-mismatch", f"{requires_api} vs {HOOK_API}")
        if graph not in GRAPHS:
            _refuse(seam, owner, "graph-unsupported", f"{graph!r} not in {GRAPHS}")
        if not callable(impl) or (select is not None and not callable(select)):
            _refuse(seam, owner, "not-callable", "impl and select must be callable")
        if _loading is not None and owner != _loading:
            _refuse(seam, owner, "owner-mismatch", f"loading entry point {_loading}")
        if owner in _registered.get(seam, {}):
            _refuse(seam, owner, "duplicate", f"{owner} already serves {seam}")
        if _kill_switch or _plan is None or owner not in _plan.providers:
            reason = "kill-switch" if _kill_switch else "not-enabled"
            _refusals.append(Registration(seam, owner, False, reason))
            return _refusals[-1]
        if not _planned(seam, owner):
            _refusals.append(Registration(seam, owner, False, "not-planned"))
            return _refusals[-1]
        _registered.setdefault(seam, {})[owner] = _Route(
            owner, impl, select, graph, tuple(owns_collectives), contract
        )
        return Registration(seam, owner, True, "registered")


# -- module takeover ----------------------------------------------------------
#
# A recipe v2 may hand a whole `nn.Module` subtree to a plugin: ``modules``
# maps a slot (its path in the built model, `*` matching one path segment,
# e.g. ``layers.*.attn`` or ``draft.mtp.*.ffn``) to ordered owners. ATOM needs
# no seam for it and nothing in the model changes: after the load the module's
# ``forward`` is routed per call (`install_modules`), its parameters, buffers
# and names stay where they are, and the owner's callable reads the live
# module it was built for. Slots install deepest first, so an outer owner that
# declines a call falls through to the module's own forward, which reaches the
# inner owners: overlapping spans compose by construction (non-orthogonal
# plugins never need to know about each other).


@dataclass(frozen=True)
class _ModuleRoute:
    owner: str
    factory: Callable[..., Any]
    select: Callable[..., Any] | None
    graph: str
    fake: Callable[..., Any] | None


_module_registered: dict[str, dict[str, _ModuleRoute]] = {}  # slot -> owner -> route
_module_tables: dict[Any, dict[str, tuple[_ModuleRoute, ...]]] = {}
_module_active: dict[Any, dict[str, tuple[_ModuleRoute, ...]]] = {}
_install_roots: list[tuple[Any, str]] = []  # (model, path prefix), install order
_installed: list[Any] = []  # _ModuleRouter, in install order


def _planned_module(slot: str, owner: str) -> bool:
    if _plan is None:
        return False
    for table in getattr(_plan, "modules", {}).values():
        if table is None or owner in table.get(slot, ()):
            return True
    return False


def register_module(
    *,
    slot: str,
    factory: Callable[..., Any],
    owner: str,
    select: Callable[..., Any] | None = None,
    graph: str = "capture_safe",
    contract: int = 1,
    requires_api: str = f">={HOOK_API},<{HOOK_API + 1}",
    fake: Callable[..., Any] | None = None,
    method: str | None = None,
) -> Registration:
    """Offer ``owner`` for every module at ``slot`` (whole-subtree takeover).

    A slot is a module path (``layers.*.ffn``, ``*`` one segment) or a class,
    ``@<qualified.Class>``: every instance of that class (``isinstance``),
    wherever it sits, so one plugin serves the class in any model that uses
    it. ``method`` (or a ``slot`` written ``...:method``) takes over that
    method of the matched modules instead of ``forward``: any Python method
    boundary is a takeover point without a declared seam. The host installs and restores it,
    routes it per call and per capture bucket, and makes it one opaque op under
    compile, exactly as for ``forward``; the factory's callable is called as
    that method.

    ``factory(module, capability)`` runs once per matched module after the
    load, before warmup and capture, and returns the callable that serves it:
    called exactly as the module's ``forward``. ``select(capability, *args,
    **kwargs)`` is a pure per-call predicate (None: always); declining runs the
    module's own forward, inner owners included. ``fake`` gives the output of
    a tensor-only call as a compiled graph sees it (default: the module's own
    forward on fake inputs).
    """
    if method is not None:
        slot = f"{slot}:{method}"
    if not callable(factory) or (select is not None and not callable(select)):
        _refuse(slot, owner, "not-callable", "factory and select must be callable")
    with _lock:
        if not isinstance(slot, str) or not slot or not _SLOT_RE.match(slot):
            _refuse(slot, owner, "unknown-slot", f"{slot!r} is not a module path")
        if not api_satisfies(requires_api):
            _refuse(slot, owner, "api-mismatch", f"{requires_api} vs {HOOK_API}")
        if graph not in GRAPHS:
            _refuse(slot, owner, "graph-unsupported", f"{graph!r} not in {GRAPHS}")
        if _loading is not None and owner != _loading:
            _refuse(slot, owner, "owner-mismatch", f"loading entry point {_loading}")
        if owner in _module_registered.get(slot, {}):
            _refuse(slot, owner, "duplicate", f"{owner} already serves {slot}")
        if _kill_switch or _plan is None or owner not in _plan.providers:
            reason = "kill-switch" if _kill_switch else "not-enabled"
            _refusals.append(Registration(slot, owner, False, reason))
            return _refusals[-1]
        if not _planned_module(slot, owner):
            _refusals.append(Registration(slot, owner, False, "not-planned"))
            return _refusals[-1]
        _module_registered.setdefault(slot, {})[owner] = _ModuleRoute(
            owner, factory, select, graph, fake
        )
        return Registration(slot, owner, True, "registered")


# A module path (``*`` one segment) or ``@<qualified.Class>``, the module's
# ``forward``, or ``...:method`` (one of its methods).
_SLOT_RE = re.compile(
    r"^(@[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+"
    r"|[A-Za-z0-9_*]+(\.[A-Za-z0-9_*]+)*)"
    r"(:[A-Za-z_][A-Za-z0-9_]*)?$"
)


def _split_slot(slot: str) -> tuple[str, str]:
    path, _, method = slot.partition(":")
    return path, method or "forward"


def _is_class_slot(pattern: str) -> bool:
    return pattern.startswith("@")


@functools.cache
def _slot_class(pattern: str) -> type | None:
    """The class a ``@<qualified.Class>`` slot names; None when it does not
    import (no module of it can be in this model)."""
    module_name, _, name = pattern[1:].rpartition(".")
    try:
        cls = getattr(importlib.import_module(module_name), name)
    except (ImportError, AttributeError):
        return None
    return cls if isinstance(cls, type) else None


def _slot_matches(slot: str, path: str, module: Any = None) -> bool:
    if _is_class_slot(slot):
        cls = _slot_class(slot)
        return cls is not None and isinstance(module, cls)
    want, have = slot.split("."), path.split(".")
    return (
        bool(path)
        and len(want) == len(have)
        and all(fnmatch.fnmatchcase(h, w) for w, h in zip(want, have))
    )


def _module_route_tables(plan):
    from atom import plugin_recipe

    tables = {}
    for key, table in getattr(plan, "modules", {}).items():
        tables[key] = {}
        for slot, owners in (table or {}).items():
            missing = [o for o in owners if o not in _module_registered.get(slot, {})]
            if missing:
                raise plugin_recipe.RecipeRefused(
                    "recipe-coverage-miss", f"{slot}: {missing} did not register"
                )
            tables[key][slot] = tuple(_module_registered[slot][o] for o in owners)
    return tables


def _set_module_arm(arm: str, allowed) -> None:
    global _module_active
    _module_active = {
        key: {
            slot: tuple(r for r in routes if allowed is None or r.owner in allowed)
            for slot, routes in table.items()
        }
        for key, table in _module_tables.items()
    }


def _module_routes(slots, captured: bool, bucket: int | None):
    if captured:
        table = _module_active.get(bucket, _module_active.get("any"))
    else:
        table = _module_active.get("eager", _module_active.get("any"))
    if not table:
        return ()
    return tuple(r for slot in slots for r in table.get(slot, ()))


class _ModuleRouter:
    """A taken-over module's ``forward`` (or ``attr``, a method): the planned
    owners, else its own."""

    def __init__(self, path, slots, module, native, attr="forward"):
        self.path, self.slots, self.module, self.native = path, slots, module, native
        self.attr = attr
        self.impls: dict[str, Callable[..., Any]] = {}
        self._op = None

    def build(self) -> None:
        """Every owner's callable for this module, once, outside any capture."""
        cap = Capability(
            f"module:{self.slots[0]}",
            flags=MappingProxyType(
                {"path": self.path, "module": self.module, "method": self.attr}
            ),
        )
        routes = {
            r.owner: r
            for table in _module_tables.values()
            for slot in self.slots
            for r in table.get(slot, ())
        }
        for owner, route in routes.items():
            try:
                self.impls[owner] = route.factory(self.module, cap)
            except Exception as error:
                raise RuntimeError(
                    f"plugin {owner!r} could not take over {self.path}: "
                    f"{type(error).__name__}: {error}"
                ) from error

    def __call__(self, *args, **kwargs):
        if torch.compiler.is_compiling():
            if args and not kwargs and all(isinstance(a, torch.Tensor) for a in args):
                return self._opaque()(*args)
            return self.native(*args, **kwargs)
        return self.run(*args, **kwargs)

    def _opaque(self):
        if self._op is None:
            fake = next(
                (
                    r.fake
                    for table in _module_tables.values()
                    for slot in self.slots
                    for r in table.get(slot, ())
                    if r.fake is not None
                ),
                self._native_fake,
            )
            key = self.path if self.attr == "forward" else f"{self.path}:{self.attr}"
            self._op = span_n(f"module:{key}", self.run, fake)
        return self._op

    def _native_fake(self, *xs):
        """The default fake of a taken-over module: its own forward on the
        fake inputs (an owner returns what the module returns), else a tensor
        like the first input."""
        from torch._guards import detect_fake_mode
        from torch.nn.utils.stateless import _reparametrize_module

        try:
            mode = detect_fake_mode(xs)
            if mode is None:
                out = self.native(*xs)
            else:
                state = {
                    name: mode.from_tensor(t)
                    for name, t in (
                        *self.module.named_parameters(),
                        *self.module.named_buffers(),
                    )
                }
                with _reparametrize_module(self.module, state):
                    out = self.native(*xs)
        except Exception:  # noqa: BLE001 -- a forward a fake cannot run
            return torch.empty_like(xs[0])
        return out if isinstance(out, torch.Tensor) else torch.empty_like(xs[0])

    def run(self, *args, **kwargs):
        name = f"module:{self.slots[0]}"
        if _arm == "native":
            _record(Decision(name, "native", "arm-native"))
            return self.native(*args, **kwargs)
        captured, bucket = _graph_state()
        routes = _module_routes(self.slots, captured, bucket)
        if not routes:
            return self.native(*args, **kwargs)
        cap = Capability(
            name,
            rows=_rows_of(args[0]) if args else 0,
            captured=captured,
            bucket=bucket,
            flags=MappingProxyType(
                {"path": self.path, "module": self.module, "method": self.attr}
            ),
        )
        declined = 0
        for route in routes:
            if captured and route.graph != "capture_safe":
                declined += 1
                continue
            if route.select is not None:
                try:
                    accepted = bool(route.select(cap, *args, **kwargs))
                except Exception as error:  # noqa: BLE001 -- a broken select declines
                    logger.warning(
                        "plugin_host: %s select failed: %s", route.owner, error
                    )
                    accepted = False
                if not accepted:
                    declined += 1
                    continue
            if _arm == "identity":
                _record(Decision(name, "plugin", "identity", route.owner, declined))
                return self.native(*args, **kwargs)
            chosen = Decision(
                name, "plugin", "selected", route.owner, declined, captured, bucket
            )
            _record(chosen)
            _note_route(chosen)
            return self.impls[route.owner](*args, **kwargs)
        missed = Decision(name, "native", "declined", None, declined, captured, bucket)
        _record(missed)
        _note_route(missed)
        if _on_decline == "error":
            raise PluginDeclined(f"{name}: every owner declined")
        return self.native(*args, **kwargs)


def _module_slots() -> list[str]:
    seen: dict[str, None] = {}
    for table in _module_tables.values():
        for slot in table:
            seen[slot] = None
    return list(seen)


def _install_into(model, prefix: str) -> list[str]:
    slots = _module_slots()
    if not slots:
        return []
    found = []
    for path, module in model.named_modules(prefix=prefix):
        by_attr: dict[str, list[str]] = {}
        for s in slots:
            pattern, attr = _split_slot(s)
            if _slot_matches(pattern, path, module):
                by_attr.setdefault(attr, []).append(s)
        for attr, matched in by_attr.items():
            if not callable(getattr(module, attr, None)):
                raise TypeError(f"{path} has no method {attr!r} to take over")
            found.append((path, module, attr, matched))
    # Deepest first: an outer owner's fallback is the module's own forward,
    # which reaches the inner owners already installed.
    found.sort(key=lambda item: -item[0].count("."))
    installed = []
    for path, module, attr, matched in found:
        router = _ModuleRouter(
            path, tuple(matched), module, getattr(module, attr), attr
        )
        router.build()
        setattr(module, attr, router)
        _installed.append(router)
        installed.append(path if attr == "forward" else f"{path}:{attr}")
    return installed


def _uninstall_modules() -> None:
    while _installed:
        router = _installed.pop()
        if router.module.__dict__.get(router.attr) is router:
            delattr(router.module, router.attr)
            if getattr(router.native, "__self__", None) is not router.module:
                setattr(router.module, router.attr, router.native)


def install_modules(model, *, prefix: str = "") -> list[str]:
    """After the load (and the drafter's, with ``prefix="draft"``), before
    warmup and capture: route the planned slots' modules of ``model``.
    Returns the module paths taken over."""
    with _lock:
        if not any(m is model and p == prefix for m, p in _install_roots):
            _install_roots.append((model, prefix))
        paths = _install_into(model, prefix)
    if paths:
        logger.info("plugin_host: modules taken over: %s", paths)
    return paths


def install_built(model, drafter=None) -> None:
    """The runner's step after both loads: the takeovers of ``model`` and of
    the drafter's model (``prefix="draft"``), then the plan against the seams
    the built models declared (`check_plan`: refuses the start)."""
    install_modules(model)
    if drafter is not None:
        install_modules(drafter.model, prefix="draft")
    check_plan()


# -- opaque op of N tensors -----------------------------------------------------

_spans_n: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {}
SPAN_N_OP = "atom_plugin::span_n"


def _span_n(xs: list[torch.Tensor], key: str) -> torch.Tensor:
    return sys.modules[__name__]._spans_n[key][0](*xs)


def _span_n_fake(xs: list[torch.Tensor], key: str) -> torch.Tensor:
    return sys.modules[__name__]._spans_n[key][1](*xs)


def _define_span_n_op() -> None:
    namespace, name = SPAN_N_OP.split("::")
    if hasattr(getattr(torch.ops, namespace), name):
        return
    library = torch.library.Library(namespace, "FRAGMENT")
    schema = torch.library.infer_schema(_span_n, mutates_args=())
    library.define(name + schema)
    library.impl(name, _span_n, "CompositeExplicitAutograd")
    library._register_fake(name, _span_n_fake)
    _define_span_n_op.library = library


def span_n(
    key: str, run: Callable[..., torch.Tensor], fake: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """``run(*tensors) -> tensor`` as one opaque op a compiled graph keeps as a
    call (``atom_plugin::span_n``): its body runs eagerly on every execution,
    graph capture included. ``key`` names the body."""
    _define_span_n_op()
    _spans_n[key] = (run, fake)
    op = torch.ops.atom_plugin.span_n.default
    return lambda *xs: op(list(xs), key)


# -- recipe options and the plugin model ---------------------------------------


def plugin_model_active() -> bool:
    """Whether this process builds the plugin variant of a model."""
    return not _kill_switch and _plan is not None


def _module_file(module: Any) -> Path | None:
    """The file a module was imported from, resolved; None for a fileless one."""
    file = getattr(module, "__file__", None)
    if not isinstance(file, str) or not file:
        return None
    try:
        return Path(file).resolve()
    except OSError:
        return None


_CODE_SUFFIXES = frozenset({".py", ".so", ".h", ".hpp", ".hh", ".hxx"})


def plugin_code_digest(roots: Iterable[str]) -> str:
    """A stable fingerprint of the plugin code under ``roots`` (what a
    recipe's ``requires.plugins`` pins), in a fixed, documented layout so an
    out-of-tree tool can compute exactly the same string:

    - recurse the roots; regular files only; skip ``__pycache__``;
    - keep the extensions ``.py .so .h .hpp .hh .hxx``, case-insensitive;
    - de-duplicate by resolved absolute path (the first root wins);
    - sort records by the path relative to its root, POSIX form;
    - per record, feed ``<relative>\\0<sha256 hex of the file bytes>\\n``.

    Returns ``"sha256:" + hexdigest()``. A missing root or an unreadable file
    raises, never returns a digest for code it did not read.
    """
    resolved_roots = []
    for root in roots:
        path = Path(root).resolve()
        if not path.is_dir():
            raise ValueError(f"plugin code root {root!r} is not a directory")
        if path not in resolved_roots:
            resolved_roots.append(path)
    if not resolved_roots:
        raise ValueError("plugin_code_digest needs at least one root")
    digest = hashlib.sha256()
    seen: set[Path] = set()
    entries: list[tuple[str, Path]] = []
    for root_path in resolved_roots:
        for path in sorted(root_path.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() not in _CODE_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            entries.append((path.relative_to(root_path).as_posix(), path))
    for relative, path in sorted(entries, key=lambda item: item[0]):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


# The model ATOM builds while a plugin plan is in place is the recipe's
# ``model_options.target_model_cls`` / ``draft_model_cls`` (a plugin package's
# model, with its seams in it); without them ATOM's own model, unchanged.


def _recipe_model_options(config: Any):
    """The recipe's model options from ``config.plugin_source`` in a process that
    builds the model without configuring the host (the weight-cache daemon:
    it must build the class the servers it serves build); None without one."""
    from atom.plugin_recipe import RecipeSource
    from atom.utils import envs

    source = getattr(config, "plugin_source", None)
    if not envs.ATOM_PLUGIN_HOST or not isinstance(source, RecipeSource):
        return None
    plan = source.resolve()
    return None if plan is None else plan.model_options


def model_class_for(arch: str, default: str, config: Any = None) -> str:
    """The qualname to build for ``arch``: the recipe's
    ``model_options.target_model_cls``, else ATOM's."""
    if plugin_model_active():
        return model_options().target_model_cls or default
    options = None if _plan is not None else _recipe_model_options(config)
    return getattr(options, "target_model_cls", None) or default


def draft_model_class_for(arch: str, default: str, config: Any = None) -> str:
    """As `model_class_for`, for a draft architecture."""
    if plugin_model_active():
        return model_options().draft_model_cls or default
    options = None if _plan is not None else _recipe_model_options(config)
    return getattr(options, "draft_model_cls", None) or default


def model_classes_key(config: Any) -> dict[str, str | None] | None:
    """The model classes a plugin recipe builds in place of ATOM's
    (``model_options.target_model_cls`` / ``draft_model_cls``), for the
    weight-cache key: None without a recipe, without such a class, or under the
    plugin kill switch, where ATOM builds its own. A daemon's weights are never
    mapped into another model class."""
    options = _recipe_model_options(config)
    if options is None or (
        options.target_model_cls is None and options.draft_model_cls is None
    ):
        return None
    return {"target": options.target_model_cls, "draft": options.draft_model_cls}


def new_model_build() -> None:
    """A model is about to be built (again): forget the opaque spans of an
    earlier build of this process (a weight-cache client that fell back to a
    disk load builds the model twice), so its layers register anew."""
    with _lock:
        _spans.clear()
        _spans_n.clear()


def model_options():
    """The plan's ``model_options`` (`plugin_recipe.ModelOptions`): build-time
    model choices; the defaults without a plan. Any arm, native included,
    builds the same model: an A/B arm changes routes, not topology."""
    from atom.plugin_recipe import ModelOptions

    plan = _plan
    return getattr(plan, "model_options", None) or ModelOptions()


def plugin_options(name: str) -> Mapping[str, Any]:
    """The plan's ``plugin_options[name]`` (frozen HOOK_API surface): options a
    recipe passes to a plugin package unread by ATOM; empty without a plan."""
    plan = _plan
    raw = getattr(plan, "plugin_options", None) or {}
    return MappingProxyType(dict(raw.get(name, {})))


def owns(seam: str, rows: int | None = None) -> bool:
    """Whether the active arm has an owner of ``seam`` in any table. A native
    region that spans ``seam`` (fused attention spans qk_norm and attend)
    yields to it, so a finer plugin is never bypassed."""
    return any(table.get(seam) for table in _active.values())


def _entry_points() -> dict[str, metadata.EntryPoint]:
    return {ep.name: ep for ep in metadata.entry_points(group=ENTRY_POINT_GROUP)}


# -- provider modules and their prepare hooks -----------------------------------


def _entry_point_module(entry_point: metadata.EntryPoint) -> types.ModuleType | None:
    """The module an entry point's ``attr`` is defined in (loading it if needed).

    ``prepare(device)``/``close()`` are module-level conventions, so the module
    is found from the entry point value (``package.module:register``), not from
    the register callable: an entry point may export a wrapper defined
    elsewhere.
    """
    loaded = entry_point.load()
    if isinstance(loaded, types.ModuleType):
        return loaded
    module_name = getattr(entry_point, "module", None)
    if not isinstance(module_name, str) or not module_name:
        module_name = getattr(loaded, "__module__", None)
    if not isinstance(module_name, str) or not module_name:
        return None
    module = sys.modules.get(module_name)
    if module is None:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - the caller reports the provider
            return None
    return module


def _jsonable(value: Any) -> Any:
    """A prepare() return value as plain JSON data (the reply carries it)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _prepare_providers(plan: Any, device: Any) -> tuple[list[str], dict[str, Any]]:
    """Call the optional module-level ``prepare(device)`` of every planned provider.

    A provider declares it by exporting ``prepare`` next to its ``register``;
    the call is where a plugin loads and warms its versioned native library.
    Its return value is reported under ``native``. A
    prepare that raises is a refusal, never a silent native fallback.
    """
    if plan is None:
        return [], {}
    available = _entry_points()
    prepared: list[str] = []
    native: dict[str, Any] = {}
    for name in plan.providers:
        entry_point = available.get(name)
        if entry_point is None:
            continue  # an AITER-only provider: prepared below
        try:
            module = _entry_point_module(entry_point)
        except Exception as error:
            raise RuntimeError(
                f"plugin provider {name!r} could not be loaded for "
                f"prepare(device): {type(error).__name__}: {error}"
            ) from error
        prepare = None if module is None else getattr(module, "prepare", None)
        if not callable(prepare):
            continue
        try:
            result = prepare(device)
        except Exception as error:
            raise RuntimeError(
                f"plugin provider {name!r} prepare(device={device!r}) failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        prepared.append(name)
        native[name] = _jsonable(result)
        logger.info(
            "plugin_host: provider %s prepared (device=%r): %s",
            name,
            device,
            native[name],
        )
    # The plan's AITER providers: their ``prepare(device)`` (an AITER Provider
    # builds and warms its kernels there), before any capture, like the rest.
    aiter_points = _aiter_entry_points()
    for name in getattr(plan, "aiter", ()):
        if name in prepared or name not in aiter_points:
            continue
        try:
            provider = aiter_points[name].load()
            prepare = getattr(provider, "prepare", None)
            result = prepare(device) if callable(prepare) else None
        except Exception as error:
            raise RuntimeError(
                f"AITER provider {name!r} prepare(device={device!r}) failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        if callable(prepare):
            prepared.append(name)
            native[name] = _jsonable(result)
    return prepared, native


def _aiter_entry_points() -> dict[str, metadata.EntryPoint]:
    return {ep.name: ep for ep in metadata.entry_points(group="aiter.plugins")}


# -- plugin pins -------------------------------------------------------------


def provider_code_roots(name: str) -> tuple[Path, ...]:
    """The source trees a provider's code digest covers: its entry point
    module's package directory, plus the absolute paths it lists in an optional
    module-level ``PLUGIN_CODE_ROOTS`` (e.g. its kernel sources)."""
    points = {**_aiter_entry_points(), **_entry_points()}
    if name not in points:
        raise LookupError(f"plugin provider {name!r} is not installed")
    module = _entry_point_module(points[name])
    path = None if module is None else _module_file(module)
    if path is None:
        raise LookupError(f"plugin provider {name!r} has no source on disk")
    roots = [path.parent]
    for extra in getattr(module, "PLUGIN_CODE_ROOTS", ()) or ():
        roots.append(Path(extra))
    return tuple(dict.fromkeys(r.resolve() for r in roots))


def provider_digest(name: str) -> str:
    """``sha256:`` digest of a provider's code (``provider_code_roots``, the
    ``plugin_code_digest`` layout): what a recipe's ``requires.plugins`` pins."""
    return plugin_code_digest(str(r) for r in provider_code_roots(name))


def _check_plugin_pins(plan) -> None:
    """A recipe's ``requires.plugins``: each pinned provider's code must hash
    to the pin, else the start is refused (no ATOM or AITER version is pinned)."""
    from atom import plugin_recipe

    pins = {} if plan is None else dict(plan.requires.get("plugins", {}) or {})
    for name, expected in pins.items():
        try:
            got = provider_digest(name)
        except (LookupError, OSError, ValueError) as error:
            raise plugin_recipe.RecipeRefused(
                "recipe-plugin-drift", f"{name}: {error}"
            ) from error
        if got != expected:
            raise plugin_recipe.RecipeRefused(
                "recipe-plugin-drift", f"{name}: code {got} != pinned {expected}"
            )


def configure(
    plan=None,
    *,
    arm: str = "plugin",
    config: Any = None,
    kill_switch: bool | None = None,
    strict: bool | None = None,
) -> None:
    """Load and route ``plan`` (a ``plugin_recipe.Plan``, or the
    ``RecipeSource`` a Config carries) before the model is built.

    A plan is all or nothing: a plugin that fails to load, or a planned owner
    that did not register, refuses the start. The plan's AITER providers go to
    ``aiter.plugin.configure`` when AITER has it. A plan is fixed for the
    process: to change it, restart (``ATOM_PRELOAD=1`` keeps the weights)."""
    from atom import plugin_recipe
    from atom.utils import envs

    global _plan, _kill_switch, _strict, _on_decline, _loading, _owned, _requested
    if isinstance(plan, plugin_recipe.RecipeSource):
        plan = plan.resolve()  # Config carries the recipes; each process resolves
    requested = None if plan is None else plan.plan_id
    if _bound:
        # A second runner in this process may restate the plan the built binds
        # were made with; any other plan would leave them on the old one.
        if requested != _requested:
            raise RuntimeError("plugin_host.configure after model build")
        set_arm(arm if _plan is not None else "plugin")
        return
    _requested = requested
    if plan is not None and config is not None:
        plugin_recipe.check_runtime(plan, config)
    kill = not envs.ATOM_PLUGIN_HOST if kill_switch is None else kill_switch
    with _lock:
        _plan = None if kill else plan
        _kill_switch = kill
        _strict = (
            envs.ATOM_PLUGIN_HOST_STRICT or bool(plan and plan.strict_overlap)
            if strict is None
            else strict
        )
        _on_decline = plan.on_decline if plan is not None else "native"
        _registered.clear()
        _tables.clear()
        _module_registered.clear()
        _module_tables.clear()
        _refusals.clear()
        _spans.clear()
        if _plan is not None:
            available = _entry_points()
            for name in _plan.providers:
                if name not in available:
                    continue  # an AITER-only provider
                _loading = name
                try:
                    available[name].load()(sys.modules[__name__])
                except Exception as error:
                    raise plugin_recipe.RecipeRefused(
                        "recipe-plugin-drift", f"{name} failed to load: {error}"
                    ) from error
                finally:
                    _loading = None
            _tables.update(_route_tables(_plan))
            _module_tables.update(_module_route_tables(_plan))
        _owned = frozenset(s for table in _tables.values() for s in table)
        set_arm(arm if _plan is not None else "plugin", sync_aiter=False)
    _check_plugin_pins(_plan)
    _apply_aiter(force=True)
    if plan is not None:
        logger.info("plugin_host: %s", status())


def configure_runner(config: Any) -> None:
    """`configure` for the runner: the plan and the arm ride the engine
    ``Config`` (``plugin_source`` / ``plugin_arm``); a config without them (a
    stand-in) is native."""
    configure(
        getattr(config, "plugin_source", None),
        arm=getattr(config, "plugin_arm", "plugin"),
        config=config,
    )


def _route_tables(plan) -> dict[Any, dict[str, tuple[_Route, ...]]]:
    from atom import plugin_recipe

    tables = {}
    for key, table in plan.tables.items():
        tables[key] = {}
        for seam, owners in table.items():
            missing = [o for o in owners if o not in _registered.get(seam, {})]
            if missing:
                raise plugin_recipe.RecipeRefused(
                    "recipe-coverage-miss", f"{seam}: {missing} did not register"
                )
            tables[key][seam] = tuple(_registered[seam][o] for o in owners)
    return tables


def set_arm(arm: str, *, sync_aiter: bool = True) -> None:
    """Which planned providers serve: ``plugin`` (all), ``native`` (none, the
    plugin topology kept), ``identity`` (decide as plugin, run native) or a
    recipe arm. Only the route table changes; binds and opaque ops stay. The
    AITER providers of the plan follow the same arm (``_aiter_for_arm``)."""
    global _arm, _active
    with _lock:
        arms = _plan.arms if _plan is not None else {}
        if arm not in ("plugin", "native", "identity") and arm not in arms:
            raise ValueError(f"unknown arm {arm!r}")
        allowed = None if arm in ("plugin", "identity") else set(arms.get(arm, ()))
        _active = {
            key: {
                seam: tuple(r for r in routes if allowed is None or r.owner in allowed)
                for seam, routes in table.items()
            }
            for key, table in _tables.items()
        }
        _set_module_arm(arm, allowed)
        _arm = arm
        _cache.clear()
    if sync_aiter:
        _apply_aiter(force=False)


def _aiter_for_arm() -> tuple[str, ...]:
    """The plan's AITER providers the arm lets serve: all on ``plugin`` (and on
    ``identity``, which AITER has no form of: its providers keep serving), none
    on ``native``, a named arm's members otherwise."""
    if _kill_switch or _plan is None:
        return ()
    names = tuple(_plan.aiter)
    if _arm in ("plugin", "identity"):
        return names
    if _arm == "native":
        return ()
    members = set(_plan.arms.get(_arm, ()))
    return tuple(n for n in names if n in members)


def _apply_aiter(*, force: bool) -> None:
    """Hand AITER the arm's enable list (``aiter.plugin.configure``): always on
    ``configure``, on an arm change only when the list changed."""
    global _aiter_applied
    names = _aiter_for_arm()
    if not force and names == _aiter_applied:
        return
    _configure_aiter(names)
    _aiter_applied = names


def prepare_plan(device: Any = None) -> dict[str, Any]:
    """Let the boot plan's providers prepare native code once, after the build.

    Every planned provider that exports a module-level ``prepare(device)`` is
    asked once; that is where a plugin loads and warms its versioned native
    library before warmup and CUDA-graph capture freeze the call sites. A
    ``prepare`` that raises refuses the start with the provider named -- a
    provider that advertised prepare never silently falls back to native.

    Returns ``{"prepared": [provider, ...], "native": {provider: <return>}}``;
    ``native`` carries what each prepare returned (JSON-safe) so the caller can
    log or report it. No plan, or no prepare hook: both lists are empty.
    """
    with _lock:
        if _plan is None:
            return {"prepared": [], "native": {}}
        prepared, native = _prepare_providers(_plan, device)
    logger.info("plugin_host: boot plan prepared providers: %s", prepared)
    return {"prepared": prepared, "native": native}


# -- dispatch -----------------------------------------------------------------


_capture_rows = ContextVar("atom_plugin_capture_rows", default=None)


@contextmanager
def capture_bucket(rows: int):
    """Explicit rows of a child graph, not the enclosing target forward's rows."""
    marker = _capture_rows.set(rows)
    try:
        yield
    finally:
        _capture_rows.reset(marker)


def has_capture_owner(seam: str, rows: int, *, explicit=False) -> bool:
    """Check the installed plan, even in its native arm, to keep A/B topology."""
    table = _tables.get(rows, {} if explicit else _tables.get("any", {}))
    return any(route.graph == "capture_safe" for route in table.get(seam, ()))


@dataclass(frozen=True)
class Context:
    """Where a call runs, for code a seam's capability does not reach (an
    AITER provider, a module owner's inner kernels): inside a CUDA-graph
    capture or not, the capture bucket (padded rows), and the rows of the
    step that are real, as a device tensor a captured kernel reads."""

    captured: bool
    bucket: int | None
    valid_rows: torch.Tensor | None


def context(rows: int | None = None) -> Context:
    """The execution context of the current forward (frozen HOOK_API surface).
    ``rows``: the padded rows of the caller's tensor, for ``valid_rows``."""
    captured, bucket = _graph_state()
    padded = rows if rows is not None else bucket
    return Context(captured, bucket, None if padded is None else _valid_rows(padded))


def _graph_state() -> tuple[bool, int | None]:
    """``(captured, bucket)``: inside a CUDA-graph capture, the pass's padded
    rows (``running_tokens``). A target decode graph of batch ``bs`` runs
    ``bs * q_len`` rows (a DSpark verify: ``bs * (1 + k)``); a draft pass,
    recorded by its own graph outside ``in_hipgraph``, ``bs`` times its
    tokens per sequence."""
    rows = _capture_rows.get()
    if rows is not None:
        return True, rows
    from atom.utils.forward_context import get_forward_context

    try:
        forward = get_forward_context()
    except Exception:  # noqa: BLE001 -- outside an ATOM forward
        return _stream_capturing(), None
    if not (getattr(forward, "in_hipgraph", False) or _stream_capturing()):
        return False, None
    return True, getattr(getattr(forward, "context", None), "running_tokens", None)


def _stream_capturing() -> bool:
    try:
        return torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001 -- a build without CUDA
        return False


def _valid_rows(rows: int) -> torch.Tensor | None:
    from atom.utils.forward_context import pass_scheduled_tokens_gpu

    try:
        return pass_scheduled_tokens_gpu(rows)
    except Exception:  # noqa: BLE001 -- outside an ATOM forward: all rows real
        return None


def _routes(seam: str, captured: bool, bucket: int | None) -> tuple[_Route, ...]:
    if captured:
        table = _active.get(bucket, _active.get("any"))
    else:
        table = _active.get("eager", _active.get("any"))
    return () if table is None else table.get(seam, ())


_warned_misses: set[tuple[str, int | None]] = set()


def _no_table(seam: str, captured: bool, bucket: int | None) -> None:
    """A seam with owners elsewhere has none for this pass: never silent."""
    _record(Decision(seam, "native", "no-table", captured=captured, bucket=bucket))
    if _strict and captured and (seam, bucket) not in _warned_misses:
        declared = _declared_rows()
        if declared:
            _warned_misses.add((seam, bucket))
            logger.warning(
                "plugin_host: %s has no table for captured rows %s (the plan "
                "declares rows %s); it runs native in that graph",
                seam,
                bucket,
                declared,
            )


def _declared_rows() -> list[int]:
    """The capture-bucket keys (padded rows) the plan's tables declare."""
    return sorted(k for k in _tables if isinstance(k, int))


def _key_of(value: Any) -> Any:
    """A key input's part of the route-cache key: what a select may read."""
    if isinstance(value, torch.Tensor):
        # Layout, not just shape: a sliced view has the same shape and other
        # strides, and a kernel's select may check strides and alignment.
        return (
            tuple(value.shape),
            value.dtype,
            value.stride(),
            value.data_ptr() % 16 if value.device.type != "meta" else None,
        )
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape), getattr(value, "dtype", None)
    if isinstance(value, (list, tuple)):
        return tuple(_key_of(v) for v in value)
    if value is None or isinstance(value, (bool, int, float, str, torch.dtype)):
        return value
    return type(value).__name__


def _rows_of(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    shape = getattr(value, "shape", None)
    return int(shape[0]) if shape else 0


def _decide(seam, routes, cap, inputs, kwargs, strict) -> tuple[Decision, Any]:
    chosen, declined, errors, unsafe = None, 0, 0, 0
    for route in routes:
        if cap.captured and route.graph != "capture_safe":
            unsafe += 1
            declined += chosen is None
            continue
        answer = True
        if route.select is not None:
            try:
                answer = bool(route.select(cap, *inputs, **kwargs))
            except Exception as error:  # noqa: BLE001 -- a broken select declines
                logger.warning("plugin_host: %s select failed: %s", route.owner, error)
                answer, errors = False, errors + 1
        if not answer:
            declined += chosen is None
            continue
        if chosen is not None:
            raise PluginConflict(
                f"{seam}: {chosen.owner} and {route.owner} both accept this call"
            )
        chosen = route
        if not strict:
            break
    where = {"captured": cap.captured, "bucket": cap.bucket}
    if chosen is None:
        reason = "select-error" if errors else "declined"
        if unsafe and unsafe == declined and not errors:
            reason = "graph-unsafe"
        return Decision(seam, "native", reason, None, declined, **where), None
    return Decision(seam, "plugin", "selected", chosen.owner, declined, **where), chosen


# -- the plan against the built model -----------------------------------------


def _scope_inside(scope: str, slot: str) -> bool:
    """Whether a seam's call-site module path lies in (or at) module ``slot``."""
    have, want = scope.split("."), slot.split(".")
    return len(have) >= len(want) and all(
        h == w or fnmatch.fnmatchcase(h, w) or fnmatch.fnmatchcase(w, h)
        for h, w in zip(have, want)
    )


def check_plan() -> dict[str, Any]:
    """Once the model (and its drafter) are built and their modules installed:
    the recipe against the seams the model declared at its call sites.

    Refuses (``RecipeRefused``) a planned seam the model never declared
    (``recipe-seam-unbound``), a required contract the model does not have
    (``recipe-contract-mismatch``), an owned seam whose ``excludes`` is owned
    too (``recipe-excludes``), a taken-over module whose inner seams the recipe
    does not plan (``recipe-inner-unplanned``), and, under
    ``shadow_policy: error``, an owner an outer owner (``parent`` / ``covers``)
    makes unreachable (``recipe-shadow``). Returns the shadowed owners."""
    from atom import plugin_recipe

    plan = _plan
    if plan is None or _kill_switch:
        return {"shadowed": []}
    owned: dict[str, set[str]] = {}
    for table in _tables.values():
        for name, routes in table.items():
            if routes:
                owned.setdefault(name, set()).update(r.owner for r in routes)
    planned = set(owned) | set(getattr(plan, "native_seams", ()))
    unbound = sorted(s for s in planned if s not in SEAMS)
    if unbound:
        raise plugin_recipe.RecipeRefused(
            "recipe-seam-unbound",
            f"the built model has no call site for {unbound}",
        )
    for name, contract in dict(plan.requires.get("seams", {}) or {}).items():
        spec = SEAMS.get(name)
        if spec is None:
            raise plugin_recipe.RecipeRefused(
                "recipe-seam-unbound", f"requires.seams: {name} has no call site"
            )
        if contract != spec.contract:
            raise plugin_recipe.RecipeRefused(
                "recipe-contract-mismatch",
                f"{name}: recipe {contract} != model {spec.contract}",
            )
    for name in owned:
        clash = [e for e in SEAMS[name].excludes if e in owned]
        if clash:
            raise plugin_recipe.RecipeRefused(
                "recipe-excludes", f"{name} needs {clash} native, the recipe owns them"
            )
    shadowed = []
    for name, owners in owned.items():
        spec = SEAMS[name]
        outers = [spec.parent] if spec.parent else []
        outers += [s for s, sp in SEAMS.items() if name in sp.covers]
        for outer in outers:
            if outer not in owned:
                continue
            for owner in sorted(owners - owned[outer]):
                shadowed.append((name, owner, outer))
    if shadowed and getattr(plan, "shadow_policy", "nested") == "error":
        raise plugin_recipe.RecipeRefused(
            "recipe-shadow", f"outer owners make unreachable: {shadowed}"
        )
    for table in _module_tables.values():
        for slot, routes in table.items():
            pattern, method = _split_slot(slot)
            if not routes or method != "forward":
                continue  # a method takeover replaces one method, no inner seam
            # A class slot is where its instances were installed.
            paths = (
                [r.path for r in _installed if slot in r.slots and r.attr == method]
                if _is_class_slot(pattern)
                else [pattern]
            )
            missing = sorted(
                n
                for n, s in SEAMS.items()
                if s.scope is not None
                and any(_scope_inside(s.scope, path) for path in paths)
                and n not in planned
            )
            if missing:
                raise plugin_recipe.RecipeRefused(
                    "recipe-inner-unplanned",
                    f"modules[{slot}] takes over a module whose seams {missing} "
                    "the recipe does not plan (list owners, or [] for native)",
                )
    if shadowed:
        logger.info(
            "plugin_host: owners an outer owner may make unreachable "
            "(reached only where it declines or calls them): %s",
            shadowed,
        )
    return {"shadowed": shadowed}


# -- recipe reachability: what each seam / module actually routed to ----------

# (seam, "captured" | "eager") -> owner (or "native") -> buckets (captured) or
# the rows seen (eager, capped).
_route_log: dict[tuple[str, str], dict[str, set]] = {}
_ROUTE_LOG_ROWS = 64


def _note_route(decision: Decision) -> None:
    mode = "captured" if decision.captured else "eager"
    owners = _route_log.setdefault((decision.seam, mode), {})
    seen = owners.setdefault(decision.owner or "native", set())
    if decision.captured or len(seen) < _ROUTE_LOG_ROWS:
        seen.add(decision.bucket)


def route_report() -> dict[str, Any]:
    """What the plan's seams and module slots routed to, captured (per bucket)
    and eager, and the recipe owners that no call reached
    (``planned_but_unreached``): the check that captured graphs and eager
    prefill follow the recipe."""
    routed: dict[str, dict[str, dict[str, list]]] = {}
    reached: set[tuple[str, str]] = set()
    for (seam, mode), owners in sorted(_route_log.items()):
        for owner, seen in owners.items():
            routed.setdefault(seam, {}).setdefault(mode, {})[owner] = sorted(
                (x for x in seen if x is not None), key=str
            )
            reached.add((seam, owner))
    planned: dict[str, set[str]] = {}
    for table in _tables.values():
        for seam, routes in table.items():
            planned.setdefault(seam, set()).update(r.owner for r in routes)
    for table in _module_tables.values():
        for slot, routes in table.items():
            planned.setdefault(f"module:{slot}", set()).update(r.owner for r in routes)
    unreached = sorted(
        f"{seam}:{owner}"
        for seam, owners in planned.items()
        for owner in owners
        if (seam, owner) not in reached
    )
    unplanned = sorted(
        f"{seam}:{owner}"
        for seam, owner in reached
        if owner != "native" and owner not in planned.get(seam, ())
    )
    return {
        "routed": routed,
        "planned_but_unreached": unreached,
        "unplanned": unplanned,
    }


# -- one weight preparation for every configuration of the process ------------

# Every captured graph, prefill and decode of a process (and every arm) shares
# one set of prepared weights: an owner may keep private copies, never change
# the model's own parameters / buffers after the boot preparation.
_weight_marks: dict[str, tuple] = {}
_weight_models: list[tuple[Any, str]] = []


_SAMPLES = 512  # elements a tensor's content checksum reads


def _content_sums(tensors: list[torch.Tensor]) -> list[int]:
    """An order-sensitive checksum of a strided sample of each tensor's bytes
    (one device-to-host copy for all of them). A re-preparation (repacking,
    re-quantizing, padding) rewrites every element, so a sample sees it; the
    in-place version counter cannot be used (custom ops that only read a weight
    still bump it)."""
    sums = []
    for tensor in tensors:
        flat = tensor.detach().reshape(-1)
        n = flat.numel()
        if n == 0:
            sums.append(torch.zeros((), dtype=torch.int64, device=tensor.device))
            continue
        step = max(1, n // _SAMPLES)
        raw = flat[::step][:_SAMPLES].contiguous().view(torch.uint8).to(torch.int64)
        weights = torch.arange(1, raw.numel() + 1, device=raw.device, dtype=torch.int64)
        sums.append((raw * weights).sum())
    if not sums:
        return []
    return [int(v) for v in torch.stack([s.to("cpu") for s in sums]).tolist()]


def _weights_now() -> dict[str, tuple]:
    named = []
    for model, prefix in _weight_models:
        for name, tensor in (*model.named_parameters(), *model.named_buffers()):
            if isinstance(tensor, torch.Tensor) and tensor.device.type != "meta":
                named.append((f"{prefix}.{name}" if prefix else name, tensor))
    unique = list({id(t): t for _, t in named}.values())  # a tied weight once
    sums = dict(zip((id(t) for t in unique), _content_sums(unique)))
    return {
        name: (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tensor.dtype,
            tuple(tensor.stride()),
            sums[id(tensor)],
        )
        for name, tensor in named
    }


def freeze_weights(*models, prefixes: Iterable[str] = ()) -> int:
    """After the load, the boot hook, module installs and ``prepare_plan``:
    record every parameter and buffer (storage, layout, in-place version) of
    ``models``. ``weights_changed`` names any that changed since. Returns the
    number recorded."""
    names = list(prefixes) or [""] * len(models)
    _weight_models[:] = [(m, p) for m, p in zip(models, names) if m is not None]
    _weight_marks.clear()
    _weight_marks.update(_weights_now())
    return len(_weight_marks)


def prepare_built(device: Any, model, drafter=None) -> None:
    """The runner's step before warmup and capture: the plan's providers
    prepare their native code (`prepare_plan`), then every parameter and
    buffer of ``model`` and the drafter's model is recorded (`freeze_weights`)."""
    prepare_plan(device)
    freeze_weights(model, getattr(drafter, "model", None), prefixes=("", "draft"))


def weights_changed() -> list[str]:
    """The recorded parameters / buffers whose storage, layout or contents
    (in-place version) changed, appeared or went since ``freeze_weights``: a
    configuration that re-prepared shared weights its siblings also read."""
    if not _weight_models:
        return []
    now = _weights_now()
    return sorted(
        name
        for name in set(now) | set(_weight_marks)
        if now.get(name) != _weight_marks.get(name)
    )


def log_route_report(stage: str) -> dict[str, Any]:
    """Log ``route_report`` (after capture: every bucket's route is fixed)."""
    report = route_report()
    changed = weights_changed()
    report["weights_changed"] = changed
    if changed:
        now = _weights_now()
        fields_ = ("storage", "shape", "dtype", "stride", "contents")
        detail = {
            name: [
                f
                for f, a_, b_ in zip(
                    fields_, _weight_marks.get(name, ()), now.get(name, ())
                )
                if a_ != b_
            ]
            or ["added or removed"]
            for name in changed[:32]
        }
        logger.error(
            "plugin_host: shared weights changed after the boot preparation "
            "(%s; every graph, prefill and decode must share one): %s",
            stage,
            detail,
        )
        if _strict:
            raise PluginContractError(f"shared weights changed: {changed[:8]}")
    if _plan is not None:
        logger.info("plugin_host: routes after %s: %s", stage, report["routed"])
        if report["planned_but_unreached"]:
            logger.warning(
                "plugin_host: planned owners no call reached after %s: %s",
                stage,
                report["planned_but_unreached"],
            )
        if report["unplanned"]:
            logger.error(
                "plugin_host: owners outside the plan: %s", report["unplanned"]
            )
    return report


# -- cross-rank agreement for owners that run collectives ---------------------


def _agree(decision: Decision, route):
    """A captured decision whose candidates run collectives: every TP rank must
    take the same owner, or a rank waits in a collective the others never
    enter. The ranks exchange their choice (CPU group); any disagreement makes
    all of them run native for that call site and bucket."""
    try:
        import torch.distributed as dist
        from aiter.dist.parallel_state import get_tp_group

        group = get_tp_group()
        if group.world_size <= 1 or group.cpu_group is None:
            return decision, route
        mine = decision.owner
        agreed = [None] * group.world_size
        dist.all_gather_object(agreed, mine, group=group.cpu_group)
    except Exception:  # noqa: BLE001 -- no TP group (CPU tests, single rank)
        return decision, route
    if all(owner == mine for owner in agreed):
        return decision, route
    logger.warning(
        "plugin_host: %s bucket %s: TP ranks chose %s; all run native",
        decision.seam,
        decision.bucket,
        agreed,
    )
    return (
        Decision(
            decision.seam,
            "native",
            "rank-disagree",
            None,
            decision.declined,
            decision.captured,
            decision.bucket,
        ),
        None,
    )


def _record(decision: Decision) -> None:
    global _last
    _last = decision
    _counts[(decision.seam, decision.route, decision.reason, decision.owner)] += 1


def _record_traced(ctx) -> None:
    # Runs once, while Dynamo traces `_dispatch`, reading its own local.
    seam = ctx.get_local("seam").as_python_constant()
    _record(Decision(seam, "native", "traced"))


try:
    from torch._dynamo.comptime import comptime as _comptime
except Exception:  # noqa: BLE001 -- recording a traced decision is best effort
    _comptime = None


def _dispatch(seam, native, capability, inputs, kwargs):
    if torch.compiler.is_compiling():
        if _comptime is not None:
            _comptime(_record_traced)
        return native(*inputs, **kwargs)
    if _arm == "native":
        _record(Decision(seam, "native", "arm-native"))
        return native(*inputs, **kwargs)
    captured, bucket = _graph_state()
    routes = _routes(seam, captured, bucket)
    if not routes:
        _no_table(seam, captured, bucket)
        return native(*inputs, **kwargs)
    spec = SEAMS[seam]
    named = dict(zip(spec.inputs, inputs))
    key = (
        seam,
        # A bound seam's capability is its call site's own (per layer: flags.module),
        # so a decision and its Capability are cached per call site, never shared.
        id(capability),
        captured,
        bucket,
        tuple(_key_of(named.get(k)) for k in spec.key),
        tuple((k, _key_of(v)) for k, v in sorted(kwargs.items())),
    )
    hit = _cache.get(key)
    if hit is None:
        static = capability(*inputs, **kwargs) if callable(capability) else capability
        static = static or {}
        cap = Capability(
            seam,
            quant=static.get("quant"),
            rows=_rows_of(named.get(spec.key[0])) if spec.key else 0,
            captured=captured,
            bucket=bucket,
            flags=MappingProxyType(dict(static.get("flags", {}))),
        )
        decision, route = _decide(seam, routes, cap, inputs, kwargs, _strict)
        if captured and route is not None and any(r.collectives for r in routes):
            decision, route = _agree(decision, route)
        _note_route(decision)
        if len(_cache) >= _CACHE_LIMIT:  # keys with per-step values grow it
            _cache.clear()
        hit = (decision, route, cap)
        # A captured bucket's eager warmup (in_hipgraph, not yet recording)
        # shares its key with the recording right after it. An owner that
        # declined there because it was not warm yet (its kernels load on that
        # warmup) must be asked again by the recording, not refused from cache.
        cold = (
            route is None
            and captured
            and torch.cuda.is_initialized()
            and not _stream_capturing()
        )
        if not cold:
            _cache[key] = hit
    decision, route, cap = hit
    if route is None:
        _record(decision)
        if _on_decline == "error":
            raise PluginDeclined(f"{seam}: every owner declined ({decision.reason})")
        return native(*inputs, **kwargs)
    if _arm == "identity":
        _record(Decision(seam, "plugin", "identity", route.owner, decision.declined))
        return native(*inputs, **kwargs)
    _record(decision)
    result = route.impl(
        *inputs, capability=cap, valid_rows=_valid_rows(cap.rows), **kwargs
    )
    arity = len(spec.returns)
    if arity > 1 and not (isinstance(result, tuple) and len(result) == arity):
        raise PluginContractError(
            f"{route.owner} returned {type(result).__name__} for {seam}, "
            f"expected {spec.returns}"
        )
    return result


class Router:
    """A bound seam that has a planned owner (``bind`` returns native else)."""

    __slots__ = ("capability", "native", "seam")

    def __init__(self, seam, native, capability):
        self.seam, self.native, self.capability = seam, native, capability

    def __call__(self, *inputs: Any, **kwargs: Any) -> Any:
        return _dispatch(self.seam, self.native, self.capability, inputs, kwargs)

    def explain(self, *inputs: Any, **kwargs: Any) -> Decision:
        return explain(self.seam, *inputs, _capability=self.capability, **kwargs)


def bind(
    seam: str,
    native: Callable[..., Any],
    *,
    capability: Mapping[str, Any] | Callable[..., Any] | None = None,
    layer: str | None = None,
    fake: Callable[..., torch.Tensor] | None = None,
    spec: Any = None,
) -> Callable[..., Any]:
    """At model build (after ``configure``): how the call site runs ``seam``.
    ``spec`` is the seam's contract, written here at the call site (a model
    hard-codes its own seams); a seam declared before may omit it.

    No planned owner: ``native`` itself, so the call site is today's call.
    Else a ``Router``; an ``opaque_op`` seam given ``layer`` and ``fake`` is
    wrapped in one opaque op so a compiled graph decides per execution.
    ``capability`` is ``{"quant": ..., "flags": {...}}`` or a callable taking
    the call's inputs and returning one.
    """
    if spec is not None:
        seam = globals()["seam"](spec)  # the module function, not the argument
    spec = SEAMS.get(seam)
    if spec is None:
        raise KeyError(f"bind of {seam!r} without its spec (pass spec=)")
    if spec.form != "bind":
        raise KeyError(f"seam {seam!r} is a {spec.form} seam, not a bound one")
    _bound.add(seam)
    if seam not in _owned:
        return native
    router = Router(seam, native, capability)
    if spec.graph == "opaque_op" and layer is not None and fake is not None:
        return span_op(
            seam, layer, lambda x, extra: router(x), lambda x, extra: fake(x)
        )
    return router


_call_capability: dict[str, Callable[..., Mapping[str, Any]]] = {
    "v41.moe.topk": lambda *a, score_func=None, **kw: {
        "flags": {"score_func": score_func}
    },
    "v41.op.quant_fp8_g32": lambda *a, **kw: {"quant": MX},
    "v41.attn.index_topk": lambda *a, **kw: {"flags": {"packed": True}},
    "v41.attn.attend": lambda *a, unified_kv_rope=None, **kw: {
        "flags": {"kv_layout": "bf16" if unified_kv_rope is None else "fp8_2buff"}
    },
}


def call(seam: str, native: Callable[..., Any], *inputs: Any, **kwargs: Any) -> Any:
    """``native(*inputs, **kwargs)``, unless a planned owner of ``seam`` serves."""
    if seam not in _owned:
        return native(*inputs, **kwargs)
    return _dispatch(seam, native, _call_capability.get(seam), inputs, kwargs)


def site(
    name: str, spec: Any = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Declare a whole module-level function as the native of call seam
    ``name``; ``spec`` is the seam's contract, written here."""
    if spec is not None:
        seam(spec)
    spec = SEAMS.get(name)
    if spec is not None and spec.form != "call":
        raise KeyError(f"seam {name!r} is a {spec.form} seam, not a call seam")

    def decorate(native):
        @functools.wraps(native)
        def wrapper(*args, **kwargs):
            return call(name, native, *args, **kwargs)

        return wrapper

    return decorate


# -- opaque op ----------------------------------------------------------------

# Seam spans a compiled graph contains: layer -> (run, fake). See `span_op`.
_spans: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {}
SPAN_OP = "atom_plugin::span"


def _span(x: torch.Tensor, extra: torch.Tensor | None, layer: str) -> torch.Tensor:
    # Through sys.modules: the op outlives a re-import of this module.
    return sys.modules[__name__]._spans[layer][0](x, extra)


def _span_fake(x: torch.Tensor, extra: torch.Tensor | None, layer: str) -> torch.Tensor:
    return sys.modules[__name__]._spans[layer][1](x, extra)


def _define_span_op() -> None:
    namespace, name = SPAN_OP.split("::")
    if hasattr(getattr(torch.ops, namespace), name):
        return
    library = torch.library.Library(namespace, "FRAGMENT")
    schema = torch.library.infer_schema(_span, mutates_args=())
    library.define(name + schema)
    library.impl(name, _span, "CompositeExplicitAutograd")
    library._register_fake(name, _span_fake)
    _define_span_op.library = library  # the registration lives as long as it


def span_op(
    seam: str,
    layer: str,
    run: Callable[..., torch.Tensor],
    fake: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """At model build: ``run(x, extra)`` as a compiled graph should see it.

    No planned owner of ``seam``: ``run`` itself, inline, the graph ATOM has
    without plugins. Else one opaque op (``atom_plugin::span``, shapes from
    ``fake``) whose body runs ``run`` eagerly, where the seam's owners are
    asked on every execution.
    """
    _bound.add(seam)
    if seam not in _owned:
        return run
    if layer in _spans:
        raise ValueError(f"span layer name {layer!r} is not unique")
    _define_span_op()
    _spans[layer] = (run, fake)
    op = torch.ops.atom_plugin.span.default
    return lambda x, extra=None: op(x, extra, layer)


# -- diagnostics --------------------------------------------------------------


def explain(seam: str, *inputs: Any, captured=False, bucket=None, **kwargs) -> Decision:
    """What a call would decide now, without running a kernel or recording."""
    capability = kwargs.pop("_capability", _call_capability.get(seam))
    routes = _routes(seam, captured, bucket)
    if not routes:
        reason = "kill-switch" if _kill_switch else "no-owner"
        return Decision(seam, "native", reason, captured=captured, bucket=bucket)
    spec = SEAMS[seam]
    named = dict(zip(spec.inputs, inputs))
    static = (
        capability(*inputs, **kwargs) if callable(capability) else capability
    ) or {}
    cap = Capability(
        seam,
        quant=static.get("quant"),
        rows=_rows_of(named.get(spec.key[0])) if spec.key else 0,
        captured=captured,
        bucket=bucket,
        flags=MappingProxyType(dict(static.get("flags", {}))),
    )
    return _decide(seam, routes, cap, inputs, kwargs, _strict)[0]


def owners(seam: str, table: Any = "eager") -> tuple[str, ...]:
    """The order a seam's owners are asked in, for a table (``eager``,
    ``any`` or a capture bucket) under the current arm."""
    if table == "eager":
        routes = _routes(seam, False, None)
    elif table == "any":
        routes = _active.get("any", {}).get(seam, ())
    else:
        routes = _routes(seam, True, table)
    return tuple(r.owner for r in routes)


def last_decision() -> Decision | None:
    return _last


def decision_counts() -> Counter:
    """Decisions so far, keyed ``(seam, route, reason, owner)``."""
    return Counter(_counts)


def route_proof(seam: str, owner: str, n: int = 1) -> int:
    """Assert ``owner`` served ``seam`` at least ``n`` times; the count."""
    served = _counts[(seam, "plugin", "selected", owner)]
    if served < n:
        raise AssertionError(f"{seam}: {owner} served {served} < {n} calls")
    return served


def status() -> dict[str, Any]:
    return {
        "hook_api": HOOK_API,
        "plan_id": None if _plan is None else _plan.plan_id,
        "strategy": None if _plan is None else _plan.strategy,
        "arm": _arm,
        "kill_switch": _kill_switch,
        "strict": _strict,
        "on_decline": _on_decline,
        "seams": {
            seam: {
                "tables": {
                    str(key): tuple(r.owner for r in table.get(seam, ()))
                    for key, table in _active.items()
                    if seam in table
                }
            }
            for seam in sorted(_owned)
        },
        # A captured pass routes by its padded rows; these are the ones planned.
        "capture_rows": _declared_rows(),
        "span_ops": tuple(_spans),
        "bound": sorted(_bound),
        "refusals": tuple(_refusals),
        "modules": {
            slot: {
                str(key): tuple(r.owner for r in table.get(slot, ()))
                for key, table in _module_active.items()
                if slot in table
            }
            for slot in _module_slots()
        },
        "modules_installed": [router.path for router in _installed],
        "plugin_model": plugin_model_active(),
        "aiter": {"enabled": list(_aiter_applied or ()), "status": _aiter_status()},
        "routes": route_report(),
    }


def _aiter_status() -> Any:
    try:
        from aiter import plugin as aiter_plugin
    except Exception:  # noqa: BLE001 -- absent or broken
        return None
    status_fn = getattr(aiter_plugin, "status", None)
    try:
        full = status_fn() if callable(status_fn) else None
    except Exception as error:  # noqa: BLE001 -- status is best effort
        return f"error: {type(error).__name__}: {error}"
    if not isinstance(full, Mapping):
        return None
    # The serving view only: what is enabled and whether routing is on.
    keep = ("enabled", "excluded", "kill_switch", "config_source")
    return _jsonable({k: full.get(k) for k in keep if k in full})


def overhead_probe(
    seam: str, native: Callable[..., Any], inputs: tuple, n: int = 10000
) -> dict[str, float]:
    """µs per call: native direct, and through a Router on the native and
    plugin arms (benchmark use; restores the arm)."""

    def timed(fn) -> float:
        start = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - start) * 1e6 / n

    router = Router(seam, native, _call_capability.get(seam))
    previous = _arm
    result = {"native_direct_us": timed(lambda: native(*inputs))}
    try:
        for arm in ("native", "plugin"):
            set_arm(arm)
            result[f"router_{arm}_arm_us"] = timed(lambda: router(*inputs))
    finally:
        set_arm(previous)
    return result


def _reset_for_testing() -> None:
    """Forget the plan, the binds and the record (tests only)."""
    global _plan, _owned, _last, _kill_switch, _strict, _on_decline, _requested
    with _lock:
        _uninstall_modules()
        _install_roots.clear()
        _bound.clear()
        _registered.clear()
        _tables.clear()
        _module_registered.clear()
        _module_tables.clear()
        _spans.clear()
        _spans_n.clear()
        _refusals.clear()
        _counts.clear()
        _route_log.clear()
        _weight_marks.clear()

        _weight_models.clear()
        _warned_misses.clear()
        _plan, _owned, _last, _requested = None, frozenset(), None, None
        _kill_switch, _strict, _on_decline = False, False, "native"
        set_arm("plugin", sync_aiter=False)
        global _aiter_applied
        _aiter_applied = None


def _configure_aiter(names: Iterable[str]) -> bool:
    """The AITER half of the plan; False when AITER has no ``aiter.plugin``
    (an older build), which leaves AITER routing untouched."""
    try:
        from aiter import plugin as aiter_plugin
    except Exception:  # noqa: BLE001 -- absent or broken: nothing to drive
        return False
    configure_aiter = getattr(aiter_plugin, "configure", None)
    if not callable(configure_aiter):
        return False
    configure_aiter(source=AITER_SOURCE, enable=list(names))
    return True


# The host's own seam is declared from import on.
for _spec in _core_specs():
    seam(_spec, source="atom.plugin_host")
